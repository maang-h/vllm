# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
N-gram Proposer for Speculative Decoding (推测解码的N-gram提议器)

这个模块实现了基于N-gram匹配的推测解码策略：
- 在已生成的token序列中查找匹配当前上下文的N-gram
- 如果找到匹配，使用匹配后面的tokens作为候选（draft tokens）
- 这些候选tokens会被并行验证，加速生成过程

核心思想：
如果上下文中出现了"the quick brown"，在prompt的其他地方也出现过，
那么后面很可能跟着"fox"。我们可以直接提议"fox"作为候选。

优势：
- 无需额外模型（零成本）
- 对重复性文本效果好（如代码、格式化文档）
- 使用Numba JIT加速，性能高
"""
import os

import numpy as np
from numba import get_num_threads, jit, njit, prange, set_num_threads

from vllm.config import VllmConfig


class NgramProposer:
    """
    N-gram 提议器 - 基于历史token匹配的推测解码器
    
    工作原理：
    1. 在已生成的token序列中查找与当前上下文匹配的N-gram
    2. 提取匹配位置后面的k个tokens作为候选
    3. 这些候选会被并行验证，如果正确则跳过生成过程
    
    示例：
        已生成: "Hello world! Hello"
        当前上下文后缀: "Hello"
        找到匹配: 位置0的"Hello"
        提议: " world!" (匹配后的tokens)
    """
    
    def __init__(self, vllm_config: VllmConfig):
        """
        初始化N-gram提议器
        
        Args:
            vllm_config: vLLM配置，包含推测解码相关参数
        """
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # ============================================================
        # 1. N-gram匹配参数
        # ============================================================
        # 最小N-gram长度：匹配的最短token数
        # 示例：min_n=3 表示至少匹配3个连续的token
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        
        # 最大N-gram长度：匹配的最长token数
        # 示例：max_n=10 表示最多匹配10个连续的token
        # 限制：防止匹配过长导致内存消耗过大
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        
        # 提议的候选token数量：匹配后提取的token数
        # 示例：k=5 表示在找到匹配后，提取后续5个token作为候选
        # 权衡：k越大，命中时收益越大，但验证失败的概率也越高
        self.k = vllm_config.speculative_config.num_speculative_tokens
        
        # 模型最大长度：防止超出模型容量
        self.max_model_len = vllm_config.model_config.max_model_len

        # ============================================================
        # 2. 预分配缓冲区（避免重复内存分配）
        # ============================================================
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        
        # 存储每个请求的候选tokens
        # 形状: [max_num_seqs, k]
        # 示例：valid_ngram_draft[0] = [123, 456, 789, 0, 0] 表示请求0有3个候选token
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        
        # 存储每个请求实际的候选token数量
        # 形状: [max_num_seqs]
        # 示例：valid_ngram_num_drafts[0] = 3 表示请求0有3个有效候选
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)

        # ============================================================
        # 3. 多线程配置（Numba并行加速）
        # ============================================================
        # 启用多线程的token数阈值
        # 当batch中总token数 >= 8192时，才使用多线程
        # 原因：小batch时多线程开销大于收益
        self.num_tokens_threshold = 8192
        
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        
        # 计算可用的Numba线程数
        if cpu_count:
            # 除以2：只使用物理核心，避免超线程
            # 原因：超线程对计算密集型任务收益不大
            # 上限设为1：当前版本暂不启用多线程
            # TODO(ekagra-ranjan): 当实现TP并行化后，将上限从1提升到8
            # 原因：避免与其他组件（tokenization、结构化输出）竞争CPU资源
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            
            # 除以tp_size：确保每个TP rank都有线程
            # 原因：所有TP rank都会运行N-gram proposer
            self.num_numba_thread_available //= tp_size
        else:
            # 无法获取CPU数量时，使用单线程
            self.num_numba_thread_available = 1

        # ============================================================
        # 4. 预热Numba JIT编译（首次调用触发编译）
        # ============================================================
        # 使用dummy数据触发JIT编译
        # 目的：避免首次真实调用时的编译延迟（约1秒）
        # 策略：在初始化时完成编译，推理时直接使用编译后的代码
        self.propose(
            [[]] * 1024,  # 1024个空的采样token列表
            [""] * 1024,  # 1024个空的请求ID
            np.zeros(1024, dtype=np.int32),  # 1024个token计数（都为0）
            np.zeros((1024, self.max_model_len), dtype=np.int32),  # token IDs
            set(),  # 空的不支持推测解码的请求集合
        )

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """
        批量N-gram提议（使用Numba加速）
        
        这是核心的批量处理方法，使用Numba JIT编译和并行化来加速N-gram匹配。
        
        工作流程：
        1. 动态调整线程数（根据batch大小）
        2. 并行处理所有需要N-gram提议的请求
        3. 收集每个请求的候选tokens
        
        Args:
            num_requests: batch中的总请求数
            valid_ngram_requests: 
                需要N-gram提议的请求索引列表
                示例：[0, 2, 5] 表示第0、2、5个请求需要提议
            num_tokens_no_spec:
                不含推测token的实际token数，形状 (batch_size,)
                示例：[100, 150, 200] 表示3个请求分别有100、150、200个token
            token_ids_cpu:
                所有请求的token IDs，形状 (batch_size, max_model_len)
                每一行是一个请求的完整token序列

        Returns:
            list[list[int]]:
                每个请求的候选token列表
                示例：[[123, 456], [], [789, 101, 102]]
                     表示请求0有2个候选，请求1无候选，请求2有3个候选
        """
        draft_token_ids: list[list[int]] = []

        # ============================================================
        # 1. 只有在有有效请求时才执行（避免Numba错误）
        # ============================================================
        # 避免调用numba函数时传入空列表，会导致错误：
        # ValueError: cannot compute fingerprint of empty list
        if num_ngram_requests := len(valid_ngram_requests):
            # 保存原始线程数，处理完后恢复
            original_num_numba_threads = get_num_threads()
            
            # --------------------------------------------------------
            # 1.1 动态调整线程数（性能优化）
            # --------------------------------------------------------
            # 计算batch中的总token数
            total_tokens = np.sum(num_tokens_no_spec)
            
            if total_tokens >= self.num_tokens_threshold:
                # 大batch：启用多线程
                # 线程数 = min(可用线程数, 请求数)
                # 原因：请求数少时，过多线程反而降低效率
                final_num_threads = max(
                    1, min(self.num_numba_thread_available, num_ngram_requests)
                )
                set_num_threads(final_num_threads)
            else:
                # 小batch：使用单线程
                # 原因：多线程开销 > 并行收益
                set_num_threads(1)

            # --------------------------------------------------------
            # 1.2 调用Numba加速的批量提议函数
            # --------------------------------------------------------
            batch_propose_numba(
                valid_ngram_requests,  # 需要处理的请求索引
                num_tokens_no_spec,  # 每个请求的token数
                token_ids_cpu,  # 所有token IDs
                self.min_n,  # 最小N-gram长度
                self.max_n,  # 最大N-gram长度
                self.max_model_len,  # 模型最大长度
                self.k,  # 候选token数
                self.valid_ngram_draft,  # 输出：候选tokens（就地修改）
                self.valid_ngram_num_drafts,  # 输出：每个请求的候选数量（就地修改）
            )

            # 恢复原始线程数（避免影响其他组件）
            set_num_threads(original_num_numba_threads)

        # ============================================================
        # 2. 收集结果（从预分配的缓冲区中提取）
        # ============================================================
        for i in range(num_requests):
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                # 请求i有有效的N-gram匹配
                # 提取前 valid_ngram_num_drafts[i] 个候选token
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                # 请求i无匹配或不需要提议
                draft_token_ids.append([])

        return draft_token_ids

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        req_ids: list[str],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        spec_decode_unsupported_reqs: set,
    ) -> list[list[int]]:
        """
        主提议方法 - 为batch中的每个请求生成候选tokens
        
        这是对外暴露的主要接口，负责：
        1. 过滤出可以使用N-gram提议的请求
        2. 调用batch_propose进行批量处理
        
        过滤条件：
        - 必须有已采样的tokens（不跳过推测解码）
        - 不在不支持推测解码的请求集合中
        - 未达到最大长度限制
        
        Args:
            sampled_token_ids: 
                每个请求已采样的token IDs
                示例：[[123], [456, 789], []] 表示3个请求分别采样了1、2、0个token
            req_ids: 
                请求ID列表
                示例：["req_1", "req_2", "req_3"]
            num_tokens_no_spec:
                不含推测token的实际token数
            token_ids_cpu:
                所有token IDs（在CPU上）
            spec_decode_unsupported_reqs:
                不支持推测解码的请求ID集合
                原因：某些采样参数（如temperature、top_p）与推测解码不兼容

        Returns:
            list[list[int]]: 每个请求的候选token列表
        """
        # ============================================================
        # 1. 筛选需要N-gram提议的请求
        # ============================================================
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            
            # --------------------------------------------------------
            # 过滤条件1：必须有已采样的tokens
            # --------------------------------------------------------
            if not num_sampled_ids:
                # 空列表表示跳过推测解码
                # 场景：特殊请求或首次采样
                continue

            # --------------------------------------------------------
            # 过滤条件2：采样参数必须兼容推测解码
            # --------------------------------------------------------
            # 某些采样参数会导致推测解码失效或不准确
            # 例如：高temperature会使分布更随机，N-gram匹配命中率低
            req_id = req_ids[i]
            if req_id in spec_decode_unsupported_reqs:
                continue

            # --------------------------------------------------------
            # 过滤条件3：未达到最大长度
            # --------------------------------------------------------
            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # 已经达到模型最大长度，无法继续生成
                continue

            # 通过所有过滤条件，添加到有效请求列表
            valid_ngram_requests.append(i)

        # ============================================================
        # 2. 批量处理有效请求
        # ============================================================
        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),  # 总请求数
            valid_ngram_requests,  # 需要处理的请求索引
            num_tokens_no_spec,  # token计数
            token_ids_cpu,  # token IDs
        )

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        """
        加载模型（占位方法）
        
        N-gram proposer不需要额外的模型，所有逻辑基于已生成的tokens。
        这个方法存在是为了与其他proposer（如draft model）保持接口一致。
        """
        # No model to load.
        pass


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    """
    Numba加速的批量N-gram提议（并行版本）
    
    使用Numba的@njit装饰器和parallel=True启用JIT编译和并行化：
    - JIT编译：将Python代码编译为机器码，速度接近C
    - 并行化：使用prange自动并行处理多个请求
    
    性能优化：
    - 避免Python解释器开销
    - 利用多核CPU并行
    - 减少GIL（全局解释器锁）影响
    
    Args:
        valid_ngram_requests: 需要处理的请求索引列表
        num_tokens_no_spec: 每个请求的token数量
        token_ids_cpu: 所有token IDs
        min_n: 最小N-gram长度
        max_n: 最大N-gram长度
        max_model_len: 模型最大长度
        k: 候选token数量
        valid_ngram_draft: 输出缓冲区（就地修改）
        valid_ngram_num_drafts: 输出的候选数量（就地修改）
    """
    # ============================================================
    # 并行处理每个请求（prange = parallel range）
    # ============================================================
    # prange会自动将循环分配到多个线程
    # 每个线程处理一部分请求，互不干扰
    for i in prange(len(valid_ngram_requests)):
        # 获取实际的请求索引
        idx = valid_ngram_requests[i]
        
        # 提取该请求的有效token序列
        num_tokens = num_tokens_no_spec[idx]
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        
        # --------------------------------------------------------
        # 核心算法：查找最长匹配的N-gram并提议后续tokens
        # --------------------------------------------------------
        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,  # 完整的token序列
            min_ngram=min_n,  # 最小匹配长度
            max_ngram=max_n,  # 最大匹配长度
            max_model_len=max_model_len,  # 长度限制
            k=k,  # 提议的token数量
        )

        # --------------------------------------------------------
        # 将结果写入预分配的缓冲区
        # --------------------------------------------------------
        # 记录候选token的数量
        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        
        # 如果有候选token，复制到缓冲区
        if len(drafter_output):
            valid_ngram_draft[idx, : drafter_output.shape[0]] = drafter_output


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """
    查找最长匹配的N-gram并提议后续tokens（基于KMP算法）
    
    核心思想：
    在已生成的token序列中，查找与当前上下文后缀匹配的最长N-gram，
    然后提取匹配位置后面的k个tokens作为候选。
    
    算法：改进的KMP（Knuth-Morris-Pratt）算法
    - 时间复杂度：O(n)，n为token序列长度
    - 空间复杂度：O(max_ngram)
    
    示例：
        origin_tokens = [10, 20, 30, 40, 10, 20, 50, 60]
                         ^^^^^^^^^      ^^^^^
                         匹配的N-gram    当前后缀
        
        找到匹配：位置2的[10, 20]与位置6的[10, 20]匹配（长度=2）
        提议：[30, 40] (匹配后的k=2个tokens)
    
    Args:
        origin_tokens: 完整的token序列
        min_ngram: 最小N-gram长度（如3）
        max_ngram: 最大N-gram长度（如10）
        max_model_len: 模型最大长度
        k: 要提议的token数量
    
    Returns:
        np.ndarray: 候选tokens，如果没有匹配则返回空数组
    """
    # ============================================================
    # 1. 边界条件检查
    # ============================================================
    total_token = origin_tokens.shape[0]
    
    # 检查1：序列长度必须 >= 最小N-gram长度
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # 检查2：不超过模型最大长度
    # 限制k，确保 current_length + k <= max_model_len
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # ============================================================
    # 2. 反转token序列（KMP算法技巧）
    # ============================================================
    # 反转后，问题变为：
    # "在反转序列的最右侧位置，查找与前缀匹配的最长N-gram"
    # 
    # 为什么反转？
    # - 我们关心的是序列末尾（当前生成位置）
    # - 反转后，末尾变成开头，可以用KMP的前缀匹配算法
    tokens = origin_tokens[::-1]

    # ============================================================
    # 3. 初始化LPS数组（Longest Proper Prefix which is also Suffix）
    # ============================================================
    # lps[i] = 在 tokens[0:i+1] 中，最长的"既是前缀又是后缀"的子串长度
    # 
    # 示例：tokens = [a, b, c, a, b]
    #      lps[0] = 0 (只有'a'，无proper前缀)
    #      lps[1] = 0 ('ab'中，无匹配)
    #      lps[2] = 0 ('abc'中，无匹配)
    #      lps[3] = 1 ('abca'中，'a'既是前缀又是后缀)
    #      lps[4] = 2 ('abcab'中，'ab'既是前缀又是后缀)
    # 
    # 优化：只存储前max_ngram个位置的lps（节省内存）
    lps = np.zeros(max_ngram, dtype=np.int32)

    # 跟踪找到的最长N-gram长度和位置
    longest_ngram = 0  # 最长匹配长度
    position = 0  # 匹配位置（在反转序列中）

    # ============================================================
    # 4. KMP算法主循环（计算LPS并查找最长N-gram）
    # ============================================================
    # lps[0]总是0，从索引1开始
    prev_lps = 0  # 前一个位置的LPS值
    i = 1  # 当前处理的位置
    
    while i < total_token:
        # --------------------------------------------------------
        # 情况1：当前token匹配
        # --------------------------------------------------------
        if tokens[prev_lps] == tokens[i]:
            # tokens[0:prev_lps+1] 是 tokens[0:i+1] 的前缀后缀
            prev_lps += 1
            
            # 检查是否找到更长的有效N-gram
            # 条件：prev_lps >= longest_ngram
            # 目标：找到最早出现的匹配（在原序列中最早 = 反转序列中最晚）
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps  # 更新最长长度
                position = i  # 记录位置
            
            # 存储LPS值（仅前max_ngram个）
            if i < max_ngram:
                lps[i] = prev_lps
            
            # 防止超过max_ngram限制
            if prev_lps == max_ngram:
                # 达到最大N-gram长度，回退到次长匹配
                # 避免匹配长度超过max_ngram
                prev_lps = lps[max_ngram - 1]
            
            i += 1
        
        # --------------------------------------------------------
        # 情况2：当前token不匹配，尝试次长前缀
        # --------------------------------------------------------
        elif prev_lps != 0:
            # 回退到次长的"前缀即后缀"
            # 这是KMP算法的核心：避免重复比较
            prev_lps = lps[prev_lps - 1]
        
        # --------------------------------------------------------
        # 情况3：无可用的前缀，继续下一个位置
        # --------------------------------------------------------
        else:
            i += 1

    # ============================================================
    # 5. 验证结果并提取候选tokens
    # ============================================================
    # 检查找到的N-gram是否满足最小长度要求
    if longest_ngram < min_ngram:
        # 没有找到有效的N-gram
        return np.empty((0,), dtype=origin_tokens.dtype)

    # --------------------------------------------------------
    # 将反转序列中的位置转换回原序列位置
    # --------------------------------------------------------
    # 在反转序列中：tokens[0:longest_ngram] 匹配 tokens[position-longest_ngram+1:position+1]
    # 在原序列中：匹配的N-gram结束位置 = total_token - 1 - (position - longest_ngram)
    # 候选tokens起始位置 = 匹配的N-gram结束位置 + 1
    start_position = total_token - 1 - position + longest_ngram
    
    # 确保不超出序列范围
    k = min(k, total_token - start_position)
    
    # 提取候选tokens
    # 示例：如果匹配了[10, 20]，位置在2-3，则提取[30, 40, ...]
    return origin_tokens[start_position : start_position + k]
