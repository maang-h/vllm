# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
FusedMoE Layer 实现 - vLLM中Mixture of Experts (MoE) 模型的融合层实现

这个模块实现了高效的MoE计算，包括：
1. 专家选择（Top-K路由）
2. 专家权重管理和加载
3. 多种并行策略支持（TP, EP, DP, PP）
4. 量化支持（FP8, INT8, INT4等）
5. 专家负载均衡（EPLB）

核心类：FusedMoE - 包含多个专家MLP网络的融合层
"""

from collections.abc import Callable, Iterable  # 可调用对象和可迭代对象类型提示 - 用于函数签名和迭代器类型标注
from contextlib import nullcontext  # 空上下文管理器 - 提供一个什么都不做的上下文管理器，用于条件性地应用上下文
from enum import Enum  # 枚举类型基类 - 用于定义枚举常量（如权重缩放支持类型）
from typing import Literal, cast, get_args, overload  # 类型提示工具 - 用于更精确的类型标注和重载

import torch  # PyTorch核心库 - 深度学习框架的核心功能
import torch.nn.functional as F  # PyTorch函数式API - 提供激活函数、填充等操作
from torch.nn.parameter import UninitializedParameter  # 未初始化的参数 - 用于GGUF等延迟初始化场景

import vllm.envs as envs  # vLLM环境变量配置 - 提供各种运行时配置选项（如MoE chunk size）
from vllm._aiter_ops import rocm_aiter_ops  # ROCm (AMD GPU) 的AITER操作支持 - AMD GPU上的优化MoE kernel
from vllm.config import VllmConfig, get_current_vllm_config  # vLLM配置管理 - 获取全局配置对象
from vllm.config.parallel import ExpertPlacementStrategy  # 专家放置策略 - "linear"或"round_robin"两种方式分配专家到GPU
from vllm.distributed import (  # 分布式通信相关工具 - 多GPU并行计算的核心支持
    get_dp_group,  # 获取数据并行组 - Data Parallelism组，用于在不同batch数据间并行
    get_ep_group,  # 获取专家并行组 - Expert Parallelism组，将专家分散到不同GPU
    get_pcp_group,  # 获取部分上下文并行组 - Partial Context Parallelism组，处理超长序列
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小 - 参与TP的GPU总数
    tensor_model_parallel_all_reduce,  # 张量并行的all-reduce操作 - 在TP组内聚合结果
)
from vllm.distributed.eplb.eplb_state import EplbState  # 专家并行负载均衡状态管理 - 管理冗余专家和负载统计
from vllm.forward_context import ForwardContext, get_forward_context  # 前向传播上下文 - 存储当前forward pass的元数据
from vllm.logger import init_logger  # 日志初始化器 - 创建模块级logger实例
from vllm.model_executor.custom_op import CustomOp  # 自定义算子基类 - 支持torch.compile的自定义操作
from vllm.model_executor.layers.fused_moe.config import (  # MoE配置类 - 定义MoE层的各种配置参数
    FusedMoEConfig,  # MoE层配置 - 包含专家数、hidden_dim等
    FusedMoEParallelConfig,  # 并行配置 - TP/EP/DP的配置信息
    FusedMoEQuantConfig,  # 量化配置 - FP8/INT8等量化方案配置
    RoutingMethodType,  # 路由方法类型 - Softmax、Sigmoid、TopK等不同路由策略
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    init_aiter_topK_meta_data,  # 初始化AITER TopK元数据 - ROCm平台的专家选择元数据
)
from vllm.model_executor.layers.fused_moe.routing_simulator import RoutingSimulator  # 路由模拟器 - 用于测试不同路由策略
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,  # 量化配置基类 - 所有量化方案的基础接口
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    is_flashinfer_supporting_global_sf,  # 检查FlashInfer是否支持全局缩放因子 - 用于某些量化方案
)
from vllm.platforms import current_platform  # 当前平台检测 - 区分CUDA/ROCm/TPU/CPU等平台
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe  # 检查是否有FlashInfer TRT-LLM MoE kernel
from vllm.utils.math_utils import cdiv, round_up  # 数学工具 - ceil除法和向上取整
from vllm.utils.torch_utils import (
    aux_stream,  # 获取辅助CUDA stream - 用于shared experts的并行执行
    current_stream,  # 获取当前CUDA stream - 主计算流
    direct_register_custom_op,  # 直接注册自定义算子 - 支持torch.compile
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id  # 获取当前microbatch ID - DBO (Dynamic Batch Orchestration) 支持

# 平台特定的EPLB (Expert Parallel Load Balancing) 函数
if current_platform.is_cuda_alike():  # CUDA/ROCm平台 - 使用原生CUDA kernel
    from .fused_moe import eplb_map_to_physical_and_record  # 将逻辑专家ID映射到物理ID并记录负载
else:  # CPU或其他平台 - 使用Python fallback

    def _eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,  # Top-K选中的专家ID (逻辑ID)
        expert_load_view: torch.Tensor,  # 专家负载视图 - 记录每个专家的使用情况
        logical_to_physical_map: torch.Tensor,  # 逻辑到物理专家的映射表
        logical_replica_count: torch.Tensor,  # 每个逻辑专家的副本数量
    ) -> torch.Tensor:
        # CPU fallback: no EPLB so just return as is
        # CPU回退：不支持EPLB，直接返回原始ID
        return topk_ids

    eplb_map_to_physical_and_record = _eplb_map_to_physical_and_record

from vllm.model_executor.layers.fused_moe.fused_moe import GroupedTopk  # 分组TopK路由 - DeepSeek等模型使用

# TPU平台特定实现
if current_platform.is_tpu():  # Google TPU - 使用Pallas实现
    from .moe_pallas import fused_moe as fused_moe_pallas  # TPU上的MoE实现
else:
    fused_moe_pallas = None  # type: ignore  # 其他平台不使用

# MoE方法基类和实现
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,  # MoE方法基类 - 定义apply等接口
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,  # 模块化MoE方法 - 支持DeepEP等高级backend
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,  # 未量化MoE方法 - 标准FP16/BF16计算
)

logger = init_logger(__name__)  # 创建模块级日志记录器


class FusedMoeWeightScaleSupported(Enum):
    """
    MoE权重缩放支持的类型枚举
    
    用于标识量化时权重缩放因子的粒度：
    - TENSOR: 每个tensor一个缩放因子（per-tensor quantization）
    - CHANNEL: 每个通道一个缩放因子（per-channel quantization）
    - GROUP: 每组元素一个缩放因子（group quantization，如GPTQ）
    - BLOCK: 每个块一个缩放因子（block quantization，如DeepSeek的128x128块）
    """
    TENSOR = "tensor"    # 张量级缩放 - 整个权重矩阵共享一个缩放因子
    CHANNEL = "channel"  # 通道级缩放 - 每个输出通道有独立缩放因子
    GROUP = "group"      # 组级缩放 - 如GPTQ的128元素一组
    BLOCK = "block"      # 块级缩放 - 如DeepSeek的128x128块


def determine_expert_map(
    ep_size: int,  # 专家并行组的大小（参与EP的GPU数量）
    ep_rank: int,  # 当前进程在专家并行组中的排名（0到ep_size-1）
    global_num_experts: int,  # 模型中专家的总数量（全局）
    expert_placement_strategy: ExpertPlacementStrategy = "linear",  # 专家放置策略："linear"或"round_robin"
    num_fused_shared_experts: int = 0,  # 融合的共享专家数量（用于某些模型如DeepSeekV2）
    return_expert_mask: bool = False,  # 是否返回expert_mask（AITER ROCm backend需要）
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    计算在专家并行(EP)场景下，应该为每个rank分配多少专家，
    并创建从全局专家索引到本地专家索引的映射。
    
    核心思想：将专家尽可能均匀地分配到各个GPU上。
    
    Args:
        ep_size: 专家并行组的大小（参与EP的GPU数量）
        ep_rank: 当前进程在专家并行组中的排名（0-based索引）
        global_num_experts: 模型中专家的总数量
        expert_placement_strategy: 专家放置策略
            - "linear": 线性分配，连续的专家分配给同一GPU
            - "round_robin": 轮询分配，专家交替分配给不同GPU
        num_fused_shared_experts: 融合的共享专家数量（AITER MOE使用）
        return_expert_mask: 是否返回expert_mask（ROCm AITER需要）

    Returns:
        tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]: 包含三个元素的元组：
            - local_num_experts (int): 分配给当前rank的专家数量
            - expert_map (Optional[torch.Tensor]): 形状为(global_num_experts,)的张量
                从全局索引映射到本地索引，未分配给当前rank的专家标记为-1
                如果ep_size=1则返回None（无需映射）
            - expert_mask (Optional[torch.Tensor]): 形状为
                (global_num_experts + num_fused_shared_experts + 1,)的张量
                分配给当前rank的专家标记为1，sentinel标记为0
                仅在AITER MOE启用时使用，否则返回None
    
    示例1 - Linear策略，64个专家，4个GPU：
        ep_size=4, global_num_experts=64
        每个GPU分配: 64 // 4 = 16个专家
        
        GPU 0 (ep_rank=0): 专家 0-15   (local_num_experts=16)
        GPU 1 (ep_rank=1): 专家 16-31  (local_num_experts=16)
        GPU 2 (ep_rank=2): 专家 32-47  (local_num_experts=16)
        GPU 3 (ep_rank=3): 专家 48-63  (local_num_experts=16)
        
        GPU 0的expert_map: [0,1,2,...,15, -1,-1,...,-1]
                          ^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
                          本地专家0-15    未分配专家
    
    示例2 - Linear策略，65个专家，4个GPU（不能整除）：
        ep_size=4, global_num_experts=65
        base_experts = 65 // 4 = 16
        remainder = 65 % 4 = 1
        
        GPU 0 (ep_rank=0): 专家 0-16   (17个，因为rank < remainder)
        GPU 1 (ep_rank=1): 专家 17-32  (16个)
        GPU 2 (ep_rank=2): 专家 33-48  (16个)
        GPU 3 (ep_rank=3): 专家 49-64  (16个)
    
    示例3 - Round-Robin策略，8个专家，2个GPU：
        ep_size=2, global_num_experts=8
        
        GPU 0 (ep_rank=0): 专家 0,2,4,6 (本地索引0,1,2,3)
        GPU 1 (ep_rank=1): 专家 1,3,5,7 (本地索引0,1,2,3)
        
        GPU 0的expert_map: [0,-1,1,-1,2,-1,3,-1]
                          ^   ^   ^   ^   ^
                          本  无  本  无  本
    """
    assert ep_size > 0  # 确保EP组大小为正数
    if ep_size == 1:  # 如果只有1个GPU（无EP）
        # 无需专家映射，所有专家都在本地
        return (global_num_experts, None, None)

    # Distribute experts as evenly as possible to each rank.
    # 将专家尽可能均匀地分配给每个rank
    # 算法：先每个rank分配base_experts个，剩余的分配给前remainder个rank
    base_experts = global_num_experts // ep_size  # 每个rank的基础专家数量
    remainder = global_num_experts % ep_size      # 无法整除的剩余专家数量
    # 如果当前rank在前remainder个rank中，则多分配一个专家
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Create a tensor of size num_experts filled with -1
    # 创建一个大小为num_experts的张量，初始值全部为-1
    # -1表示该全局专家ID未分配给当前rank
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    
    # Create an expert map for the local experts
    # 为本地专家创建映射关系
    if expert_placement_strategy == "linear":  # 线性分配策略
        # 计算当前rank的专家起始索引
        # 前ep_rank个rank每个有base_experts+1或base_experts个专家
        # 所以起始位置 = ep_rank * base_experts + min(ep_rank, remainder)
        # 示例：64专家,4GPU,remainder=0 -> rank0从0开始,rank1从16开始,rank2从32开始
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        # 将连续的local_num_experts个全局索引映射到本地索引0,1,2,...
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "round_robin":  # 轮询分配策略
        # 以轮询方式分配专家：rank0得0,4,8...; rank1得1,5,9...; rank2得2,6,10...
        # 生成当前rank负责的全局专家ID序列
        # 示例：8专家,2GPU,rank0 -> [0,2,4,6], rank1 -> [1,3,5,7]
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int32
        )

        # 将这些全局ID映射到本地索引0,1,2,...
        expert_map[local_log_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None  # 初始化expert_mask为None
    if return_expert_mask:  # 如果需要返回mask（AITER MOE使用）
        # 创建mask张量：1表示分配给当前rank，0表示未分配或sentinel
        # 大小包括：路由专家 + 共享专家 + 1个sentinel位置
        expert_mask = torch.ones(
            (global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32
        )
        expert_mask[-1] = 0  # 最后一位是sentinel，标记为0
        # 前global_num_experts位根据expert_map设置：有映射(>-1)则为1，否则为0
        expert_mask[:global_num_experts] = expert_map > -1
        # 将共享专家的本地索引追加到expert_map中
        # 共享专家的本地ID从local_num_experts开始
        expert_map = torch.cat(
            (
                expert_map,  # 原有的路由专家映射
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),  # 共享专家的本地索引
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


def determine_expert_placement_strategy(
    expert_placement_strategy: ExpertPlacementStrategy,  # 用户请求的放置策略
    moe_parallel_config: FusedMoEParallelConfig,  # MoE并行配置
    num_expert_group: int | None,  # 专家组数量（DeepSeek模型使用）
    num_redundant_experts: int,  # 冗余专家数量（EPLB使用）
    enable_eplb: bool,  # 是否启用专家并行负载均衡
) -> ExpertPlacementStrategy:
    """
    确定最终使用的专家放置策略，如果请求的策略不支持则回退到linear。
    
    Round-robin策略的限制条件：
    1. 必须有多个专家组（num_expert_group > 1）
    2. 不能有冗余专家（num_redundant_experts == 0）
    3. 不能启用EPLB（enable_eplb == False）
    4. 如果使用all2all kernels，必须是DeepEP low-latency backend
    
    Args:
        expert_placement_strategy: 用户请求的策略 ("linear"或"round_robin")
        moe_parallel_config: MoE并行配置对象
        num_expert_group: 专家组数量，None表示无分组
        num_redundant_experts: 冗余专家数量（EPLB特性）
        enable_eplb: 是否启用负载均衡
    
    Returns:
        ExpertPlacementStrategy: 最终确定的策略，可能回退到"linear"
    
    示例：
        # 场景1：DeepSeekV2-Lite，有8个专家组，要求round-robin
        num_expert_group=8, num_redundant_experts=0, enable_eplb=False
        -> 返回"round_robin" ✅ 支持
        
        # 场景2：普通MoE，无专家组，要求round-robin
        num_expert_group=None, num_redundant_experts=0, enable_eplb=False
        -> 返回"linear" ❌ 不支持（无专家组），回退
        
        # 场景3：启用EPLB，要求round-robin
        num_expert_group=8, num_redundant_experts=2, enable_eplb=True
        -> 返回"linear" ❌ 不支持（EPLB与round-robin不兼容），回退
    """
    if expert_placement_strategy == "round_robin":  # 如果请求round-robin策略
        # 检查是否满足round-robin的所有前提条件
        round_robin_supported = (
            (num_expert_group is not None and num_expert_group > 1)  # 条件1：必须有多个专家组
            and num_redundant_experts == 0  # 条件2：不能有冗余专家
            and not enable_eplb  # 条件3：不能启用EPLB
        )

        if not round_robin_supported:  # 如果不满足前提条件
            logger.warning(
                "Round-robin expert placement is only supported for "
                "models with multiple expert groups and no redundant "
                "experts. Falling back to linear expert placement."
            )
            return "linear"  # 回退到linear策略
            
        # 检查all2all backend兼容性
        if (
            moe_parallel_config.use_all2all_kernels  # 如果使用all2all kernels
            and not moe_parallel_config.use_deepep_ll_kernels  # 但不是DeepEP-LL backend
        ):
            logger.warning(
                "Round-robin expert placement currently only supports "
                "the DeepEP low-latency backend, but '%s' was configured. "
                "Falling back to linear expert placement.",
                moe_parallel_config.all2all_backend,
            )
            return "linear"  # 回退到linear策略

    return expert_placement_strategy  # 返回原策略（可能是linear或通过检查的round-robin）


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    压缩专家映射表，移除所有-1条目，生成本地到全局索引的映射字符串。
    
    这个函数用于日志记录，将expert_map压缩成易读的字符串格式。
    使用字符串支持哈希，确保同样的映射只记录一次日志。

    Args:
        expert_map (torch.Tensor): 形状为(global_num_experts,)的张量
            从全局索引映射到本地索引，-1表示未分配给当前rank的专家

    Returns:
        str: 本地索引到全局索引的映射字符串
            格式："本地索引0->全局索引X, 本地索引1->全局索引Y, ..."
    
    示例：
        # GPU 0有expert_map = [0, 1, 2, -1, -1, -1, -1, -1]
        # 表示全局专家0,1,2映射到本地专家0,1,2，其他专家不在此GPU
        get_compressed_expert_map(expert_map)
        -> "0->0, 1->1, 2->2"
        
        # GPU 1在round-robin下有expert_map = [-1, 0, -1, 1, -1, 2, -1, 3]
        # 表示全局专家1,3,5,7映射到本地专家0,1,2,3
        get_compressed_expert_map(expert_map)
        -> "0->1, 1->3, 2->5, 3->7"
    """
    # 找出所有非-1的位置（即分配给当前rank的专家的全局索引）
    global_indices = torch.where(expert_map != -1)[0]
    # 获取这些位置对应的本地索引
    local_indices = expert_map[global_indices]
    # 构建 "本地索引->全局索引" 的映射字符串
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


def maybe_roundup_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    quant_config: QuantizationConfig | None,
    moe_parallel_config: FusedMoEParallelConfig,
    is_lora_enabled: bool,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        quant_config: Fused MoE quantization configuration.
        moe_parallel_config: Fused MoE parallelization strategy configuration.
        is_lora_enabled: True if the engine is enabled with LoRA. This
            is used in the case of mxfp4 quantization in selecting the
            MxFP4Backend.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs.
        Original hidden size otherwise.
    """
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_roundup_layer_hidden_size,
    )

    hidden_size = maybe_roundup_layer_hidden_size(
        hidden_size, act_dtype, moe_parallel_config
    )

    # we are padding globally so EP buffer allocation works
    if quant_config and quant_config.get_name() == "mxfp4":
        from vllm.model_executor.layers.quantization.mxfp4 import (
            Mxfp4Backend,
            get_mxfp4_backend,
        )

        current_mxfp4_backend = get_mxfp4_backend(is_lora_enabled)
        if (
            current_mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
        ):
            hidden_size = round_up(hidden_size, 128)
        elif (
            current_platform.is_rocm()
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
        ):
            hidden_size = round_up(hidden_size, 256)

    return hidden_size


@CustomOp.register("fused_moe")  # 注册为自定义算子，支持torch.compile
class FusedMoE(CustomOp):
    """
    FusedMoE层 - 用于MoE（Mixture of Experts）模型的融合层实现。
    
    这个层包含了专家的所有权重，并提供高效的MoE计算：
    1. MergedColumnParallel权重 (gate_up_proj / w13) - 门控和上投影合并
    2. RowParallelLinear权重 (down_proj / w2) - 下投影
    
    权重命名约定：
    - w1: gate_proj (门控投影)
    - w2: down_proj (下投影)  
    - w3: up_proj (上投影)
    - w13: w1和w3合并的权重
    
    注意：Mixtral等模型使用w1/w2/w3命名。我们在这里采用这个约定，
    并在每个模型实现的load_weights函数中处理任何必要的重映射。

    主要参数说明：
        num_experts: 模型中专家的数量（不包括冗余专家）
        top_k: 每个token选择的专家数量（如Top-2、Top-4）
        hidden_size: Transformer的输入隐藏状态大小
        intermediate_size: 专家的中间层大小（通常是hidden_size的4倍）
        params_dtype: 参数的数据类型（如torch.float16、torch.bfloat16）
        reduce_results: 是否在层输出上执行all_reduce（TP/EP并行时）  # all_reduce是什么？？
        renormalize: 是否在fused_moe kernel中重新归一化logits
        quant_config: 量化配置（支持FP8、INT8、INT4等）
        enable_eplb: 是否启用专家并行负载均衡器（EPLB）
    
    核心功能：
    1. 专家选择：根据router logits选择Top-K专家
    2. 专家计算：并行计算选中的专家输出
    3. 结果融合：加权聚合各专家输出
    4. 负载均衡：通过EPLB优化专家利用率
    
    支持的并行策略：
    - TP (Tensor Parallelism): 张量并行
    - EP (Expert Parallelism): 专家并行
    - DP (Data Parallelism): 数据并行
    - PP (Pipeline Parallelism): 流水线并行（通过模型级支持）
    - SP (Sequence Parallelism): 序列并行
    """

    def __init__(
        self,
        num_experts: int,  # 全局专家数量（逻辑专家数，不含冗余专家）
        top_k: int,  # 每个token选择的专家数量（如2表示Top-2路由）
        hidden_size: int,  # 隐藏层维度（输入/输出维度）
        intermediate_size: int,  # 中间层维度（FFN的放大倍数，通常4倍hidden_size） FFN intermediate size mlp中上投影的维度
        params_dtype: torch.dtype | None = None,  # 参数数据类型（如fp16/bf16，None则使用默认）
        reduce_results: bool = False,  # 是否在输出时all_reduce（TP/EP时可能需要）
        renormalize: bool = True,  # 是否重新归一化专家权重（softmax后的Top-K权重）
        use_grouped_topk: bool = False,  # 是否使用分组TopK（DeepSeekV2/V3特性）
        num_expert_group: int | None = None,  # 专家组数量（分组TopK时使用）
        topk_group: int | None = None,  # 每组选择的专家数（分组TopK时使用）
        quant_config: QuantizationConfig | None = None,  # 量化配置（FP8/INT8/INT4等）
        tp_size: int | None = None,  # 张量并行大小（None则自动获取）
        ep_size: int | None = None,  # 专家并行大小（None则自动获取）
        dp_size: int | None = None,  # 数据并行大小（None则自动获取）
        pcp_size: int | None = None,  # 部分上下文并行大小（处理超长序列）
        prefix: str = "",  # 层名称前缀（用于日志和权重加载）
        custom_routing_function: Callable | None = None,  # 自定义路由函数（可覆盖默认TopK）
        scoring_func: str = "softmax",  # 评分函数类型："softmax"或"sigmoid"
        routed_scaling_factor: float = 1.0,  # 路由权重的缩放因子（某些模型使用）
        e_score_correction_bias: torch.Tensor | None = None,  # 专家分数修正偏置（DeepSeek使用）
        apply_router_weight_on_input: bool = False,  # 是否在输入上应用路由权重
        activation: str = "silu",  # 激活函数类型（默认SiLU，即Swish）
        is_act_and_mul: bool = True,  # 是否使用激活-乘法融合（SiluAndMul，大多数模型）
        enable_eplb: bool = False,  # 是否启用专家并行负载均衡（动态专家分配）
        num_redundant_experts: int = 0,  # 冗余专家数量（EPLB特性，用于负载均衡）
        has_bias: bool = False,  # 专家MLP是否有bias项
        is_sequence_parallel=False,  # 是否启用序列并行（在TP基础上切分序列）
        expert_mapping: list[tuple[str, str, int, str]] | None = None,  # 专家权重映射表（用于权重加载）
        n_shared_experts: int | None = None,  # 共享专家数量（DeepSeekV2等模型）
        routing_method_type: int | None = None,  # 路由方法类型（None则根据scoring_func推断）
    ):
        super().__init__()  # 初始化CustomOp基类

        # ============================================================
        # 1. 设置Shared Experts的独立CUDA Stream
        # ============================================================
        # 目的：让共享专家和路由专家并行执行，提高吞吐量
        # 原理：使用独立stream，在路由专家计算时同时计算共享专家
        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:  # 环境变量控制是否禁用
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            self.shared_experts_stream = None  # 禁用独立stream
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self.shared_experts_stream = aux_stream()  # 获取辅助CUDA stream
            if self.shared_experts_stream is not None:  # 如果成功获取（CUDA平台）
                logger.debug_once(
                    "Enabled separate cuda stream for MoE shared_experts", scope="local"
                )

        # ============================================================
        # 2. 设置参数数据类型
        # ============================================================
        if params_dtype is None:  # 如果未指定
            params_dtype = torch.get_default_dtype()  # 使用PyTorch默认类型（通常fp32）
        self.params_dtype = params_dtype  # 保存参数类型（权重的存储类型）

        # ============================================================
        # 3. 获取vLLM全局配置
        # ============================================================
        vllm_config = get_current_vllm_config()  # 获取全局配置对象
        self.vllm_config = vllm_config  # 保存配置引用

        # ============================================================
        # 4. 推断MoE输入激活值的数据类型
        # ============================================================
        # 关键：激活值类型 ≠ 参数类型（激活值通常不量化）
        # FIXME (varun): We should have a better way of inferring the activation
        # datatype. This works for now as the tensor datatype entering the MoE
        # operation is typically unquantized (i.e. float16/bfloat16).
        if vllm_config.model_config is not None:  # 正常情况
            moe_in_dtype = vllm_config.model_config.dtype  # 从模型配置获取（如fp16/bf16）
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype  # 测试时的fallback

        # ============================================================
        # 5. 确定各种并行策略的大小
        # ============================================================
        # 如果未指定则从分布式组自动获取
        tp_size_ = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )  # 张量并行度：将模型权重切分到多个GPU
        # 示例：权重[1024,4096]，TP=4 -> 每GPU持有[1024,1024]
        
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
        # 数据并行度：不同GPU处理不同batch数据
        # 示例：batch=32，DP=2 -> 每GPU处理16个样本
        
        pcp_size_ = pcp_size if pcp_size is not None else get_pcp_group().world_size
        # 部分上下文并行度：将超长序列切分到多个GPU
        # 示例：序列长度8192，PCP=2 -> 每GPU处理4096个token

        # ============================================================
        # 6. 设置序列并行
        # ============================================================
        self.is_sequence_parallel = is_sequence_parallel  # 是否启用序列并行
        self.sp_size = tp_size_ if is_sequence_parallel else 1
        # 序列并行通常与TP共享同一组GPU
        # 示例：TP=4且启用SP -> SP也在这4个GPU上工作，每GPU处理1/4序列

        # ============================================================
        # 7. 创建MoE并行配置对象
        # ============================================================
        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,  # 传入TP大小
            pcp_size_=pcp_size_,  # 传入PCP大小
            dp_size_=dp_size_,  # 传入DP大小
            vllm_parallel_config=vllm_config.parallel_config,  # 全局并行配置
        )  # 这个对象封装了所有并行相关的配置和rank信息

        # ============================================================
        # 8. 计算专家数量
        # ============================================================
        self.global_num_experts = num_experts + num_redundant_experts
        # 全局物理专家数 = 逻辑专家数 + 冗余专家数
        # 示例：64个逻辑专家 + 2个冗余专家 = 66个物理专家
        
        self.logical_num_experts = num_experts
        # 逻辑专家数：模型定义的专家数量（如Mixtral-8x7B中的8）

        # ============================================================
        # 9. 保存专家映射表（用于权重加载）
        # ============================================================
        # Expert mapping used in self.load_weights
        self.expert_mapping = expert_mapping
        # 格式：[(param_name, weight_name, expert_id, shard_id), ...]
        # 用于将checkpoint的权重名称映射到模型参数

        # ============================================================
        # 10. 对齐hidden_size以优化性能
        # ============================================================
        # Round up hidden size if needed.
        hidden_size = maybe_roundup_hidden_size(
            hidden_size,  # 原始hidden_size（如1024）
            moe_in_dtype,  # 激活值类型（如fp16）
            quant_config,  # 量化配置
            self.moe_parallel_config,  # 并行配置
            is_lora_enabled=self.vllm_config.lora_config is not None,  # 是否启用LoRA
        )
        # 可能会将hidden_size向上取整到128或256的倍数
        # 目的：优化GPU内存访问和kernel性能
        # 示例：1024 -> 1024（已对齐），1536 -> 1664（取整到128倍数）

        # ============================================================
        # 11. 注册层到编译上下文（支持torch.compile）
        # ============================================================
        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:  # 检查重复
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self  # 注册当前层
        self.layer_name = prefix  # 保存层名称（如"model.layers.0.mlp"）

        # ============================================================
        # 12. 初始化EPLB（专家并行负载均衡）相关状态
        # ============================================================
        self.enable_eplb = enable_eplb  # 是否启用负载均衡
        # EPLB通过动态专家副本分配来平衡负载
        self.expert_load_view: torch.Tensor | None = None  # 专家负载视图（记录每个专家的使用情况）
        self.logical_to_physical_map: torch.Tensor | None = None  # 逻辑专家ID到物理专家ID的映射
        self.logical_replica_count: torch.Tensor | None = None  # 每个逻辑专家的副本数量
        # 这些张量会在后续的set_eplb_state中设置
        
        self.expert_placement_strategy: ExpertPlacementStrategy = (
            vllm_config.parallel_config.expert_placement_strategy
        )  # 专家放置策略："linear"或"round_robin"
        # 指的是在ep场景下，如何将多个专家分配到不同GPU上
        # linear: 线性分配，连续的专家分配给同一GPU
        # round_robin: 轮询分配，专家交替分配给不同GPU

        # ============================================================
        # 13. 检查ROCm AITER MoE特性支持
        # ============================================================
        # ROCm aiter shared experts fusion
        # AITER是AMD GPU上的优化MoE实现
        self.rocm_aiter_fmoe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        # 检查基础AITER MoE是否可用
        
        self.aiter_fmoe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )  # 检查AITER是否支持共享专家融合

        self.num_fused_shared_experts = (
            n_shared_experts  # 如果指定了共享专家数量
            if n_shared_experts is not None and self.aiter_fmoe_shared_expert_enabled  # 且AITER支持
            else 0  # 否则为0
        )
        # 共享专家：所有token都会经过的专家（如DeepSeekV2）
        # 与路由专家不同，共享专家不需要路由选择
        
        if (
            not self.aiter_fmoe_shared_expert_enabled  # 如果AITER不支持共享专家
            and self.num_fused_shared_experts != 0  # 但指定了共享专家数量
        ):
            raise ValueError(
                "n_shared_experts is only supported on ROCm aiter when "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
            )  # 抛出错误

        # ============================================================
        # 14. 确定专家映射（核心：决定哪些专家在当前GPU上）
        # ============================================================
        # Determine expert maps
        if self.use_ep:  # 如果使用专家并行
            # --------------------------------------------------------
            # 14.1 验证EPLB和冗余专家的兼容性
            # --------------------------------------------------------
            if self.enable_eplb:  # 如果启用专家并行负载均衡
                assert self.global_num_experts % self.ep_size == 0, (
                    "EPLB currently only supports even distribution of "
                    "experts across ranks."
                )
                # EPLB要求专家数能被EP size整除，以便均匀分配
                # 示例：66个专家（64+2冗余），2个GPU -> 每GPU 33个专家 ✅
                #      65个专家（64+1冗余），2个GPU -> 无法均分 ❌
            else:  # 如果未启用EPLB
                assert num_redundant_experts == 0, (
                    "Redundant experts are only supported with EPLB."
                )
                # 冗余专家只在EPLB模式下支持
                # 因为EPLB负责管理冗余专家的动态映射

            # --------------------------------------------------------
            # 14.2 确定最终使用的专家放置策略
            # --------------------------------------------------------
            self.expert_placement_strategy = determine_expert_placement_strategy(
                expert_placement_strategy=self.expert_placement_strategy,  # 用户请求的策略
                moe_parallel_config=self.moe_parallel_config,  # 并行配置
                num_expert_group=num_expert_group,  # 专家组数量
                num_redundant_experts=num_redundant_experts,  # 冗余专家数
                enable_eplb=self.enable_eplb,  # 是否启用EPLB
            )
            # 这个函数会检查round-robin策略的所有前提条件
            # 如果不满足则回退到linear策略

            # --------------------------------------------------------
            # 14.3 计算专家映射：全局ID -> 本地ID
            # --------------------------------------------------------
            self._expert_map: torch.Tensor | None  # 类型标注
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,  # EP组大小（如4个GPU）
                ep_rank=self.ep_rank,  # 当前rank（0-3）
                global_num_experts=self.global_num_experts,  # 全局专家数（如64）
                expert_placement_strategy=self.expert_placement_strategy,  # 放置策略
                num_fused_shared_experts=self.num_fused_shared_experts,  # 共享专家数
                return_expert_mask=self.rocm_aiter_fmoe_enabled,  # 是否需要mask（ROCm）
            )
            # 返回值：
            # - local_num_experts: 当前GPU上的专家数（如16）
            # - expert_map: 全局ID到本地ID的映射 [global_num_experts]
            #   示例：[0,1,2,...,15,-1,-1,...,-1] 表示全局专家0-15在本地，其他不在
            # - expert_mask: 用于ROCm AITER kernel的mask（可选）
            
            self.local_num_experts = local_num_experts  # 保存本地专家数
            
            # --------------------------------------------------------
            # 14.4 注册映射tensor为buffer（不参与梯度，但会保存/加载）
            # --------------------------------------------------------
            self.register_buffer("_expert_map", expert_map)  # 注册为persistent buffer
            self.register_buffer("expert_mask", expert_mask)  # 注册expert mask
            # register_buffer的作用：
            # 1. 这些tensor会随模型一起保存/加载
            # 2. 会自动移动到正确的device
            # 3. 不会被优化器更新（非参数）
            
            # --------------------------------------------------------
            # 14.5 初始化专家路由表（round-robin策略需要）
            # --------------------------------------------------------
            self._maybe_init_expert_routing_tables()
            # 如果使用round-robin + DeepEP backend，创建额外的路由表
            # 包括：global_to_physical, physical_to_global, local_to_global映射
            
            # --------------------------------------------------------
            # 14.6 记录EP配置信息（只记录一次）
            # --------------------------------------------------------
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,  # 当前rank（如0）
                self.ep_size,  # 总EP size（如4）
                self.expert_placement_strategy,  # 策略名（linear/round_robin）
                self.local_num_experts,  # 本地专家数（如16）
                self.global_num_experts,  # 全局专家数（如64）
                get_compressed_expert_map(self._expert_map),  # 压缩的映射字符串
            )
            # 日志示例：
            # [EP Rank 0/4] Expert parallelism is enabled. Expert placement strategy: linear.
            # Local/global number of experts: 16/64. Experts local to global index map:
            # 0->0, 1->1, 2->2, ..., 15->15.
            
        else:  # 如果不使用专家并行（单GPU或TP only）
            # --------------------------------------------------------
            # 14.7 单GPU情况：所有专家都在本地
            # --------------------------------------------------------
            self.local_num_experts, self._expert_map, self.expert_mask = (
                self.global_num_experts,  # 本地 = 全局
                None,  # 不需要映射（所有专家都在本地）
                None,  # 不需要mask
            )
            # 示例：64个专家，单GPU -> local_num_experts=64, 无需映射

        # ============================================================
        # 15. 保存Top-K配置
        # ============================================================
        self.top_k = top_k  # 每个token选择的专家数量（如2表示Top-2）

        # ============================================================
        # 16. 初始化AITER共享专家TopK缓冲区（ROCm特定）
        # ============================================================
        self._init_aiter_shared_experts_topK_buffer(
            vllm_config=vllm_config,  # vLLM配置
            dp_size=dp_size_,  # 数据并行大小
        )
        # AITER是AMD ROCm平台的优化MoE实现
        # 这个buffer用于处理共享专家和路由专家的TopK元数据
        # 如果num_fused_shared_experts > 0，会创建额外的缓冲区
        
        # --------------------------------------------------------
        # 16.1 验证AITER expert_mask的格式
        # --------------------------------------------------------
        if self.use_ep and self.rocm_aiter_fmoe_enabled:  # 如果使用EP且启用AITER
            assert self.expert_mask is None or torch.all(
                (expert_mask == 0) | (expert_mask == 1)
            ), "Aiter Fused MoE kernel only supports expert_map with 0 and 1s."
            # AITER kernel要求expert_mask只包含0和1
            # 0表示专家不在当前GPU，1表示在当前GPU
            # 这是ROCm AITER实现的硬性要求

        # ============================================================
        # 16. 设置MoE层的基本参数
        # ============================================================
        assert intermediate_size % self.tp_size == 0  # 确保中间层大小可以被TP整除
        self.hidden_size = hidden_size  # 隐藏层大小（可能已向上取整）
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        # 每个TP分区的中间层大小
        # 示例：intermediate_size=4096，TP=4 -> 每分区1024
        
        self.reduce_results = reduce_results  # 是否reduce输出（TP/EP时）
        self.renormalize = renormalize  # 是否重新归一化TopK权重
        
        # ============================================================
        # 17. 配置分组TopK（DeepSeek等模型使用）
        # ============================================================
        self.use_grouped_topk = use_grouped_topk  # 是否使用分组TopK
        # 分组TopK：先将专家分成若干组，每组内选Top-K，然后跨组再选
        # 示例：64专家分8组，每组8专家，每组选Top-2，最后从8组中选Top-4
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
            # 分组TopK必须指定组数和每组的K
        self.num_expert_group = num_expert_group  # 专家组数量
        self.topk_group = topk_group  # 每组选择的专家数
        
        # ============================================================
        # 18. 配置路由相关参数
        # ============================================================
        self.custom_routing_function = custom_routing_function  # 自定义路由函数（可选）
        self.scoring_func = scoring_func  # 评分函数："softmax"或"sigmoid"
        # - softmax: 归一化后选TopK（传统方法）
        # - sigmoid: 独立评分后选TopK（DeepSeekV3/Llama4）
        
        self.routed_scaling_factor = routed_scaling_factor  # 路由权重缩放因子
        # 某些模型会缩放专家输出权重，如0.5倍
        
        self.e_score_correction_bias = e_score_correction_bias  # 专家分数修正偏置
        # DeepSeek等模型使用，用于校正专家偏好
        
        self.apply_router_weight_on_input = apply_router_weight_on_input
        # 是否在输入上应用路由权重（不常用）
        
        self.activation = activation  # 激活函数类型（如"silu"）

        # ============================================================
        # 19. 验证scoring_func与topk模式的兼容性
        # ============================================================
        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            # 非softmax的评分函数（如sigmoid）只支持分组TopK
            raise ValueError(
                "Only softmax scoring function is supported for non-grouped topk."
            )

        # ============================================================
        # 20. 确定路由方法类型（影响kernel选择）
        # ============================================================
        # ToDo: Better logic to determine the routing method type
        if routing_method_type is not None:  # 如果显式指定了
            self.routing_method_type = routing_method_type
        else:  # 否则根据其他参数自动推断
            if scoring_func == "sigmoid":  # Sigmoid评分
                if self.use_grouped_topk:  # 分组TopK
                    self.routing_method_type = RoutingMethodType.DeepSeekV3
                    # DeepSeekV3路由：Sigmoid + 分组TopK
                elif self.top_k == 1:  # Top-1选择
                    self.routing_method_type = RoutingMethodType.Llama4
                    # Llama4路由：Sigmoid + Top-1
            elif self.scoring_func == "softmax":  # Softmax评分（最常见）
                self.routing_method_type = (
                    RoutingMethodType.Renormalize  # Softmax后再做TopK再归一化
                    if not self.renormalize
                    else RoutingMethodType.RenormalizeNaive  # 简单的Softmax+TopK
                )
            else:  # 其他情况
                self.routing_method_type = RoutingMethodType.TopK  # 基础TopK

        # ============================================================
        # 21. 创建MoE层配置对象
        # ============================================================
        self.moe_config: FusedMoEConfig = FusedMoEConfig(
            num_experts=self.global_num_experts,  # 全局专家数（含冗余）
            experts_per_token=top_k,  # 每个token选择的专家数
            hidden_dim=hidden_size,  # 隐藏层维度
            num_local_experts=self.local_num_experts,  # 本地专家数（EP后）
            moe_parallel_config=self.moe_parallel_config,  # 并行配置
            in_dtype=moe_in_dtype,  # 输入激活值类型（fp16/bf16）
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,  # DP分块的最大token数
            has_bias=has_bias,  # 专家MLP是否有bias
            is_act_and_mul=is_act_and_mul,  # 是否使用SiluAndMul融合
            is_lora_enabled=vllm_config.lora_config is not None,  # 是否启用LoRA
        )
        # 这个配置对象会传递给各种MoE kernel
        
        self.moe_config_use_flashinfer_cutlass_kernels = (
            self.moe_config.use_flashinfer_cutlass_kernels
        )  # 缓存是否使用FlashInfer CUTLASS kernels

        # ============================================================
        # 22. 设置量化配置
        # ============================================================
        self.quant_config = quant_config  # 保存量化配置对象（如FP8Config）

        def _get_quant_method() -> FusedMoEMethodBase:
            """
            获取量化方法对象的辅助函数。
            
            确保self.quant_method永远不为None且类型正确。
            
            选择逻辑：
            1. 如果有quant_config，从config获取对应的quant_method
            2. 否则使用UnquantizedFusedMoEMethod（标准fp16/bf16）
            
            Returns:
                FusedMoEMethodBase: 量化方法对象
            """
            quant_method = None
            if self.quant_config is not None:  # 如果指定了量化
                # 从量化配置获取对应的方法（如Fp8MoEMethod）
                quant_method = self.quant_config.get_quant_method(self, prefix)
            if quant_method is None:  # 如果没有量化或获取失败
                # 使用未量化方法（标准计算）
                quant_method = UnquantizedFusedMoEMethod(self.moe_config)
            assert isinstance(quant_method, FusedMoEMethodBase)  # 类型检查
            return quant_method

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        # 注意：get_quant_method会查看local_num_experts来做启发式选择，
        # 所以必须先初始化local_num_experts
        self.quant_method: FusedMoEMethodBase = _get_quant_method()
        # 保存量化方法对象，后续forward时会调用其apply方法

        # ============================================================
        # 23. 验证激活-乘法模式的兼容性
        # ============================================================
        if not self.moe_config.is_act_and_mul:
            # is_act_and_mul=False: 不使用SiluAndMul融合（较罕见）
            # 只有特定量化方法支持这种模式
            
            # Avoid circular import
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8MoEMethod,
                ModelOptNvFp4FusedMoE,
            )

            if not isinstance(
                self.quant_method,
                (
                    UnquantizedFusedMoEMethod,  # 未量化
                    ModelOptFp8MoEMethod,  # ModelOpt FP8
                    ModelOptNvFp4FusedMoE,  # ModelOpt NvFp4
                ),
            ):
                raise NotImplementedError(
                    "is_act_and_mul=False is supported only for unquantized "
                    ", ModelOpt FP8, and ModelOpt NvFp4 checkpoints"
                )
            if not current_platform.is_cuda():  # 且只支持CUDA
                raise NotImplementedError(
                    "is_act_and_mul=False is supported only for CUDA for now"
                )

        # ============================================================
        # 24. 验证EPLB与量化方法的兼容性
        # ============================================================
        if self.enable_eplb and not self.quant_method.supports_eplb:
            # EPLB目前只支持部分量化方法
            # TODO: Add support for additional quantization methods.
            # The implementation for other quantization methods does not
            # contain essential differences, but the current quant API
            # design causes duplicated work when extending to new
            # quantization methods, so I'm leaving it for now.
            # If you plan to add support for more quantization methods,
            # please refer to the implementation in `Fp8MoEMethod`.
            raise NotImplementedError(
                f"EPLB is not supported {self.quant_method.__class__.__name__}. "
                "EPLB is only supported for FP8 quantization for now."
            )

        # ============================================================
        # 25. 创建专家权重
        # ============================================================
        # 准备传递给create_weights的参数
        moe_quant_params = {
            "num_experts": self.local_num_experts,  # 本地专家数
            "hidden_size": hidden_size,  # 隐藏层大小
            "intermediate_size_per_partition": self.intermediate_size_per_partition,  # 每分区中间层大小
            "params_dtype": params_dtype,  # 参数类型
            "weight_loader": self.weight_loader,  # 权重加载函数
            "global_num_experts": self.global_num_experts,  # 全局专家数
        }
        # need full intermediate size pre-sharding for WNA16 act order
        # WNA16（Weight-only INT4/8 Activation FP16）量化需要完整的中间层大小
        if self.quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",  # GPTQ Marlin量化
            "CompressedTensorsWNA16MarlinMoEMethod",  # CompressedTensors WNA16 Marlin
            "CompressedTensorsWNA16MoEMethod",  # CompressedTensors WNA16
        ):
            moe_quant_params["intermediate_size_full"] = intermediate_size
            # 传入完整大小（TP切分前）用于正确的act order处理

        # 调用量化方法的create_weights创建权重张量
        # 这会注册w13_weight, w2_weight等参数到模块
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # ============================================================
        # 26. 初始化DP分块的staging tensors
        # ============================================================
        # Chunked all2all staging tensor
        # 用于数据并行分块计算时的临时缓冲区
        self.batched_hidden_states: torch.Tensor | None = None  # 批处理的隐藏状态缓冲
        self.batched_router_logits: torch.Tensor | None = None  # 批处理的路由logits缓冲
        # 这些会在第一次forward时根据需要创建

    # Note: maybe_init_modular_kernel should only be called by
    # prepare_communication_buffer_for_model.
    # This is called after all weight loading and post-processing, so it
    # should be safe to swap out the quant_method.
    def maybe_init_modular_kernel(self) -> None:
        self.ensure_moe_quant_config_init()
        # routing_tables only needed for round-robin expert placement with
        # DeepEP all2all backend.
        routing_tables = self._maybe_init_expert_routing_tables()
        prepare_finalize = self.quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            self.quant_method = FusedMoEModularMethod.make(
                self, self.quant_method, prepare_finalize, self.shared_experts
            )

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return None

    @property
    def gate(self) -> torch.nn.Module | None:
        return None

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def pcp_size(self):
        return self.moe_parallel_config.pcp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def pcp_rank(self):
        return self.moe_parallel_config.pcp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels

    @property
    def use_flashinfer_cutlass_kernels(self):
        return (
            self.moe_quant_config is not None
            and self.moe_quant_config.quant_dtype == "nvfp4"
            and self.moe_config_use_flashinfer_cutlass_kernels
        )

    @property
    def use_marlin_kernels(self):
        return getattr(self.quant_method, "use_marlin", False)

    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_parallel_config.use_pplx_kernels
            or self.moe_parallel_config.use_deepep_ll_kernels
            or (self.dp_size > 1 and self.use_flashinfer_cutlass_kernels)
        ) and envs.VLLM_ENABLE_MOE_DP_CHUNK

    @property
    def is_internal_router(self) -> bool:
        # By default, router/gate is called before FusedMoE forward pass
        return False

    def _maybe_init_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        # Currently routing_tables only needed for round-robin expert placement
        # with DeepEP-ll all2all backend.
        if (
            self.expert_placement_strategy != "round_robin"
            or not self.use_deepep_ll_kernels
        ):
            return None

        if hasattr(self, "expert_global_to_physical"):
            return cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                (
                    self.expert_global_to_physical,
                    self.expert_physical_to_global,
                    self.expert_local_to_global,
                ),
            )

        if self._expert_map is None:
            return None

        routing_tables = self.ensure_round_robin_expert_routing_tables(
            global_num_experts=self.global_num_experts,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            local_num_experts=self.local_num_experts,
            device=self._expert_map.device,
        )

        global_to_physical, physical_to_global, local_global = routing_tables
        self.register_buffer("expert_global_to_physical", global_to_physical)
        self.register_buffer("expert_physical_to_global", physical_to_global)
        self.register_buffer("expert_local_to_global", local_global)

        return routing_tables

    @staticmethod
    def ensure_round_robin_expert_routing_tables(
        global_num_experts: int,
        ep_size: int,
        ep_rank: int,
        local_num_experts: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device_kwargs = {"device": device} if device is not None else {}
        global_indices = torch.arange(
            global_num_experts, dtype=torch.long, **device_kwargs
        )
        owner = torch.remainder(global_indices, ep_size)
        local_index = torch.div(global_indices, ep_size, rounding_mode="floor")
        base = global_num_experts // ep_size
        remainder = global_num_experts % ep_size
        physical_offset = owner * base
        if remainder > 0:
            remainder_tensor = torch.tensor(
                remainder, dtype=torch.long, **device_kwargs
            )
            physical_offset = physical_offset + torch.minimum(owner, remainder_tensor)

        global_to_physical = physical_offset + local_index
        physical_to_global = torch.empty_like(global_to_physical)
        physical_to_global[global_to_physical] = global_indices

        local_global = torch.arange(
            ep_rank,
            global_num_experts,
            ep_size,
            dtype=torch.long,
            **device_kwargs,
        )
        if local_global.numel() != local_num_experts:
            local_global = local_global[:local_num_experts]

        return (global_to_physical, physical_to_global, local_global)

    def update_expert_map(self):
        # ep_size and ep_rank should already be updated
        assert self._expert_map is not None
        with self._expert_map.device:
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_fmoe_enabled,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("_expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            self._maybe_init_expert_routing_tables()
            if self.aiter_fmoe_shared_expert_enabled:
                self._init_aiter_shared_experts_topK_buffer(
                    vllm_config=get_current_vllm_config(),
                    dp_size=get_dp_group().world_size,
                )

    def _maybe_setup_shared_experts_stream(
        self,
        hidden_states: torch.Tensor,
        has_separate_shared_experts: bool,
        use_chunked_impl: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        use_shared_experts_stream = (
            current_platform.is_cuda()
            and has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
        )

        hidden_states_clone: torch.Tensor | None = None
        if use_shared_experts_stream:
            assert self.shared_experts_stream is not None

            # Clone BEFORE switching streams to avoid race condition
            # where routed_expert kernel may mutate hidden_states.
            hidden_states_clone = hidden_states.clone()

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            hidden_states_clone.record_stream(self.shared_experts_stream)

            # Mark sync start point for the separate shared experts
            # stream here since we want to run in parallel with the
            # router/gate (next op below)
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())

        return use_shared_experts_stream, hidden_states_clone

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_combined_w13_weight_scale(
        self,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        param: torch.Tensor,
        tp_rank: int,
    ):
        """
        Load w13 weight scales assuming that w1 weight scales and w3 weight
        scales are stored in the same loaded_weight tensor.
        """
        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        param.copy_(loaded_weight)

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full_w2: bool = False,
    ):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full_w2,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        if self.moe_config.is_act_and_mul:
            shard_size = expert_data.shape[shard_dim] // 2
        else:
            shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self._expert_map is None:
            return expert_id
        return self._expert_map[expert_id].item()

    def _init_aiter_shared_experts_topK_buffer(
        self, vllm_config: VllmConfig, dp_size: int
    ):
        if self.num_fused_shared_experts > 0:
            init_aiter_topK_meta_data(
                n_routed_experts=self.global_num_experts,
                n_shared_experts=self.num_fused_shared_experts,
                top_k=self.top_k,
                tp_rank=self.ep_rank if self.use_ep else self.tp_rank,
                tp_size=self.ep_size if self.use_ep else self.tp_size,
                shared_experts_score=1.0,
                max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens
                * dp_size,
                is_EP=self.use_ep,
            )
        self.local_num_experts += self.num_fused_shared_experts

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[False],
    ) -> None: ...

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[True],
    ) -> bool: ...

    def weight_loader(
        self,
        param: torch.nn.Parameter,  # 目标参数（如self.w13_weight）
        loaded_weight: torch.Tensor,  # 从checkpoint加载的权重
        weight_name: str,  # 权重的完整名称（如"model.layers.0.mlp.experts.0.gate_proj.weight"）
        shard_id: str,  # 分片ID："w1"(gate)/"w2"(down)/"w3"(up)
        expert_id: int,  # 全局专家ID（0到global_num_experts-1）
        return_success: bool = False,  # 是否返回加载成功标志
    ) -> bool | None:
        """
        MoE专家权重加载器 - 将checkpoint中的权重加载到模型参数中。
        
        这个方法处理各种复杂情况：
        1. 专家并行(EP)：只加载分配给当前rank的专家
        2. 张量并行(TP)：对权重进行切分
        3. 量化：处理量化权重和缩放因子
        4. 权重合并：w1和w3合并为w13
        
        Args:
            param: 目标参数张量（如w13_weight [num_experts, 2*intermediate, hidden]）
            loaded_weight: 从checkpoint加载的单个专家权重
            weight_name: 权重的完整限定名称
            shard_id: 标识是哪个权重："w1"/"w2"/"w3"
            expert_id: 全局专家索引
            return_success: 是否返回bool标志
        
        Returns:
            bool | None: 如果return_success=True，返回加载是否成功；否则返回None
        
        示例：
            # 加载Mixtral-8x7B的第2个专家的gate_proj权重
            weight_name = "model.layers.0.mlp.experts.2.gate_proj.weight"
            shard_id = "w1"  # gate_proj对应w1
            expert_id = 2
            loaded_weight shape: [intermediate_size, hidden_size]
            
            # EP=2，当前rank=0，专家0-3在rank0，专家4-7在rank1
            if expert_id < 4:  # 专家2在当前rank
                # 映射到本地索引2
                # 加载到param.data[2][0:intermediate_size, :]（w13的前半部分）
        """
        if self.quant_config and self.quant_config.get_name() == "mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        quant_method_name = self.quant_method.__class__.__name__
        global_expert_id = expert_id
        expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)

        allow_flashinfer = getattr(self.quant_method, "allow_flashinfer", False)
        moe_backend = getattr(self.quant_method, "flashinfer_moe_backend", None)

        use_global_sf = (
            allow_flashinfer
            and is_flashinfer_supporting_global_sf(moe_backend)
            and "input_scale" in weight_name
            and quant_method_name == "ModelOptNvFp4FusedMoE"
        )

        if expert_id == -1 and not use_global_sf:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            loaded_weight = loaded_weight.t().contiguous()

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BNB inflight quantization has already sharded the weights
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter accounting merged weights
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            # To materialize a tensor, we must have full shape including
            # number of experts, making this portion to require `full_load`.
            assert full_load
            final_shape = list(loaded_weight.shape)
            # w1 and w3 are merged per expert.
            if shard_id in {"w1", "w3"}:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                "compressed" in quant_method_name.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=global_expert_id if use_global_sf else expert_id,
            )
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            is_per_tensor = (
                "weight_scale_2" in weight_name
                if uses_weight_scale_2
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=expert_data,
                        tp_rank=self.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        return False if return_success else None

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        if (expert_mapping := self.expert_mapping) is None:
            raise ValueError(
                "`self.expert_mapping` must be provided to "
                "load weights using `self.load_weights`."
            )
        for expert_name, loaded_weight in weights:
            qual_name = f"{self.layer_name}.{expert_name}"
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in qual_name:
                    continue
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                param = getattr(self, param_name)
                success = self.weight_loader(
                    param=param,
                    loaded_weight=loaded_weight,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    logger.debug(
                        "Loaded %s for expert %d into %s",
                        param_name,
                        expert_id,
                        self.layer_name,
                    )
                    yield param_name

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        def _maybe_make_contiguous(
            name: str, p: torch.nn.Parameter
        ) -> torch.nn.Parameter:
            """
            In some cases, the last 2 dimensions (the non-expert dimensions)
            of the weight scale tensor are transposed. This function
            transforms the tensor (view update) so the tensor is contiguous().
            Example: A non-contiguous scale tensor,
              `x` of shape (E, 32, 16) and stride (512, 1, 32) is transformed to
              `x_` of shape (E, 16, 32) and stride (512, 32, 1).
              Note that we specifically use torch.transpose() so `x_` refers
              to the same underlying memory. The tensors `x` and `x_`, pointing
              to the same underlying memory make this transformation safe in the
              context of EPLB. i.e. It is the same memory and just the view
              is different.
            Note: This function handles the "weight_scale" tensors specifically.
            This could however be generalized to handle similar tensors.
            """
            if p.ndim != 3:
                return p
            if p.is_contiguous():
                # Already contiguous. do nothing.
                return p
            # p is non-contiguous. We only handle the case where the last 2
            # dimensions of the scales tensor is transposed. We can handle
            # other cases when they become relevant.
            is_transposed_12 = p.stride(1) == 1 and p.stride(2) != 1
            if "weight_scale" not in name or not is_transposed_12:
                # do nothing.
                return p

            # Do not update the layer parameter as the layer's MoE operations would
            # expect the parameter's tensor to the same shape / stride. Instead,
            # make a new torch.nn.Parameter that is used just in the context of
            # EPLB.
            return torch.nn.Parameter(
                torch.transpose(p.data, 1, 2), requires_grad=False
            )

        weights = list(self.named_parameters())
        weights = [(name, _maybe_make_contiguous(name, p)) for name, p in weights]

        assert all(
            weight.is_contiguous()
            for name, weight in weights
            if not name.startswith("_shared_experts.")
        )

        # Filter out the non-expert weights.
        # `e_score_correction_bias` is a bias for each logical expert,
        # with shape (num_logical_experts,), not an expert weight.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        return [
            weight.view(self.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
            and weight.shape != torch.Size([])
            and not name.startswith("_shared_experts.")
            # exclude parameters from non-expert submodules (e.g. gate/shared)
            and not name.startswith("_gate.")
        ]

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]

    def ensure_moe_quant_config_init(self):
        if self.quant_method.moe_quant_config is None:
            # Note: the moe_quant_config can't be constructed until after
            # weight loading post processing.
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self)
            )

    @property
    def moe_quant_config(self) -> FusedMoEQuantConfig | None:
        self.ensure_moe_quant_config_init()
        return self.quant_method.moe_quant_config

    def ensure_dp_chunking_init(self):
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.vllm_config.parallel_config.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.hidden_size)
            logits_shape = (2, moe.max_num_tokens, self.logical_num_experts)
        else:
            states_shape = (moe.max_num_tokens, self.hidden_size)
            logits_shape = (moe.max_num_tokens, self.logical_num_experts)

        self.batched_hidden_states = torch.zeros(
            states_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

        self.batched_router_logits = torch.zeros(
            logits_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

    def select_experts(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态 [num_tokens, hidden_size]
        router_logits: torch.Tensor,  # 路由器输出的logits [num_tokens, num_experts]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据路由器logits将输入隐藏状态路由到Top-K专家。
        
        这是MoE的核心路由逻辑，决定每个token使用哪些专家。
        
        主要流程：
        1. 计算专家分数（softmax/sigmoid）
        2. 选择Top-K专家
        3. （可选）应用EPLB负载均衡
        4. 返回专家权重和ID
        
        支持的路由策略：
        - 标准TopK：Softmax -> TopK -> 重归一化
        - 分组TopK：先组内TopK，再跨组TopK（DeepSeekV2/V3）
        - Sigmoid路由：Sigmoid -> TopK（Llama4）
        - 自定义路由：通过custom_routing_function
        
        Args:
            hidden_states: 输入隐藏状态张量
                shape: [num_tokens, hidden_size]
            router_logits: 路由器输出的原始logits
                shape: [num_tokens, num_experts]
                每个token对每个专家的原始分数

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (topk_weights, topk_ids)
                - topk_weights: Top-K专家的归一化权重
                    shape: [num_tokens, top_k]
                    示例：Top-2时可能是[[0.6, 0.4], [0.7, 0.3], ...]
                - topk_ids: 选中的专家ID
                    shape: [num_tokens, top_k]
                    示例：Top-2时可能是[[2, 5], [1, 7], ...]
                    
            **兼容性说明**: 
            - 当EPLB未启用时，返回的ID是全局逻辑ID，与普通MoE实现兼容
            - 当EPLB启用时，返回的ID是物理ID（可能包含冗余专家）
        
        示例1 - 标准TopK（Softmax）：
            router_logits = [[1.0, 2.0, 0.5, 1.5], ...]  # 4个专家
            top_k = 2
            
            # 计算softmax
            scores = softmax([[1.0, 2.0, 0.5, 1.5]]) = [[0.15, 0.41, 0.09, 0.27]]
            
            # 选择Top-2
            topk_ids = [[1, 3]]  # 专家1和3分数最高
            
            # 重归一化（只对选中的专家）
            topk_weights = [[0.41, 0.27]] / (0.41+0.27) = [[0.60, 0.40]]
            
        示例2 - 分组TopK（DeepSeekV2）：
            num_experts = 64, num_expert_group = 8, topk_group = 2, top_k = 4
            
            # 第1步：将64个专家分成8组，每组8个专家
            # 第2步：每组内选Top-2 -> 得到8组 x 2专家 = 16个候选
            # 第3步：从16个候选中选Top-4 -> 最终4个专家
        """
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk,
            fused_topk_bias,
        )

        if self.enable_eplb:
            if self.quant_method.supports_eplb:
                if self.expert_load_view is None:
                    raise ValueError(
                        "enable_eplb=True requiere expert_load_view != None"
                    )
                if self.logical_to_physical_map is None:
                    raise ValueError(
                        "enable_eplb=True requiere logical_to_physical_map != None"
                    )
                if self.logical_replica_count is None:
                    raise ValueError(
                        "enable_eplb=True requiere logical_replica_count != None"
                    )
            else:
                raise NotImplementedError(
                    f"EPLB is not supported for {self.quant_method.method_name}."
                )

        def valid_grouping() -> bool:
            # Check if num_experts is greater than num_expert_group
            # and is divisible by num_expert_group
            num_experts = router_logits.shape[-1]
            if num_experts <= self.num_expert_group:
                return False
            return num_experts % self.num_expert_group == 0

        indices_type = self.quant_method.topk_indices_dtype

        # Check if we should use a routing simulation strategy
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=self.top_k,
                indices_type=indices_type,
            )

        # DeepSeekv2 uses grouped_top_k
        elif self.use_grouped_topk and valid_grouping():
            assert self.topk_group is not None
            assert self.num_expert_group is not None
            grouped_topk_impl = GroupedTopk(
                topk=self.top_k,
                renormalize=self.renormalize,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                num_fused_shared_experts=self.num_fused_shared_experts,
            )

            topk_weights, topk_ids = grouped_topk_impl(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=self.e_score_correction_bias,
            )
        elif self.e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=self.e_score_correction_bias.data,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
            if self.routed_scaling_factor != 1.0:
                topk_weights *= self.routed_scaling_factor
        elif self.custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = self.custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )

        if self.enable_eplb:
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None

        return topk_weights, topk_ids

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        assert self.quant_method is not None
        return (
            isinstance(self.quant_method, FusedMoEModularMethod)
            and self.quant_method.fused_experts.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(
                hidden_states,
                (0, self.hidden_size - og_hidden_states),
                mode="constant",
                value=0.0,
            )

        def reduce_output(states: torch.Tensor) -> torch.Tensor:
            if (
                not self.is_sequence_parallel
                and not self.use_dp_chunking
                and self.reduce_results
                and (self.tp_size > 1 or self.ep_size > 1)
            ):
                states = self.maybe_all_reduce_tensor_model_parallel(states)
            return states

        if self.shared_experts is None:
            if current_platform.is_tpu() or current_platform.is_cpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                # Note: CPU doesn't require wrapped forward_impl.
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name
                )
            return reduce_output(fused_output)[..., :og_hidden_states]
        else:
            if current_platform.is_tpu() or current_platform.is_cpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                # Note: CPU doesn't require wrapped forward_impl.
                shared_output, fused_output = self.forward_impl(
                    hidden_states, router_logits
                )
            else:
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states, router_logits, self.layer_name
                )
            return (
                reduce_output(shared_output)[..., :og_hidden_states],
                reduce_output(fused_output)[..., :og_hidden_states],
            )

    @property
    def expert_map(self) -> torch.Tensor | None:
        return (
            self._expert_map if not self.rocm_aiter_fmoe_enabled else self.expert_mask
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(hidden_states, router_logits)

    def forward_impl_chunked(
        self,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
        has_separate_shared_experts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype
        assert self.batched_router_logits.dtype == full_router_logits.dtype
        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=staged_hidden_states,
                router_logits=staged_router_logits,
            )

            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                shared_output = self.shared_experts(staged_hidden_states)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
                    )

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.sp_size
            )

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(
                self.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态 [num_tokens, hidden_size]
        router_logits: torch.Tensor,  # 路由器logits [num_tokens, num_experts]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        MoE层的前向传播实现 - 核心计算逻辑。
        
        完整流程：
        1. 初始化量化配置和DP分块缓冲区
        2. 设置共享专家的并行stream（如果有）
        3. （可选）应用路由器
        4. 选择Top-K专家（通过select_experts）
        5. 执行专家计算（路由专家 + 共享专家）
        6. 聚合结果
        7. （可选）执行all2all通信（DP/EP）
        
        支持的模式：
        - 标准模式：路由专家 only
        - 共享专家模式：路由专家 + 共享专家（DeepSeekV2）
        - DP分块模式：大batch时分chunk计算
        - 独立stream模式：共享专家与路由专家并行计算
        
        Args:
            hidden_states: 输入张量 [num_tokens, hidden_size]
            router_logits: 路由器输出 [num_tokens, num_experts]
        
        Returns:
            torch.Tensor 或 tuple[torch.Tensor, torch.Tensor]:
                - 如果只有路由专家：返回 [num_tokens, hidden_size]
                - 如果有共享专家：返回 (shared_output, routed_output)
                    两个张量都是 [num_tokens, hidden_size]
        
        示例：
            # Mixtral-8x7B：只有路由专家
            hidden_states: [1024, 4096]  # 1024个token
            router_logits: [1024, 8]     # 8个专家
            output: [1024, 4096]          # 聚合后的输出
            
            # DeepSeekV2：路由专家 + 共享专家
            hidden_states: [1024, 4096]
            router_logits: [1024, 64]    # 64个路由专家
            output: ([1024, 4096], [1024, 4096])  # (共享输出, 路由输出)
        """
        assert self.quant_method is not None  # 确保量化方法已初始化

        # ============================================================
        # 1. 初始化量化配置和DP分块
        # ============================================================
        self.ensure_moe_quant_config_init()  # 确保量化配置已初始化（延迟初始化）
        self.ensure_dp_chunking_init()  # 确保DP分块缓冲区已创建

        # ============================================================
        # 2. 检查是否有独立的共享专家
        # ============================================================
        has_separate_shared_experts = (
            not isinstance(self.quant_method, FusedMoEModularMethod)  # 不是模块化方法
            and self.shared_experts is not None  # 且有共享专家
        )
        # 共享专家：所有token都会经过（如DeepSeekV2）
        # 如果是FusedMoEModularMethod，共享专家已融合到主kernel中

        use_chunked_impl = self.use_dp_chunking  # 是否使用分块实现（大batch时）

        # ============================================================
        # 3. 设置共享专家的独立CUDA stream
        # ============================================================
        use_shared_experts_stream, hidden_states_clone = (
            self._maybe_setup_shared_experts_stream(
                hidden_states, has_separate_shared_experts, use_chunked_impl
            )
        )
        # 如果启用，共享专家会在独立stream上并行计算
        # hidden_states_clone: 为共享专家准备的输入副本（避免race condition）

        # ============================================================
        # 4. （可选）应用内部路由器
        # ============================================================
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:  # 如果MoE层有内部gate（不常见）
            router_logits, _ = self.gate(hidden_states)  # 计算路由logits
            # 大多数模型的gate在MoE层外部，所以self.gate通常为None

        # ============================================================
        # 4. （可选）使用分块实现（大batch场景）
        # ============================================================
        if use_chunked_impl:
            # 如果batch太大，使用分块处理避免OOM
            # 适用场景：DP + PPLX/DeepEP-LL/FlashInfer-CUTLASS
            return self.forward_impl_chunked(
                hidden_states, router_logits, has_separate_shared_experts
            )

        # ============================================================
        # 5. 判断是否需要 naive dispatch/combine（DP场景）
        # ============================================================
        # naive dispatch/combine: 简单的DP all-to-all + combine
        # 条件：DP size > 1 且不使用ModularMethod（ModularMethod有更优化的实现）
        do_naive_dispatch_combine: bool = self.dp_size > 1 and not isinstance(
            self.quant_method, FusedMoEModularMethod
        )

        # ============================================================
        # 6. 获取序列并行上下文
        # ============================================================
        ctx = get_forward_context()  # 获取当前forward的全局上下文
        sp_ctx = (
            # 如果有DP metadata，设置序列并行的local sizes
            # 用于在SP场景下正确切分序列
            ctx.dp_metadata.sp_local_sizes(self.sp_size)
            if ctx.dp_metadata
            else nullcontext()  # 否则使用空上下文（无操作）
        )

        # ============================================================
        # 7. 在序列并行上下文中执行 DP dispatch
        # ============================================================
        with sp_ctx:
            extra_tensors = None  # 额外的tensor（用于某些量化方法）
            
            if do_naive_dispatch_combine:
                # --------------------------------------------------------
                # 7.1 检查是否使用 post-quantization all-gather
                # --------------------------------------------------------
                # Avoid circular import
                from vllm.model_executor.layers.quantization.modelopt import (
                    ModelOptNvFp4FusedMoE,
                )

                # post_quant_allgather: 在量化后执行all-gather
                # 适用于 FlashInfer + ModelOpt NvFp4 + DP + EP 的组合
                # 优势：减少通信量（传输量化后的数据）
                post_quant_allgather = (
                    has_flashinfer_trtllm_fused_moe()  # 有FlashInfer TRT-LLM kernel
                    and self.quant_method is not None  # 有量化方法
                    and self.dp_size > 1  # 使用数据并行
                    and self.use_ep  # 使用专家并行
                    and isinstance(self.quant_method, ModelOptNvFp4FusedMoE)  # 是NvFp4量化
                )
                
                # --------------------------------------------------------
                # 7.2 准备dispatch的数据
                # --------------------------------------------------------
                if post_quant_allgather:
                    # 如果使用post-quant all-gather，需要准备额外的量化tensor
                    # extra_tensors: 量化相关的辅助数据（如scale、zero_point）
                    hidden_states_to_dispatch, extra_tensors = (
                        self.quant_method.prepare_dp_allgather_tensor(
                            self, hidden_states, router_logits
                        )
                    )
                else:
                    # 标准情况：直接dispatch原始hidden_states
                    hidden_states_to_dispatch = hidden_states

                # --------------------------------------------------------
                # 7.3 执行 EP dispatch（all-to-all通信）
                # --------------------------------------------------------
                # EP dispatch: 根据router logits将token分发到对应的专家所在GPU
                # 输入: [num_tokens, hidden_size] per GPU
                # 输出: [num_tokens_after_dispatch, hidden_size] per GPU
                #       (每个GPU接收需要本地专家处理的tokens)
                dispatch_res = get_ep_group().dispatch(
                    hidden_states_to_dispatch,  # 要分发的hidden states
                    router_logits,  # 路由器输出（决定token去哪个GPU）
                    self.is_sequence_parallel,  # 是否使用序列并行
                    extra_tensors=extra_tensors,  # 额外的量化tensor
                )
                
                # --------------------------------------------------------
                # 7.4 处理 dispatch 结果
                # --------------------------------------------------------
                if extra_tensors is not None:
                    # 如果有extra_tensors，dispatch返回3个值
                    hidden_states_combined, router_logits, extra_tensors_combined = (
                        dispatch_res
                    )
                    # 将hidden_states和extra_tensors组合成元组
                    # 后续量化kernel需要这两个tensor一起使用
                    hidden_states_combined = (
                        hidden_states_combined,
                        extra_tensors_combined[0],  # 通常是量化的scale或其他元数据
                    )
                else:
                    # 标准情况：只有hidden_states和router_logits
                    hidden_states_combined, router_logits = dispatch_res

            # ============================================================
            # 7. （可选）先计算共享专家
            # ============================================================
            # Run shared experts before matrix multiply.
            # because matrix multiply maybe modify the hidden_states.
            if has_separate_shared_experts and not use_shared_experts_stream:
                # 如果有共享专家且不使用独立stream
                # 在主stream上先计算共享专家
                assert self.shared_experts is not None
                shared_output = self.shared_experts(hidden_states)
                # shared_output: [num_tokens, hidden_size]

            # ============================================================
            # 8. （可选）PCP all-gather
            # ============================================================
            # NOTE: Similar with DP, PCP also needs dispatch and combine. For
            # simplicity, AgRsAll2All was added separately for PCP here. Maybe
            # we should modify All2AllManager abstract to better support PCP.
            if self.pcp_size > 1:  # 如果使用部分上下文并行
                # 将分散在各GPU上的序列片段聚合到一起
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,  # [num_tokens/pcp_size, hidden_size]
                    dim=0,  # 在token维度聚合
                )  # 输出: [num_tokens, hidden_size]
                
                router_logits = get_pcp_group().all_gather(
                    router_logits,  # [num_tokens/pcp_size, num_experts]
                    dim=0,
                )  # 输出: [num_tokens, num_experts]
                # 聚合后每个GPU都有完整的序列

            # ============================================================
            # 9. 执行专家计算（核心）
            # ============================================================
            # Matrix multiply.
            # 调用量化方法的apply执行实际的MoE计算
            final_hidden_states = self.quant_method.apply(
                layer=self,  # 当前MoE层
                x=hidden_states_combined  # 如果有DP dispatch则用combined
                if do_naive_dispatch_combine
                else hidden_states,  # 否则用原始hidden_states
                router_logits=router_logits,  # 路由logits
            )
            # 这个apply方法内部会：
            # 1. 调用select_experts选择Top-K专家
            # 2. 执行专家MLP计算（w13 -> activation -> w2）
            # 3. 加权聚合各专家输出
            # 输出: [num_tokens, hidden_size]

            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # Run shared experts in parallel on a separate stream
                    # NOTE: We start the separate stream here and mark the
                    # sync end point immediately after it is done. This is
                    # important to avoid excessive stream allocations by the cuda
                    # graph replay later.
                    with torch.cuda.stream(self.shared_experts_stream):
                        # Note that hidden_states clone() is necessary here to avoid
                        # conflict with the main stream
                        shared_output = self.shared_experts(hidden_states_clone)
                    current_stream().wait_stream(self.shared_experts_stream)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            def combine_output(states: torch.Tensor) -> torch.Tensor:
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(states, self.is_sequence_parallel)

                if self.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            if self.shared_experts is not None:
                return (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            else:
                return combine_output(final_hidden_states)

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = (
            EplbState.build_initial_global_physical_to_logical_map(
                num_experts, num_redundant_experts
            )
        )

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                "experts.w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else "experts.w2_",
                f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:
        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}"
        )

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s


def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is None
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 从forward context中获取FusedMoE层实例
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is not None  # 所有的moe模型一定有共享专家吗？
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_out = torch.empty_like(hidden_states)
    fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)

# Mark the FusedMoE weight_loader as supporting MoE-specific parameters
# to avoid expensive runtime reflection in model loading code
FusedMoE.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
