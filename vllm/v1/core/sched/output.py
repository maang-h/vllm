# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@dataclass
class NewRequestData:
    # 请求id
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    # 生成式模型（Chat / Completion）用的采样参数， 
    # temperature、top_p、top_k、max_tokens
    sampling_params: SamplingParams | None
    # Pooling 模型（Embedding / Classify / Rerank 等）用的参数，不做自回归「一个一个生成 token」
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    # 本步要更新的请求 ID 列表
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    # 主要给 PP 用，把上一步采样 token 传回
    new_token_ids: list[list[int]]
    # For requests not scheduled in the last step, propagate the token ids to the
    # connector. Won't contain requests that were scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    # 到本步调度为止，该请求 已经过模型前向、并写入 KV cache 的 token 总数
    num_computed_tokens: list[int]
    # 该请求 已生成的输出 token 数（并计入 async 占位），用于区分 prefill / decode、同步输出长度
    num_output_tokens: list[int]

    # Version of dataclass repr with token IDs obfuscated.
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    # 首次调度的请求列表
    # 在 Scheduler 里主要来自 RequestStatus.WAITING 刚被调度进 running 的请求
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    # 后续(非首次)调度的请求列表
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    # 每个请求调度的token数量
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    # 调度的总token数量
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    # 需要投机解码的请求的投机解码的token
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    # 每个request命中prefix缓存的block数量
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    # 已完成的请求的id
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    # 被抢占的请求的id
    preempted_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    # 是否有需要结构化输出的请求
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    # 调度请求是否具有执行语法位掩码计算所需的所有输出令牌。？？？
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation. ？？？
    # 在本轮投机解码里，每个请求有多少个 draft token 
    # 因不满足 grammar（结构化输出约束）而被判为无效
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None

    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    # 本调度步里刚从 KV block 池新分配、尚未使用过的 block ID 列表；
    new_block_ids_to_zero: list[int] | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
