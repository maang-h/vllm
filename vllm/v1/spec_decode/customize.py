import json
from typing import Any

import numpy as np
from numba import get_num_threads, set_num_threads

from vllm.config import VllmConfig


class CustomizeProposer:

    def __init__(self, vllm_config: VllmConfig):
        self.max_model_len = vllm_config.model_config.max_model_len
        self.rules: list[dict[str, list[int]]] = []
        spec_config = vllm_config.speculative_config
        if spec_config is not None:
            rule_file = getattr(spec_config, "customize_rule_file", None)
            if rule_file:
                self._read_rule(rule_file)

    def _read_rule(self, file: str):
        """Read customize speculative rules from a JSON file.

        Expected JSON format:
          [
            {"prefix": [11, 12], "fill": [657, 34, 5904]},
            {"prefix": [654, 43, -1], "fill": [77, 59, 988]}
          ]

        Semantics:
        - `prefix` matches the *suffix* of the current token sequence.
        - `-1` in `prefix` is a wildcard matching any token.
        - On first match, proposer returns `fill` as draft tokens.
        """
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"Customize spec decode rule file must be a JSON list, got: {type(data)}"
            )

        rules: list[dict[str, list[int]]] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Rule[{idx}] must be an object with keys 'prefix' and 'fill', "
                    f"got: {type(item)}"
                )
            if "prefix" not in item or "fill" not in item:
                raise ValueError(
                    f"Rule[{idx}] must contain keys 'prefix' and 'fill', got keys: "
                    f"{sorted(item.keys())}"
                )
            prefix = item["prefix"]
            fill = item["fill"]
            if (not isinstance(prefix, list) or
                    not all(isinstance(x, int) for x in prefix)):
                raise ValueError(
                    f"Rule[{idx}].prefix must be a list[int], got: {prefix!r}"
                )
            if not isinstance(fill, list) or not all(isinstance(x, int) for x in fill):
                raise ValueError(
                    f"Rule[{idx}].fill must be a list[int], got: {fill!r}"
                )
            if len(prefix) == 0:
                raise ValueError(f"Rule[{idx}].prefix must be non-empty.")
            rules.append({"prefix": prefix, "fill": fill})

        self.rules = rules

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ):
        # Must return a list with length == num_requests.
        draft_token_ids: list[list[int]] = [[] for _ in range(num_requests)]

        # Only run batch propose if there are requests needing ngram proposals.
        # avoid calling numba function with empty list which causes error
        # ValueError: cannot compute fingerprint of empty list
        if len(valid_ngram_requests):

            original_num_numba_threads = get_num_threads()
            set_num_threads(1)

            draft_token_ids = batch_propose_python(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.rules,
            )

            # Restore original number of threads.
            set_num_threads(original_num_numba_threads)
        print(f"投机解码: {draft_token_ids}")

        return draft_token_ids

    def propose(
        self,
        sampled_token_ids,
        req_ids: list[str],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        spec_decode_unsupported_reqs: set,
    ):
        # find which requests need ngram proposals
        valid_customize_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding.
            req_id = req_ids[i]
            if req_id in spec_decode_unsupported_reqs:
                continue

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                continue

            valid_customize_requests.append(i)

        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_customize_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )

        return draft_token_ids


def batch_propose_python(
    valid_requests: list[int],
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    rules: list[dict[str, list[int]]] | None,
) -> list[list[int]]:
    """Pure-python implementation (rules are Python dicts, not numba-friendly)."""
    out: list[list[int]] = []
    if not rules:
        return [[] for _ in range(len(valid_requests))]
    for req_idx in valid_requests:
        num_tokens = int(num_tokens_no_spec[req_idx])
        context = token_ids_cpu[req_idx, :num_tokens]
        out.append(merge_draft_tokens(context, rules))
    return out


def merge_draft_tokens(
    origin_tokens: np.ndarray,
    rules: list[dict[str, Any]] | None,
) -> list[int]:

    draft_tokens = []
    if rules is None:
        return draft_tokens

    # `prefix` matches the suffix of `origin_tokens`.
    for rule in rules:
        prefix_tokens = rule["prefix"]
        m = len(prefix_tokens)
        if m > len(origin_tokens):
            continue
        # Compare from the end.
        start = len(origin_tokens) - m
        ok = True
        for i in range(m):
            p = prefix_tokens[i]
            t = int(origin_tokens[start + i])
            if p != -1 and p != t:
                ok = False
                break
        if ok:
            return list(rule["fill"])

    return draft_tokens
