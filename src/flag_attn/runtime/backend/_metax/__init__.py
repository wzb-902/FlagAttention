# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MetaX attention backend."""

from __future__ import annotations

import importlib

_OPERATOR_EXPORTS = {
    "hy3_INT8DecodeInputs": (
        ".hpc_ops_attention.decode.int8",
        "INT8DecodeInputs",
    ),
    "hy3_INT8DecodeWorkspace": (
        ".hpc_ops_attention.decode.int8",
        "INT8DecodeWorkspace",
    ),
    "hy3_attention": (
        ".hpc_ops_attention",
        "hy3_attention",
    ),
    "hy3_attention_decode_bf16_dynamic": (
        ".hpc_ops_attention.decode.bf16",
        "attention_decode_bf16_dynamic",
    ),
    "hy3_attention_decode_bf16_dynamic_prepare_workspace": (
        ".hpc_ops_attention.decode.bf16",
        "prepare_dynamic_bf16_workspace",
    ),
    "hy3_attention_decode_bf16_dynamic_refresh_task_map": (
        ".hpc_ops_attention.decode.bf16",
        "refresh_dynamic_bf16_task_map",
    ),
    "hy3_attention_decode_bf16_dynamic_select_policy": (
        ".hpc_ops_attention.decode.bf16",
        "select_dynamic_bf16_policy",
    ),
    "hy3_attention_decode_bf16_dynamic_workspace_is_reset": (
        ".hpc_ops_attention.decode.bf16",
        "bf16_dynamic_workspace_is_reset",
    ),
    "hy3_attention_decode_bf16_static": (
        ".hpc_ops_attention.decode.bf16",
        "attention_decode_bf16_static",
    ),
    "hy3_attention_decode_bf16_static_prepare_workspace": (
        ".hpc_ops_attention.decode.bf16",
        "prepare_static_bf16_workspace",
    ),
    "hy3_attention_decode_bf16_static_select_policy": (
        ".hpc_ops_attention.decode.bf16",
        "select_static_bf16_policy",
    ),
    "hy3_attention_decode_int8": (
        ".hpc_ops_attention.decode.int8",
        "attention_decode_int8",
    ),
    "hy3_attention_dispatch": (
        ".hpc_ops_attention",
        "hy3_attention",
    ),
    "hy3_attention_prefill_bf16": (
        ".hpc_ops_attention.prefill.bf16",
        "attention_with_kvcache_blocksparse_prefill_bf16",
    ),
    "hy3_attention_prefill_int8": (
        ".hpc_ops_attention.prefill.int8",
        "attention_with_kvcache_blocksparse_prefill_int8",
    ),
    "hy3_attention_prepare": (
        ".hpc_ops_attention",
        "hy3_attention_prepare",
    ),
    "hy3_prepare_int8_decode_workspace": (
        ".hpc_ops_attention.decode.int8",
        "prepare_int8_decode_workspace",
    ),
    "hy3_prepare_int8_prefill_inputs": (
        ".hpc_ops_attention.prefill.int8",
        "prepare_int8_prefill_inputs",
    ),
    "hy3_quantize_int8_decode_inputs": (
        ".hpc_ops_attention.decode.int8",
        "quantize_int8_decode_inputs",
    ),
    "SPARSE_BLOCK_SIZE": (".msa.index_topk", "SPARSE_BLOCK_SIZE"),
    "chunk_gdn2": (".FLA.gdn2", "chunk_gdn2"),
    "chunk_gla": (".FLA.gla", "chunk_gla"),
    "chunk_kda": (".FLA.kda", "chunk_kda"),
    "chunk_kda_fwd_infer": (".FLA.kda.chunk_kda", "chunk_kda_fwd_infer"),
    "fused_attnres": (".FLA.attnres.fused", "fused_attnres"),
    "minimax_m3_index_decode": (".msa", "minimax_m3_index_decode"),
    "minimax_m3_index_decode_score": (".msa", "minimax_m3_index_decode_score"),
    "minimax_m3_index_score": (".msa", "minimax_m3_index_score"),
    "minimax_m3_index_topk": (".msa", "minimax_m3_index_topk"),
    "minimax_m3_sparse_attn": (".msa", "minimax_m3_sparse_attn"),
    "minimax_m3_sparse_attn_decode": (".msa", "minimax_m3_sparse_attn_decode"),
    "parallel_nsa": (".FLA.nsa.parallel_nsa", "parallel_nsa"),
    "parallel_nsa_compression": (".FLA.nsa.parallel_nsa_compression", "parallel_nsa_compression"),
    "sage_attention": (".sage_attention.attn_qk_int8_per_block", "forward"),
    "sage_attention_forward": (".sage_attention.attn_qk_int8_per_block", "forward"),
    "sage_attention_per_block_int8": (".sage_attention.quant_per_block", "per_block_int8"),
}


def __getattr__(name: str):
    try:
        module_name, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(importlib.import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_OPERATOR_EXPORTS)
