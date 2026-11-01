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

"""HPC attention operators for the prefill phase."""

from flag_attn.runtime.backend import resolve_operator as _resolve_operator

from .attention_blocksparse_prefill_fp8 import (
    attention_with_kvcache_blocksparse_prefill_fp8,
    attention_with_kvcache_blocksparse_prefill_fp8_hopper,
    attention_with_kvcache_blocksparse_prefill_fp8_tle,
    attention_with_kvcache_blocksparse_prefill_fp8_triton,
)



__all__ = [
    "attention_with_kvcache_blocksparse_prefill_int8",
    "prepare_int8_prefill_inputs",
    "attention_with_kvcache_blocksparse_prefill_bf16",
    "attention_with_kvcache_blocksparse_prefill_fp8",
    "attention_with_kvcache_blocksparse_prefill_fp8_hopper",
    "attention_with_kvcache_blocksparse_prefill_fp8_tle",
    "attention_with_kvcache_blocksparse_prefill_fp8_triton",
]




def _unsupported_bf16_prefill(*args, **kwargs):
    raise NotImplementedError("BF16 Hy3 Prefill is currently implemented for MetaX only")


def attention_with_kvcache_blocksparse_prefill_bf16(*args, **kwargs):
    """Run BF16 paged, causal block-sparse Prefill on a supported backend."""
    return _resolve_operator("hy3_attention_prefill_bf16", __name__, "_unsupported_bf16_prefill")(*args, **kwargs)




def _unsupported_int8_prefill(*args, **kwargs):
    raise NotImplementedError("Prequantized INT8 Prefill requires the MetaX backend")


def attention_with_kvcache_blocksparse_prefill_int8(*args, **kwargs):
    return _resolve_operator("hy3_attention_prefill_int8", __name__, "_unsupported_int8_prefill")(*args, **kwargs)


def prepare_int8_prefill_inputs(*args, **kwargs):
    return _resolve_operator("hy3_prepare_int8_prefill_inputs", __name__, "_unsupported_int8_prefill")(*args, **kwargs)
