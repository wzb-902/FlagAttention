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

"""Hardware-specific attention operators grouped by execution phase."""

from typing import Literal

from flag_attn.runtime.backend import resolve_operator

from . import decode, prefill

_HY3_DECODE_IMPLEMENTATIONS = {
    "int8": ("", "_unsupported_int8_decode"),
    "bf16_static": ("static.bf16_static", "attention_decode_bf16_static"),
    "bf16_dynamic": ("dynamic.bf16_dynamic", "attention_decode_bf16_dynamic"),
    "fp8_qk_static": ("static.fp8_qkpertoken_perhead_vperhead_static", "attention_decode_fp8"),
    "fp8_qk_dynamic": ("dynamic.fp8_qkpertoken_perhead_vperhead_dynamic", "attention_decode_fp8"),
    "fp8_kv_static": ("static.fp8_qpertoken_perhead_kvpertensor_static", "attention_decode_fp8"),
    "fp8_kv_dynamic": ("dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic", "attention_decode_fp8"),
}


_HY3_DECODE_PREPARE_FUNCTIONS = {
    "int8": "_unsupported_int8_decode",
    "bf16_static": "prepare_static_bf16_workspace",
    "bf16_dynamic": "prepare_dynamic_bf16_workspace",
    "fp8_qk_static": "prepare_decode_workspace",
    "fp8_qk_dynamic": "prepare_decode_workspace",
    "fp8_kv_static": "prepare_decode_workspace",
    "fp8_kv_dynamic": "prepare_decode_workspace",
}


def hy3_attention_prepare(inputs, *, stage: Literal["decode"] = "decode", variant=None, **kwargs):
    """Prepare a decode workspace using the same backend as hy3_attention.

    MetaX infers BF16/INT8 from inputs.q/k_cache/v_cache; variant is optional.
    Other backends retain explicit variants. For INT8,
    first quantize inputs with decode.quantize_int8_decode_inputs, then pass the
    outer-scale quant_type explicitly; adaptive_split is an optional keyword.
    Preparation can allocate and synchronize. Prepare outside graph capture and
    reuse the returned workspace with the same decode variant.
    """
    if stage != "decode":
        raise ValueError("Hy3 workspace preparation only supports stage='decode'")
    return resolve_operator("hy3_attention_prepare", __name__, "_generic_hy3_attention_prepare")(
        inputs, variant=variant, **kwargs
    )


def _generic_hy3_attention_prepare(inputs, *, variant, **kwargs):
    if variant not in _HY3_DECODE_IMPLEMENTATIONS:
        raise ValueError(f"Unsupported Hy3 decode variant: {variant!r}")
    module_name, _ = _HY3_DECODE_IMPLEMENTATIONS[variant]
    operator = (
        "hy3_prepare_int8_decode_workspace"
        if variant == "int8"
        else f"hy3_attention_decode_{variant}_prepare_workspace"
    )
    prepare = resolve_operator(operator, f"{__name__}.decode" + (f".{module_name}" if module_name else ""), _HY3_DECODE_PREPARE_FUNCTIONS[variant])
    return prepare(inputs, **kwargs)


def hy3_attention(*args, stage: Literal["prefill", "decode"] = "prefill", variant=None, **kwargs):
    """Run Hy3 attention through the selected backend.

    MetaX selects BF16 or INT8 from matching Q/K/V dtypes. Supply INT8 scales
    explicitly and prepare Decode workspaces with hy3_attention_prepare.
    Other backends retain their existing stage/variant contracts.
    """
    return resolve_operator("hy3_attention_dispatch", __name__, "_generic_hy3_attention")(
        *args, stage=stage, variant=variant, **kwargs
    )


def _generic_hy3_attention(
    *args,
    stage: Literal["prefill", "decode"] = "prefill",
    variant: Literal[
        "int8",
        "bf16",
        "bf16_static",
        "bf16_dynamic",
        "fp8_qk_static",
        "fp8_qk_dynamic",
        "fp8_kv_static",
        "fp8_kv_dynamic",
    ]
    | None = None,
    **kwargs,
):
    """Run Hy3 prefill or an explicitly selected decode implementation.

    Prefill variant="int8" uses prequantized INT8 Q/K/V, explicit FP32 scales,
    and BF16 output; quantization preparation is a separate API.
    Prefill variant="bf16" uses BF16 Q/K/V without quantization scales.
    With variant omitted, Prefill forwards the tensor/cache/scale arguments to
    prefill.attention_with_kvcache_blocksparse_prefill_fp8.
    Decode requires variant and forwards (inputs, workspace)
    unchanged. qk denotes per-token Q/K with per-head V quantization;
    kv denotes per-token Q with per-tensor K/V quantization. Prepare the
    workspace with hy3_attention_prepare using the same variant before calling.
    Decode variant="int8" uses prequantized INT8 inputs and returns BF16 output.
    """
    if stage == "prefill":
        if variant == "int8":
            return prefill.attention_with_kvcache_blocksparse_prefill_int8(*args, **kwargs)
        if variant == "bf16":
            return prefill.attention_with_kvcache_blocksparse_prefill_bf16(*args, **kwargs)
        if variant is not None:
            raise ValueError("Hy3 prefill variant must be omitted, 'bf16', or 'int8'")
        return resolve_operator(
            "hy3_attention",
            "flag_attn.hpc_ops_attention.prefill.attention_blocksparse_prefill_fp8",
            "attention_with_kvcache_blocksparse_prefill_fp8",
        )(*args, **kwargs)
    if stage != "decode":
        raise ValueError(f"Unsupported Hy3 stage: {stage!r}; expected 'prefill' or 'decode'")
    try:
        module_name, function_name = _HY3_DECODE_IMPLEMENTATIONS[variant]
    except KeyError as exc:
        raise ValueError(
            f"Hy3 decode requires variant in {tuple(_HY3_DECODE_IMPLEMENTATIONS)}; got {variant!r}"
        ) from exc
    implementation = resolve_operator(
        f"hy3_attention_decode_{variant}", f"{__name__}.decode" + (f".{module_name}" if module_name else ""), function_name
    )
    return implementation(*args, **kwargs)


__all__ = ["hy3_attention", "hy3_attention_prepare", "decode", "prefill"]
