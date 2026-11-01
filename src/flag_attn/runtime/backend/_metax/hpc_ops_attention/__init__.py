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

"""MetaX BF16/INT8 attention, selected by input dtype."""

import torch

from . import decode, prefill
from .decode import bf16, int8


def _input_dtype(q, k, v):
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
        raise TypeError("Q, K and V must be tensors")
    if q.dtype not in (torch.bfloat16, torch.int8):
        raise TypeError("MetaX Hy3 supports torch.bfloat16 or torch.int8 inputs only")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q, K and V must have the same dtype")
    return q.dtype


def _check_variant(variant, dtype, stage):
    if variant is None:
        return
    allowed = ("int8",) if dtype == torch.int8 else ("bf16",)
    if dtype == torch.bfloat16 and stage == "decode":
        allowed += ("bf16_static", "bf16_dynamic")
    if variant not in allowed:
        raise ValueError(f"variant={variant!r} does not match {stage} input dtype {dtype}")


def _reject_fp8(variant):
    if isinstance(variant, str) and variant.startswith("fp8"):
        raise NotImplementedError("MetaX FP8 Decode has been retired; use BF16 or INT8 inputs")


def _signature(inputs):
    return tuple(
        (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
        for t in (inputs.q, inputs.k_cache, inputs.v_cache, inputs.block_ids, inputs.kv_lens)
    )


def _schedule(variant, schedule, default="static"):
    legacy = {"bf16_static": "static", "bf16_dynamic": "dynamic"}.get(variant)
    if legacy is not None and schedule is not None and legacy != schedule:
        raise ValueError("variant and schedule disagree")
    selected = schedule or legacy or default
    if selected not in ("static", "dynamic"):
        raise ValueError("BF16 schedule must be 'static' or 'dynamic'")
    return selected


def hy3_attention_prepare(inputs, *, variant=None, schedule=None, **kwargs):
    """Prepare once outside graph capture; infer precision from Q/K/V."""
    _reject_fp8(variant)
    if inputs is None:
        raise ValueError("Decode inputs are required")
    dtype = _input_dtype(inputs.q, inputs.k_cache, inputs.v_cache)
    _check_variant(variant, dtype, "decode")
    if dtype == torch.int8:
        if schedule is not None:
            raise ValueError("schedule is a BF16 option; INT8 uses adaptive_split")
        if "quant_type" not in kwargs:
            raise ValueError("INT8 Decode requires explicit quant_type")
        for name in ("q_scale", "k_scale", "v_scale", "q_step", "k_step", "v_step"):
            if not isinstance(getattr(inputs, name, None), torch.Tensor):
                raise ValueError(f"INT8 Decode requires {name}")
        state = int8.prepare_int8_decode_workspace(inputs, **kwargs)
    else:
        selected = _schedule(variant, schedule)
        state = getattr(bf16, f"prepare_{selected}_bf16_workspace")(inputs, **kwargs)
        state._hy3_schedule = selected
    state._hy3_signature = _signature(inputs)
    return state


def _decode(inputs, workspace, *, variant=None, schedule=None):
    dtype = _input_dtype(inputs.q, inputs.k_cache, inputs.v_cache)
    _check_variant(variant, dtype, "decode")
    expected = (
        (int8.INT8DecodeWorkspace,)
        if dtype == torch.int8
        else (bf16.PureTritonMTP1Workspace, bf16.PureTritonMTPWorkspace)
    )
    if not isinstance(workspace, expected):
        raise TypeError("Decode workspace does not match input dtype")
    signature = getattr(workspace, "_hy3_signature", None)
    if signature is not None and signature != _signature(inputs):
        raise ValueError("Decode input metadata changed; prepare a new workspace")
    if dtype == torch.int8:
        if schedule is not None:
            raise ValueError("schedule is a BF16 option; INT8 uses adaptive_split")
        return int8.attention_decode_int8(inputs, workspace)
    prepared = getattr(workspace, "_hy3_schedule", None)
    selected = _schedule(variant, schedule, prepared or "static")
    if prepared is not None and selected != prepared:
        raise ValueError("Decode schedule differs from prepared workspace")
    return getattr(bf16, f"attention_decode_bf16_{selected}")(inputs, workspace)


def hy3_attention(*args, stage="prefill", variant=None, **kwargs):
    """Select BF16/INT8 from Q/K/V; stage and INT8 scales remain explicit."""
    if stage not in ("decode", "prefill"):
        raise ValueError("Hy3 stage must be 'decode' or 'prefill'")
    _reject_fp8(variant)
    if stage == "decode":
        if not args and "inputs" not in kwargs:
            raise ValueError("Decode inputs and workspace are required")
        return _decode(*args, variant=variant, **kwargs)
    tensors = []
    for index, name in enumerate(("q", "kcache", "vcache")):
        if len(args) > index:
            tensors.append(args[index])
        elif name in kwargs:
            tensors.append(kwargs[name])
        else:
            raise ValueError(f"Prefill requires {name}")
    dtype = _input_dtype(*tensors)
    _check_variant(variant, dtype, stage)
    implementation = (
        prefill.attention_with_kvcache_blocksparse_prefill_int8
        if dtype == torch.int8
        else prefill.attention_with_kvcache_blocksparse_prefill_bf16
    )
    return implementation(*args, **kwargs)


__all__ = ["decode", "prefill", "hy3_attention", "hy3_attention_prepare"]
