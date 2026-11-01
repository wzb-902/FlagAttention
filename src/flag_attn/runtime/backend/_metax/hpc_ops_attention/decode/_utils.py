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

"""Shared Decode metadata, buffers, and input-device guards."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import torch
import triton



PURE_TRITON_TILE_N = 64


PURE_TRITON_TASK_STRIDE = 12


@dataclass(frozen=True)
class DecodeWorkload:
    """Order-independent KV-length features used by policy selection."""

    histogram: tuple[tuple[int, int], ...]
    batch_size: int
    min_length: int
    max_length: int

    @classmethod
    def from_lengths(cls, lengths) -> "DecodeWorkload":
        values = tuple(int(value) for value in lengths)
        counts: dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return cls(
            histogram=tuple(sorted(counts.items())),
            batch_size=len(values),
            min_length=min(values, default=0),
            max_length=max(values, default=0),
        )

    def count(self, length: int) -> int:
        for value, count in self.histogram:
            if value == length:
                return count
        return 0

    @property
    def distinct_lengths(self) -> int:
        return len(self.histogram)

    @property
    def uniform_length(self) -> int | None:
        if len(self.histogram) == 1:
            return self.histogram[0][0]
        return None

    def is_uniform(self, length: int | None = None) -> bool:
        uniform = self.uniform_length
        return uniform is not None and (length is None or uniform == length)

    def is_mix(
        self,
        first_length: int,
        first_count: int,
        second_length: int,
        second_count: int,
    ) -> bool:
        return (
            self.batch_size == first_count + second_count
            and self.distinct_lengths == 2
            and self.count(first_length) == first_count
            and self.count(second_length) == second_count
        )

    def is_one_long_tail(self, long_length: int, short_length: int) -> bool:
        return (
            self.distinct_lengths == 2
            and self.count(long_length) == 1
            and self.count(short_length) == self.batch_size - 1
        )

    def is_two_long_tail(self, long_length: int, short_length: int) -> bool:
        return (
            self.distinct_lengths == 2
            and self.count(long_length) == 2
            and self.count(short_length) == self.batch_size - 2
        )

    @property
    def signature(self) -> tuple[int, tuple[tuple[int, int], ...]]:
        return self.batch_size, self.histogram


def pure_triton_task_map_metadata(kv_lens, *, num_head_kv, num_ctas, min_process_len):
    # Defer import until public and vendor packages are initialized.
    from flag_attn.hpc_ops_attention.decode.assign_task import pure_triton_task_map_metadata as metadata

    return metadata(kv_lens, num_head_kv=num_head_kv, num_ctas=num_ctas, min_process_len=min_process_len)


@dataclass
class PureTritonMTP1Workspace:
    """GQA8 split-K workspace for ordinary Triton kernels."""

    task_map: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    out: torch.Tensor
    quant_type: str
    num_ctas: int
    max_tasks: int
    sched_ints: int
    heads_per_group: int
    padded_heads_per_group: int
    min_process_len: int

    @property
    def mtp(self) -> int:
        return 1

    @property
    def route(self) -> str:
        return "triton-gqa8-mtp1"


@dataclass
class PureTritonMTPWorkspace:
    """Multi-token workspace with optional shared MACA BF16 paired storage."""

    workspaces: tuple[PureTritonMTP1Workspace, ...]
    paired: object | None = None

    @property
    def mtp(self) -> int:
        return len(self.workspaces)

    @property
    def route(self) -> str:
        if self.paired is not None:
            return "triton-bf16-paired"
        return "triton-per-token"


def prepare_decode_workspace_storage(inputs, quant_type):

    hkv = int(inputs.k_cache.shape[2])
    hq = int(inputs.q.shape[1])
    heads_per_group = hq // hkv
    padded_heads = triton.cdiv(heads_per_group, 8) * 8
    properties = torch.cuda.get_device_properties(inputs.q.device)
    num_ctas = properties.multi_processor_count * 4
    min_process_len = 512
    max_tasks, sched_ints, workspace_ints = pure_triton_task_map_metadata(
        inputs.kv_lens,
        num_head_kv=hkv,
        num_ctas=num_ctas,
        min_process_len=min_process_len,
    )
    task_map = torch.full(
        (workspace_ints,),
        -1,
        dtype=torch.int32,
        device=inputs.q.device,
    )
    split_out = torch.empty(
        (inputs.batch, num_ctas, hq, inputs.q.shape[-1]),
        dtype=torch.float32,
        device=inputs.q.device,
    )
    split_lse = torch.empty(
        (inputs.batch, num_ctas, hkv, padded_heads),
        dtype=torch.float32,
        device=inputs.q.device,
    )
    return PureTritonMTP1Workspace(
        task_map=task_map,
        split_out=split_out,
        split_lse=split_lse,
        out=torch.empty((inputs.batch, hq, inputs.q.shape[-1]), device=inputs.q.device, dtype=torch.bfloat16),
        quant_type=quant_type,
        num_ctas=num_ctas,
        max_tasks=max_tasks,
        sched_ints=sched_ints,
        heads_per_group=heads_per_group,
        padded_heads_per_group=padded_heads,
        min_process_len=min_process_len,
    )


_INPUT_TENSORS = (
    "q",
    "k_cache",
    "v_cache",
    "block_ids",
    "kv_lens",
    "q_scale",
    "k_scale",
    "v_scale",
    "q_step",
    "k_step",
    "v_step",
)


def _check_tensor_device(tensor, device, name):
    if not isinstance(tensor, torch.Tensor) or tensor.device != device:
        raise ValueError(f"{name} must be a tensor on the Q device {device}")


def on_input_device(function):
    """Validate Decode tensor devices and restore the caller's current device."""
    input_name = function.__code__.co_varnames[0]

    @wraps(function)
    def guarded(*args, **kwargs):
        if not args and input_name not in kwargs:
            return function(*args, **kwargs)
        inputs = args[0] if args else kwargs[input_name]
        q = inputs.q
        if not isinstance(q, torch.Tensor) or not q.is_cuda:
            raise ValueError("Q must be a CUDA tensor")
        for name in _INPUT_TENSORS:
            value = getattr(inputs, name, None)
            if value is not None:
                _check_tensor_device(value, q.device, name)
        with torch.cuda.device(q.device):
            return function(*args, **kwargs)

    return guarded


def validate_workspace_device(workspace, device):
    """Check owned Decode buffers without inspecting their values."""
    for name in ("task_map", "split_out", "split_lse", "out"):
        value = getattr(workspace, name, None)
        if value is not None:
            _check_tensor_device(value, device, f"workspace.{name}")
    for child in getattr(workspace, "workspaces", ()):
        validate_workspace_device(child, device)
    paired = getattr(workspace, "paired", None)
    if paired is not None:
        validate_workspace_device(paired.base, device)
    for _, children, split_out, split_lse in getattr(workspace, "pairs", ()):
        _check_tensor_device(split_out, device, "workspace.split_out")
        _check_tensor_device(split_lse, device, "workspace.split_lse")
        for child in children:
            validate_workspace_device(child, device)
