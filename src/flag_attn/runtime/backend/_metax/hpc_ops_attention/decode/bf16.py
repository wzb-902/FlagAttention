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

from __future__ import annotations

import torch
from ._utils import pure_triton_task_map_metadata
import triton
import triton.language as tl
from dataclasses import replace
from types import SimpleNamespace
from ._utils import PureTritonMTP1Workspace, PureTritonMTPWorkspace
from ._utils import DecodeWorkload
from ._utils import on_input_device, validate_workspace_device
from dataclasses import dataclass


# schedule

PURE_TRITON_TASK_STRIDE = 12

PURE_TRITON_TILE_N = 64

_PURE_TRITON_TASK_STRIDE = tl.constexpr(12)

_PURE_TRITON_TILE_N = tl.constexpr(64)


@triton.jit
def assign_pure_triton_task_map_kernel(
    SEQLENS_KV,
    TASK_MAP,
    B: tl.constexpr,
    H_KV: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    MIN_PROCESS_LEN: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    SCHED_INTS: tl.constexpr,
):
    """Build the portable MTP1 split-K schedule without TLE primitives."""
    total_tiles_per_head = 0
    b_scan = 0
    while b_scan < B:
        total_len = tl.load(SEQLENS_KV + b_scan)
        total_tiles_per_head += (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N
        b_scan += 1

    total_tiles = total_tiles_per_head * H_KV
    tiles_per_cta = (total_tiles + NUM_CTAS - 1) // NUM_CTAS
    minimum_tiles = MIN_PROCESS_LEN // _PURE_TRITON_TILE_N
    if minimum_tiles < 1:
        minimum_tiles = 1
    tiles_per_cta = tl.maximum(tiles_per_cta, minimum_tiles)
    if tl.program_id(0) == 0:
        tl.store(TASK_MAP + 0, tiles_per_cta + 1)
        tl.store(TASK_MAP + 1, NUM_CTAS)
        tl.store(TASK_MAP + 2, H_KV)
        tl.store(TASK_MAP + 3, B)
        tl.store(TASK_MAP + 4, SCHED_INTS * 4)

    num_chunks_base = (MAX_TASKS * NUM_CTAS + 1) * _PURE_TRITON_TASK_STRIDE
    hkv = 0
    batch = 0
    chunks = 0
    start_tiles = 0
    total_len = tl.load(SEQLENS_KV)
    tiles_left = (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N

    cta = 0
    while cta < NUM_CTAS:
        bucket = tiles_per_cta
        slot = 0
        while (bucket > 0) & (hkv < H_KV) & (slot < MAX_TASKS - 1):
            if tiles_left <= 0:
                batch += 1
                if batch >= B:
                    batch = 0
                    hkv += 1
                chunks = 0
                start_tiles = 0
                if hkv < H_KV:
                    total_len = tl.load(SEQLENS_KV + batch)
                    tiles_left = (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N

            if (tiles_left > 0) & (hkv < H_KV):
                cur_hkv = hkv
                cur_batch = batch
                total_len = tl.load(SEQLENS_KV + cur_batch)
                chunk = chunks
                add_tiles = tl.minimum(tiles_left, bucket)
                if chunks == NUM_CTAS - 1:
                    add_tiles = tiles_left
                seq_start = start_tiles * _PURE_TRITON_TILE_N
                seq_len = tl.minimum(
                    add_tiles * _PURE_TRITON_TILE_N,
                    total_len - seq_start,
                )
                start_tiles += add_tiles
                tiles_left -= add_tiles
                bucket -= add_tiles
                chunks += 1

                task_base = (MAX_TASKS * cta + slot + 1) * _PURE_TRITON_TASK_STRIDE
                tl.store(TASK_MAP + task_base + 0, cur_hkv)
                tl.store(TASK_MAP + task_base + 1, cur_batch)
                tl.store(TASK_MAP + task_base + 2, chunk)
                tl.store(TASK_MAP + task_base + 3, seq_start)
                tl.store(TASK_MAP + task_base + 4, seq_len)
                tl.store(TASK_MAP + task_base + 5, seq_len)
                tl.store(
                    TASK_MAP + task_base + 6,
                    (seq_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N,
                )
                tl.store(
                    TASK_MAP + task_base + 7,
                    seq_len // _PURE_TRITON_TILE_N,
                )
                tl.store(TASK_MAP + task_base + 8, tiles_left <= 0)
                tl.store(TASK_MAP + task_base + 9, 0)
                tl.store(TASK_MAP + task_base + 10, 0)
                tl.store(TASK_MAP + task_base + 11, 0)
                slot += 1

                if tiles_left <= 0:
                    tl.store(
                        TASK_MAP + num_chunks_base + cur_hkv * B + cur_batch,
                        chunks,
                    )
                    batch = cur_batch + 1
                    if batch >= B:
                        batch = 0
                        hkv = cur_hkv + 1
                    chunks = 0
                    start_tiles = 0
                    if hkv < H_KV:
                        total_len = tl.load(SEQLENS_KV + batch)
                        tiles_left = (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N

        terminator = (MAX_TASKS * cta + slot + 1) * _PURE_TRITON_TASK_STRIDE
        tl.store(TASK_MAP + terminator + 0, -1)
        tl.store(TASK_MAP + terminator + 1, -1)
        cta += 1


@triton.jit
def _assign_maca_parallel_task_map_kernel(
    L,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    MIN: tl.constexpr,
    MAX: tl.constexpr,
    SCHED: tl.constexpr,
    NB: tl.constexpr,
):
    c = tl.program_id(0)
    bs = tl.arange(0, NB)
    lens = tl.load(L + bs, bs < B, 0)
    tiles = (lens + 63) // 64
    ends = tl.cumsum(tiles, 0)
    total = tl.sum(tiles, 0)
    width = tl.maximum(tl.cdiv(total * H, C), tl.maximum(MIN // 64, 1))
    if c == 0:
        tl.store(T + 0, width + 1)
        tl.store(T + 1, C)
        tl.store(T + 2, H)
        tl.store(T + 3, B)
        tl.store(T + 4, SCHED * 4)
    pos = c * width
    stop = tl.minimum(pos + width, total * H)
    slot = 0
    while (pos < stop) & (slot < MAX - 1):
        h = pos // total
        local = pos - h * total
        batch = tl.sum(((ends <= local) & (bs < B)).to(tl.int32), 0)
        begin = tl.sum(tl.where(bs < batch, tiles, 0), 0) + h * total
        length = tl.load(L + batch)
        seqend = begin + tl.cdiv(length, 64)
        end = tl.minimum(seqend, stop)
        chunk = c - begin // width
        start = (pos - begin) * 64
        count = tl.minimum((end - pos) * 64, length - start)
        base = (MAX * c + slot + 1) * 12
        tl.store(T + base + 0, h)
        tl.store(T + base + 1, batch)
        tl.store(T + base + 2, chunk)
        tl.store(T + base + 3, start)
        tl.store(T + base + 4, count)
        tl.store(T + base + 5, count)
        tl.store(T + base + 6, tl.cdiv(count, 64))
        tl.store(T + base + 7, count // 64)
        tl.store(T + base + 8, end == seqend)
        tl.store(T + base + 9, 0)
        tl.store(T + base + 10, 0)
        tl.store(T + base + 11, 0)
        if end == seqend:
            tl.store(T + (MAX * C + 1) * 12 + h * B + batch, chunk + 1)
        pos = end
        slot += 1
    term = (MAX * c + slot + 1) * 12
    tl.store(T + term, -1)
    tl.store(T + term + 1, -1)


def launch_pure_triton_task_map(
    kv_lens: torch.Tensor,
    task_map: torch.Tensor,
    *,
    num_head_kv: int,
    num_ctas: int,
    min_process_len: int,
    max_tasks: int,
    sched_ints: int,
) -> None:
    if triton.runtime.driver.active.get_current_target().backend == "maca":
        _assign_maca_parallel_task_map_kernel[(num_ctas,)](
            kv_lens,
            task_map,
            kv_lens.numel(),
            num_head_kv,
            num_ctas,
            min_process_len,
            max_tasks,
            sched_ints,
            triton.next_power_of_2(kv_lens.numel()),
            num_warps=4,
            num_stages=1,
        )
        return
    assign_pure_triton_task_map_kernel[(1,)](
        kv_lens,
        task_map,
        B=kv_lens.numel(),
        H_KV=num_head_kv,
        NUM_CTAS=num_ctas,
        MIN_PROCESS_LEN=min_process_len,
        MAX_TASKS=max_tasks,
        SCHED_INTS=sched_ints,
        num_warps=1,
        num_stages=1,
    )


# mtp1


@triton.jit
def _bf16_mtp1_pure_triton_bf16_gqa8_splitk_kernel(
    Q,
    K,
    V,
    BLOCK_IDS,
    TASK_MAP,
    SPLIT_OUT,
    SPLIT_LSE,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
):
    """BF16 MTP1 producer: GQA8 reuse, tl.dot, and dynamic split-K."""
    cta = tl.program_id(0)
    group_block = tl.program_id(1)
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    scale = tl.rsqrt(tl.full((), D, tl.float32))
    for slot in tl.static_range(0, MAX_TASKS):
        task = ((MAX_TASKS * cta + slot) + 1) * 12
        hkv = tl.load(TASK_MAP + task + 0)
        batch = tl.load(TASK_MAP + task + 1)
        if hkv < 0:
            return
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        hq = hkv * HEADS_PER_GROUP + offs_h
        q = tl.load(
            Q + batch * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((GROUP_M,), tl.float32)
        acc = tl.zeros((D, GROUP_M), tl.float32)
        start = 0
        while start < seq_len:
            local_n = start + offs_n
            logical = seq_start + local_n
            page = logical // BLOCK_SIZE
            pos = logical - page * BLOCK_SIZE
            valid_n = local_n < seq_len
            physical = tl.load(
                BLOCK_IDS + batch * MAX_BLOCKS + page,
                mask=valid_n,
                other=0,
            )
            k = tl.load(
                K + physical[:, None] * K_SBLOCK + pos[:, None] * K_STOKEN + hkv * K_SHEAD + offs_d[None, :] * K_SD,
                mask=valid_n[:, None],
                other=0.0,
            )
            v = tl.load(
                V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD,
                mask=valid_n[:, None],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(
                valid_h[:, None] & valid_n[None, :],
                scores,
                -float("inf"),
            )
            tile_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, tile_max)
            active = m_new != -float("inf")
            safe_new = tl.where(active, m_new, 0.0)
            safe_old = tl.where(m_i == -float("inf"), safe_new, m_i)
            p = tl.where(active[:, None], tl.exp(scores - safe_new[:, None]), 0.0)
            alpha = tl.where(active, tl.exp(safe_old - safe_new), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = tl.dot(
                tl.trans(v),
                tl.trans(p.to(tl.bfloat16)),
                acc * alpha[None, :],
                out_dtype=tl.float32,
            )
            m_i = m_new
            start += BLOCK_N
        has_value = l_i > 0.0
        partial = tl.where(has_value[:, None], tl.trans(acc) / l_i[:, None], 0.0)
        lse = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
        tl.store(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
            partial,
            mask=valid_h[:, None],
        )
        tl.store(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + offs_h * SL_SH,
            lse,
            mask=valid_h,
        )


# mtp1_empty


@triton.jit
def _bf16_mtp1_empty_direct_producer(
    Q,
    K,
    V,
    BLOCK_IDS,
    KV_LENS,
    SPLIT_OUT,
    SPLIT_LSE,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
):
    """BF16 MTP1 producer: GQA8 reuse, tl.dot, and dynamic split-K."""
    cta = tl.program_id(0)
    group_block = 0
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    scale = tl.rsqrt(tl.full((), D, tl.float32))
    chunk = cta
    batch_head = tl.program_id(1)
    hkv = batch_head % (H_Q // HEADS_PER_GROUP)
    batch = batch_head // (H_Q // HEADS_PER_GROUP)
    seq_start = chunk * CHUNK_SIZE
    seq_len = tl.minimum(tl.maximum(tl.load(KV_LENS + batch) - seq_start, 0), CHUNK_SIZE)
    hq = hkv * HEADS_PER_GROUP + offs_h
    if seq_len == 0:
        if MAX_BLOCKS * BLOCK_SIZE <= CHUNK_SIZE:
            tl.store(
                SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
                0.0,
                mask=valid_h[:, None],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + offs_h * SL_SH, -float("inf"), mask=valid_h
            )
        return
    q = tl.load(
        Q + batch * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_h[:, None],
        other=0.0,
    )
    m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((GROUP_M,), tl.float32)
    acc = tl.zeros((D, GROUP_M), tl.float32)
    start = 0
    while start < seq_len:
        local_n = start + offs_n
        logical = seq_start + local_n
        page = logical // BLOCK_SIZE
        pos = logical - page * BLOCK_SIZE
        valid_n = local_n < seq_len
        physical = tl.load(
            BLOCK_IDS + batch * MAX_BLOCKS + page,
            mask=valid_n,
            other=0,
        )
        k = tl.load(
            K + physical[:, None] * K_SBLOCK + pos[:, None] * K_STOKEN + hkv * K_SHEAD + offs_d[None, :] * K_SD,
            mask=valid_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD,
            mask=valid_n[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(
            valid_h[:, None] & valid_n[None, :],
            scores,
            -float("inf"),
        )
        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        active = m_new != -float("inf")
        safe_new = tl.where(active, m_new, 0.0)
        safe_old = tl.where(m_i == -float("inf"), safe_new, m_i)
        p = tl.where(active[:, None], tl.exp(scores - safe_new[:, None]), 0.0)
        alpha = tl.where(active, tl.exp(safe_old - safe_new), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(
            tl.trans(v),
            tl.trans(p.to(tl.bfloat16)),
            acc * alpha[None, :],
            out_dtype=tl.float32,
        )
        m_i = m_new
        start += BLOCK_N
    has_value = l_i > 0.0
    partial = tl.where(has_value[:, None], tl.trans(acc) / l_i[:, None], 0.0)
    lse = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
    tl.store(
        SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
        partial,
        mask=valid_h[:, None],
    )
    tl.store(
        SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + offs_h * SL_SH,
        lse,
        mask=valid_h,
    )


@triton.jit
def _bf16_mtp1_empty_direct_combine(
    SO,
    SL,
    O,
    KV_LENS,
    HQ: tl.constexpr,
    C: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    valid_chunks = tl.minimum(C, tl.cdiv(tl.load(KV_LENS + batch), 256))
    c = tl.arange(0, CP)
    d = tl.arange(0, D)
    lse = tl.load(SL + batch * SL_SB + c * SL_SC + head, mask=c < valid_chunks, other=-float("inf"))
    maximum = tl.max(lse, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(lse - maximum)
    values = tl.load(
        SO + batch * SO_SB + c[:, None] * SO_SC + head * D + d[None, :], mask=c[:, None] < valid_chunks, other=0.0
    )
    denominator = tl.sum(weights, 0)
    result = tl.sum(weights[:, None] * values, 0) / tl.maximum(denominator, 1.0e-20)
    tl.store(O + (batch * HQ + head) * D + d, result)




def _bf16_mtp1_empty_run(i, w):
    b, hq, d = i.q.shape
    hkv = hq // 8
    _bf16_mtp1_empty_direct_producer[(w.chunks, b * hkv)](
        i.q,
        i.k_cache,
        i.v_cache,
        i.block_ids,
        i.kv_lens,
        w.split_out,
        w.split_lse,
        H_Q=hq,
        HEADS_PER_GROUP=8,
        GROUP_M=16,
        D=d,
        BLOCK_SIZE=64,
        MAX_BLOCKS=i.block_ids.shape[1],
        CHUNK_SIZE=w.chunk_size,
        BLOCK_N=64,
        Q_SB=i.q.stride(0),
        Q_SH=i.q.stride(1),
        K_SBLOCK=i.k_cache.stride(0),
        K_STOKEN=i.k_cache.stride(1),
        K_SHEAD=i.k_cache.stride(2),
        K_SD=i.k_cache.stride(3),
        V_SBLOCK=i.v_cache.stride(0),
        V_STOKEN=i.v_cache.stride(1),
        V_SHEAD=i.v_cache.stride(2),
        V_SD=i.v_cache.stride(3),
        SO_SB=w.split_out.stride(0),
        SO_SC=w.split_out.stride(1),
        SO_SH=w.split_out.stride(2),
        SL_SB=w.split_lse.stride(0),
        SL_SC=w.split_lse.stride(1),
        SL_SK=w.split_lse.stride(2),
        SL_SH=w.split_lse.stride(3),
        num_warps=4,
        num_stages=1,
    )
    if w.chunks > 1:
        _bf16_mtp1_empty_direct_combine[(b, hq)](
            w.split_out,
            w.split_lse,
            w.out,
            KV_LENS=i.kv_lens,
            HQ=hq,
            C=w.chunks,
            CP=triton.next_power_of_2(w.chunks),
            D=d,
            SO_SB=w.split_out.stride(0),
            SO_SC=w.split_out.stride(1),
            SL_SB=w.split_lse.stride(0),
            SL_SC=w.split_lse.stride(1),
            num_warps=4,
            num_stages=1,
        )
    return w.out


def _bf16_mtp1_empty_run_with_workspace(inputs, base):
    from types import SimpleNamespace

    chunks = triton.cdiv(inputs.block_ids.shape[1] * 64, 256)
    if chunks > base.num_ctas:
        raise ValueError("direct split capacity exceeds workspace")
    workspace = SimpleNamespace(
        out=base.out,
        split_out=base.out[:, None] if chunks == 1 else base.split_out,
        split_lse=base.split_lse,
        chunks=chunks,
        chunk_size=256,
    )
    return _bf16_mtp1_empty_run(inputs, workspace)


# mtp1_direct


@triton.jit
def _bf16_mtp1_direct_direct_producer(
    Q,
    K,
    V,
    BLOCK_IDS,
    KV_LENS,
    SPLIT_OUT,
    SPLIT_LSE,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
):
    """BF16 MTP1 producer: GQA8 reuse, tl.dot, and dynamic split-K."""
    cta = tl.program_id(0)
    group_block = 0
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    scale = tl.rsqrt(tl.full((), D, tl.float32))
    chunk = cta
    batch_head = tl.program_id(1)
    hkv = batch_head % (H_Q // HEADS_PER_GROUP)
    batch = batch_head // (H_Q // HEADS_PER_GROUP)
    seq_start = chunk * CHUNK_SIZE
    seq_len = tl.minimum(tl.maximum(tl.load(KV_LENS + batch) - seq_start, 0), CHUNK_SIZE)
    hq = hkv * HEADS_PER_GROUP + offs_h
    if seq_len == 0:
        tl.store(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
            0.0,
            mask=valid_h[:, None],
        )
        tl.store(SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + offs_h * SL_SH, -float("inf"), mask=valid_h)
        return
    q = tl.load(
        Q + batch * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_h[:, None],
        other=0.0,
    )
    m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((GROUP_M,), tl.float32)
    acc = tl.zeros((D, GROUP_M), tl.float32)
    start = 0
    while start < seq_len:
        local_n = start + offs_n
        logical = seq_start + local_n
        page = logical // BLOCK_SIZE
        pos = logical - page * BLOCK_SIZE
        valid_n = local_n < seq_len
        physical = tl.load(
            BLOCK_IDS + batch * MAX_BLOCKS + page,
            mask=valid_n,
            other=0,
        )
        k = tl.load(
            K + physical[:, None] * K_SBLOCK + pos[:, None] * K_STOKEN + hkv * K_SHEAD + offs_d[None, :] * K_SD,
            mask=valid_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD,
            mask=valid_n[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(
            valid_h[:, None] & valid_n[None, :],
            scores,
            -float("inf"),
        )
        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        active = m_new != -float("inf")
        safe_new = tl.where(active, m_new, 0.0)
        safe_old = tl.where(m_i == -float("inf"), safe_new, m_i)
        p = tl.where(active[:, None], tl.exp(scores - safe_new[:, None]), 0.0)
        alpha = tl.where(active, tl.exp(safe_old - safe_new), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(
            tl.trans(v),
            tl.trans(p.to(tl.bfloat16)),
            acc * alpha[None, :],
            out_dtype=tl.float32,
        )
        m_i = m_new
        start += BLOCK_N
    has_value = l_i > 0.0
    partial = tl.where(has_value[:, None], tl.trans(acc) / l_i[:, None], 0.0)
    lse = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
    tl.store(
        SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
        partial,
        mask=valid_h[:, None],
    )
    tl.store(
        SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + offs_h * SL_SH,
        lse,
        mask=valid_h,
    )


@triton.jit
def _bf16_mtp1_direct_direct_combine(
    SO,
    SL,
    O,
    HQ: tl.constexpr,
    C: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    c = tl.arange(0, CP)
    d = tl.arange(0, D)
    lse = tl.load(SL + batch * SL_SB + c * SL_SC + head, mask=c < C, other=-float("inf"))
    maximum = tl.max(lse, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(lse - maximum)
    values = tl.load(SO + batch * SO_SB + c[:, None] * SO_SC + head * D + d[None, :], mask=c[:, None] < C, other=0.0)
    denominator = tl.sum(weights, 0)
    result = tl.sum(weights[:, None] * values, 0) / tl.maximum(denominator, 1.0e-20)
    tl.store(O + (batch * HQ + head) * D + d, result)




def _bf16_mtp1_direct_run(i, w):
    b, hq, d = i.q.shape
    hkv = hq // 8
    _bf16_mtp1_direct_direct_producer[(w.chunks, b * hkv)](
        i.q,
        i.k_cache,
        i.v_cache,
        i.block_ids,
        i.kv_lens,
        w.split_out,
        w.split_lse,
        H_Q=hq,
        HEADS_PER_GROUP=8,
        GROUP_M=16,
        D=d,
        BLOCK_SIZE=64,
        MAX_BLOCKS=i.block_ids.shape[1],
        CHUNK_SIZE=w.chunk_size,
        BLOCK_N=64,
        Q_SB=i.q.stride(0),
        Q_SH=i.q.stride(1),
        K_SBLOCK=i.k_cache.stride(0),
        K_STOKEN=i.k_cache.stride(1),
        K_SHEAD=i.k_cache.stride(2),
        K_SD=i.k_cache.stride(3),
        V_SBLOCK=i.v_cache.stride(0),
        V_STOKEN=i.v_cache.stride(1),
        V_SHEAD=i.v_cache.stride(2),
        V_SD=i.v_cache.stride(3),
        SO_SB=w.split_out.stride(0),
        SO_SC=w.split_out.stride(1),
        SO_SH=w.split_out.stride(2),
        SL_SB=w.split_lse.stride(0),
        SL_SC=w.split_lse.stride(1),
        SL_SK=w.split_lse.stride(2),
        SL_SH=w.split_lse.stride(3),
        num_warps=4,
        num_stages=1,
    )
    if w.chunks > 1:
        _bf16_mtp1_direct_direct_combine[(b, hq)](
            w.split_out,
            w.split_lse,
            w.out,
            HQ=hq,
            C=w.chunks,
            CP=triton.next_power_of_2(w.chunks),
            D=d,
            SO_SB=w.split_out.stride(0),
            SO_SC=w.split_out.stride(1),
            SL_SB=w.split_lse.stride(0),
            SL_SC=w.split_lse.stride(1),
            num_warps=4,
            num_stages=1,
        )
    return w.out


def _bf16_mtp1_direct_run_with_workspace(inputs, base):
    from types import SimpleNamespace

    chunks = triton.cdiv(inputs.block_ids.shape[1] * 64, 256)
    if chunks > 2 and chunks <= base.num_ctas:
        run_empty_aware = _bf16_mtp1_empty_run_with_workspace

        return run_empty_aware(inputs, base)
    if chunks > base.num_ctas:
        raise ValueError("direct split capacity exceeds workspace")
    workspace = SimpleNamespace(
        out=base.out,
        split_out=base.out[:, None] if chunks == 1 else base.split_out,
        split_lse=base.split_lse,
        chunks=chunks,
        chunk_size=256,
    )
    return _bf16_mtp1_direct_run(inputs, workspace)


# paired


@triton.jit
def _bf16_paired_paired_producer(
    Q,
    KV_LENS,
    MTP: tl.constexpr,
    SO_ST: tl.constexpr,
    SL_ST: tl.constexpr,
    K,
    V,
    BLOCK_IDS,
    TASK_MAP,
    SPLIT_OUT,
    SPLIT_LSE,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
):
    """Pair two GQA8 queries in a 16-row BF16 tile and share KV loads."""
    cta = tl.program_id(1)
    group_block = tl.program_id(0)
    rows = tl.arange(0, GROUP_M)
    token = group_block * 2 + rows // 8
    offs_h = rows % 8
    valid_h = token < MTP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    scale = tl.rsqrt(tl.full((), D, tl.float32))
    for slot in tl.static_range(0, MAX_TASKS):
        task = ((MAX_TASKS * cta + slot) + 1) * 12
        hkv = tl.load(TASK_MAP + task + 0)
        batch = tl.load(TASK_MAP + task + 1)
        if hkv < 0:
            return
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        limit = tl.load(KV_LENS + batch) - (MTP - 1 - token)
        hq = hkv * HEADS_PER_GROUP + offs_h
        q = tl.load(
            Q + (batch * MTP + token[:, None]) * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((GROUP_M,), tl.float32)
        acc = tl.zeros((D, GROUP_M), tl.float32)
        start = 0
        while start < seq_len:
            local_n = start + offs_n
            logical = seq_start + local_n
            # The task scheduler emits starts aligned to its 64-token tile.
            # BLOCK_N equals the physical page size, so this tile has one ID.
            tl.static_assert(BLOCK_N == BLOCK_SIZE)
            page = (seq_start + start) // BLOCK_SIZE
            pos = offs_n
            valid_n = local_n < seq_len
            physical = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + page)
            k = tl.load(
                K + physical * K_SBLOCK + pos[:, None] * K_STOKEN + hkv * K_SHEAD + offs_d[None, :] * K_SD,
                mask=valid_n[:, None],
                other=0.0,
            )
            v = tl.load(
                V + physical * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD,
                mask=valid_n[:, None],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(
                valid_h[:, None] & valid_n[None, :] & (logical[None, :] < limit[:, None]),
                scores,
                -float("inf"),
            )
            tile_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, tile_max)
            active = m_new != -float("inf")
            safe_new = tl.where(active, m_new, 0.0)
            safe_old = tl.where(m_i == -float("inf"), safe_new, m_i)
            p = tl.where(active[:, None], tl.exp(scores - safe_new[:, None]), 0.0)
            alpha = tl.where(active, tl.exp(safe_old - safe_new), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = tl.dot(
                tl.trans(v),
                tl.trans(p.to(tl.bfloat16)),
                acc * alpha[None, :],
                out_dtype=tl.float32,
            )
            m_i = m_new
            start += BLOCK_N
        has_value = l_i > 0.0
        partial = tl.where(has_value[:, None], tl.trans(acc) / l_i[:, None], 0.0)
        lse = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
        tl.store(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC + token[:, None] * SO_ST + hq[:, None] * SO_SH + offs_d[None, :],
            partial,
            mask=valid_h[:, None],
        )
        tl.store(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + token * SL_ST + hkv * SL_SK + offs_h * SL_SH,
            lse,
            mask=valid_h,
        )


@triton.jit
def _bf16_paired_paired_combine(
    MTP: tl.constexpr,
    SO_ST: tl.constexpr,
    SL_ST: tl.constexpr,
    SPLIT_OUT,
    SPLIT_LSE,
    TASK_MAP,
    OUT,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    flat = tl.program_id(0)
    batch = flat // MTP
    token = flat % MTP
    hq = tl.program_id(1)
    hkv = hq // HEADS_PER_GROUP
    h_in_group = hq - hkv * HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    counts = (MAX_TASKS * NUM_CTAS + 1) * 12
    batch_count = tl.load(TASK_MAP + 3)
    n_chunks = tl.load(TASK_MAP + counts + hkv * batch_count + batch)
    # Runtime GPU dispatch preserves short-split latency and Graph updates.
    if n_chunks <= 8:
        max_lse = tl.full((), -float("inf"), tl.float32)
        chunk = 0
        while chunk < n_chunks:
            value = tl.load(
                SPLIT_LSE + batch * SL_SB + chunk * SL_SC + token * SL_ST + hkv * SL_SK + h_in_group * SL_SH,
            )
            max_lse = tl.maximum(max_lse, value)
            chunk += 1
        safe_max = tl.where(max_lse == -float("inf"), 0.0, max_lse)
        denom = tl.zeros((), tl.float32)
        out = tl.zeros((D,), tl.float32)
        chunk = 0
        while chunk < n_chunks:
            value = tl.load(
                SPLIT_LSE + batch * SL_SB + chunk * SL_SC + token * SL_ST + hkv * SL_SK + h_in_group * SL_SH,
            )
            weight = tl.exp(value - safe_max)
            partial = tl.load(
                SPLIT_OUT + batch * SO_SB + chunk * SO_SC + token * SO_ST + hq * SO_SH + offs_d,
            )
            out += weight * partial
            denom += weight
            chunk += 1
    else:
        # Reduce all LSE values together, then merge split outputs in small tiles.
        offsets = tl.arange(0, triton.next_power_of_2(MAX_SPLIT))
        lse_base = SPLIT_LSE + batch * SL_SB + token * SL_ST + hkv * SL_SK + h_in_group * SL_SH
        all_values = tl.load(lse_base + offsets * SL_SC, mask=offsets < n_chunks, other=-float("inf"))
        max_lse = tl.max(all_values, axis=0)
        safe_max = tl.where(max_lse == -float("inf"), 0.0, max_lse)
        denom = tl.zeros((), tl.float32)
        out = tl.zeros((D,), tl.float32)
        chunk = 0
        while chunk < n_chunks:
            chunks = chunk + tl.arange(0, 16)
            valid = chunks < n_chunks
            values = tl.load(lse_base + chunks * SL_SC, mask=valid, other=-float("inf"))
            weights = tl.exp(values - safe_max)
            partial = tl.load(
                SPLIT_OUT + batch * SO_SB + chunks[:, None] * SO_SC + token * SO_ST + hq * SO_SH + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            out += tl.sum(weights[:, None] * partial, axis=0)
            denom += tl.sum(weights, axis=0)
            chunk += 16
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        OUT + flat * O_SB + hq * O_SH + offs_d,
        tl.where(denom > 0.0, out / safe_denom, 0.0),
    )


@triton.jit
def _bf16_paired_paired_combine_serial(
    MTP: tl.constexpr,
    SO_ST: tl.constexpr,
    SL_ST: tl.constexpr,
    SPLIT_OUT,
    SPLIT_LSE,
    TASK_MAP,
    OUT,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    flat = tl.program_id(0)
    batch = flat // MTP
    token = flat % MTP
    hq = tl.program_id(1)
    hkv = hq // HEADS_PER_GROUP
    h_in_group = hq - hkv * HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    counts = (MAX_TASKS * NUM_CTAS + 1) * 12
    batch_count = tl.load(TASK_MAP + 3)
    n_chunks = tl.load(TASK_MAP + counts + hkv * batch_count + batch)
    max_lse = tl.full((), -float("inf"), tl.float32)
    chunk = 0
    while chunk < n_chunks:
        value = tl.load(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + token * SL_ST + hkv * SL_SK + h_in_group * SL_SH,
        )
        max_lse = tl.maximum(max_lse, value)
        chunk += 1
    safe_max = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    denom = tl.zeros((), tl.float32)
    out = tl.zeros((D,), tl.float32)
    chunk = 0
    while chunk < n_chunks:
        value = tl.load(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + token * SL_ST + hkv * SL_SK + h_in_group * SL_SH,
        )
        weight = tl.exp(value - safe_max)
        partial = tl.load(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC + token * SO_ST + hq * SO_SH + offs_d,
        )
        out += weight * partial
        denom += weight
        chunk += 1
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        OUT + flat * O_SB + hq * O_SH + offs_d,
        tl.where(denom > 0.0, out / safe_denom, 0.0),
    )


def _bf16_paired_prepare(inputs):
    b, m = inputs.batch, inputs.mtp
    assert m in (2, 3, 4) and inputs.q.shape[-1] == 128
    assert inputs.q.shape[1] == inputs.k_cache.shape[2] * 8
    assert inputs.q.dtype == torch.bfloat16
    assert int(inputs.kv_lens.min()) >= m
    last = replace(inputs, q=inputs.q.reshape(b, m, *inputs.q.shape[1:])[:, -1])
    w = prepare_pure_triton_mtp1_workspace(last, "bf16")
    hq = inputs.q.shape[1]
    hkv = hq // 8
    c = w.num_ctas
    w.split_out = torch.empty((b, c, m, hq, 128), device=inputs.q.device, dtype=torch.float32)
    w.split_lse = torch.empty((b, c, m, hkv, 8), device=inputs.q.device, dtype=torch.float32)
    w.out = torch.empty(inputs.q.shape, device=inputs.q.device, dtype=torch.bfloat16)
    return SimpleNamespace(base=w, mtp=m)


def _bf16_paired_run(inputs, workspace):
    if inputs.mtp != workspace.mtp or tuple(inputs.q.shape) != tuple(workspace.base.out.shape):
        raise ValueError("input shape changed; refresh or prepare a new workspace")
    w = workspace.base
    m = workspace.mtp
    b = inputs.batch
    hq = inputs.q.shape[1]
    launch_pure_triton_task_map(
        inputs.kv_lens,
        w.task_map,
        num_head_kv=hq // 8,
        num_ctas=w.num_ctas,
        min_process_len=w.min_process_len,
        max_tasks=w.max_tasks,
        sched_ints=w.sched_ints,
    )
    common = dict(
        H_Q=hq,
        HEADS_PER_GROUP=8,
        D=128,
        MAX_TASKS=w.max_tasks,
        SO_SB=w.split_out.stride(0),
        SO_SC=w.split_out.stride(1),
        SO_ST=w.split_out.stride(2),
        SO_SH=w.split_out.stride(3),
        SL_SB=w.split_lse.stride(0),
        SL_SC=w.split_lse.stride(1),
        SL_ST=w.split_lse.stride(2),
        SL_SK=w.split_lse.stride(3),
        SL_SH=w.split_lse.stride(4),
        MTP=m,
        num_warps=4,
        num_stages=1,
    )
    _bf16_paired_paired_producer[(triton.cdiv(m, 2), w.num_ctas)](
        inputs.q,
        inputs.kv_lens,
        **common,
        K=inputs.k_cache,
        V=inputs.v_cache,
        BLOCK_IDS=inputs.block_ids,
        TASK_MAP=w.task_map,
        SPLIT_OUT=w.split_out,
        SPLIT_LSE=w.split_lse,
        GROUP_M=16,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        BLOCK_N=64,
        Q_SB=inputs.q.stride(0),
        Q_SH=inputs.q.stride(1),
        K_SBLOCK=inputs.k_cache.stride(0),
        K_STOKEN=inputs.k_cache.stride(1),
        K_SHEAD=inputs.k_cache.stride(2),
        K_SD=inputs.k_cache.stride(3),
        V_SBLOCK=inputs.v_cache.stride(0),
        V_STOKEN=inputs.v_cache.stride(1),
        V_SHEAD=inputs.v_cache.stride(2),
        V_SD=inputs.v_cache.stride(3),
    )
    # Each scheduler CTA handles at least min_process_len tokens. A sequence
    # may also begin in a partially occupied CTA, hence the extra split.
    split_bound = triton.cdiv(inputs.block_ids.shape[1] * 64, w.min_process_len) + 1
    combine = _bf16_paired_paired_combine_serial if split_bound <= 16 else _bf16_paired_paired_combine
    combine[(b * m, hq)](
        **common,
        SPLIT_OUT=w.split_out,
        SPLIT_LSE=w.split_lse,
        TASK_MAP=w.task_map,
        OUT=w.out,
        MAX_SPLIT=w.num_ctas,
        NUM_CTAS=w.num_ctas,
        O_SB=w.out.stride(0),
        O_SH=w.out.stride(1),
    )
    return w.out


# Public entry points

# Deprecated flags retained for callers; no extension is loaded.
HAS_TLE = False
USE_TLE = False


@triton.jit
def _pure_triton_bf16_gqa8_splitk_kernel(
    Q,
    K,
    V,
    BLOCK_IDS,
    TASK_MAP,
    SPLIT_OUT,
    SPLIT_LSE,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
):
    """BF16 MTP1 producer: GQA8 reuse, tl.dot, and dynamic split-K."""
    cta = tl.program_id(0)
    group_block = tl.program_id(1)
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    scale = tl.rsqrt(tl.full((), D, tl.float32))
    for slot in tl.static_range(0, MAX_TASKS):
        task = ((MAX_TASKS * cta + slot) + 1) * 12
        hkv = tl.load(TASK_MAP + task + 0)
        batch = tl.load(TASK_MAP + task + 1)
        if hkv < 0:
            return
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        hq = hkv * HEADS_PER_GROUP + offs_h
        q = tl.load(
            Q + batch * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((GROUP_M,), tl.float32)
        acc = tl.zeros((GROUP_M, D), tl.float32)
        start = 0
        while start < seq_len:
            local_n = start + offs_n
            logical = seq_start + local_n
            page = logical // BLOCK_SIZE
            pos = logical - page * BLOCK_SIZE
            valid_n = local_n < seq_len
            physical = tl.load(
                BLOCK_IDS + batch * MAX_BLOCKS + page,
                mask=valid_n,
                other=0,
            )
            k = tl.load(
                K + physical[:, None] * K_SBLOCK + pos[:, None] * K_STOKEN + hkv * K_SHEAD + offs_d[None, :] * K_SD,
                mask=valid_n[:, None],
                other=0.0,
            )
            v = tl.load(
                V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD,
                mask=valid_n[:, None],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(
                valid_h[:, None] & valid_n[None, :],
                scores,
                -float("inf"),
            )
            tile_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, tile_max)
            active = m_new != -float("inf")
            safe_new = tl.where(active, m_new, 0.0)
            safe_old = tl.where(m_i == -float("inf"), safe_new, m_i)
            p = tl.where(active[:, None], tl.exp(scores - safe_new[:, None]), 0.0)
            alpha = tl.where(active, tl.exp(safe_old - safe_new), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16),
                v,
                out_dtype=tl.float32,
            )
            m_i = m_new
            start += BLOCK_N
        has_value = l_i > 0.0
        partial = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
        lse = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
        tl.store(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
            partial,
            mask=valid_h[:, None],
        )
        tl.store(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + offs_h * SL_SH,
            lse,
            mask=valid_h,
        )


@triton.jit
def _pure_triton_splitk_combine_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    TASK_MAP,
    OUT,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    batch = tl.program_id(0)
    hq = tl.program_id(1)
    hkv = hq // HEADS_PER_GROUP
    h_in_group = hq - hkv * HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    counts = (MAX_TASKS * NUM_CTAS + 1) * 12
    batch_count = tl.load(TASK_MAP + 3)
    n_chunks = tl.load(TASK_MAP + counts + hkv * batch_count + batch)
    max_lse = tl.full((), -float("inf"), tl.float32)
    chunk = 0
    while chunk < n_chunks:
        value = tl.load(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + h_in_group * SL_SH,
        )
        max_lse = tl.maximum(max_lse, value)
        chunk += 1
    safe_max = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    denom = tl.zeros((), tl.float32)
    out = tl.zeros((D,), tl.float32)
    chunk = 0
    while chunk < n_chunks:
        value = tl.load(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + h_in_group * SL_SH,
        )
        weight = tl.exp(value - safe_max)
        partial = tl.load(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq * SO_SH + offs_d,
        )
        out += weight * partial
        denom += weight
        chunk += 1
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        OUT + batch * O_SB + hq * O_SH + offs_d,
        tl.where(denom > 0.0, out / safe_denom, 0.0),
    )


@on_input_device
def prepare_pure_triton_mtp1_workspace(inputs, quant_type: str) -> PureTritonMTP1Workspace:
    if inputs.mtp != 1:
        raise NotImplementedError("the single-token workspace requires MTP=1")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != 128:
        raise ValueError("q must have shape [batch, Hq, 128] for MTP=1")
    if inputs.k_cache.ndim != 4 or inputs.v_cache.ndim != 4:
        raise ValueError("K/V caches must be rank-4 tensors")
    if inputs.k_cache.shape[1] != 64 or inputs.v_cache.shape[1] != 64:
        raise ValueError("K/V caches must use 64-token blocks")
    if inputs.k_cache.shape[-1] != 128 or inputs.v_cache.shape[-1] != 128:
        raise ValueError("K/V cache head dimension must be 128")
    if inputs.block_ids.ndim != 2 or inputs.block_ids.dtype != torch.int32:
        raise ValueError("block_ids must be rank-2 int32")
    if inputs.kv_lens.ndim != 1 or inputs.kv_lens.dtype != torch.int32:
        raise ValueError("kv_lens must be rank-1 int32")
    if inputs.q.shape[0] != inputs.kv_lens.numel():
        raise ValueError("MTP1 q batch must match kv_lens")
    if quant_type != "bf16":
        raise NotImplementedError("Use the explicit INT8 Decode API on MetaX")
    if any(t.dtype != torch.bfloat16 for t in (inputs.q, inputs.k_cache, inputs.v_cache)):
        raise ValueError("BF16 Decode requires BF16 Q/K/V")
    from ._utils import prepare_decode_workspace_storage

    return prepare_decode_workspace_storage(inputs, quant_type)


def _pure_triton_mtp1_inputs(inputs, query_index: int):
    """View one MTP query as an independent MTP=1 workload."""
    batch = inputs.batch
    mtp = inputs.mtp
    q = inputs.q.reshape(batch, mtp, *inputs.q.shape[1:])[:, query_index]
    offset = mtp - 1 - query_index
    effective_kv_lens = inputs.kv_lens if offset == 0 else inputs.kv_lens - offset
    fields = {"q": q, "kv_lens": effective_kv_lens}
    q_scale = getattr(inputs, "q_scale", None)
    if q_scale is not None:
        fields["q_scale"] = q_scale.reshape(batch, mtp, -1)[:, query_index]
    return replace(inputs, **fields)


@on_input_device
def prepare_pure_triton_mtp_workspace(inputs, quant_type: str) -> PureTritonMTPWorkspace:
    """Prepare paired MACA BF16 storage or per-token portable workspaces."""
    if inputs.mtp < 2:
        raise ValueError("the MTP workspace requires MTP >= 2")
    if quant_type != "bf16":
        raise NotImplementedError("Use the explicit INT8 Decode API on MetaX")
    if (
        quant_type == "bf16"
        and inputs.mtp in (2, 3, 4)
        and inputs.q.shape[-1] == 128
        and inputs.q.shape[1] == inputs.k_cache.shape[2] * 8
        and triton.runtime.driver.active.get_current_target().backend == "maca"
    ):
        prepare = _bf16_paired_prepare

        paired = prepare(inputs)
        base = paired.base
        # Child views retain the MTP workspace interface without duplicating storage.
        children = tuple(
            replace(
                base,
                split_out=base.split_out[:, :, token],
                split_lse=base.split_lse[:, :, token],
                out=base.out.reshape(inputs.batch, inputs.mtp, *inputs.q.shape[1:])[:, token],
            )
            for token in range(inputs.mtp)
        )
        return PureTritonMTPWorkspace(workspaces=children, paired=paired)
    workspaces = tuple(
        prepare_pure_triton_mtp1_workspace(
            _pure_triton_mtp1_inputs(inputs, query_index),
            quant_type,
        )
        for query_index in range(inputs.mtp)
    )
    return PureTritonMTPWorkspace(workspaces=workspaces)


@on_input_device
def attention_decode_pure_triton_mtp(
    inputs,
    workspace: PureTritonMTPWorkspace,
):
    """Run the selected paired or per-token portable implementation."""
    validate_workspace_device(workspace, inputs.q.device)
    if workspace.paired is not None:
        run = _bf16_paired_run

        return run(inputs, workspace.paired)
    output_all = torch.empty(inputs.q.shape, dtype=torch.bfloat16, device=inputs.q.device)
    output_view = output_all.view(inputs.batch, inputs.mtp, *inputs.q.shape[1:])
    for query_index, mtp1_workspace in enumerate(workspace.workspaces):
        mtp1_inputs = _pure_triton_mtp1_inputs(inputs, query_index)
        output = attention_decode_pure_triton_mtp1(mtp1_inputs, mtp1_workspace)
        output_view[:, query_index].copy_(output)
    return output_all


@on_input_device
def attention_decode_pure_triton_mtp1(inputs, workspace: PureTritonMTP1Workspace):
    validate_workspace_device(workspace, inputs.q.device)
    if workspace.quant_type != "bf16":
        raise NotImplementedError("Use the explicit INT8 Decode API on MetaX")
    if inputs.mtp != 1:
        raise NotImplementedError("the single-token workspace requires MTP=1")
    # Bounded MACA MTP1 direct split avoids task-map construction.
    # Capacity is shape metadata; actual lengths remain GPU inputs on replay.
    if (
        workspace.quant_type == "bf16"
        and inputs.q.shape[-1] == 128
        and inputs.q.shape[1] == inputs.k_cache.shape[2] * 8
        and inputs.q.is_contiguous()
        and workspace.out.is_contiguous()
        and 0 < inputs.block_ids.shape[1] * 64 <= 16384
        and triton.cdiv(inputs.block_ids.shape[1] * 64, 256) <= workspace.num_ctas
        and triton.runtime.driver.active.get_current_target().backend == "maca"
    ):
        run_with_workspace = _bf16_mtp1_direct_run_with_workspace

        return run_with_workspace(inputs, workspace)
    batch = inputs.batch
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if hq % hkv:
        raise ValueError("query heads must be divisible by KV heads")

    launch_pure_triton_task_map(
        inputs.kv_lens,
        workspace.task_map,
        num_head_kv=hkv,
        num_ctas=workspace.num_ctas,
        min_process_len=workspace.min_process_len,
        max_tasks=workspace.max_tasks,
        sched_ints=workspace.sched_ints,
    )
    common = dict(
        H_Q=hq,
        HEADS_PER_GROUP=workspace.heads_per_group,
        GROUP_M=(16 if triton.runtime.driver.active.get_current_target().backend == "maca" else 8),
        D=inputs.q.shape[-1],
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        MAX_TASKS=workspace.max_tasks,
        BLOCK_N=64,
        Q_SB=inputs.q.stride(0),
        Q_SH=inputs.q.stride(1),
        K_SBLOCK=inputs.k_cache.stride(0),
        K_STOKEN=inputs.k_cache.stride(1),
        K_SHEAD=inputs.k_cache.stride(2),
        K_SD=inputs.k_cache.stride(3),
        V_SBLOCK=inputs.v_cache.stride(0),
        V_STOKEN=inputs.v_cache.stride(1),
        V_SHEAD=inputs.v_cache.stride(2),
        V_SD=inputs.v_cache.stride(3),
        SO_SB=workspace.split_out.stride(0),
        SO_SC=workspace.split_out.stride(1),
        SO_SH=workspace.split_out.stride(2),
        SL_SB=workspace.split_lse.stride(0),
        SL_SC=workspace.split_lse.stride(1),
        SL_SK=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        num_warps=4,
        num_stages=3,
    )
    grid = (
        workspace.num_ctas,
        triton.cdiv(workspace.heads_per_group, common["GROUP_M"]),
    )
    if workspace.quant_type == "bf16":
        producer = _pure_triton_bf16_gqa8_splitk_kernel
        # Use the measured pipeline setting for MACA BF16 GQA8 tiles.
        if workspace.heads_per_group == 8 and triton.runtime.driver.active.get_current_target().backend == "maca":
            common["num_stages"] = 1
            if inputs.q.shape[-1] == 128:
                producer = _bf16_mtp1_pure_triton_bf16_gqa8_splitk_kernel
        producer[grid](
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            workspace.task_map,
            workspace.split_out,
            workspace.split_lse,
            **common,
        )
    _pure_triton_splitk_combine_kernel[(batch, hq)](
        workspace.split_out,
        workspace.split_lse,
        workspace.task_map,
        workspace.out,
        H_Q=hq,
        HEADS_PER_GROUP=workspace.heads_per_group,
        D=inputs.q.shape[-1],
        MAX_SPLIT=workspace.num_ctas,
        MAX_TASKS=workspace.max_tasks,
        NUM_CTAS=workspace.num_ctas,
        SO_SB=workspace.split_out.stride(0),
        SO_SC=workspace.split_out.stride(1),
        SO_SH=workspace.split_out.stride(2),
        SL_SB=workspace.split_lse.stride(0),
        SL_SC=workspace.split_lse.stride(1),
        SL_SK=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SH=workspace.out.stride(1),
        num_warps=4,
        num_stages=1,
    )
    return workspace.out


# static

_bf16_entry__HEAD_DIM = 128

_bf16_entry__BLOCK_SIZE = 64

_bf16_entry__OFFICIAL_CASES = {
    "uniform_512": (512,) * 64,
    "uniform_4096": (4096,) * 64,
    "skewed_mix": (128,) * 32 + (4096,) * 32,
    "skewed_extreme": (64,) * 15 + (16 * 1024,),
    "one_64k_7x4k": (64 * 1024,) + (4096,) * 7,
    "one_64k_15x4k": (64 * 1024,) + (4096,) * 15,
    "one_64k_31x4k": (64 * 1024,) + (4096,) * 31,
    "one_128k_31x4k": (128 * 1024,) + (4096,) * 31,
    "two_32k_30x4k": (32 * 1024,) * 2 + (4096,) * 30,
}


@dataclass(frozen=True)
class _bf16_entry__StaticBF16Policy:
    workload: DecodeWorkload
    layout: str
    mtp: int
    cluster_size: int
    chunk_tokens: int
    kernel: str
    narrow: bool
    finalizer_heads: int

    @property
    def label(self) -> str:
        return f"c{self.cluster_size}t{self.chunk_tokens}/{self.kernel}"


@dataclass
class _bf16_entry__StaticBF16Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    layout: str

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)


def _bf16_entry___validate(inputs: _bf16_entry__StaticBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ("NHD", "HND"):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype != torch.bfloat16:
        raise ValueError("q must be bfloat16")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != _bf16_entry__HEAD_DIM:
        raise ValueError("q must have shape [batch * MTP, Hq, 128]")
    if inputs.mtp not in (1, 2, 3):
        raise ValueError("final BF16 static decode supports MTP 1, 2, or 3")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if cache.ndim != 4 or cache.shape[1] != 64 or cache.shape[3] != 128:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    if inputs.block_ids.ndim != 2 or inputs.block_ids.dtype != torch.int32:
        raise ValueError("block_ids must be rank-2 int32")
    if inputs.kv_lens.dtype != torch.int32:
        raise ValueError("kv_lens must be int32")
    hq, hkv = (int(inputs.q.shape[1]), int(inputs.k_cache.shape[2]))
    if (hkv, hq) not in ((1, 8), (4, 32)):
        raise ValueError("final BF16 decode requires official GQA8 heads (Hkv,Hq)=(1,8) or (4,32)")
    if hkv > 1:
        inferred_layout = "HND" if inputs.k_cache.stride(2) > inputs.k_cache.stride(1) else "NHD"
        if inferred_layout != inputs.layout:
            raise ValueError("explicit layout does not match K-cache strides")
        inferred_v_layout = "HND" if inputs.v_cache.stride(2) > inputs.v_cache.stride(1) else "NHD"
        if inferred_v_layout != inputs.layout:
            raise ValueError("explicit layout does not match V-cache strides")
    return (inputs.mtp, hq, hkv)


def _bf16_entry__select_static_bf16_policy(inputs):
    _bf16_entry___validate(inputs)
    workload = DecodeWorkload.from_lengths(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    return _bf16_entry__StaticBF16Policy(workload, inputs.layout, inputs.mtp, 1, 512, "triton-gqa8", False, 1)


BLOCK_SIZE = _bf16_entry__BLOCK_SIZE

HEAD_DIM = _bf16_entry__HEAD_DIM

OFFICIAL_CASES = _bf16_entry__OFFICIAL_CASES

StaticBF16Inputs = _bf16_entry__StaticBF16Inputs

StaticBF16Policy = _bf16_entry__StaticBF16Policy

StaticBF16Workspace = PureTritonMTP1Workspace | PureTritonMTPWorkspace


def prepare_static_bf16_workspace(inputs):
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, "bf16")
    return prepare_pure_triton_mtp_workspace(inputs, "bf16")


select_static_bf16_policy = _bf16_entry__select_static_bf16_policy


def attention_decode_bf16_triton(inputs, workspace):
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    raise TypeError(f"unsupported Triton workspace: {type(workspace)!r}")


# Deprecated flag; dispatch is always ordinary Triton.

# Compatibility alias; executes ordinary Triton.
attention_decode_bf16_tle = attention_decode_bf16_triton
attention_decode_bf16_static = attention_decode_bf16_triton


# dynamic






@dataclass(frozen=True)
class DynamicBF16Policy:
    layout: str
    mtp: int
    workload: DecodeWorkload
    cluster_size: int
    chunk_tokens: int
    kernel: str
    narrow: bool
    compact_direct_narrow: bool = False
    finalizer_heads: int = 1
    pdl_finalizer: bool = False
    deterministic_tail_election: bool = False
    reduction_only: bool = False
    aligned_full_chunk: bool = False

    @property
    def label(self) -> str:
        label = f"c{self.cluster_size}t{self.chunk_tokens}/{self.kernel}"
        if self.aligned_full_chunk:
            label += "-aligned"
        elif self.reduction_only:
            label += "-reduce"
        if self.deterministic_tail_election and self.cluster_size > 1:
            label += "-det"
        if self.finalizer_heads > 1:
            label += f"-h{self.finalizer_heads}-pdl"
        elif self.pdl_finalizer:
            label += "-pdl"
        return label


@dataclass
class DynamicBF16Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    layout: str

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)


DynamicBF16Workspace = PureTritonMTP1Workspace | PureTritonMTPWorkspace


def _dynamic__validate(inputs: DynamicBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ("NHD", "HND"):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype != torch.bfloat16 or inputs.q.ndim != 3 or inputs.q.shape[-1] != 128:
        raise ValueError("q must be BF16 [batch * MTP, Hq, 128]")
    if inputs.mtp not in (1, 2, 3):
        raise ValueError("BF16 dynamic decode supports MTP 1, 2, or 3")
    for name, cache in (
        ("k_cache", inputs.k_cache),
        ("v_cache", inputs.v_cache),
    ):
        if cache.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if cache.ndim != 4 or cache.shape[1] != 64 or cache.shape[3] != 128:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    if inputs.block_ids.dtype != torch.int32 or inputs.block_ids.ndim != 2:
        raise ValueError("block_ids must be rank-2 int32")
    if inputs.kv_lens.dtype != torch.int32 or not inputs.kv_lens.is_cuda:
        raise ValueError("kv_lens must be a CUDA int32 tensor")
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if (hkv, hq) not in ((1, 8), (4, 32)):
        raise ValueError("BF16 dynamic decode requires official GQA8 heads")
    if hkv > 1:
        inferred_k = "HND" if inputs.k_cache.stride(2) > inputs.k_cache.stride(1) else "NHD"
        inferred_v = "HND" if inputs.v_cache.stride(2) > inputs.v_cache.stride(1) else "NHD"
        if inferred_k != inputs.layout or inferred_v != inputs.layout:
            raise ValueError("explicit layout does not match cache strides")
    return inputs.mtp, hq, hkv


def select_dynamic_bf16_policy(inputs):
    _dynamic__validate(inputs)
    workload = DecodeWorkload.from_lengths(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    return DynamicBF16Policy(inputs.layout, inputs.mtp, workload, 1, 512, "triton-gqa8", False)


prepare_dynamic_bf16_workspace = prepare_static_bf16_workspace


attention_decode_bf16_dynamic = attention_decode_bf16_triton


def refresh_dynamic_bf16_task_map(inputs, workspace) -> None:
    """Rebuild state after input changes; recapture graphs using old buffers."""
    if isinstance(workspace, PureTritonMTPWorkspace):
        if inputs.mtp != len(workspace.workspaces):
            raise ValueError("MTP changed; prepare a new workspace")
        replacement = prepare_pure_triton_mtp_workspace(inputs, "bf16")
        workspace.workspaces = replacement.workspaces
        workspace.paired = replacement.paired
        return
    if isinstance(workspace, PureTritonMTP1Workspace):
        if inputs.mtp != 1:
            raise ValueError("MTP changed; prepare a new workspace")
        from dataclasses import fields

        replacement = prepare_pure_triton_mtp1_workspace(inputs, "bf16")
        for field in fields(replacement):
            setattr(workspace, field.name, getattr(replacement, field.name))
        return
    raise TypeError(f"unsupported BF16 Triton workspace: {type(workspace)!r}")


def bf16_dynamic_workspace_is_reset(workspace):
    if isinstance(workspace, PureTritonMTPWorkspace):
        return all(bf16_dynamic_workspace_is_reset(child) for child in workspace.workspaces)
    if isinstance(workspace, PureTritonMTP1Workspace):
        return True
    raise TypeError(f"unsupported Triton workspace: {type(workspace)!r}")


# Deprecated flag; dispatch is always ordinary Triton.


__all__ = [
    "PURE_TRITON_TASK_STRIDE",
    "PURE_TRITON_TILE_N",
    "assign_pure_triton_task_map_kernel",
    "pure_triton_task_map_metadata",
    "launch_pure_triton_task_map",
    "DecodeWorkload",
    "HAS_TLE",
    "USE_TLE",
    "PureTritonMTP1Workspace",
    "PureTritonMTPWorkspace",
    "attention_decode_pure_triton_mtp",
    "attention_decode_pure_triton_mtp1",
    "prepare_pure_triton_mtp_workspace",
    "prepare_pure_triton_mtp1_workspace",
    "BLOCK_SIZE",
    "HEAD_DIM",
    "OFFICIAL_CASES",
    "StaticBF16Inputs",
    "StaticBF16Policy",
    "StaticBF16Workspace",
    "attention_decode_bf16_tle",
    "prepare_static_bf16_workspace",
    "select_static_bf16_policy",
    "attention_decode_bf16_triton",
    "DynamicBF16Inputs",
    "DynamicBF16Policy",
    "DynamicBF16Workspace",
    "attention_decode_bf16_dynamic",
    "bf16_dynamic_workspace_is_reset",
    "prepare_dynamic_bf16_workspace",
    "refresh_dynamic_bf16_task_map",
    "select_dynamic_bf16_policy",
]

__all__.append("attention_decode_bf16_static")
