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

import triton
import triton.language as tl
import torch
from dataclasses import dataclass, replace
from ._utils import prepare_decode_workspace_storage
from ._utils import on_input_device, validate_workspace_device


# kernels


@triton.jit
def _int8_single_kernel(
    Q,
    K,
    V,
    BLOCK_IDS,
    TASK_MAP,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
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
    QUANT_TYPE: tl.constexpr,
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
    QS_SB: tl.constexpr,
    QS_SH: tl.constexpr,
    KS_SBLOCK: tl.constexpr,
    KS_STOKEN: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    Q_STEP,
    K_STEP,
    V_STEP,
):
    """INT8 QK and FP16 PV MMA with FP32 softmax and accumulation."""
    cta = tl.program_id(0)
    group_block = tl.program_id(1)
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    dot_scale = tl.rsqrt(tl.full((), D, tl.float32))
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
        q_step = tl.load(Q_STEP + batch * (Q_SB // D) + hq, mask=valid_h, other=1.0)
        qi = q.to(tl.int8)
        q_scale = tl.load(
            Q_SCALE + batch * QS_SB + hq * QS_SH,
            mask=valid_h,
            other=1.0,
        ).to(tl.float32)
        value_scale = tl.load(V_SCALE + (hkv if QUANT_TYPE == 0 else 0)).to(tl.float32)
        tensor_k_scale = tl.load(K_SCALE).to(tl.float32) if QUANT_TYPE == 1 else 1.0
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
            k_step = tl.load(
                K_STEP + (physical * BLOCK_SIZE + pos) * (H_Q // 8) + hkv,
                mask=valid_n,
                other=1.0,
            )
            ki = k.to(tl.int8)
            scores = (
                tl.trans(tl.dot(ki, tl.trans(qi), out_dtype=tl.int32)).to(tl.float32)
                * q_step[:, None]
                * k_step[None, :]
            )
            if QUANT_TYPE == 0:
                byte_offset = physical * KS_SBLOCK + (pos // 32) * KS_STOKEN + hkv * KS_SHEAD + (pos % 32) * 4 * KS_SD
                scale_ptr = (K_SCALE + byte_offset).to(tl.pointer_type(tl.float32))
                k_scale = tl.load(scale_ptr, mask=valid_n, other=0.0).to(tl.float32)
                scores *= q_scale[:, None] * k_scale[None, :] * dot_scale
            else:
                scores *= q_scale[:, None] * tensor_k_scale * dot_scale
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
            physical_page = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
            v_step = tl.load(V_STEP + physical_page * (H_Q // 8) + hkv)
            v = tl.load(
                V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD,
                mask=valid_n[:, None],
                other=0.0,
            )
            vi = v.to(tl.int8)
            pi = p.to(tl.float16)
            pv = tl.dot(tl.trans(vi.to(tl.float16)), tl.trans(pi), out_dtype=tl.float32)
            acc = acc * alpha[None, :] + pv * v_step
            m_i = m_new
            start += BLOCK_N
        has_value = l_i > 0.0
        partial = tl.where(
            has_value[:, None],
            tl.trans(acc) / l_i[:, None] * value_scale,
            0.0,
        )
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
def _int8_pair_kernel(
    LENS,
    MTP: tl.constexpr,
    TOKEN_BASE: tl.constexpr,
    SO_ST: tl.constexpr,
    SL_ST: tl.constexpr,
    Q,
    K,
    V,
    BLOCK_IDS,
    TASK_MAP,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
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
    QUANT_TYPE: tl.constexpr,
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
    QS_SB: tl.constexpr,
    QS_SH: tl.constexpr,
    KS_SBLOCK: tl.constexpr,
    KS_STOKEN: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    Q_STEP,
    K_STEP,
    V_STEP,
):
    """INT8 QK and FP16 PV MMA with FP32 softmax and accumulation."""
    cta = tl.program_id(0)
    group_block = tl.program_id(1)
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    token = offs_h // 8
    valid_h = offs_h < 16
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    dot_scale = tl.rsqrt(tl.full((), D, tl.float32))
    for slot in tl.static_range(0, MAX_TASKS):
        task = ((MAX_TASKS * cta + slot) + 1) * 12
        hkv = tl.load(TASK_MAP + task + 0)
        batch = tl.load(TASK_MAP + task + 1)
        if hkv < 0:
            return
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        hq = hkv * HEADS_PER_GROUP + offs_h % 8
        query_batch = batch * MTP + TOKEN_BASE + token
        row_length = tl.load(LENS + batch) - (MTP - 1 - TOKEN_BASE - token)
        q = tl.load(
            Q + query_batch[:, None] * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        q_step = tl.load(Q_STEP + query_batch * (Q_SB // D) + hq, mask=valid_h, other=1.0)
        qi = q.to(tl.int8)
        q_scale = tl.load(
            Q_SCALE + query_batch * QS_SB + hq * QS_SH,
            mask=valid_h,
            other=1.0,
        ).to(tl.float32)
        value_scale = tl.load(V_SCALE + (hkv if QUANT_TYPE == 0 else 0)).to(tl.float32)
        tensor_k_scale = tl.load(K_SCALE).to(tl.float32) if QUANT_TYPE == 1 else 1.0
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
            k_step = tl.load(
                K_STEP + (physical * BLOCK_SIZE + pos) * (H_Q // 8) + hkv,
                mask=valid_n,
                other=1.0,
            )
            ki = k.to(tl.int8)
            scores = (
                tl.trans(tl.dot(ki, tl.trans(qi), out_dtype=tl.int32)).to(tl.float32)
                * q_step[:, None]
                * k_step[None, :]
            )
            if QUANT_TYPE == 0:
                byte_offset = physical * KS_SBLOCK + (pos // 32) * KS_STOKEN + hkv * KS_SHEAD + (pos % 32) * 4 * KS_SD
                scale_ptr = (K_SCALE + byte_offset).to(tl.pointer_type(tl.float32))
                k_scale = tl.load(scale_ptr, mask=valid_n, other=0.0).to(tl.float32)
                scores *= q_scale[:, None] * k_scale[None, :] * dot_scale
            else:
                scores *= q_scale[:, None] * tensor_k_scale * dot_scale
            scores = tl.where(
                valid_h[:, None] & valid_n[None, :] & (logical[None, :] < row_length[:, None]),
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
            physical_page = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
            v_step = tl.load(V_STEP + physical_page * (H_Q // 8) + hkv)
            v = tl.load(
                V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD,
                mask=valid_n[:, None],
                other=0.0,
            )
            vi = v.to(tl.int8)
            pi = p.to(tl.float16)
            pv = tl.dot(tl.trans(vi.to(tl.float16)), tl.trans(pi), out_dtype=tl.float32)
            acc = acc * alpha[None, :] + pv * v_step
            m_i = m_new
            start += BLOCK_N
        has_value = l_i > 0.0
        partial = tl.where(
            has_value[:, None],
            tl.trans(acc) / l_i[:, None] * value_scale,
            0.0,
        )
        lse = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
        tl.store(
            SPLIT_OUT + token[:, None] * SO_ST + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
            partial,
            mask=valid_h[:, None],
        )
        tl.store(
            SPLIT_LSE + token * SL_ST + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + (offs_h % 8) * SL_SH,
            lse,
            mask=valid_h,
        )


@triton.jit
def _int8_four_kernel(
    LENS,
    MTP: tl.constexpr,
    TOKEN_BASE: tl.constexpr,
    SO_ST: tl.constexpr,
    SL_ST: tl.constexpr,
    Q,
    K,
    V,
    BLOCK_IDS,
    TASK_MAP,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
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
    QUANT_TYPE: tl.constexpr,
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
    QS_SB: tl.constexpr,
    QS_SH: tl.constexpr,
    KS_SBLOCK: tl.constexpr,
    KS_STOKEN: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    Q_STEP,
    K_STEP,
    V_STEP,
):
    """INT8 QK and FP16 PV MMA with FP32 softmax and accumulation."""
    cta = tl.program_id(0)
    group_block = tl.program_id(1)
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    token = offs_h // 8
    valid_h = offs_h < GROUP_M
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    dot_scale = tl.rsqrt(tl.full((), D, tl.float32))
    for slot in tl.static_range(0, MAX_TASKS):
        task = ((MAX_TASKS * cta + slot) + 1) * 12
        hkv = tl.load(TASK_MAP + task + 0)
        batch = tl.load(TASK_MAP + task + 1)
        if hkv < 0:
            return
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        hq = hkv * HEADS_PER_GROUP + offs_h % 8
        query_batch = batch * MTP + TOKEN_BASE + token
        row_length = tl.load(LENS + batch) - (MTP - 1 - TOKEN_BASE - token)
        q = tl.load(
            Q + query_batch[:, None] * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        q_step = tl.load(Q_STEP + query_batch * (Q_SB // D) + hq, mask=valid_h, other=1.0)
        qi = q.to(tl.int8)
        q_scale = tl.load(
            Q_SCALE + query_batch * QS_SB + hq * QS_SH,
            mask=valid_h,
            other=1.0,
        ).to(tl.float32)
        value_scale = tl.load(V_SCALE + (hkv if QUANT_TYPE == 0 else 0)).to(tl.float32)
        tensor_k_scale = tl.load(K_SCALE).to(tl.float32) if QUANT_TYPE == 1 else 1.0
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
            k_step = tl.load(
                K_STEP + (physical * BLOCK_SIZE + pos) * (H_Q // 8) + hkv,
                mask=valid_n,
                other=1.0,
            )
            ki = k.to(tl.int8)
            scores = (
                tl.trans(tl.dot(ki, tl.trans(qi), out_dtype=tl.int32)).to(tl.float32)
                * q_step[:, None]
                * k_step[None, :]
            )
            if QUANT_TYPE == 0:
                byte_offset = physical * KS_SBLOCK + (pos // 32) * KS_STOKEN + hkv * KS_SHEAD + (pos % 32) * 4 * KS_SD
                scale_ptr = (K_SCALE + byte_offset).to(tl.pointer_type(tl.float32))
                k_scale = tl.load(scale_ptr, mask=valid_n, other=0.0).to(tl.float32)
                scores *= q_scale[:, None] * k_scale[None, :] * dot_scale
            else:
                scores *= q_scale[:, None] * tensor_k_scale * dot_scale
            scores = tl.where(
                valid_h[:, None] & valid_n[None, :] & (logical[None, :] < row_length[:, None]),
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
            physical_page = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
            v_step = tl.load(V_STEP + physical_page * (H_Q // 8) + hkv)
            v = tl.load(
                V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD,
                mask=valid_n[:, None],
                other=0.0,
            )
            vi = v.to(tl.int8)
            pi = p.to(tl.float16)
            pv = tl.dot(tl.trans(vi.to(tl.float16)), tl.trans(pi), out_dtype=tl.float32)
            acc = acc * alpha[None, :] + pv * v_step
            m_i = m_new
            start += BLOCK_N
        has_value = l_i > 0.0
        partial = tl.where(
            has_value[:, None],
            tl.trans(acc) / l_i[:, None] * value_scale,
            0.0,
        )
        lse = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
        tl.store(
            SPLIT_OUT + token[:, None] * SO_ST + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
            partial,
            mask=valid_h[:, None],
        )
        tl.store(
            SPLIT_LSE + token * SL_ST + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + (offs_h % 8) * SL_SH,
            lse,
            mask=valid_h,
        )


# merge


@triton.jit
def _merge_single(
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
    if n_chunks <= 8:
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
    else:
        splits = tl.arange(0, 32)
        running_max = tl.full((), -float("inf"), tl.float32)
        denom = tl.zeros((), tl.float32)
        out = tl.zeros((D,), tl.float32)
        base = 0
        while base < n_chunks:
            chunks = base + splits
            valid = chunks < n_chunks
            values = tl.load(
                SPLIT_LSE + batch * SL_SB + chunks * SL_SC + hkv * SL_SK + h_in_group * SL_SH,
                mask=valid,
                other=-float("inf"),
            )
            tile_max = tl.max(values, 0)
            new_max = tl.maximum(running_max, tile_max)
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            alpha = tl.exp(running_max - safe_max)
            weights = tl.exp(values - safe_max)
            partial = tl.load(
                SPLIT_OUT + batch * SO_SB + chunks[:, None] * SO_SC + hq * SO_SH + offs_d[None, :],
                mask=(valid & (values != -float("inf")))[:, None],
                other=0.0,
            )
            out = out * alpha + tl.sum(weights[:, None] * partial, 0)
            denom = denom * alpha + tl.sum(weights, 0)
            running_max = new_max
            base += 32
        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        tl.store(
            OUT + batch * O_SB + hq * O_SH + offs_d,
            tl.where(denom > 0.0, out / safe_denom, 0.0),
        )


@triton.jit
def _merge_tokens(
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
    SO_ST: tl.constexpr,
    SL_ST: tl.constexpr,
    O_ST: tl.constexpr,
):
    token = tl.program_id(2)
    SPLIT_OUT += token * SO_ST
    SPLIT_LSE += token * SL_ST
    OUT += token * O_ST
    batch = tl.program_id(0)
    hq = tl.program_id(1)
    hkv = hq // HEADS_PER_GROUP
    h_in_group = hq - hkv * HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    counts = (MAX_TASKS * NUM_CTAS + 1) * 12
    batch_count = tl.load(TASK_MAP + 3)
    n_chunks = tl.load(TASK_MAP + counts + hkv * batch_count + batch)
    if n_chunks <= 8:
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
    else:
        splits = tl.arange(0, 32)
        running_max = tl.full((), -float("inf"), tl.float32)
        denom = tl.zeros((), tl.float32)
        out = tl.zeros((D,), tl.float32)
        base = 0
        while base < n_chunks:
            chunks = base + splits
            valid = chunks < n_chunks
            values = tl.load(
                SPLIT_LSE + batch * SL_SB + chunks * SL_SC + hkv * SL_SK + h_in_group * SL_SH,
                mask=valid,
                other=-float("inf"),
            )
            tile_max = tl.max(values, 0)
            new_max = tl.maximum(running_max, tile_max)
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            alpha = tl.exp(running_max - safe_max)
            weights = tl.exp(values - safe_max)
            partial = tl.load(
                SPLIT_OUT + batch * SO_SB + chunks[:, None] * SO_SC + hq * SO_SH + offs_d[None, :],
                mask=(valid & (values != -float("inf")))[:, None],
                other=0.0,
            )
            out = out * alpha + tl.sum(weights[:, None] * partial, 0)
            denom = denom * alpha + tl.sum(weights, 0)
            running_max = new_max
            base += 32
        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        tl.store(
            OUT + batch * O_SB + hq * O_SH + offs_d,
            tl.where(denom > 0.0, out / safe_denom, 0.0),
        )


# schedule


@triton.jit
def _assign_int8_tasks(
    L,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    MIN: tl.constexpr,
    MAX: tl.constexpr,
    SCHED: tl.constexpr,
    NB: tl.constexpr,
    SM_COUNT: tl.constexpr,
    ADAPTIVE: tl.constexpr,
):
    c = tl.program_id(0)
    bs = tl.arange(0, NB)
    lens = tl.load(L + bs, bs < B, 0)
    tiles = (lens + 63) // 64
    ends = tl.cumsum(tiles, 0)
    total = tl.sum(tiles, 0)
    width = tl.maximum(tl.cdiv(total * H, C), tl.maximum(MIN // 64, 1))
    active_ctas = tl.cdiv(total * H, width)
    if ADAPTIVE and MIN == 512:
        width = tl.where(active_ctas < SM_COUNT, tl.maximum(tl.cdiv(total * H, C), 4), width)
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


# quantize


@triton.jit
def _quantize_rows(
    X,
    Y,
    S,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    h = row % HEADS
    p = row // HEADS
    d = tl.arange(0, D)
    if PAGE:
        off = p * S0 + (d // 128) * S1 + h * S2 + (d % 128) * S3
    else:
        off = (p // TOKENS) * S0 + (p % TOKENS) * S1 + h * S2 + d * S3
    x = tl.load(X + off).to(tl.float32)
    # PyTorch scalar division multiplies by an FP32 reciprocal. Preserve
    # that rounding before quantization, including exact half-way inputs.
    step = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-10) * 0.007874015718698502
    # Match the existing preparation contract, including half-way rounding.
    y = tl.minimum(tl.maximum(tl.floor(tl.div_rn(x, step) + 0.5), -127.0), 127.0)
    tl.store(Y + off, y.to(tl.int8))
    tl.store(S + row, step)


@triton.jit
def _check_lengths(
    X,
    RESULT,
    N: tl.constexpr,
    MINIMUM: tl.constexpr,
    MAXIMUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=MINIMUM)
    invalid = (i < N) & ((x < MINIMUM) | (x > MAXIMUM))
    tl.store(RESULT, tl.max(invalid.to(tl.int32), 0))


def validate_lengths(lengths, minimum, maximum):
    flag = torch.empty((), device=lengths.device, dtype=torch.int32)
    _check_lengths[(1,)](
        lengths,
        flag,
        lengths.numel(),
        minimum,
        maximum,
        triton.next_power_of_2(lengths.numel()),
    )
    if flag.item():
        raise ValueError("KV lengths must cover MTP and fit the page table")


def quantize_inputs(source):
    q, k, v = source.q, source.k_cache, source.v_cache
    if q.ndim != 3 or q.shape[-1] != 128 or not q.is_contiguous():
        raise ValueError("quantization expects contiguous Q with D=128")
    for cache in (k, v):
        if cache.ndim != 4 or cache.shape[1] != 64 or cache.shape[-1] != 128:
            raise ValueError("quantization expects [pages,64,Hkv,128] caches")
    if k.shape != v.shape:
        raise ValueError("K/V cache shapes must match")
    qi = torch.empty_like(q, dtype=torch.int8)
    ki = torch.empty_strided(k.shape, k.stride(), device=k.device, dtype=torch.int8)
    vi = torch.empty_strided(v.shape, v.stride(), device=v.device, dtype=torch.int8)
    qs = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    ks = torch.empty(k.shape[:3], device=k.device, dtype=torch.float32)
    vs = torch.empty((v.shape[0], v.shape[2]), device=v.device, dtype=torch.float32)
    _quantize_rows[(qs.numel(),)](
        q,
        qi,
        qs,
        q.shape[1],
        1,
        q.stride(0),
        0,
        q.stride(1),
        q.stride(2),
        False,
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _quantize_rows[(ks.numel(),)](
        k, ki, ks, k.shape[2], 64, *k.stride(), False, 128, num_warps=4, enable_fp_fusion=False
    )
    _quantize_rows[(vs.numel(),)](
        v, vi, vs, v.shape[2], 64, *v.stride(), True, 8192, num_warps=4, enable_fp_fusion=False
    )
    return qi, ki, vi, qs, ks, vs


# Public entry points


@dataclass
class INT8DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    q_step: torch.Tensor
    k_step: torch.Tensor
    v_step: torch.Tensor

    @property
    def batch(self):
        return self.kv_lens.numel()

    @property
    def mtp(self):
        if not self.batch or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return self.q.shape[0] // self.batch


def _signature(inputs):
    return tuple((tuple(t.shape), t.stride(), t.dtype, t.device) for t in vars(inputs).values())


@dataclass
class INT8DecodeWorkspace:
    pairs: tuple
    out: torch.Tensor
    quant_type: str
    adaptive_split: bool
    sm_count: int
    signature: tuple


def _validate(inputs, quant_type):
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        raise NotImplementedError("INT8 Decode currently supports MetaX only")
    if quant_type not in (
        "qkpertoken_perhead_vperhead",
        "qpertoken_perhead_kvpertensor",
    ):
        raise ValueError("unsupported quantization metadata format")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != 128 or inputs.mtp not in (1, 2, 4):
        raise ValueError("expected D=128 and MTP=1, 2 or 4")
    if not inputs.q.is_contiguous():
        raise ValueError("Q must be contiguous")
    for cache in (inputs.k_cache, inputs.v_cache):
        if cache.ndim != 4 or cache.shape[1] != 64 or cache.shape[-1] != 128:
            raise ValueError("K/V must have shape [pages,64,Hkv,128]")
    if inputs.k_cache.shape != inputs.v_cache.shape or inputs.q.shape[1] != 8 * inputs.k_cache.shape[2]:
        raise ValueError("INT8 Decode requires matching K/V and GQA8")
    if any(t.dtype != torch.int8 for t in (inputs.q, inputs.k_cache, inputs.v_cache)):
        raise ValueError("Q/K/V must be prequantized INT8 tensors")
    if any(t.device != inputs.q.device for t in vars(inputs).values()) or not inputs.q.is_cuda:
        raise ValueError("all tensors must be on the same GPU")
    if inputs.kv_lens.dtype != torch.int32 or inputs.kv_lens.ndim != 1:
        raise ValueError("kv_lens must be rank-1 INT32")
    if inputs.block_ids.dtype != torch.int32 or inputs.block_ids.ndim != 2 or inputs.block_ids.shape[0] != inputs.batch:
        raise ValueError("block_ids must be rank-2 INT32 with matching batch")
    if not inputs.block_ids.is_contiguous() or not inputs.kv_lens.is_contiguous():
        raise ValueError("block_ids and kv_lens must be contiguous")
    pages, _, heads, _ = inputs.k_cache.shape
    for t, shape in (
        (inputs.q_step, inputs.q.shape[:2]),
        (inputs.k_step, (pages, 64, heads)),
        (inputs.v_step, (pages, heads)),
    ):
        if t.dtype != torch.float32 or tuple(t.shape) != tuple(shape) or not t.is_contiguous():
            raise ValueError("INT8 steps must be contiguous FP32 tensors of the documented shape")
    if inputs.q_scale.dtype != torch.float32 or inputs.q_scale.shape != inputs.q.shape[:2]:
        raise ValueError("q_scale must be FP32 [batch*MTP,Hq]")
    if (
        inputs.v_scale.dtype != torch.float32
        or not inputs.v_scale.is_contiguous()
        or inputs.v_scale.numel() != (heads if quant_type == "qkpertoken_perhead_vperhead" else 1)
    ):
        raise ValueError("outer V scale must be contiguous FP32 with the expected number of elements")
    if quant_type == "qpertoken_perhead_kvpertensor":
        if inputs.k_scale.dtype != torch.float32 or inputs.k_scale.numel() != 1:
            raise ValueError("tensor K scale must be one FP32 value")
    elif (
        inputs.k_scale.ndim != 4
        or tuple(inputs.k_scale.shape) != (pages, 2, heads, 128)
        or inputs.k_scale.element_size() != 1
    ):
        raise ValueError("packed K scale must use source-format [pages,2,Hkv,128] bytes")

    validate_lengths(inputs.kv_lens, inputs.mtp, inputs.block_ids.shape[1] * 64)


@on_input_device
def prepare_int8_decode_workspace(inputs, quant_type, *, adaptive_split=True):
    """Prepare capacity for fixed 512 or adaptive 256/512 scheduling.

    Preparation may synchronize. Capture only attention_decode_int8. Rebuild
    for changed shapes/strides. In-place lengths must stay within the prepared
    page-table and prepared task capacity (see docs/metax_hy3_attention.rst);
    Q/K/V and their step tensors can be updated in place.
    """
    _validate(inputs, quant_type)
    output = torch.empty_like(inputs.q, dtype=torch.bfloat16)
    view = output.reshape(inputs.batch, inputs.mtp, *inputs.q.shape[1:])
    base = prepare_decode_workspace_storage(inputs, quant_type)
    tokens = inputs.mtp
    split_out = torch.empty(
        (inputs.batch, base.num_ctas, tokens, inputs.q.shape[1], 128),
        device=inputs.q.device,
        dtype=torch.float32,
    )
    split_lse = torch.empty(
        (inputs.batch, base.num_ctas, tokens, inputs.k_cache.shape[2], 8),
        device=inputs.q.device,
        dtype=torch.float32,
    )
    children = tuple(
        replace(
            base,
            split_out=split_out[:, :, token],
            split_lse=split_lse[:, :, token],
            out=view[:, token],
        )
        for token in range(tokens)
    )
    sm = torch.cuda.get_device_properties(inputs.q.device).multi_processor_count
    return INT8DecodeWorkspace(
        ((0, children, split_out, split_lse),),
        output,
        quant_type,
        bool(adaptive_split),
        sm,
        _signature(inputs),
    )


@on_input_device
def quantize_int8_decode_inputs(source):
    """Requantize source cache values to INT8 while retaining their outer scales.

    This preserves the existing two source metadata formats. It is not an
    implicit conversion performed by attention_decode_int8.
    """

    q, ki, vi, qs, ks, vs = quantize_inputs(source)
    return INT8DecodeInputs(
        q,
        ki,
        vi,
        source.block_ids,
        source.kv_lens,
        source.q_scale,
        source.k_scale,
        source.v_scale,
        qs,
        ks,
        vs,
    )


@on_input_device
def attention_decode_int8(inputs, state):
    validate_workspace_device(state, inputs.q.device)
    if _signature(inputs) != state.signature:
        raise ValueError("input metadata changed; prepare a new INT8 workspace")
    batch, hq, hkv = inputs.batch, inputs.q.shape[1], inputs.k_cache.shape[2]
    quant_type = 0 if state.quant_type == "qkpertoken_perhead_vperhead" else 1
    tokens = inputs.mtp
    for token_base, children, split_out, split_lse in state.pairs:
        workspace = children[-1]
        _assign_int8_tasks[(workspace.num_ctas,)](
            inputs.kv_lens,
            workspace.task_map,
            inputs.batch,
            hkv,
            workspace.num_ctas,
            512,
            workspace.max_tasks,
            workspace.sched_ints,
            triton.next_power_of_2(inputs.batch),
            SM_COUNT=state.sm_count,
            ADAPTIVE=state.adaptive_split,
            num_warps=4,
            num_stages=1,
        )
        common = dict(
            H_Q=hq,
            HEADS_PER_GROUP=workspace.heads_per_group,
            GROUP_M=32 if tokens == 4 else 16,
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
        common["num_stages"] = 1
        grid = (workspace.num_ctas, 1)
        args = (
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            workspace.task_map,
            inputs.q_scale,
            inputs.k_scale,
            inputs.v_scale,
            split_out,
            split_lse,
        )
        producer = {1: _int8_single_kernel, 2: _int8_pair_kernel, 4: _int8_four_kernel}[tokens]
        if tokens != 1:
            args = (
                inputs.kv_lens,
                inputs.mtp,
                token_base,
                split_out.stride(2),
                split_lse.stride(2),
            ) + args
        producer[grid](
            *args,
            Q_STEP=inputs.q_step,
            K_STEP=inputs.k_step,
            V_STEP=inputs.v_step,
            QUANT_TYPE=quant_type,
            QS_SB=inputs.q_scale.stride(0),
            QS_SH=inputs.q_scale.stride(1),
            KS_SBLOCK=inputs.k_scale.stride(0),
            KS_STOKEN=inputs.k_scale.stride(1) if inputs.k_scale.ndim > 1 else 0,
            KS_SHEAD=inputs.k_scale.stride(2) if inputs.k_scale.ndim > 2 else 0,
            KS_SD=inputs.k_scale.stride(3) if inputs.k_scale.ndim > 3 else 0,
            **common,
        )

        workspace = children[0]
        merge_kernel = _merge_single if tokens == 1 else _merge_tokens
        merge_grid = (batch, hq) if tokens == 1 else (batch, hq, tokens)
        token_strides = (
            {}
            if tokens == 1
            else dict(
                SO_ST=split_out.stride(2),
                SL_ST=split_lse.stride(2),
                O_ST=state.out.stride(0),
            )
        )
        merge_kernel[merge_grid](
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
            **token_strides,
            num_warps=4,
            num_stages=1,
        )

    return state.out
