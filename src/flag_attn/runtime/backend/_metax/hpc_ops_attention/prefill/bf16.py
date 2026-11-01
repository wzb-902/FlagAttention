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

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_q_ptr,
    block_ids_ptr,
    seqlens_kv_ptr,
    block_mask_ptr,
    out_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_block_ids_batch,
    stride_block_ids_page,
    stride_mask_batch,
    stride_mask_head,
    stride_mask_qtile,
    stride_mask_kvtile,
    stride_out_token,
    stride_out_head,
    stride_out_dim,
    num_head_q,
    num_head_kv,
    num_mask_kv_tiles,
    PAGE_SIZE: tl.constexpr,
    MAX_KV_TOKENS: tl.constexpr,
    HAS_BLOCK_MASK: tl.constexpr,
    SCALE_LOG2E_OVER_SQRT_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PV_TILE: tl.constexpr,
):
    """One program computes one [BLOCK_M, HEAD_DIM] output tile."""
    tl.static_assert(BLOCK_M == 32 or BLOCK_M == 64)
    tl.static_assert(PV_TILE == 32 or PV_TILE == 128)
    tl.static_assert(BLOCK_N == 128)
    tl.static_assert(BLOCK_N % PAGE_SIZE == 0)
    q_tile = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)

    q_begin = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(seqlens_kv_ptr + batch)

    q_local_start = q_tile * BLOCK_M
    program_valid = q_local_start < q_len
    kv_group_size = num_head_q // num_head_kv
    kv_head = q_head // kv_group_size
    q_start_in_kv = kv_len - q_len

    offs_m = q_local_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_valid = offs_m < q_len

    q_offsets = (q_begin + offs_m[:, None]) * stride_q_token + q_head * stride_q_head + offs_d[None, :] * stride_q_dim
    q = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)

    # The CUDA implementation accumulates both softmax state and PV in FP32.
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((HEAD_DIM, BLOCK_M), tl.float32)

    # Only tiles up to the causal end of this Q tile can contribute.  The loop
    # still has a compile-time maximum so Triton can pipeline it.
    q_tile_end = tl.minimum(q_local_start + BLOCK_M, q_len)
    causal_kv_end = q_start_in_kv + q_tile_end
    num_causal_kv_tiles = (causal_kv_end + BLOCK_N - 1) // BLOCK_N

    offs_n = tl.arange(0, BLOCK_N)
    max_kv_tiles = (MAX_KV_TOKENS + BLOCK_N - 1) // BLOCK_N
    # Keep indirect page-table loads at the validated single-stage depth.
    for kv_tile in tl.range(0, max_kv_tiles, 1, num_stages=1):
        tile_active = program_valid & (kv_tile < num_causal_kv_tiles)
        if HAS_BLOCK_MASK:
            mask_in_range = kv_tile < num_mask_kv_tiles
            mask_offset = (
                batch * stride_mask_batch
                + q_head * stride_mask_head
                + (q_local_start // 128) * stride_mask_qtile
                + kv_tile * stride_mask_kvtile
            )
            selected = (
                tl.load(
                    block_mask_ptr + mask_offset,
                    mask=mask_in_range,
                    other=0,
                )
                != 0
            )
            # Match the CUDA compatibility rule: if the supplied mask is
            # shorter than the causal range, retain exactly the first tile
            # after the represented range.
            selected = selected | (kv_tile == num_mask_kv_tiles)
            tile_active = tile_active & selected

        # This is a program-uniform branch.  A skipped BSA tile performs no
        # page-table loads, K/V loads, or Tensor-Core work.
        if tile_active:
            kv_tokens = kv_tile * BLOCK_N + offs_n
            kv_valid = kv_tokens < kv_len
            page_base = (
                block_ids_ptr
                + batch * stride_block_ids_batch
                + (kv_tile * BLOCK_N // PAGE_SIZE) * stride_block_ids_page
            )
            page0 = tl.load(page_base, mask=kv_tile * BLOCK_N < kv_len, other=0)
            page1 = tl.load(page_base + stride_block_ids_page, mask=kv_tile * BLOCK_N + PAGE_SIZE < kv_len, other=0)
            if PAGE_SIZE == 32:
                page2 = tl.load(page_base + 2 * stride_block_ids_page, mask=kv_tile * BLOCK_N + 64 < kv_len, other=0)
                page3 = tl.load(page_base + 3 * stride_block_ids_page, mask=kv_tile * BLOCK_N + 96 < kv_len, other=0)
                physical_pages = tl.where(
                    offs_n < 64, tl.where(offs_n < 32, page0, page1), tl.where(offs_n < 96, page2, page3)
                )
            else:
                physical_pages = tl.where(offs_n < 64, page0, page1)
            tokens_in_page = kv_tokens % PAGE_SIZE

            k_offsets = (
                physical_pages[:, None] * stride_k_page
                + tokens_in_page[:, None] * stride_k_token
                + kv_head * stride_k_head
                + offs_d[None, :] * stride_k_dim
            )
            k = tl.load(k_ptr + k_offsets, mask=kv_valid[:, None], other=0.0)

            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

            q_positions = q_start_in_kv + offs_m
            causal_valid = kv_tokens[None, :] <= q_positions[:, None]
            score_valid = q_valid[:, None] & kv_valid[None, :] & causal_valid
            scores = tl.where(
                score_valid,
                scores * SCALE_LOG2E_OVER_SQRT_D,
                -float("inf"),
            )

            # Online softmax in base 2, equivalent to the CUDA implementation.
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m_i, tile_max)
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            alpha = tl.exp2(m_i - safe_max)
            p = tl.exp2(scores - safe_max[:, None])
            tile_sum = tl.sum(p, axis=1)

            l_i = l_i * alpha + tile_sum
            acc *= alpha[None, :]

            p_bf16 = p.to(tl.bfloat16)

            if PV_TILE == 32:
                p0, p1 = tl.split(tl.permute(tl.reshape(p_bf16, (BLOCK_M, 2, 64)), (0, 2, 1)))
                p00, p01 = tl.split(tl.permute(tl.reshape(p0, (BLOCK_M, 2, 32)), (0, 2, 1)))
                p10, p11 = tl.split(tl.permute(tl.reshape(p1, (BLOCK_M, 2, 32)), (0, 2, 1)))
                for quarter in tl.static_range(4):
                    vn = kv_tile * BLOCK_N + quarter * 32 + tl.arange(0, 32)
                    vvalid = vn < kv_len
                    if PAGE_SIZE == 64:
                        if quarter < 2:
                            vp = page0
                        else:
                            vp = page1
                    else:
                        if quarter == 0:
                            vp = page0
                        elif quarter == 1:
                            vp = page1
                        elif quarter == 2:
                            vp = page2
                        else:
                            vp = page3
                    vo = (
                        vp * stride_v_page
                        + (vn % PAGE_SIZE)[None, :] * stride_v_token
                        + kv_head * stride_v_head
                        + offs_d[:, None] * stride_v_dim
                    )
                    vpart = tl.load(v_ptr + vo, mask=vvalid[None, :], other=0.0)
                    if quarter == 0:
                        ppart = p00
                    elif quarter == 1:
                        ppart = p01
                    elif quarter == 2:
                        ppart = p10
                    else:
                        ppart = p11
                    acc = tl.dot(vpart, tl.trans(ppart), acc, out_dtype=tl.float32)
            else:
                v_offsets = (
                    physical_pages[:, None] * stride_v_page
                    + tokens_in_page[:, None] * stride_v_token
                    + kv_head * stride_v_head
                    + offs_d[None, :] * stride_v_dim
                )
                v = tl.load(v_ptr + v_offsets, mask=kv_valid[:, None], other=0.0)
                acc = tl.dot(tl.trans(v), tl.trans(p_bf16), acc, out_dtype=tl.float32)
            m_i = new_max

    # A completely empty sparse row intentionally follows the CUDA contract
    # and produces NaN (0 / 0).  Callers are expected to retain the diagonal
    # BSA tile for every valid Q tile.
    output = tl.trans(acc) / l_i[:, None]
    out_offsets = (
        (q_begin + offs_m[:, None]) * stride_out_token + q_head * stride_out_head + offs_d[None, :] * stride_out_dim
    )
    tl.store(out_ptr + out_offsets, output.to(tl.bfloat16), mask=q_valid[:, None])


def attention_with_kvcache_blocksparse_prefill_bf16(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    block_mask: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16 paged causal Prefill with optional 128x128 sparse blocks.

    Q is [total_q, Hq, 128]; caches are logically [pages, page_size, Hkv, 128].
    Both NHD and HND strides are supported. Q positions are right-aligned to
    each sequence's KV length. The caller supplies valid page IDs, cumulative
    Q lengths and KV lengths, with 0 <= q_len <= kv_len and q_len <= max_seqlens_q.
    Metadata values are not copied to the CPU, preserving CUDA Graph replay.
    As in the FP8 API, a short mask retains the first unrepresented KV tile;
    a completely empty sparse row returns NaN. This is a forward-only API.
    """
    tensors = dict(
        q=q,
        kcache=kcache,
        vcache=vcache,
        cu_seqlens_q=cu_seqlens_q,
        block_ids=block_ids,
        seqlens_kvcache=seqlens_kvcache,
    )
    for name, tensor in tensors.items():
        if not tensor.is_cuda or tensor.device != q.device:
            raise ValueError(f"{name} must be a CUDA tensor on the Q device")
    if any(t.dtype != torch.bfloat16 for t in (q, kcache, vcache)):
        raise TypeError("q, kcache and vcache must use torch.bfloat16")
    if q.ndim != 3 or kcache.ndim != 4 or vcache.ndim != 4:
        raise ValueError("q must be rank 3 and caches must be rank 4")
    if q.shape[-1] != 128 or kcache.shape[-1] != 128 or vcache.shape != kcache.shape:
        raise ValueError("head_dim must be 128 and K/V cache shapes must match")
    if q.shape[1] <= 0 or kcache.shape[2] <= 0 or q.shape[1] % kcache.shape[2]:
        raise ValueError("positive query head count must be divisible by KV head count")
    page_size = kcache.shape[1]
    if page_size not in (32, 64) or kcache.shape[0] <= 0:
        raise ValueError("nonempty caches with page sizes 32 and 64 are supported")
    if any(t.dtype != torch.int32 for t in (cu_seqlens_q, block_ids, seqlens_kvcache)):
        raise TypeError("sequence metadata must be int32")
    if cu_seqlens_q.ndim != 1 or seqlens_kvcache.ndim != 1 or block_ids.ndim != 2:
        raise ValueError("invalid sequence metadata rank")
    batch = seqlens_kvcache.numel()
    if batch <= 0 or cu_seqlens_q.numel() != batch + 1 or block_ids.shape[0] != batch:
        raise ValueError("sequence metadata batch dimensions must match")
    if not cu_seqlens_q.is_contiguous() or not seqlens_kvcache.is_contiguous():
        raise ValueError("sequence length arrays must be contiguous")
    if block_ids.shape[1] <= 0 or not isinstance(max_seqlens_q, int) or max_seqlens_q <= 0:
        raise ValueError("page-table capacity and max_seqlens_q must be positive")
    if block_mask is not None:
        if block_mask.device != q.device or block_mask.dtype != torch.uint8:
            raise ValueError("block_mask must be uint8 on the Q device")
        if block_mask.ndim != 4 or block_mask.shape[:3] != (batch, q.shape[1], triton.cdiv(max_seqlens_q, 128)):
            raise ValueError("block_mask must have shape [batch, Hq, ceil(max_seqlens_q/128), Kb]")
        if block_mask.shape[3] <= 0:
            raise ValueError("block_mask must contain at least one KV tile")
    if output is None:
        output = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    elif (
        output.device != q.device
        or output.dtype != torch.bfloat16
        or output.shape != q.shape
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 with the Q shape and device")
    if q.numel() == 0:
        return output
    if any(output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in (q, kcache, vcache)):
        raise ValueError("output must not alias Q/K/V storage")
    has_mask = block_mask is not None
    capacity = block_ids.shape[1] * page_size
    block_m = 64 if capacity <= 128 else 32
    num_warps = 8 if capacity <= 128 else 4
    with torch.cuda.device(q.device):
        _bf16_prefill_kernel[(triton.cdiv(max_seqlens_q, block_m), q.shape[1], batch)](
            q,
            kcache,
            vcache,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            block_mask if has_mask else q,
            output,
            *q.stride(),
            *kcache.stride(),
            *vcache.stride(),
            *block_ids.stride(),
            *(block_mask.stride() if has_mask else (0, 0, 0, 0)),
            *output.stride(),
            q.shape[1],
            kcache.shape[2],
            block_mask.shape[3] if has_mask else 0,
            PAGE_SIZE=page_size,
            MAX_KV_TOKENS=capacity,
            HAS_BLOCK_MASK=has_mask,
            SCALE_LOG2E_OVER_SQRT_D=1.4426950408889634 / (128**0.5),
            BLOCK_M=block_m,
            BLOCK_N=128,
            HEAD_DIM=128,
            PV_TILE=128 if capacity <= 128 else 32,
            num_warps=num_warps,
            num_stages=1,
        )
    return output
