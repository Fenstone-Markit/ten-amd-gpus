import os  # [claude-mqa-01]
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
import math
from importlib.util import find_spec

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.glm5_meta_watch import meta_watch_check_c
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950
else:
    _ON_GFX942 = False
    _ON_GFX950 = False


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    # [n_blocks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    # The packed KV layout makes per-block strides large
    # enough that block_id * stride can exceed 32-bit range.
    block_id = (slot_id // block_size).to(tl.int64)
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_value_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
        )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)


def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    kv_cache_value = kv_cache[:, : block_size * head_dim].view(fp8_dtype)
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)
    head_tile_size = head_tile_size // kv_cache.element_size()
    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        layout,
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    valid_tid = tid < NUM_TOKENS
    batch_id = tl.load(token_to_seq_ptr + tid, mask=valid_tid, other=-1)
    valid_batch = (batch_id >= 0) & (batch_id < NUM_BATCHES)
    safe_batch_id = tl.where(valid_batch, batch_id, 0)
    batch_start = tl.load(cu_seqlen_ptr + safe_batch_id, mask=valid_batch, other=0)
    batch_end = tl.load(cu_seqlen_ptr + safe_batch_id + 1, mask=valid_batch, other=0)
    batch_offset = tid - batch_start
    valid_token = valid_tid & valid_batch & (tid >= batch_start) & (tid < batch_end)
    if not valid_token:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    valid_block_table = (
        valid_token
        & (block_table_id >= 0)
        & (block_table_id < BLOCK_TABLE_WIDTH)
        & (block_offset >= 0)
        & (block_offset < block_size)
    )
    safe_block_table_id = tl.where(valid_block_table, block_table_id, 0)
    block_table_offset = safe_batch_id * block_table_stride + safe_block_table_id
    block_id = tl.load(
        block_table_ptr + block_table_offset, mask=valid_block_table, other=-1
    )
    valid_block = valid_block_table & (block_id >= 0) & (block_id < NUM_BLOCKS)
    # The packed KV layout makes per-block strides large
    # enough that block_id * stride can exceed 32-bit range.
    safe_block_id = tl.where(valid_block, block_id, 0).to(tl.int64)
    safe_block_offset = tl.where(valid_block, block_offset, 0)
    tiled_block_offset = safe_block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            safe_block_id * kv_cache_stride
            + (safe_block_offset // BLOCK_TILE_SIZE) * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = (
            safe_block_id * kv_cache_stride + safe_block_offset * HEAD_DIM
        )
    src_scale_offset = safe_block_id * kv_cache_scale_stride + safe_block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr, mask=valid_block, other=0.0)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val, mask=valid_block)


def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)
    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
    # [glm5next-port-17] Host-side index audit of this launch, gated on GLM5_DSA_AUDIT
    # (off by default; the audit is also behind GLM5_DSA_BREAK per the brief, but a
    # separate opt-in keeps it from adding a per-launch DtoH sync to every run).
    # Computes the largest index this kernel will READ from the kv-cache and WRITE
    # to the fp8 outputs, from the block table / seqlens / block size / grid, and
    # prints the first launch where one exceeds the corresponding tensor size.
    import os as _os
    if _os.environ.get("GLM5_DSA_AUDIT"):
        try:
            _seq = cu_seqlen.detach().to(torch.int64).cpu()
            _bt = block_table.detach().to(torch.int64).cpu()
            _B = int(_seq.shape[0]) - 1
            _stride = int(k_cache_value.stride(0))
            _blk_w = int(_bt.shape[1]) if _bt.dim() > 1 else 0
            _max_bi = -1
            _nvalid = 0
            for _b in range(_B):
                _s = int(_seq[_b]); _e = int(_seq[_b + 1])
                if _e <= _s:
                    continue
                _n = _e - _s
                _bti = torch.arange(_n) // int(block_size)
                _bti = _bti[_bti < _blk_w]
                if _bti.numel() == 0:
                    continue
                _ids = _bt[_b][_bti]
                _ids = _ids[_ids >= 0]
                if _ids.numel():
                    _nvalid += int(_ids.numel())
                    _max_bi = max(_max_bi, int(_ids.max()))
            _total = int(k_cache_value.numel())
            _max_read = (_max_bi + 1) * _stride if _max_bi >= 0 else 0
            _max_write = int(num_tokens) * int(head_dim)
            _fp8_total = int(k_fp8.numel())
            _exceed = (_max_bi >= int(num_blocks)) or (_max_read > _total) or (_max_write > _fp8_total)
            print(
                "[DSA-GATHER-AUDIT] B=%d num_tokens=%d num_blocks=%d block_size=%d head_dim=%d "
                "max_block_id=%d (valid tokens=%d) | kv_cache total=%d max_read_idx=%d | "
                "fp8 total=%d max_write=%d | EXCEEDS_BOUNDS=%s"
                % (_B, num_tokens, int(num_blocks), int(block_size), int(head_dim),
                   _max_bi, _nvalid, _total, _max_read, _fp8_total, _max_write,
                   "YES" if _exceed else "no"),
                flush=True)
        except Exception as _e:  # noqa: BLE001
            print("[DSA-GATHER-AUDIT] audit error: %r" % (_e,), flush=True)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        layout,
        head_dim,
        block_tile_size,
        head_tile_size,
        num_tokens,
        cu_seqlen.shape[0] - 1,
        block_table.shape[1],
        num_blocks,
    )


# [glm5-noaiter R1d] Split a uint8 indexer page into logical (values, scales).
def _unshuffle_indexer_page(cache, block_size, dim, fp8_dtype):
    """Undo the indexer page byte layout for the torch logits readers.

    The indexer cache page is a SEPARATED layout -- ``block_size * dim`` fp8 K
    bytes followed by ``block_size`` fp32 scales -- and whenever
    ``block_size > 1`` the K bytes are tiled
    ``[block_size/16, dim/16, 16, 16]`` ("SHUFFLE"; both
    ``indexer_k_quant_and_cache_triton`` and the kpool compress kernels write
    it that way, and ``cp_gather_indexer_k_quant_cache_triton`` unshuffles on
    read). The old interleaved ``[..., :dim]`` / ``[..., dim:]`` slicing in
    ``fp8_paged_mqa_logits_torch`` only lined up with the writer for
    ``block_size == 1`` (DeepSeek indexer caches); GLM pool pages (32 slots)
    were read back garbled.

    cache: ``[N, block_size * (dim + 4)]`` uint8 ->
        (fp8 values ``[N, block_size, dim]``, fp32 scales ``[N, block_size]``).
    """
    values = cache[..., : block_size * dim].view(fp8_dtype)
    scales = cache[..., block_size * dim :].view(torch.float32)
    if block_size > 1:
        values = (
            values.view(-1, block_size // 16, dim // 16, 16, 16)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, block_size, dim)
        )
    return values, scales


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    if next_n == 1:
        block_size = kv_cache.shape[1]
        logits = torch.full(
            [batch_size, max_model_len],
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        kv_cache_flat = kv_cache.view(-1, block_size * (dim + 4))
        for i in range(batch_size):
            q_i = q[i, 0].to(torch.float32)
            q_scale = weights[i]
            seq_len = int(context_lens[i].item())
            assert seq_len <= max_model_len
            num_pages = cdiv(seq_len, block_size)
            padded_seq_len = num_pages * block_size
            pages = block_tables[i, :num_pages]
            cache = kv_cache_flat[pages]
            # [glm5-noaiter R1d] separated + (bs>1) shuffled page layout.
            cache_value, cache_scale = _unshuffle_indexer_page(
                cache, block_size, dim, fp8_dtype
            )
            cache_value = cache_value.to(torch.float32).view(padded_seq_len, dim)
            cache_scale = cache_scale.reshape(padded_seq_len)
            score = F.linear(cache_value, q_i)
            score = F.relu(score)
            score *= q_scale[None, :]
            score = score.sum(dim=1)
            score *= cache_scale
            logits[i, :seq_len] = score[:seq_len]
        return logits

    q = q.float()
    num_block, block_size = kv_cache.shape[0], kv_cache.shape[1]
    # [glm5-noaiter R1d] undo the separated + (bs>1) shuffled page layout and
    # dequantize per token (the old interleaved slicing only lined up with
    # the writer for block_size == 1). Shape is kept 4D for the code below.
    kv_values, kv_scales = _unshuffle_indexer_page(
        kv_cache.reshape(num_block, block_size * (dim + 4)),
        block_size,
        dim,
        fp8_dtype,
    )
    kv_cache = (kv_values.to(torch.float32) * kv_scales[:, :, None]).unsqueeze(2)
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = context_lens[i]
        if context_len.ndim == 0:
            context_len_i = int(context_len.item())
            q_offsets = torch.arange(
                context_len_i - next_n, context_len_i, device=q.device
            )
            context_limit = torch.full(
                (next_n,), context_len_i, dtype=torch.int32, device=q.device
            )
        else:
            context_limit = context_len.to(device=q.device, dtype=torch.int32)
            q_offsets = context_limit - 1
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        max_context_len = int(context_limit.max().item())
        for block_rk in range(cdiv(max_context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_limit[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


# [glm5-noaiter R2b] Cudagraph-safe Triton replacement for
# fp8_paged_mqa_logits_torch (rung R2b of research/dsa-backend-ladder.md).
#
# Why: the torch reference resolves per-request context lengths with
# ``int(context_lens[i].item())`` (+ a data-dependent page loop for next_n>1)
# -- host syncs that are ILLEGAL under cudagraph capture
# (boot8: "operation not permitted when stream is capturing"). In FULL
# cudagraph mode (uniform decode batches) the breakable-CG wrapper runs the
# enclosing indexer custom op INLINE (breakable_cudagraph.py:
# ``if mode == CUDAGraphMode.FULL: return fn(...)``), so the decode logits
# path must be genuinely capture-safe, not just break-decorated.
#
# Semantics are byte-exact the torch reference (validated on CPU by
# tests/test_r2b_kernels.py against fp8_paged_mqa_logits_torch, and on GPU by
# preflight-gpu.sh P7): per row r=(b,n),
#   logits[r, p] = sum_h relu(q[b,n,h,:] . K[p,:]) * weights[r,h] * scale[p]
# for p < limit (else -inf), with K/scale read from the indexer page layout:
# separated fp8 K bytes + fp32 scales, K tiled [bs/16, D/16, 16, 16]
# ("SHUFFLE") when block_size > 1 (see _unshuffle_indexer_page for the same
# addressing in torch form).
#
# Structure mirrors the aiter deepgemm_fp8_paged_mqa_logits contract that the
# official vllm-openai-rocm:glm53-flash image dispatches to here (that module
# is aiter-only and absent on this image, so this is the from-scratch
# capture-safe equivalent for the same call contract).
@triton.jit
def _fp8_paged_mqa_logits_triton_kernel(
    q_ptr,  # [B, N, H, D] uint8 view of the fp8 query
    kv_ptr,  # uint8 paged cache base [num_blocks, block_size, 1, D+4]
    w_ptr,  # [B*N, H] fp32 head weights
    ctx_ptr,  # [B, C] context (pool-granular seq) lengths, C in {1, N}
    bt_ptr,  # [B, max_pages] int32 block tables
    out_ptr,  # [B*N, max_model_len] fp32 logits (-inf prefilled)
    q_stride_b,
    q_stride_n,
    q_stride_h,
    w_stride_r,
    ctx_stride_b,
    bt_stride_b,
    out_stride_r,
    max_model_len,
    next_n,
    ctx_cols,
    num_heads,
    kv_page_stride,  # bytes per cache page (uint8 elements)
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_P: tl.constexpr,  # kv positions per program (== BLOCK_SIZE if > 1)
    BLOCK_SIZE: tl.constexpr,  # cache page slots; 1 => NORMAL layout
    IS_FNUZ: tl.constexpr,
):
    row = tl.program_id(0)
    pid_p = tl.program_id(1)
    b = row // next_n
    n = row % next_n
    if ctx_cols == next_n:
        c = n
    else:
        c = 0
    limit = tl.load(ctx_ptr + b * ctx_stride_b + c)
    p0 = pid_p * BLOCK_P
    if p0 >= limit:
        return
    slots = p0 + tl.arange(0, BLOCK_P)
    valid = slots < limit
    dims = tl.arange(0, HEAD_DIM)
    if BLOCK_SIZE == 1:
        # NORMAL layout (DeepSeek indexer caches): one page per slot.
        pages = tl.load(bt_ptr + b * bt_stride_b + slots, mask=valid, other=0).to(
            tl.int64
        )
        k_addr = pages[:, None] * kv_page_stride + dims[None, :]
        s_addr = pages * kv_page_stride + HEAD_DIM
    else:
        # SHUFFLE layout: one page per program (BLOCK_P == BLOCK_SIZE), K
        # tiled [bs/16, D/16, 16, 16]; scales after the K bytes, fp32.
        page = tl.load(bt_ptr + b * bt_stride_b + pid_p).to(tl.int64)
        s_in = tl.arange(0, BLOCK_P)
        off = (
            (s_in // 16)[:, None] * (16 * HEAD_DIM)
            + (dims // 16)[None, :] * 256
            + (s_in % 16)[:, None] * 16
            + (dims % 16)[None, :]
        )
        k_addr = page * kv_page_stride + off
        s_addr = page * kv_page_stride + BLOCK_SIZE * HEAD_DIM + s_in * 4
    x_uint8 = tl.load(kv_ptr + k_addr, mask=valid[:, None], other=0)
    if IS_FNUZ:
        k = x_uint8.to(tl.float8e4b8, bitcast=True).to(tl.float32)
    else:
        k = x_uint8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    k_scale = tl.load(
        (kv_ptr + s_addr).to(tl.pointer_type(tl.float32)), mask=valid, other=0.0
    )
    hq = tl.arange(0, BLOCK_H)
    h_mask = hq < num_heads
    q_uint8 = tl.load(
        q_ptr
        + b * q_stride_b
        + n * q_stride_n
        + hq[:, None] * q_stride_h
        + dims[None, :],
        mask=h_mask[:, None],
        other=0,
    )
    if IS_FNUZ:
        qf = q_uint8.to(tl.float8e4b8, bitcast=True).to(tl.float32)
    else:
        qf = q_uint8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # ieee: fp8 products are exact in fp32 and the reference sums in fp32 --
    # no TF32 (ranking fidelity for the downstream top-k selection).
    scores = tl.dot(qf, tl.trans(k), input_precision="ieee")
    scores = tl.maximum(scores, 0.0)  # relu
    w = tl.load(w_ptr + row * w_stride_r + hq, mask=h_mask, other=0.0)
    s = tl.sum(scores * w[:, None], axis=0)
    s = s * k_scale
    s = tl.where(valid, s, float("-inf"))
    tl.store(out_ptr + row * out_stride_r + slots, s, mask=slots < max_model_len)


def _triton_paged_logits_supported(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    context_lens: torch.Tensor,
    next_n: int,
) -> bool:
    """Contracts the Triton kernel covers: head_dim % 16 (fp8 tile geometry +
    tl.dot), block_size 1 (NORMAL) or % 16 (SHUFFLE), and context lens in the
    (B,), (B,1) or (B,next_n) shapes vLLM actually passes. Anything else keeps
    the torch reference semantics (eager/piecewise only)."""
    block_size = kv_cache.shape[1]
    if q.shape[-1] % 16 != 0:
        return False
    if block_size > 1 and block_size % 16 != 0:
        return False
    if context_lens.dim() == 2 and context_lens.shape[1] not in (1, next_n):
        return False
    if context_lens.dim() > 2:
        return False
    return True


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Cudagraph-safe drop-in for fp8_paged_mqa_logits_torch (same contract:
    q [B,N,H,D] fp8, kv_cache [nb,bs,1,D+4] uint8, weights [B*N,H] fp32,
    context_lens, block_tables -> logits [B*N, max_model_len] fp32)."""
    batch_size, next_n, num_heads, head_dim = q.shape
    block_size = kv_cache.shape[1]
    rows = batch_size * next_n
    ctx = context_lens
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(1)
    ctx = ctx.contiguous()
    ctx_cols = ctx.shape[1]
    # -inf prefill covers every position the kernel early-exits past; under
    # graph replay the buffer address is fixed, so all live positions are
    # (re)written by the kernel and the rest stay -inf every replay.
    logits = torch.full(
        (rows, max_model_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    if block_size == 1:
        block_p = 64
        grid_p = triton.cdiv(block_tables.shape[1], block_p)
    else:
        block_p = block_size
        grid_p = block_tables.shape[1]
    _fp8_paged_mqa_logits_triton_kernel[(rows, grid_p)](
        q.view(torch.uint8),
        kv_cache,
        weights,
        ctx,
        block_tables,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        ctx.stride(0),
        block_tables.stride(0),
        logits.stride(0),
        max_model_len,
        next_n,
        ctx_cols,
        num_heads,
        kv_cache.stride(0),
        BLOCK_H=max(16, triton.next_power_of_2(num_heads)),
        HEAD_DIM=head_dim,
        BLOCK_P=block_p,
        BLOCK_SIZE=block_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        num_warps=8,
    )
    return logits


@functools.lru_cache
def paged_mqa_logits_module():
    paged_mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.pa_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.attention.pa_mqa_logits"

    if paged_mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(paged_mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_paged_mqa_logits_module = None
    # if rocm_aiter_ops.is_enabled():
    batch_size, next_n = q_fp8.shape[:2]
    block_size = kv_cache_fp8.shape[1]

    if rocm_aiter_ops.is_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()

    if aiter_paged_mqa_logits_module is not None:
        if _ON_GFX942 or _ON_GFX950:
            deepgemm_fp8_paged_mqa_logits = (
                aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits
            )
            batch_size, next_n, heads, _ = q_fp8.shape
            (out_logits,) = current_workspace_manager().get_simultaneous(
                ((batch_size * next_n, max_model_len), torch.float32),
            )
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                out_logits,
                context_lens,
                block_tables,
                max_model_len,
                ChunkK=256,
                Preshuffle=block_size > 1,
                KVBlockSize=block_size,
                WavePerEU=2,
            )
            out_logits.nan_to_num_(float("-inf"))
            return out_logits
        deepgemm_fp8_paged_mqa_logits_stage1 = (
            aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits_stage1
        )
        batch_size, next_n, heads, _ = q_fp8.shape
        (out_qk,) = current_workspace_manager().get_simultaneous(
            ((heads, batch_size * next_n, max_model_len), torch.float32),
        )
        out_qk.fill_(float("-inf"))
        deepgemm_fp8_paged_mqa_logits_stage1(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
            ChunkQ=heads,
        )
        return out_qk.sum(dim=0)
    else:
        # [glm5-noaiter R2b] default: the cudagraph-safe Triton paged logits
        # (no aiter on this image, so the official image's aiter-module path
        # above never engages). GLM5_DSA_TORCH=1 restores the host-loop torch
        # reference (eager/piecewise A/B only -- capture-illegal by design);
        # off-contract shapes keep the torch semantics as well.
        from vllm.models.glm5next.noaiter import dsa_torch

        if not dsa_torch() and _triton_paged_logits_supported(
            q_fp8, kv_cache_fp8, context_lens, next_n
        ):
            return fp8_paged_mqa_logits_triton(
                q_fp8,
                kv_cache_fp8,
                weights,
                context_lens,
                block_tables,
                max_model_len,
            )
        return fp8_paged_mqa_logits_torch(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )


# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    device = q.device

    mask_lo = (
        torch.arange(0, seq_len_kv, device=device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    # ``score`` is [H, M, N]; ``scale`` is the per-KV-token scale, which
    # vLLM callers hand us as ``[N, 1]`` (a ``[N, 4]`` uint8 buffer cast
    # to fp32). PyTorch right-aligns dimensions for broadcasting, so a
    # naked ``score * scale`` would align ``scale``'s leading dim with
    # ``score``'s M dim and raise a shape mismatch. Flatten to ``[N]`` so
    # broadcasting lines up with the last dim of ``score``.
    #
    # [glm5-fix 2026-08-28] head-sliced accumulation: the monolithic
    # einsum materializes ``score`` [H, M, N] fp32 = H x the caller's
    # logits-budgeted [M, N]. The indexer prefill planner chunks rows to
    # M*N*4 <= VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (backends/mla/indexer.py:91),
    # which sizes the FUSED kernels' output — but this torch fallback's
    # intermediate is H x larger, so at H=64 a 128 MB-planned chunk peaked
    # at ~8 GiB transient and OOMed long-chunk prefills (boot39, p5000,
    # 1024 MiB alloc failure). Slice the head dim so the intermediate
    # stays under the same budget; fp32 block accumulation reassociates
    # the 64-term head sum (top-k ranking unaffected at ~1e-7 rel).
    score_scale = scale.reshape(-1)
    num_heads = q.shape[1]
    rows_bytes = q.shape[0] * seq_len_kv * 4
    cap_bytes = max(1, envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024)
    head_block = max(1, min(num_heads, cap_bytes // max(rows_bytes, 1)))
    logits = torch.zeros(
        (q.shape[0], seq_len_kv), dtype=torch.float32, device=device
    )
    for h0 in range(0, num_heads, head_block):
        h1 = min(num_heads, h0 + head_block)
        score = torch.einsum("mhd,nd->hmn", q[:, h0:h1], k).float()
        score *= score_scale
        logits += (
            score.relu_() * weights[:, h0:h1].transpose(0, 1).unsqueeze(-1)
        ).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@functools.lru_cache
def mqa_logits_module():
    mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.fp8_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.attention.fp8_mqa_logits"

    if mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    # TODO(ganyi): Temporarily workaround, will remove the module check and reference
    # path after aiter merge this kernel into main
    from vllm._aiter_ops import rocm_aiter_ops

    k_fp8, scale = kv

    # Temporarily route gfx942 to the vendored ROCm/aiter#3257 workaround.
    # Remove this branch once vLLM bumps AITER to a version that includes
    # ROCm/aiter#3257.
    if _ON_GFX942 and rocm_aiter_ops.is_enabled():
        from vllm.v1.attention.ops.triton_fp8_mqa_logits import (
            fp8_mqa_logits_gfx942,
        )

        return fp8_mqa_logits_gfx942(
            q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke
        )

    aiter_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_mqa_logits_module = mqa_logits_module()

    if aiter_mqa_logits_module is not None:
        fp8_mqa_logits = aiter_mqa_logits_module.fp8_mqa_logits
        return fp8_mqa_logits(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    else:
        if os.environ.get("GLM5_MQA_TRITON", "0") == "1":  # [claude-mqa-01]
            from vllm.v1.attention.ops.glm5_mqa_logits_rdna3 import fp8_mqa_logits_rdna3
            return fp8_mqa_logits_rdna3(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


# [glm5-noaiter R1c] torch top-k replacement for the _C.top_k_per_row_* ops.
# The function below is ported VERBATIM from the DSV4-on-R9700 patch file
#   host:/path/to/patches/rocm_aiter_mla_sparse.py:608
# where its output was validated bit-exact against the CUDA top_k_per_row op.
# Probe nuance (verified 2026-08-27): bare `import vllm._C` leaves
# torch.ops._C.top_k_per_row_* UNREGISTERED, but `import vllm.platforms.rocm`
# loads vllm._C_stable_libtorch (rocm.py:50) which DOES register them -- so in
# the real serving process the CUDA op path engages via this probe, and the
# torch fallback covers any process where the stable lib never loaded
# (minimal launchers, foreign images, future build regressions).
@functools.lru_cache(maxsize=1)
def _has_cuda_topk_per_row() -> bool:
    """True iff the _C top_k_per_row ops are registered in this process."""
    return hasattr(torch.ops._C, "top_k_per_row_prefill") and hasattr(
        torch.ops._C, "top_k_per_row_decode"
    )


def _topk_indices_torch(
    logits: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.topk(logits, k=k, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1, dtype=torch.int32),
        indices,
    )
    if row_starts is not None:
        # Match the CUDA top_k_per_row_prefill contract: indices are local to
        # each row's valid [row_start, row_end) range, not columns in the
        # concatenated chunk logits matrix.
        starts = row_starts.to(dtype=torch.int32).view(-1, 1)
        indices = torch.where(indices < 0, indices, indices - starts)
    if k == topk_tokens:
        return indices
    padded = torch.full(
        (logits.shape[0], topk_tokens),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    padded[:, :k] = indices
    return padded


def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


@eager_break_during_capture
def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    from vllm.utils.torch_utils import _resolve_layer_name

    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Profiling early-exit: reserve memory to account for runtime
        # allocations. Must be in the real impl, not the fake impl —
        # torch.compile calls the fake impl under FakeTensor mode where
        # workspace manager operations on the locked real workspace
        # would corrupt PyTorch's dispatch state.
        workspace_manager = current_workspace_manager()

        # Prefill k_fp8 and k_scale buffers, used by
        # rocm_aiter_sparse_attn_indexer's prefill path
        workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )

        # Decode logits buffer, used by rocm_fp8_paged_mqa_logits.
        # batch_size * next_n <= hidden_states.shape[0] == max_num_batched_tokens
        # [glm5-noaiter R1b] Compact profiler workspace on RDNA4 (gfx1201).
        # Ported from the DSV4-on-R9700 patch
        #   host:/path/to/patches_aiter/
        #     patch_sparse_attn_workspace_gfx1201.py
        # whose header reads: "Upstream computes the decode logits workspace
        # with an extra NUM_Q_HEADS multiplier for any arch that is NOT
        # gfx942/gfx950 ... OOMs the engine at startup. The gfx942/950 branch
        # uses the compact 2D layout which is NUM_Q_HEADS smaller and is
        # functionally equivalent on RDNA4."
        # For GLM flags the fat formula reserves
        #   32 heads x 8192 tokens x 65536 len x fp32 ~= 64 GiB per rank
        # -> instant OOM on 32 GiB R9700; the compact form is ~2 GiB.
        # The DSV4 patch keyed on on_rdna4(); we key on the ACTUAL runtime
        # decider instead: rocm_fp8_paged_mqa_logits only requests the fat
        # (heads, ...) stage-1 workspace when the aiter module path is live
        # (see :420-465). When aiter is unusable the torch fallback allocates
        # its own logits outside the workspace, so the compact reservation is
        # both sufficient and arch-agnostic. aiter present on gfx942/950 keeps
        # the stock branch via the first two disjuncts.
        from vllm._aiter_ops import rocm_aiter_ops as _rocm_aiter_ops_ws

        if _ON_GFX942 or _ON_GFX950 or not _rocm_aiter_ops_ws.is_enabled():
            workspace_manager.get_simultaneous(
                ((hidden_states.shape[0], max_model_len), torch.float32),
            )
        else:
            workspace_manager.get_simultaneous(
                (
                    (q_fp8.shape[1], hidden_states.shape[0], max_model_len),
                    torch.float32,
                ),
            )
        # Transient logits tensor peak memory, produced by
        # rocm_fp8_mqa_logits (prefill) and rocm_fp8_paged_mqa_logits
        # (decode). Prefill logits are bounded by
        # VLLM_SPARSE_INDEXER_MAX_LOGITS_MB via chunking in
        # split_indexer_prefill_chunks; decode logits are smaller.
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
        )
    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None
    assert scale_fmt is not None
    slot_mapping = layer_attn_metadata.slot_mapping
    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens

    # Initialize the complete row width, including alignment padding. The
    # attention backend may inspect columns beyond topk_tokens when adapting a
    # kpool-shaped buffer to AITER's fixed-width input contract (official
    # vllm-openai-rocm:glm53-flash, same function).
    topk_indices_buffer[: hidden_states.shape[0]] = -1

    # during speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    elif not skip_k_cache_insert:
        raise ValueError("k must be provided when skip_k_cache_insert is False")

    if not skip_k_cache_insert:
        indexer_k_quant_and_cache_triton(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None

        workspace_manager = current_workspace_manager()
        k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )
        for chunk in prefill_metadata.chunks:
            k_fp8 = k_fp8_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            cp_gather_indexer_k_quant_cache_triton(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
                token_to_seq=chunk.token_to_seq,
            )
            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            num_rows = logits.shape[0]

            # [glm5-noaiter R1c] CUDA op when registered (real serve loads the
            # stable lib), DSV4-proven torch fallback otherwise.
            if _has_cuda_topk_per_row():
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                topk_indices.copy_(
                    _topk_indices_torch(
                        logits, topk_tokens, row_starts=chunk.cu_seqlen_ks
                    )
                )

    if has_decode:
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        num_rows = logits.shape[0]

        # [glm5-noaiter R1c] CUDA op when registered, DSV4-proven torch
        # fallback otherwise. Decode logits columns ARE request-local
        # positions, so the fallback needs no row_starts offset.
        if _has_cuda_topk_per_row():
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                decode_metadata.seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
        else:
            topk_indices.copy_(_topk_indices_torch(logits, topk_tokens))

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, next_n, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


@triton.jit
def _inverse_rope_gptj_kernel(
    o_ptr,  # [T, H, D] input
    out_ptr,  # [T, H, D] bf16 output
    pos_ptr,  # [T] positions
    cos_sin_ptr,  # [P, rope_dim] fp32 (cos[:half] | sin[half:])
    s_t,
    s_h,  # input row strides (last dim contiguous)
    os_t,
    os_h,  # output row strides
    cs_stride,  # cos_sin_cache row stride
    NOPE: tl.constexpr,  # non-rope head dims (passed through)
    HALF: tl.constexpr,  # rope_dim // 2
    BLOCK_NOPE: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """Fused inverse GPT-J RoPE on the trailing rope_dim of each (token, head).

    Mirrors ``DeepseekV4ScalingRotaryEmbedding.forward_native(inverse=True)``
    for the GPT-J (non-neox) layout, writing bf16 directly. Replaces the
    clone + index_select + repeat_interleave + neg + stack + cat + cast chain
    (~10 small kernels) with a single launch.
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    in_base = t * s_t + h * s_h
    out_base = t * os_t + h * os_h

    # NoPE lanes pass through unchanged (only cast to bf16).
    n = tl.arange(0, BLOCK_NOPE)
    nmask = n < NOPE
    vals = tl.load(o_ptr + in_base + n, mask=nmask)
    tl.store(out_ptr + out_base + n, vals.to(tl.bfloat16), mask=nmask)

    # RoPE lanes: out_even = a*cos + b*sin, out_odd = b*cos - a*sin
    # (a = even lane, b = odd lane; sin negated for the inverse rotation).
    pos = tl.load(pos_ptr + t).to(tl.int64)
    k = tl.arange(0, BLOCK_HALF)
    kmask = k < HALF
    a = tl.load(o_ptr + in_base + NOPE + 2 * k, mask=kmask).to(tl.float32)
    b = tl.load(o_ptr + in_base + NOPE + 2 * k + 1, mask=kmask).to(tl.float32)
    cos = tl.load(cos_sin_ptr + pos * cs_stride + k, mask=kmask)
    sin = tl.load(cos_sin_ptr + pos * cs_stride + HALF + k, mask=kmask)
    out_even = a * cos + b * sin
    out_odd = b * cos - a * sin
    tl.store(out_ptr + out_base + NOPE + 2 * k, out_even.to(tl.bfloat16), mask=kmask)
    tl.store(out_ptr + out_base + NOPE + 2 * k + 1, out_odd.to(tl.bfloat16), mask=kmask)


def _fused_inverse_rope_gptj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """bf16 inverse GPT-J RoPE via a single fused Triton kernel."""
    assert o.dim() == 3 and o.stride(-1) == 1, (
        "_fused_inverse_rope_gptj expects a [T, H, D] input with a contiguous last dim"
    )
    assert rope_head_dim > 0 and rope_head_dim % 2 == 0, (
        f"_fused_inverse_rope_gptj expects an even rope_head_dim, got {rope_head_dim}"
    )
    assert cos_sin_cache.shape[-1] == rope_head_dim, (
        "_fused_inverse_rope_gptj expects cos_sin_cache laid out as "
        f"[P, {rope_head_dim}] = cos | sin, got {tuple(cos_sin_cache.shape)}"
    )
    num_tokens, num_heads, head_dim = o.shape
    out = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=o.device
    )
    if num_tokens == 0:
        return out
    _inverse_rope_gptj_kernel[(num_tokens, num_heads)](
        o,
        out,
        positions,
        cos_sin_cache,
        o.stride(0),
        o.stride(1),
        out.stride(0),
        out.stride(1),
        cos_sin_cache.stride(0),
        NOPE=head_dim - rope_head_dim,
        HALF=rope_head_dim // 2,
        BLOCK_NOPE=triton.next_power_of_2(head_dim - rope_head_dim),
        BLOCK_HALF=triton.next_power_of_2(rope_head_dim // 2),
    )
    return out


def _get_cached_wo_a_bf16(
    wo_a: torch.nn.Module,
    n_local_groups: int,
    o_lora_rank: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Dequantize wo_a to bf16 once and cache it on the module.

    wo_a weights are static, so the fp8 -> fp32 -> (* block scale) -> bf16
    dequant only needs to run once. Recomputing it every decode step shows up
    in the profile as the largest copy/mul kernels (``direct_copy float`` ~55us
    and ``MulFunctor float`` ~31us per two layers). SGLang / ATOM keep wo_a in
    bf16 and feed a plain bf16 GEMM; this mirrors that.
    """
    cached = getattr(wo_a, "_dsv4_wo_a_bf16", None)
    if cached is not None:
        return cached
    if hasattr(wo_a, "weight_scale_inv"):
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.float32
        )
        wo_a_scale = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(
                n_local_groups, -1, wo_a.weight_scale_inv.shape[-1]
            ),
            o_lora_rank,
            hidden_dim,
        )
        cached = (wo_a_weight * wo_a_scale).to(torch.bfloat16)
    else:
        cached = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.bfloat16
        )
    wo_a._dsv4_wo_a_bf16 = cached
    return cached


def rocm_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """Inverse-RoPE + WO_A bmm path used on ROCm.

    Fuses the inverse GPT-J RoPE into one Triton kernel and caches the bf16
    wo_a weight so the per-step dequant disappears.
    """
    o_ref = _fused_inverse_rope_gptj(
        o, positions, rotary_emb.cos_sin_cache, rope_head_dim
    )
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)

    wo_a_weight = _get_cached_wo_a_bf16(
        wo_a, n_local_groups, o_lora_rank, o_ref.shape[-1]
    )

    return torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)


_DSV4_SPARSE_NOPE_DIM = 448
_DSV4_SPARSE_ROPE_DIM = 64


def _validate_dsv4_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    assert head_dim == nope_head_dim + rope_head_dim, (
        f"{op_name} expected head_dim={nope_head_dim + rope_head_dim}, got {head_dim}"
    )
    # [glm5-noaiter R2b] GLM NoPE dims (512, 0) whitelisted verbatim from the
    # official vllm-openai-rocm:glm53-flash image (same file) -- the rung-R2b
    # "relax _validate_dsv4_sparse_dims 512+64 -> 512+0" of
    # research/dsa-backend-ladder.md. The kernels below are dim-agnostic
    # (BLOCK_D = next_power_of_2(head_dim)); the whitelist only guards
    # untested geometry.
    supported_dims = {
        (_DSV4_SPARSE_NOPE_DIM, _DSV4_SPARSE_ROPE_DIM),
        (512, 0),
    }
    assert (nope_head_dim, rope_head_dim) in supported_dims, (
        f"{op_name} does not support NoPE/RoPE dims ({nope_head_dim}, {rope_head_dim})"
    )


@triton.jit
def _pack_dense_prefix_to_ragged_kernel(
    indices_ptr,
    lengths_ptr,
    indptr_ptr,
    out_ptr,
    indices_stride0,
    num_rows_limit,
    row_width,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_len = tl.load(lengths_ptr + row_idx)
    if block_idx * BLOCK_SIZE >= row_len:
        return

    mask = offsets < row_len
    safe_offsets = tl.where(offsets < row_width, offsets, 0)
    vals = tl.load(
        indices_ptr + row_idx * indices_stride0 + safe_offsets,
        mask=mask & (offsets < row_width),
        other=-1,
    ).to(tl.int32)
    if num_rows_limit >= 0:
        vals = tl.where((vals >= 0) & (vals < num_rows_limit), vals, -1)

    out_start = tl.load(indptr_ptr + row_idx)
    tl.store(out_ptr + out_start + offsets, vals, mask=mask)


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert lengths.numel() == indices.shape[0], (
        f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"
    )

    max_width = indices.shape[1] if indices.ndim == 2 else 0
    lengths = lengths.clamp(min=0, max=max_width).contiguous()

    indptr = torch.zeros(indices.shape[0] + 1, dtype=torch.int32, device=indices.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])

    if indices.numel() == 0:
        flat = torch.empty(0, dtype=torch.int32, device=indices.device)
    else:
        flat = torch.empty(
            indices.shape[0] * max_width,
            dtype=torch.int32,
            device=indices.device,
        )
        if flat.numel() > 0:
            block_size = 128
            _pack_dense_prefix_to_ragged_kernel[
                (indices.shape[0], triton.cdiv(max_width, block_size))
            ](
                indices,
                lengths,
                indptr,
                flat,
                indices.stride(0),
                int(num_rows),
                max_width,
                BLOCK_SIZE=block_size,
            )

    return flat, indptr


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


# [glm5next-port-22] GLM5_CAPTURE_AUDIT: diagnostic audit of the ragged prefill
# sparse-attention kernel inputs, gated on an env switch (off by default, read at
# call time). No behaviour change when off. See part3.md.
_glm5_capture_audit_count = 0
_glm5_capture_audit_enabled = False


def _gc_aud_enabled():
    global _glm5_capture_audit_enabled
    if not _glm5_capture_audit_enabled:
        import os as _gca_os
        _glm5_capture_audit_enabled = _gca_os.environ.get("GLM5_CAPTURE_AUDIT") == "1"
    return _glm5_capture_audit_enabled


def _gc_aud_info(t):
    try:
        if not (isinstance(t, torch.Tensor) and t.is_cuda and t.numel() > 0):
            return "non-cuda-or-empty"
        n = int(t.numel())
        s = int(t.element_size())
        return f"shape={list(t.shape)} dtype={t.dtype} nbytes={n*s} ptr=[{t.data_ptr()},{t.data_ptr()+n*s})"
    except Exception as e:
        return f"(info failed: {e})"


def _gc_aud_stats(t):
    try:
        if not (isinstance(t, torch.Tensor) and t.is_cuda and t.numel() > 0):
            return "non-cuda-or-empty"
        c = t.detach().cpu().to(torch.int64)
        return f"shape={list(t.shape)} min={int(c.min())} max={int(c.max())} numel={int(t.numel())}"
    except Exception as e:
        return f"(stats failed: {e})"


_glm5_ragged_indptr_clamp_cache = False


def _glm5_ragged_indptr_clamp_enabled():
    global _glm5_ragged_indptr_clamp_cache
    if not _glm5_ragged_indptr_clamp_cache:
        import os as _gri_os
        _glm5_ragged_indptr_clamp_cache = _gri_os.environ.get(
            "GLM5_RAGGED_INDPTR_CLAMP"
        ) == "1"
    return _glm5_ragged_indptr_clamp_cache


def _glm5_capture_audit(
    q, kv, indices, indptr, attn_sink, out, k_scale,
    num_queries, num_heads, kv_rows, scale,
):
    """Print the inputs to _sparse_attn_prefill_ragged_kernel (first 20 calls).
    Skips any host read while a CUDA graph is being captured. No behaviour change."""
    global _glm5_capture_audit_count
    try:
        if not _gc_aud_enabled():
            return
        if _glm5_capture_audit_count >= 20:
            return
        if torch.cuda.is_current_stream_capturing():
            _glm5_capture_audit_count += 1
            print("[GLM5_CAPTURE_AUDIT] ragged prefill kernel call #%d: SKIP (graph capturing)" % _glm5_capture_audit_count, flush=True)
            return
        _glm5_capture_audit_count += 1
        kv2 = kv.squeeze(1)
        print(
            "[GLM5_CAPTURE_AUDIT] ragged prefill kernel call #%d: "
            "num_queries=%d num_heads=%d head_dim=%d kv_rows(num_kv)=%d kv_rows_full=%d scale=%r"
            % (_glm5_capture_audit_count, num_queries, num_heads, q.shape[2],
               kv_rows, int(kv2.shape[0]), scale),
            flush=True)
        print("  q        : " + _gc_aud_info(q), flush=True)
        print("  kv       : " + _gc_aud_info(kv2) + " (kv2 = kv.squeeze(1))", flush=True)
        print("  indices  : " + _gc_aud_info(indices), flush=True)
        print("  indices  : " + _gc_aud_stats(indices), flush=True)
        print("  indptr   : " + _gc_aud_info(indptr), flush=True)
        print("  indptr   : " + _gc_aud_stats(indptr), flush=True)
        i = indptr.detach().cpu().to(torch.int64)
        e = i[-1].item()
        l = int(indices.numel())
        print(
            "[GLM5_CAPTURE_AUDIT] call #%d: indptr[0]=%d indptr[last]=%d  |indices|=%d  OOB=%s (indptr[-1] > |indices|)"
            % (_glm5_capture_audit_count, int(i[0]), int(e), l, "YES" if int(e) > l else "no"),
            flush=True)
    except Exception as e:
        print("[GLM5_CAPTURE_AUDIT] FAILED: %r" % (e,), flush=True)


@triton.jit
def _sparse_attn_prefill_ragged_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    k_scale_ptr,  # fp32 scalar pointer (fp8 KV dequant); unused for bf16/fp16
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    # [glm5-noaiter P3] fp8 KV dequant. When the cache is fp8 the query was
    # quantized to the same fp8 dtype by forward_mqa. k_scale_ptr is a
    # host-side 1.0 scalar for bf16/fp16 caches (wrapper default), so loading
    # it is always safe. The per-layer q scale is constant and cancels in
    # softmax, so it is NOT applied; only the K/V latent is scaled by k_scale.
    # Casting q to bf16 unconditionally (fp8 -> bf16 dequant, bf16 -> bf16
    # no-op) avoids the Triton SSA violation from redefining q inside a
    # conditional block (fp8e4nv -> bf16 reassign was a CompilationError).
    ks = tl.load(k_scale_ptr)
    q = q.to(tl.bfloat16)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    slot = tl.load(
        kv_indices_ptr + kv_start + k_offsets, mask=k_offsets < kv_len, other=-1
    )
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        kv = tl.load(
            kv_ptr
            + safe_slot[:, None] * kv_stride_n
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        # [glm5-noaiter P3] dequant the latent KV: e4m3 -> fp32 -> * k_scale
        # -> bf16. bf16/fp16 caches are a no-op (ks == 1.0). Unconditional to
        # avoid a Triton SSA reassign-inside-block CompilationError.
        kv = (kv.to(tl.float32) * ks).to(tl.bfloat16)

        next_k_pos = k_start + BLOCK_K + k_offsets
        slot = tl.load(
            kv_indices_ptr + kv_start + next_k_pos, mask=next_k_pos < kv_len, other=-1
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _sparse_attn_decode_ragged_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    # SWA K-cache (main): C++ encoder writes FNUZ on gfx942, OCP on gfx950.
    # Compressed K-cache (extra): Triton encoder writes OCP everywhere.
    IS_FNUZ_MAIN: tl.constexpr,
    IS_FNUZ_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    for k_start in tl.range(0, main_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_len
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ_MAIN:
            x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start

        for k_start in tl.range(0, extra_len, BLOCK_K):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_len
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            if IS_FNUZ_EXTRA:
                x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
            else:
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out_nope = tl.where(
            l_final[:, None] > 0.0,
            (acc_nope * alpha[:, None]) / denom[:, None],
            0.0,
        )
        out_rope = tl.where(
            l_final[:, None] > 0.0,
            (acc_rope * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out_nope = tl.where(l_i[:, None] > 0.0, acc_nope / denom[:, None], 0.0)
        out_rope = tl.where(l_i[:, None] > 0.0, acc_rope / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + nope_offsets[None, :],
        out_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets[None, :],
        out_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_partial_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    q_stride0,
    q_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    # `main_cache` is the SWA K-cache (written by the C++ encoder, FNUZ on
    # gfx942 / OCP on gfx950). `extra_cache` is the compressed K-cache
    # (Triton encoder, OCP on every platform). Reading both with the same
    # `IS_FNUZ` would decode one of them with the wrong FNUZ/OCP scale ratio.
    IS_FNUZ_MAIN: tl.constexpr,
    IS_FNUZ_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_h = tl.program_id(2)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    # Each split processes a contiguous slice of this query's main (SWA) and
    # extra (topk) segments. Slices are handled independently so a block never
    # straddles the main/extra boundary.
    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    main_chunk = (main_len + NUM_SPLITS - 1) // NUM_SPLITS
    main_lo = split_id * main_chunk
    main_hi = tl.minimum(main_lo + main_chunk, main_len)

    for k_start in tl.range(main_lo, main_hi, BLOCK_K, num_stages=NUM_STAGES):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_hi
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ_MAIN:
            x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
        extra_chunk = (extra_len + NUM_SPLITS - 1) // NUM_SPLITS
        extra_lo = split_id * extra_chunk
        extra_hi = tl.minimum(extra_lo + extra_chunk, extra_len)

        for k_start in tl.range(extra_lo, extra_hi, BLOCK_K, num_stages=NUM_STAGES):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_hi
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            if IS_FNUZ_EXTRA:
                x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
            else:
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    # Store raw (un-normalized) partial state for this split. Softmax sink and
    # final normalization happen in the reduce kernel.
    pm_base = query_idx * pm_stride0 + split_id * pm_stride_s + head_offsets
    tl.store(part_m_ptr + pm_base, m_i, mask=head_mask)
    tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
    acc_base = (
        part_acc_ptr
        + query_idx * pa_stride0
        + split_id * pa_stride_s
        + head_offsets[:, None] * pa_stride_h
    )
    tl.store(
        acc_base + nope_offsets[None, :],
        acc_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        acc_base + NOPE_DIM + rope_offsets[None, :],
        acc_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_reduce_kernel(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    attn_sink_ptr,
    out_ptr,
    out_stride0,
    out_stride1,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    COMB_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    comb_offsets = tl.arange(0, COMB_DIM)
    # SPLITS_PAD is NUM_SPLITS rounded up to a power of two so the parallel
    # split-axis load is a legal arange for any split count; padding lanes are
    # masked off.
    split_offsets = tl.arange(0, SPLITS_PAD)
    split_mask = split_offsets < NUM_SPLITS

    neg_large = -3.4028234663852886e38

    # Phase 1: load every split's running max/sum at once and reduce the max
    # in parallel (tl.max over the split axis) instead of walking the splits
    # serially. This breaks the long online-softmax dependency chain that made
    # the reduce latency-bound.
    load_mask = split_mask[:, None] & head_mask[None, :]
    pm_split = (
        part_m_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :]
    )
    m_all = tl.load(pm_split, mask=load_mask, other=neg_large)  # [S, H]
    l_all = tl.load(
        part_l_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :],
        mask=load_mask,
        other=0.0,
    )

    m_comb = tl.max(m_all, axis=0)  # [H]
    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_comb, sink)
    else:
        m_final = m_comb

    w_all = tl.exp(m_all - m_final[None, :])  # [S, H]
    w_all = tl.where(load_mask, w_all, 0.0)
    l_final = tl.sum(w_all * l_all, axis=0)  # [H]
    if HAS_ATTN_SINK:
        l_final = l_final + tl.exp(sink - m_final)
    denom = tl.maximum(l_final, 1.0e-30)

    # Phase 2: weighted sum of the per-split accumulators. The combine weight
    # for each split only depends on the (already known) global max, so the
    # acc loads carry no cross-split dependency and the compiler can pipeline
    # them; only the cheap FMA into `acc` is loop-carried.
    acc = tl.zeros((BLOCK_H, COMB_DIM), dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(
            part_m_ptr + query_idx * pm_stride0 + s * pm_stride_s + head_offsets,
            mask=head_mask,
            other=neg_large,
        )
        w_s = tl.exp(m_s - m_final)
        acc_base = (
            part_acc_ptr
            + query_idx * pa_stride0
            + s * pa_stride_s
            + head_offsets[:, None] * pa_stride_h
        )
        acc_s = tl.load(
            acc_base + comb_offsets[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
        acc += w_s[:, None] * acc_s

    out = tl.where(l_final[:, None] > 0.0, acc / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + comb_offsets[None, :],
        out,
        mask=head_mask[:, None],
    )


def _rocm_sparse_attn_prefill_ragged_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    k_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert not q.is_cpu and not kv.is_cpu and not indices.is_cpu and not indptr.is_cpu

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    # [glm5next-port-22] GLM5_RAGGED_INDPTR_CLAMP (off by default): the kernel reads
    # kv_indices[kv_start:k_end] where k_end = indptr[query+1]. If indptr carries a
    # garbage value past the buffer (see part3.md), that read is out of bounds.
    # Clamping indptr to [0, |indices|] is a no-op on a correct ragged layout
    # (indptr[-1] == |indices|, monotone) and bounds every indices read.
    if _glm5_ragged_indptr_clamp_enabled():
        indptr = torch.clamp(indptr, min=0, max=int(indices.numel()))
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    # [glm5-noaiter P3] fp8 KV: pass the per-tensor dequant scale to the kernel
    # (only read when the cache dtype is fp8). A host-side scalar default keeps
    # the bf16/fp16 path untouched.
    # [glm5next-port-02] torch.tensor(1.0, device=...) performs a host->device
    # copy that is illegal during CUDA-graph capture (P7 7d). Allocate the scalar
    # directly on the device instead: no copy, capture-safe, same value.
    if k_scale is None:
        k_scale = torch.ones((), dtype=torch.float32, device=q.device)
    k_scale = k_scale.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_prefill_ragged_triton",
    )

    block_h = 16
    block_d = triton.next_power_of_2(head_dim)
    block_k = 16 if head_dim >= 256 else 32
    num_warps = 4
    out = torch.empty_like(q, dtype=torch.bfloat16)
    # [glm5next-port-22] GLM5_CAPTURE_AUDIT: audit the ragged prefill kernel inputs
    # immediately before the launch (host reads skipped while a graph is captured).
    _glm5_capture_audit(
        q, kv, indices, indptr, attn_sink, out, k_scale,
        num_queries, num_heads, kv.shape[0], float(scale),
    )
    # [glm5next-port-23] GLM5_META_WATCH (c): immediately before the ragged
    # prefill kernel launch, compare indptr[:num_queries+1] against the build
    # reference. Reports only the first difference per build. Off by default.
    meta_watch_check_c("before ragged kernel", indptr, num_queries, 0)
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        k_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


def _rocm_sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
        indices,
        topk_length
        if topk_length is not None
        else (indices >= 0).sum(dim=-1, dtype=torch.int32),
        num_rows=kv.shape[0],
    )
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=ragged_indices,
        indptr=ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
    )


@functools.lru_cache
def _decode_cu_count() -> int:
    try:
        return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        return 256  # For gfx950 arch, gated behind a fallback path for other archs.


def _decode_partial_iters(
    avg_main_len: float, avg_extra_len: float, splits: int, block_k: int
) -> int:
    """BLOCK_K iterations one partial workgroup walks for ``splits`` splits.

    Each split processes ``ceil(seg_len / splits)`` tokens of a segment, walked
    ``BLOCK_K`` at a time, and the main/extra segments are handled separately.
    """
    main_iters = (
        math.ceil(math.ceil(avg_main_len / splits) / block_k) if avg_main_len > 0 else 0
    )
    extra_iters = (
        math.ceil(math.ceil(avg_extra_len / splits) / block_k)
        if avg_extra_len > 0
        else 0
    )
    return main_iters + extra_iters


def _decode_num_splits(
    num_queries: int,
    heads_blocks: int,
    avg_main_len: float = 0.0,
    avg_extra_len: float = 0.0,
    block_k: int = 32,
) -> int:
    """Pick a flash-decode split count to keep the GPU busy across batch sizes.

    Decode launches only ``num_queries * heads_blocks`` workgroups otherwise,
    which severely under-fills the device for the low-concurrency regime that
    dominates latency. Splitting the KV sequence adds parallelism.

    We model the relative partial-kernel latency for a given split count ``s``
    as ``waves * (1/s + mu)`` where ``waves = ceil(base * s / CU)`` and ``mu``
    is a small per-wave overhead penalty:

      - ``waves / s`` captures the partial compute: each wave walks roughly
        ``total_tokens / s`` tokens and there are ``waves`` of them, so dividing
        by ``s`` makes more splits cheaper *until* they spill into extra waves.
      - ``mu * waves`` charges per-wave launch/tail overhead so we do not
        over-split into many mostly-idle waves (e.g. batch 224 on 256 CUs is
        best left at 1 split rather than 8 splits across 7 waves).

    The minimiser naturally prefers split counts that pack the device into full
    waves (``base * s`` near a multiple of ``CU``) and falls back to 1 split
    once the batch already fills the device. Ties favour the smaller split
    count (less reduce work).

    Finally we "snap down" the chosen split count to the smallest value that
    yields the same wave count *and* the same per-workgroup BLOCK_K iteration
    count. Because latency tracks iteration count (not raw token count), extra
    splits that do not lower the iteration count add only reduce/HBM overhead
    for no parallelism gain (e.g. batch 24: s8 and s10 both walk 4 extra iters
    in one wave, so s8 is strictly better). Snapping needs the average segment
    lengths, which the caller derives sync-free from the ragged index sizes.
    """
    base = max(1, num_queries * heads_blocks)
    # Target ~1 workgroup per CU: enough to fill the device while keeping the
    # reduce cost (which grows with split count) small. Tuned on gfx950.
    cu = max(1, _decode_cu_count())
    # Per-wave overhead penalty: higher values discourage split counts that
    # spill into extra GPU waves. Tuned on gfx950.
    mu = 0.04
    best_splits = 1
    best_cost = None
    # Search up to 16 splits; beyond that the reduce/HBM overhead dominates.
    for splits in range(1, 17):
        waves = (base * splits + cu - 1) // cu
        cost = waves * (1.0 / splits + mu)
        if best_cost is None or cost < best_cost - 1e-9:
            best_splits = splits
            best_cost = cost

    if best_splits > 1 and (avg_main_len > 0 or avg_extra_len > 0):
        target_waves = (base * best_splits + cu - 1) // cu
        target_iters = _decode_partial_iters(
            avg_main_len, avg_extra_len, best_splits, block_k
        )
        for splits in range(1, best_splits):
            waves = (base * splits + cu - 1) // cu
            iters = _decode_partial_iters(avg_main_len, avg_extra_len, splits, block_k)
            if waves == target_waves and iters == target_iters:
                best_splits = splits
                break
    return best_splits


def _rocm_sparse_attn_decode_ragged_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {main_cache.shape}"
    )
    assert main_indices.ndim == 1, (
        f"expected main_indices=[nnz], got {main_indices.shape}"
    )
    assert main_indptr.ndim == 1, f"expected main_indptr=[b+1], got {main_indptr.shape}"
    assert (
        not q.is_cpu
        and not main_cache.is_cpu
        and not main_indices.is_cpu
        and not main_indptr.is_cpu
    )

    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr = _as_int32_contiguous_1d(main_indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert main_indptr.numel() == num_queries + 1, (
        f"expected main_indptr shape [{num_queries + 1}], got {main_indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_decode_ragged_triton",
    )

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_indptr is not None
        assert extra_indices.ndim == 1, (
            f"expected extra_indices=[nnz], got {extra_indices.shape}"
        )
        assert extra_indptr.ndim == 1, (
            f"expected extra_indptr=[b+1], got {extra_indptr.shape}"
        )
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr = _as_int32_contiguous_1d(extra_indptr)
        assert extra_indptr.numel() == num_queries + 1, (
            f"expected extra_indptr shape [{num_queries + 1}], got {extra_indptr.shape}"
        )
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    block_h = 16
    out = torch.empty_like(q, dtype=torch.bfloat16)
    heads_blocks = triton.cdiv(num_heads, block_h)
    nope_block = triton.next_power_of_2(nope_head_dim)
    comb_dim = nope_head_dim + rope_head_dim
    is_fnuz = current_platform.is_fp8_fnuz()

    if not (_ON_GFX942 or _ON_GFX950):  # Fallback path for un-tuned architectures.
        block_k = 16 if head_dim >= 256 else 32
        _sparse_attn_decode_ragged_kernel[(num_queries, heads_blocks)](
            q,
            main_cache,
            main_indices,
            main_indptr,
            extra_cache,
            extra_indices,
            extra_indptr,
            attn_sink,
            out,
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            main_cache.stride(0),
            extra_cache.stride(0),
            main_cache.shape[0] * main_cache.shape[1],
            extra_cache.shape[0] * extra_cache.shape[1],
            main_cache.shape[1],
            extra_cache.shape[1],
            scale,
            num_heads,
            HAS_ATTN_SINK=has_attn_sink,
            HAS_EXTRA=has_extra,
            NOPE_DIM=nope_head_dim,
            NOPE_BLOCK=nope_block,
            ROPE_DIM=rope_head_dim,
            IS_FNUZ_MAIN=is_fnuz,
            IS_FNUZ_EXTRA=False,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            num_warps=8,
        )
        return out

    block_k = 32  # KV tokens walked per split-K iteration. Tuned on gfx950.
    # Average per-query segment lengths, read sync-free from the ragged index
    # sizes, let the split heuristic avoid over-splitting
    # main_indices/extra_indices are flat [nnz] int32.
    inv_q = 1.0 / max(1, num_queries)
    avg_main_len = main_indices.numel() * inv_q
    avg_extra_len = (extra_indices.numel() * inv_q) if has_extra else 0.0
    num_splits = _decode_num_splits(
        num_queries, heads_blocks, avg_main_len, avg_extra_len, block_k
    )

    part_m = torch.empty(
        (num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device
    )
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty(
        (num_queries, num_splits, num_heads, comb_dim),
        dtype=torch.float32,
        device=q.device,
    )

    _sparse_attn_decode_partial_kernel[(num_queries, num_splits, heads_blocks)](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        part_m,
        part_l,
        part_acc,
        q.stride(0),
        q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        scale,
        num_heads,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=nope_block,
        ROPE_DIM=rope_head_dim,
        # main_cache = swa_k_cache (C++ encoder, FNUZ on gfx942 / OCP on gfx950).
        # extra_cache = compressed kv_cache (Triton encoder, OCP everywhere).
        # Reading both with a single IS_FNUZ would decode one of them with the
        # wrong FNUZ/OCP scale ratio (~1.87×).
        IS_FNUZ_MAIN=is_fnuz,
        IS_FNUZ_EXTRA=False,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        NUM_STAGES=1,
        num_warps=4,
    )

    _sparse_attn_decode_reduce_kernel[(num_queries, num_heads)](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        COMB_DIM=comb_dim,
        BLOCK_H=1,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    return out


def _rocm_sparse_attn_decode_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    main_lengths: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    main_ragged_indices: torch.Tensor | None = None,
    main_ragged_indptr: torch.Tensor | None = None,
    extra_ragged_indices: torch.Tensor | None = None,
    extra_ragged_indptr: torch.Tensor | None = None,
) -> torch.Tensor:
    if main_ragged_indices is None or main_ragged_indptr is None:
        main_ragged_indices, main_ragged_indptr = build_ragged_indices_from_dense(
            main_indices,
            main_lengths
            if main_lengths is not None
            else (main_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=main_cache.shape[0] * main_cache.shape[1],
        )

    if (
        (extra_ragged_indices is None or extra_ragged_indptr is None)
        and extra_cache is not None
        and extra_indices is not None
    ):
        extra_ragged_indices, extra_ragged_indptr = build_ragged_indices_from_dense(
            extra_indices,
            extra_lengths
            if extra_lengths is not None
            else (extra_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=extra_cache.shape[0] * extra_cache.shape[1],
        )

    return _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_ragged_indices,
        extra_indptr=extra_ragged_indptr,
    )


def rocm_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> None:
    assert kv.ndim == 3 and kv.shape[1] == 1, (
        f"ROCm Triton sparse prefill expects kv=[skv,1,d], got {kv.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_prefill",
    )
    if ragged_indices is not None and ragged_indptr is not None:
        output_chunk = _rocm_sparse_attn_prefill_ragged_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=ragged_indices,
            indptr=ragged_indptr,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            k_scale=k_scale,
        )
    else:
        assert indices is not None
        indices_2d = indices.reshape(indices.shape[0], -1)
        output_chunk = _rocm_sparse_attn_prefill_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=indices_2d,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            topk_length=topk_length,
        )
    output.copy_(output_chunk.to(output.dtype))


def rocm_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    assert swa_k_cache.dtype == torch.uint8, (
        "ROCm Triton sparse decode expects uint8 fp8_ds_mla SWA cache, "
        f"got {swa_k_cache.dtype}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_decode",
    )

    main_indices = swa_indices.reshape(swa_indices.shape[0], -1)

    extra_cache = None
    extra_indices = None
    if not swa_only:
        assert kv_cache is not None
        assert topk_indices is not None or (
            topk_ragged_indices is not None and topk_ragged_indptr is not None
        )
        assert kv_cache.dtype == torch.uint8, (
            "ROCm Triton sparse decode expects uint8 fp8_ds_mla extra cache, "
            f"got {kv_cache.dtype}"
        )
        extra_cache = kv_cache
        if topk_indices is not None:
            extra_indices = topk_indices.reshape(topk_indices.shape[0], -1)

    attn_out = _rocm_sparse_attn_decode_triton(
        q=q,
        main_cache=swa_k_cache,
        main_indices=main_indices,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        main_lengths=swa_lens,
        extra_lengths=topk_lens,
        main_ragged_indices=swa_ragged_indices,
        main_ragged_indptr=swa_ragged_indptr,
        extra_ragged_indices=topk_ragged_indices,
        extra_ragged_indptr=topk_ragged_indptr,
    )
    output.copy_(attn_out.to(output.dtype))
