# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import os

import torch

from vllm.triton_utils import triton

from .utils import tensor_cache


# ---------------------------------------------------------------------------
# port-16 Part 3 fix: GLM5_KDA_NOCACHE (OFF by default).
#
# In graph mode (FULL_DECODE_ONLY breakable cudagraph) the KDA decode runs as a
# captured graph (baked at capture time; Python does not run during replay) and
# the prefill runs EAGERLY. Both are batch 1 here (max-num-seqs 2, one request
# at a time). The overlay's sync-free prepare_chunk_indices (GPU-resident
# repeat_interleave + segmented cumsum, no host sync) allocates n_chunks from
# the PyTorch GPU buffer pool. During an eager prefill that pool hands out
# memory that the just-replayed decode graph is still writing, so n_chunks is
# overwritten in place: Part 2(a) observed the single element read 0 -> 1 ->
# negative within one call (cdiv cannot produce a negative) while cu_seqlens
# and a fresh recompute were correct. repeat_interleave(arange(B), n_chunks)
# then raises "repeats can not be negative" (index.py, the second request).
# Cloning n_chunks from the same aliased pool does not help (the allocation
# itself is aliased with the graph's write target).
#
# Fix: for the EAGER (non-capturing) path only, build the per-chunk index tensor
# on the HOST (the original upstream approach) from cu_seqlens.cpu(), so the
# intermediates use the CPU allocator and cannot be aliased with the graph. The
# DtoH sync is safe here because the eager prefill is not inside a captured
# graph (confirmed: capturing=False at the fault). The captured decode path is
# unchanged (it is baked at capture time; its Python never runs during replay).
# With GLM5_KDA_NOCACHE off (default) the original behaviour is used exactly.
# ---------------------------------------------------------------------------


def _kda_no_cache() -> bool:
    return os.environ.get("GLM5_KDA_NOCACHE") == "1"


def _cdiv_int(x, chunk_size):
    # Integer cdiv that works on both CPU and GPU tensors.
    return (x + chunk_size - 1) // chunk_size


def _chunk_indices_on_host(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Build the per-chunk (seq_id, offset) index tensor on the CPU, then move
    it to cu_seqlens' device. Only used on the eager (non-capturing) path."""
    cs = cu_seqlens.detach().cpu().to(torch.int64)  # DtoH sync (eager only)
    lens = cs[1:] - cs[:-1]
    n_chunks = _cdiv_int(lens, chunk_size)
    B = n_chunks.shape[0]
    seq_ids = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int64), n_chunks)
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(n_chunks, 0)[:-1]])
    offsets = torch.arange(seq_ids.shape[0], dtype=torch.int64) \
        - torch.repeat_interleave(starts, n_chunks)
    return torch.stack([seq_ids, offsets], 1).to(cu_seqlens.device).to(cu_seqlens.dtype)


def _chunk_indices_on_device(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    # GPU-resident, sync-free variant (the overlay's original code).
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks = _cdiv_int(lens, chunk_size).to(cu_seqlens.dtype)
    B = n_chunks.shape[0]
    dev = cu_seqlens.device
    seq_ids = torch.repeat_interleave(
        torch.arange(B, device=dev, dtype=cu_seqlens.dtype), n_chunks)
    ones = torch.ones_like(seq_ids)
    starts = torch.cat(
        [torch.zeros(1, device=dev, dtype=cu_seqlens.dtype),
         torch.cumsum(n_chunks, 0)[:-1]])
    offsets = torch.cumsum(ones, 0) - 1 - torch.repeat_interleave(starts, n_chunks)
    return torch.stack([seq_ids, offsets], 1).to(cu_seqlens.dtype)


def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    if _kda_no_cache() and not torch.cuda.is_current_stream_capturing():
        cs = cu_seqlens.detach().cpu()
        return (cs[1:] - cs[:-1]).to(cu_seqlens.device).to(cu_seqlens.dtype)
    return _prepare_lens_cached(cu_seqlens)


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if _kda_no_cache() and not torch.cuda.is_current_stream_capturing():
        return _chunk_indices_on_host(cu_seqlens, chunk_size)
    return _prepare_chunk_indices_cached(cu_seqlens, chunk_size)


@tensor_cache
def _prepare_lens_cached(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def _prepare_chunk_indices_cached(cu_seqlens: torch.Tensor,
                                  chunk_size: int) -> torch.Tensor:
    lens = prepare_lens(cu_seqlens)
    n_chunks = triton.cdiv(lens, chunk_size).to(cu_seqlens.dtype)
    B = n_chunks.shape[0]
    dev = cu_seqlens.device
    seq_ids = torch.repeat_interleave(
        torch.arange(B, device=dev, dtype=cu_seqlens.dtype), n_chunks)
    ones = torch.ones_like(seq_ids)
    starts = torch.cat(
        [torch.zeros(1, device=dev, dtype=cu_seqlens.dtype),
         torch.cumsum(n_chunks, 0)[:-1]])
    offsets = torch.cumsum(ones, 0) - 1 - torch.repeat_interleave(starts, n_chunks)
    return torch.stack([seq_ids, offsets], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
