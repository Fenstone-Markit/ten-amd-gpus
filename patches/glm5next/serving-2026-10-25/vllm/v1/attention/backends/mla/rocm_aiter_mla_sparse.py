# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import os
import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLAHelper,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.glm5_meta_watch import (
    meta_watch_build_b,
    meta_watch_check_e2,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
logger = init_logger(__name__)


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    cu_seqlens_ptr,  # int32 [num_tokens + 1]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load cumulative sequence lengths to get starting index of this request
    seq_start = tl.load(cu_seqlens_ptr + token_id)
    seq_end = tl.load(cu_seqlens_ptr + token_id + 1)

    if tile_id * BLOCK_N + seq_start >= seq_end:
        return

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    # # If token == -1 OR block_id OOB, output 0; else base * BLOCK_SIZE + offset
    out_val = tl.where(
        is_invalid_tok | (~valid_block), 0, base * BLOCK_SIZE + inblock_off
    )
    out_ptr_ij = out_ptr + seq_start + indice_id
    out_ptr_ij_mask = (seq_start + indice_id) < seq_end

    # store the results with mask
    tl.store(out_ptr_ij, out_val, mask=out_ptr_ij_mask)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    cu_seqlens: torch.Tensor,  # int32 [num_tokens + 1]
    paged_kv_indices: torch.Tensor,  # int32 [num_tokens * topk] out_buffer
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible byBLOCK_N ({BLOCK_N})"
    )
    # print("req_id: ", req_id, flush=True)
    num_tokens = req_id.shape[0]
    _, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        cu_seqlens,
        paged_kv_indices,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
    )
    return


# [glm5-noaiter R2b] Ported VERBATIM from the official
# vllm-openai-rocm:glm53-flash image (same file): the Triton ragged
# sparse-MLA dispatch gates. True exactly when the rope-free (NoPE),
# plain-decode/prefill shapes hold -- GLM-5.3 with --kv-cache-dtype auto and
# spec-decode off satisfies both.
#
# [glm5-noaiter P2/P3] the `not fp8` condition was REMOVED: the ragged Triton
# kernel now dequantizes fp8 KV on read (P3, `k_scale` -> `_sparse_attn_prefill_
# ragged_kernel`). Plain fp8 prefill/decode rows therefore go through the fast
# ragged path. Spec-verify batches (`num_decode_tokens == (spec+1)*num_decodes`,
# `max_query_len > 1`) miss these two gates BY SHAPE and are dispatched by the
# dedicated P4 verify gate below (`_use_rocm_sparse_spec_verify`), falling back
# to `_forward_mla_torch` (P1 sync-free decode) only when the verify shape is
# not uniform or GLM5_SPEC_VERIFY_TORCH=1 is set.
def _use_rocm_sparse_prefill(
    attn_metadata: "ROCMAiterMLASparseMetadata",
    kv_cache_dtype: str,
    head_size: int,
    kv_lora_rank: int,
) -> bool:
    """Use ragged Triton for rope-free MLA and plain decode rows."""
    return (
        head_size == kv_lora_rank
        and attn_metadata.num_prefills > 0
        and attn_metadata.num_decode_tokens == attn_metadata.num_decodes
    )


def _use_rocm_sparse_decode(
    attn_metadata: "ROCMAiterMLASparseMetadata",
    kv_cache_dtype: str,
    head_size: int,
    kv_lora_rank: int,
) -> bool:
    """Use ragged Triton for rope-free plain decode (single-token rows)."""
    return (
        head_size == kv_lora_rank
        and attn_metadata.num_decodes > 0
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decode_tokens == attn_metadata.num_decodes
        and attn_metadata.max_query_len == 1
    )


# [glm5-noaiter P4] MTP spec-verify dispatch (research/mtp-dsa-patch-plan.md
# step P4/B1). A pure verify batch is uniform: every request contributes
# exactly (spec+1) query rows. The builder's decode_threshold is spec+1
# (_init_reorder_batch_threshold(1, supports_spec_as_decode=True)), so
# split_decodes_and_prefills reports num_prefills == 0 for such a batch and
# num_decodes == num_reqs. Given num_prefills == 0 every row is <= spec+1,
# hence num_decode_tokens == (spec+1)*num_decodes FORCES per-request
# query_len == spec+1: the aggregate conditions below are exactly the uniform
# verify shape, nothing else can pass. The ragged kernel is per-token CSR
# (each query row attends only its own indptr[t]:indptr[t+1] slice; the
# indexer already emits per-token top-k under spec decode), so the SAME
# rocm_sparse_attn_prefill ragged call that serves prefill/plain-decode rows
# serves verify rows unchanged. spec_tokens comes from the impl's cached
# self._num_spec_tokens (0 without a speculative config -> this gate can
# never fire, keeping non-spec behavior bit-identical). Escape hatch:
# GLM5_SPEC_VERIFY_TORCH=1 keeps the pre-P4 torch fallback (checked by the
# caller in _forward_mla, not here, so the gate stays a pure shape predicate).
# [claude-mixspec-01] MTP verify rows (<= spec+1 per decode request) in the same step as another request's prompt chunk.
# Per-token CSR like the other three arms; see ~/fp8-rdna3-01/mixed-spec-01-edit.py. GLM5_SPEC_MIXED_TORCH=1 opts out.
def _use_rocm_sparse_mixed_spec(
    attn_metadata: "ROCMAiterMLASparseMetadata",
    head_size: int,
    kv_lora_rank: int,
    spec_tokens: int,
) -> bool:
    return (
        os.environ.get("GLM5_SPEC_MIXED_TORCH", "0") != "1"
        and head_size == kv_lora_rank
        and spec_tokens > 0
        and attn_metadata.num_prefills > 0
        and attn_metadata.num_decodes > 0
        and attn_metadata.num_decode_tokens <= (spec_tokens + 1) * attn_metadata.num_decodes
    )


def _use_rocm_sparse_spec_verify(
    attn_metadata: "ROCMAiterMLASparseMetadata",
    head_size: int,
    kv_lora_rank: int,
    spec_tokens: int,
) -> bool:
    """Use ragged Triton for a pure MTP verify batch ((spec+1) rows/request)."""
    return (
        head_size == kv_lora_rank
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decodes > 0
        and spec_tokens > 0
        and attn_metadata.max_query_len == spec_tokens + 1
        and attn_metadata.num_decode_tokens
        == (spec_tokens + 1) * attn_metadata.num_decodes
    )


# [glm5-noaiter kpool] Ported verbatim from the official
# vllm-openai-rocm:glm53-flash image (same file): the shared topk buffer is
# wider than AITER's fixed top-k contract (it also carries the kpool
# incomplete-pool tail + 128-column alignment padding), so the read side
# compresses it back to exactly topk_tokens, keeping the tail.
def fit_kpool_indices_to_aiter(
    token_indices: torch.Tensor, topk_tokens: int
) -> torch.Tensor:
    """Keep the kpool tail while fitting AITER's fixed top-k capacity."""
    if token_indices.shape[1] < topk_tokens:
        raise ValueError("token_indices width must be at least topk_tokens")
    if token_indices.shape[1] == topk_tokens:
        return token_indices

    history = token_indices[:, :topk_tokens]
    tail = token_indices[:, topk_tokens:]
    # The ROCm top-k path emits compact, left-aligned history in descending
    # score order. Keep its highest-ranked prefix when making room for the tail.
    valid_history = (history >= 0).sum(dim=1)
    valid_tail = (tail >= 0).sum(dim=1)
    keep_history = torch.minimum(valid_history, topk_tokens - valid_tail)

    columns = torch.arange(topk_tokens, device=token_indices.device).unsqueeze(0)
    tail_offsets = columns - keep_history.unsqueeze(1)
    tail_values = torch.gather(
        tail, 1, tail_offsets.clamp(min=0, max=tail.shape[1] - 1)
    )
    output = torch.where(columns < keep_history.unsqueeze(1), history, tail_values)
    valid_output = columns < (keep_history + valid_tail).unsqueeze(1)
    return torch.where(valid_output, output, -1)


@triton.jit
def generate_sparse_seqlen_kernel(
    seq_len_ptr,  # [num_seq]
    cu_query_lens_ptr,  # [num_seq]
    out_ptr,  # [num_query_tokens]
    topk_token: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    query_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    query_start = tl.load(cu_query_lens_ptr + seq_id)
    query_end = tl.load(cu_query_lens_ptr + seq_id + 1)
    if query_start + tl.program_id(1) * BLOCK_SIZE > query_end:
        return
    query_len = query_end - query_start
    query_mask = query_offset + query_start < query_end
    seq_len = tl.load(seq_len_ptr + seq_id)
    # Just return since the out_ptr is zero initialized.
    if seq_len == 0:
        return
    context_start_point = seq_len - query_len
    sparse_seqlen = context_start_point + query_offset
    sparse_seqlen_masked = tl.where(
        sparse_seqlen + 1 < topk_token, sparse_seqlen + 1, topk_token
    )
    tl.store(
        out_ptr + query_start + query_offset, sparse_seqlen_masked, mask=query_mask
    )


def generate_sparse_seqlen_triton(
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor,
    topk_token: int,
    num_tokens: int,
    max_query_len: int,
):
    num_seqs = query_lens.size(0)
    # zero initialize the tensor to make sure invalid positions will be zero
    out = torch.zeros([num_tokens], dtype=torch.int32, device=query_lens.device)
    block_size = 64
    num_block_per_row = triton.cdiv(max_query_len, block_size)
    grid = (
        num_seqs,
        num_block_per_row,
    )
    generate_sparse_seqlen_kernel[grid](
        seq_lens,
        cu_query_lens,
        out,
        topk_token,
        block_size,
    )
    return out


@triton.jit
def fetch_id_to_ragged_kernel(
    in_tensor_ptr,  # [num_seq, topk]
    cumsum_ptr,  # [num_seq + 1]
    out_tensor_ptr,  # [max_num_seq * topk]
    in_tensor_ptr_stride,
    TOPK: tl.constexpr,
    TOKEN_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    token_start = tl.load(cumsum_ptr + seq_id)
    token_end = tl.load(cumsum_ptr + seq_id + 1)
    token_num = token_end - token_start
    row_offset = block_id * BLOCK_SIZE
    if row_offset >= token_num:
        return
    in_tensor_offset = seq_id * in_tensor_ptr_stride + row_offset + offset
    in_tensor_mask = (row_offset + offset) < TOPK
    in_tensor_val = tl.load(in_tensor_ptr + in_tensor_offset, mask=in_tensor_mask)
    out_tensor_offset = token_start + row_offset + offset
    out_tensor_mask = (out_tensor_offset < token_end) & in_tensor_mask
    tl.store(out_tensor_ptr + out_tensor_offset, in_tensor_val, mask=out_tensor_mask)


def fetch_id_to_ragged_triton(
    in_tensor: torch.Tensor, cumsum: torch.Tensor, out_tensor: torch.Tensor, topk
):
    num_tokens = in_tensor.size(0)
    block_size = 64
    num_block_per_row = triton.cdiv(topk, block_size)
    grid = (
        num_tokens,
        num_block_per_row,
    )
    fetch_id_to_ragged_kernel[grid](
        in_tensor, cumsum, out_tensor, in_tensor.stride(0), topk, num_tokens, block_size
    )


class ROCMAiterMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, 64]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["ROCMAiterMLASparseMetadata"]:
        return ROCMAiterMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["ROCMAiterMLASparseMetadataBuilder"]:
        return ROCMAiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["ROCMAiterMLASparseImpl"]:
        return ROCMAiterMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True


@dataclass
class ROCMAiterMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    qo_indptr: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_indptr: torch.Tensor
    attn_out_dtype: torch.dtype

    block_size: int = 1
    topk_tokens: int = 2048

    # Fields read by the shared MLA forward. This impl has no dense-MHA prefill
    # path (supports_dense_mha_prefill=False), so it always runs the MQA path;
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    prefill_max_seq_len: int = 0
    prefill: object = None

    # Persistent MLA metadata (only populated when persistent mode is enabled,
    # i.e. when the aiter sparse decode kernel supports work-stealing splits).
    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None

    # [glm5-noaiter] Per-request total sequence lengths (incl. current query
    # tokens), device tensor [num_reqs]. Only consumed by the GLM5_DSA_FULLCTX
    # numerics-isolation branch of _forward_mla_torch; None elsewhere-safe.
    seq_lens: torch.Tensor | None = None


@dataclass
class ROCMAiterMLASparseMetadataBuilder(
    AttentionMetadataBuilder[ROCMAiterMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.model_dtype = vllm_config.model_config.dtype
        parallel_config = vllm_config.parallel_config
        self.device = device
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.vllm_config = vllm_config
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        # Bounds the KV-split heuristic (see `_sparse_decode_max_split`).
        self._num_compute_units = current_platform.num_compute_units()
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        self.req_id_per_token_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        self.qo_indptr = torch.arange(
            0, max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.paged_kv_last_page_len = torch.ones(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )

        # These two needs to be calculated in runtime,
        # but we still needs to prepare the buffer
        self.paged_kv_indices = torch.zeros(
            [max_num_batched_tokens * self.topk_tokens],
            dtype=torch.int32,
            device=device,
        )
        self.paged_kv_indptr = torch.zeros(
            [max_num_batched_tokens + 1], dtype=torch.int32, device=device
        )

        # ----- Persistent MLA metadata buffers -----
        # The aiter sparse decode kernel supports a "persistent" path that
        # uses precomputed work-splitting metadata for better load balancing
        # across CUs. Mirrors the approach used in rocm_aiter_mla.py.
        #
        # In the sparse case each query token is its own "batch" entry in the
        # qo_indptr (qo_indptr = [0, 1, 2, ..., num_tokens]) and max_qo_len=1.
        # We pad get_mla_metadata_info_v1's batch_size to max_num_batched_tokens
        # so the buffers are large enough for any decode shape we might see.
        #
        # [glm5-noaiter R2a] The stock file did an UNGUARDED
        # `from aiter import dtypes, get_mla_metadata_info_v1` here, which
        # raises ModuleNotFoundError on tcclaviger/vllm:latest (the aiter
        # package tree is physically absent) the moment ROCM_AITER_MLA_SPARSE
        # initializes -- and platforms/rocm.py makes it the ONLY sparse
        # backend. The persistent work-stealing metadata is consumed solely by
        # the aiter mla_decode_fwd kernel; the torch gather-attend fallback
        # (rung R2a, see _forward_mla) never reads it. Gate the whole block on
        # aiter availability and leave the buffers None.
        self._persistent_mla_available = bool(rocm_aiter_ops.is_enabled())
        # Aiter sparse MLA also requires num_heads >= 16 (will be padded by
        # AiterMLAHelper.get_mla_padded_q in forward).
        self._num_attention_heads = max(16, self.num_heads)
        if not self._persistent_mla_available:
            logger.info_once(
                "ROCMAiterMLASparse: aiter unavailable -- persistent MLA "
                "metadata disabled; decode/prefill use the torch "
                "gather-attend fallback (glm5 no-AITER R2a)."
            )
            self._mla_work_meta_data = None
            self._mla_work_indptr = None
            self._mla_work_info_set = None
            self._mla_reduce_indptr = None
            self._mla_reduce_final_map = None
            self._mla_reduce_partial_map = None
            self._prev_req_extent: int = 0
            self._prev_indices_extent: int = 0
            self._prev_metadata_key: tuple | None = None
            return

        from aiter import dtypes, get_mla_metadata_info_v1

        q_dtype = self.model_dtype
        kv_cache_dtype_str = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            kv_cache_dtype_str = "fp8"
        else:
            kv_cache_dtype_str = "bf16"
        kv_dtype = dtypes.d_dtypes.get(kv_cache_dtype_str, dtypes.bf16)

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max_num_batched_tokens,
            1,
            self._num_attention_heads,
            q_dtype,
            kv_dtype,
            is_sparse=True,
            fast_mode=True,
        )
        self._mla_work_meta_data = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        self._mla_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        self._mla_work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        )
        self._mla_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        self._mla_reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        self._mla_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_type,
            device=device,
        )

        self._prev_req_extent: int = 0
        self._prev_indices_extent: int = 0
        self._prev_metadata_key: tuple | None = None

    def _sparse_decode_max_split(self, max_seq_len: int) -> int:
        """Cap ``max_split_per_batch`` for the aiter sparse-MLA decode reduce.

        The reduce only covers the selected tokens per row (``<= topk_tokens``),
        so aiter's default (``-1`` => split across every CU) over-fragments it.
        Mirror ``triton_mla.py``: aim for a minimum work per split, round to a
        power of two, and cap by the CU count. Numerics are unchanged.
        """
        effective_len = min(max_seq_len, self.topk_tokens)
        min_work_per_split = 128
        ideal_splits = triton.next_power_of_2(
            max(1, effective_len // min_work_per_split)
        )
        return min(ideal_splits, self._num_compute_units)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ROCMAiterMLASparseMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        # [glm5next-port-23] Part 2 (e4): at build start, check seq_lens/qsl
        # (device buffers) against the host-staged expected values.
        meta_watch_check_e2("(e4) build start", common_attn_metadata.seq_lens, common_attn_metadata.query_start_loc)
        (num_decodes, num_prefills, num_decode_tokens, _) = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold or 1,
        )
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Only re-zero the shrink-tail. paged_kv_indptr is fully rewritten
        # by the cumsum below. paged_kv_indices entries past new_indices_extent
        # are never read (the attention kernel only touches the ranges
        # defined by paged_kv_indptr).
        new_req_extent = int(req_id_per_token.shape[0])
        new_indices_extent = num_tokens * self.topk_tokens
        if self._prev_req_extent > new_req_extent:
            self.req_id_per_token_buffer[new_req_extent : self._prev_req_extent].fill_(
                0
            )
        if self._prev_indices_extent > new_indices_extent:
            self.paged_kv_indices[new_indices_extent : self._prev_indices_extent].fill_(
                0
            )
        self._prev_req_extent = new_req_extent
        self._prev_indices_extent = new_indices_extent
        self.req_id_per_token_buffer[:new_req_extent].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        query_lens = (
            common_attn_metadata.query_start_loc[1:]
            - common_attn_metadata.query_start_loc[:-1]
        )
        seq_lens = common_attn_metadata.seq_lens
        # [glm5next-port-23] Part 2 (e5): before generate_sparse_seqlen_triton,
        # confirm seq_lens/qsl still sane (the only GPU writer in the prologue is
        # this kernel, which reads qsl as an index into its output buffer).
        meta_watch_check_e2("(e5) before sparse_seqlen kernel", seq_lens, common_attn_metadata.query_start_loc)
        sparse_seqlen = generate_sparse_seqlen_triton(
            query_lens,
            seq_lens,
            common_attn_metadata.query_start_loc,
            self.topk_tokens,
            num_tokens,
            common_attn_metadata.max_query_len,
        )
        # [glm5next-port-23] Part 2 (e6): after generate_sparse_seqlen_triton,
        # check whether seq_lens/qsl are now corrupted (if the kernel wrote OOB
        # into the region containing them). Also dump the sparse_seqlen output
        # buffer (a freshly zero-initialized tensor) to see if it is garbage.
        meta_watch_check_e2("(e6) after sparse_seqlen kernel", seq_lens, common_attn_metadata.query_start_loc)
        if os.environ.get("GLM5_META_WATCH") == "1":
            try:
                torch.cuda.synchronize()
                sse = sparse_seqlen[:num_tokens].detach().cpu().tolist()
                try:
                    sptr = hex(sparse_seqlen.data_ptr())
                except Exception:
                    sptr = "?"
                print("[GLM5_META_WATCH e] CHECKPOINT=(e6) after sparse_seqlen kernel "
                      "sparse_seqlen[:%d]=%s ptr=%s" % (num_tokens, sse, sptr), flush=True)
            except Exception:
                pass

        torch.cumsum(sparse_seqlen, dim=0, out=self.paged_kv_indptr[1 : num_tokens + 1])
        # [glm5next-port-23] Part 2 (e7): after the cumsum, re-check seq_lens/qsl.
        # If GARBAGE here, the cumsum is the writer (it writes paged_kv_indptr, but
        # if it writes OOB it could hit the seq_lens/qsl/sparse_seqlen region).
        meta_watch_check_e2("(e7) after cumsum", seq_lens, common_attn_metadata.query_start_loc)
        self.paged_kv_indptr[num_tokens + 1 :].fill_(self.paged_kv_indptr[num_tokens])
        # [glm5next-port-23] Part 2 (e8): after the fill, re-check seq_lens/qsl.
        meta_watch_check_e2("(e8) after fill", seq_lens, common_attn_metadata.query_start_loc)

        # [glm5next-port-23] GLM5_META_WATCH (b): checkpoint right after the cumsum
        # into paged_kv_indptr — record a CPU reference and dump the values the
        # build saw. No new device allocation. Off by default; no-op when off.
        meta_watch_build_b(
            seq_lens,
            common_attn_metadata.query_start_loc,
            sparse_seqlen,
            self.paged_kv_indptr,
            common_attn_metadata.num_reqs,
            num_tokens,
            self.topk_tokens,
            self.model_config.max_model_len,
        )

        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]
        qo_indptr = self.qo_indptr[: num_tokens + 1]
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_tokens]
        paged_kv_indptr = self.paged_kv_indptr[: num_tokens + 1]
        paged_kv_indices = self.paged_kv_indices[: num_tokens * self.topk_tokens]

        # ----- Compute persistent MLA metadata -----
        # The aiter sparse decode kernel uses qseqlen=1 (each query token is
        # treated as its own batch entry), so persistent metadata can always
        # be precomputed here. The kernel switches to the persistent
        # work-stealing path automatically when work_meta_data is non-None.
        # The output is a deterministic function of the per-request query and
        # context lengths (both clamped to topk_tokens, past which per-token KV
        # length saturates) and num_heads; fingerprint those CPU-side and skip
        # the launch when nothing changed.
        num_reqs = common_attn_metadata.num_reqs
        clamped_seq_lens = np.minimum(
            common_attn_metadata.seq_lens_cpu[:num_reqs].numpy(),
            self.topk_tokens,
        )
        clamped_context_lens = np.minimum(
            common_attn_metadata.seq_lens_cpu[:num_reqs].numpy() - seg_lengths,
            self.topk_tokens,
        )
        metadata_key = (
            num_tokens,
            int(common_attn_metadata.max_query_len),
            self._num_attention_heads,
            clamped_seq_lens.tobytes(),
            clamped_context_lens.tobytes(),
            seg_lengths.tobytes(),
        )
        # [glm5-noaiter R2a] get_mla_metadata_v1 is aiter-only; skipped entirely
        # when the persistent path is unavailable (buffers are None, the torch
        # gather-attend fallback reads paged_kv_indices/indptr directly).
        if self._persistent_mla_available and metadata_key != self._prev_metadata_key:
            from aiter import get_mla_metadata_v1

            max_split_per_batch = self._sparse_decode_max_split(
                int(common_attn_metadata.max_seq_len)
            )
            get_mla_metadata_v1(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                self._num_attention_heads,
                1,
                True,
                self._mla_work_meta_data,
                self._mla_work_info_set,
                self._mla_work_indptr,
                self._mla_reduce_indptr,
                self._mla_reduce_final_map,
                self._mla_reduce_partial_map,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=1,
                uni_seqlen_qo=1,
                fast_mode=True,
                max_split_per_batch=max_split_per_batch,
            )
            # The persistent metadata buffers are read by graph replay. Order
            # the async metadata write before the graph-captured decode kernel.
            torch.cuda.current_stream(self.device).synchronize()
            self._prev_metadata_key = metadata_key

        metadata = ROCMAiterMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            attn_out_dtype=self.model_dtype,
            topk_tokens=self.topk_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            qo_indptr=qo_indptr,
            paged_kv_last_page_len=paged_kv_last_page_len,
            paged_kv_indices=paged_kv_indices,
            paged_kv_indptr=paged_kv_indptr,
            work_meta_data=self._mla_work_meta_data,
            work_indptr=self._mla_work_indptr,
            work_info_set=self._mla_work_info_set,
            reduce_indptr=self._mla_reduce_indptr,
            reduce_final_map=self._mla_reduce_final_map,
            reduce_partial_map=self._mla_reduce_partial_map,
            seq_lens=common_attn_metadata.seq_lens,
        )
        return metadata


# Take from
# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_prefill.py#L72
def reference_mla_sparse_prefill(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float, d_v: int
) -> tuple[torch.Tensor, torch.Tensor]:
    import math

    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)

    skv = kv.shape[0]
    sq = q.shape[0]
    topk = indices.shape[-1]
    dqk = q.shape[-1]
    indices = indices[:, 0, :]  # [s_q, topk]
    invalid_indices_mask = (indices < 0) | (indices >= skv)
    indices[invalid_indices_mask] = 0
    qs = q  # [s_q, h_q, d_qk]
    kvs = kv[:, 0, :][indices].view(sq, topk, dqk)  # [s_q, topk, d_qk]

    attn_score = (qs @ kvs.transpose(1, 2)).float()  # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float("-inf"))
    attn_score *= sm_scale * math.log2(math.e)
    lse = log2sumexp2(attn_score, dim=-1)  # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))  # [s_q, h_q, topk]
    result = attn_score.to(q.dtype) @ kvs[:, :, :d_v]
    return (result, lse)


class ROCMAiterMLASparseImpl(MLAAttentionImpl[ROCMAiterMLASparseMetadata]):
    is_sparse = True
    supports_dense_mha_prefill = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        AiterMLAHelper.check_num_heads_validity(num_heads)

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = scale
        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )

        vllm_config = get_current_vllm_config()
        # [glm5-noaiter P4] cached spec width for the verify dispatch gate
        # (_use_rocm_sparse_spec_verify). 0 when speculative decoding is not
        # configured -> the verify gate can never fire and the non-spec
        # dispatch stays bit-identical to pre-P4.
        self._num_spec_tokens: int = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None
            else 0
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        q_concat_shape = (max_tokens, num_heads, head_size)
        (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
            (q_concat_shape, vllm_config.model_config.dtype),
        )

    def _forward_mla(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        attn_metadata: ROCMAiterMLASparseMetadata,
    ) -> torch.Tensor:
        # [glm5-noaiter R2a] Without aiter, rocm_aiter_ops.mla_decode_fwd is
        # unregistered (@if_aiter_supported gates register_ops_once) and its
        # impl would `from aiter.mla import mla_decode_fwd` -> ModuleNotFoundError.
        # Route to the torch gather-attend substitute (correctness-first).
        if not rocm_aiter_ops.is_enabled():
            from vllm.models.glm5next.noaiter import (
                dsa_fallback_disabled,
                dsa_fullctx,
                dsa_torch,
                spec_verify_torch,
            )

            if dsa_fallback_disabled():
                raise RuntimeError(
                    "ROCM_AITER_MLA_SPARSE requires the aiter package for its "
                    "mla_decode_fwd kernel, but aiter is unavailable in this "
                    "image and GLM5_DSA_DISABLE_FALLBACK=1 was set. Unset it to "
                    "use the torch gather-attend fallback (rung R2a), or set "
                    "GLM5_DSA_DENSE=1 for dense-MLA DSA layers (rung R3). See "
                    "glm5-serve-overlay/OVERLAY_NOTES.md 'no-AITER DSA port'."
                )
            # [glm5-noaiter R2b] DEFAULT: the official
            # vllm-openai-rocm:glm53-flash Triton ragged sparse-MLA
            # (_sparse_attn_prefill_ragged_kernel, serving both prefill and
            # plain decode through the ragged path) -- cudagraph-safe (no
            # host syncs; kv walk is a device-side tl.range over indptr), so
            # FULL-cudagraph uniform-decode capture works and
            # GLM5_ENFORCE_EAGER can be dropped. The dispatch gates are the
            # official image's, verbatim (module level, same file). Escape
            # hatches: GLM5_DSA_TORCH=1 (A/B) and GLM5_DSA_FULLCTX=1
            # (numerics-isolation debug rung lives in _forward_mla_torch)
            # force the torch path; gate-misses (fp8 KV, RoPE dims,
            # non-uniform spec shapes) keep the R2a torch semantics.
            # [glm5-noaiter P4] third arm: uniform MTP verify batches
            # ((spec+1) rows/request) take the same ragged Triton call --
            # the kernel is per-token CSR, so verify rows need no kernel or
            # call-site changes. GLM5_SPEC_VERIFY_TORCH=1 opts verify back
            # out to the torch fallback (escape hatch, verify-only).
            if (
                not dsa_torch()
                and not dsa_fullctx()
                and (
                    _use_rocm_sparse_prefill(
                        attn_metadata,
                        self.kv_cache_dtype,
                        q.shape[-1],
                        self.kv_lora_rank,
                    )
                    or _use_rocm_sparse_decode(
                        attn_metadata,
                        self.kv_cache_dtype,
                        q.shape[-1],
                        self.kv_lora_rank,
                    )
                    or (
                        not spec_verify_torch()
                        and _use_rocm_sparse_spec_verify(
                            attn_metadata,
                            q.shape[-1],
                            self.kv_lora_rank,
                            self._num_spec_tokens,
                        )
                    )
                    or _use_rocm_sparse_mixed_spec(  # [claude-mixspec-01]
                        attn_metadata,
                        q.shape[-1],
                        self.kv_lora_rank,
                        self._num_spec_tokens,
                    )
                )
            ):
                from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
                    rocm_sparse_attn_prefill,
                )

                num_tokens = q.shape[0]
                mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(
                    self.num_heads
                )
                output = torch.empty(
                    [num_tokens, mla_num_heads, self.kv_lora_rank],
                    dtype=attn_metadata.attn_out_dtype,
                    device=q.device,
                )
                rocm_sparse_attn_prefill(
                    q=q,
                    kv=kv_c_and_k_pe_cache.view(-1, 1, q.shape[-1]),
                    indices=None,
                    topk_length=None,
                    scale=self.scale,
                    head_dim=q.shape[-1],
                    nope_head_dim=self.kv_lora_rank,
                    rope_head_dim=q.shape[-1] - self.kv_lora_rank,
                    attn_sink=None,
                    output=output,
                    ragged_indices=attn_metadata.paged_kv_indices,
                    ragged_indptr=attn_metadata.paged_kv_indptr,
                    k_scale=layer._k_scale,
                )
                return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)
            return self._forward_mla_torch(
                q, kv_c_and_k_pe_cache, attn_metadata, layer._k_scale
            )

        num_tokens = q.shape[0]
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self.num_heads)
        output = torch.empty(
            [num_tokens, mla_num_heads, self.kv_lora_rank],
            dtype=attn_metadata.attn_out_dtype,
            device=q.device,
        )

        # Build kwargs and forward the persistent MLA metadata when it has
        # been computed. The aiter mla_decode_fwd switches to its
        # work-stealing persistent kernel path when work_meta_data is given.
        mla_kwargs: dict = dict(
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
        )
        if attn_metadata.work_meta_data is not None:
            mla_kwargs.update(
                work_meta_data=attn_metadata.work_meta_data,
                work_indptr=attn_metadata.work_indptr,
                work_info_set=attn_metadata.work_info_set,
                reduce_indptr=attn_metadata.reduce_indptr,
                reduce_final_map=attn_metadata.reduce_final_map,
                reduce_partial_map=attn_metadata.reduce_partial_map,
            )

        rocm_aiter_ops.mla_decode_fwd(
            q,
            kv_c_and_k_pe_cache,
            output,
            self.scale,
            attn_metadata.qo_indptr,
            1,
            attn_metadata.paged_kv_indptr,
            attn_metadata.paged_kv_indices,
            attn_metadata.paged_kv_last_page_len,
            **mla_kwargs,
        )

        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)

    def _forward_mla_torch(
        self,
        q: torch.Tensor,  # [sq, heads_padded, d_qk] (post get_mla_padded_q)
        kv_c_and_k_pe_cache: torch.Tensor,  # [num_blocks, block_size, d_qk]
        attn_metadata: ROCMAiterMLASparseMetadata,
        k_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[glm5-noaiter R2a] Torch gather-attend substitute for the aiter
        sparse-MLA kernel (research/dsa-backend-ladder.md rung R2a).

        Per query token, gather the top-k latent KV slots selected by the
        indexer (already resolved to GLOBAL cache slots into
        ``attn_metadata.paged_kv_indices`` by
        ``triton_convert_req_index_to_global_index`` above; the valid prefix
        per token is ``paged_kv_indptr[t]:paged_kv_indptr[t+1]``), compute MQA
        scores ``q @ kv^T * scale`` in fp32, softmax, and reduce ``P @ kv`` to
        ``[heads, kv_lora_rank]``. v_head_dim (256) is applied OUTSIDE this
        kernel by ``kv_b_proj`` (W_UV), exactly like the aiter path whose
        output is also ``[tokens, heads, kv_lora_rank]``.

        Dims-agnostic by construction: GLM NoPE (kv 512 + rope 0), DeepSeek
        (512 + 64), any topk. Correctness-first, speed-second: this is a
        Python loop over tokens (fine at decode batch sizes; slow on long
        prefill). Rung R2b (Triton kernel, relaxing _validate_dsv4_sparse_dims
        in v1/attention/ops/rocm_aiter_mla_sparse.py to 512+0) is the speed
        follow-up.

        [glm5-noaiter P0/P1] fp8 KV dequant + sync-free decode:
        * ``k_scale`` is ``layer._k_scale`` (the per-tensor fp8 KV scale; the
          write path ``concat_and_cache_mla`` stores ``kv_c / k_scale`` as
          e4m3). For fp8 caches the gathered latent is dequantized
          ``kv.to(float32) * k_scale`` (mirrors triton_decode_attention.py's
          ``(k.to(fp32) * ks)``). bf16/fp16 caches need no scale.
        * Pure-decode batches (``num_prefills == 0``) use a cudagraph-capturable
          sync-free path (no ``.tolist()``/``.item()``) instead of the host
          loop below, so MTP SPEC>=1 verify steps (``num_decode_tokens ==
          (spec+1)*num_decodes``) fall back here without breaking FULL
          cudagraph capture. Prefill/mixed batches keep the eager host loop.
        """
        num_tokens, q_heads, d_qk = q.shape
        if kv_c_and_k_pe_cache.dtype == torch.uint8:
            # uint8 == DeepSeek's packed fp8_ds_mla layout, which this backend
            # never selects (its fp8 KV is e4m3 viewed via current_platform.fp8_dtype()).
            raise RuntimeError(
                "glm5 no-AITER R2a torch gather-attend does not implement the "
                "uint8 fp8_ds_mla KV layout. Relaunch with --kv-cache-dtype "
                "auto (bf16 KV), which is the run-glm53.sh default."
            )
        assert d_qk == kv_c_and_k_pe_cache.shape[-1], (
            f"q head_dim {d_qk} != latent cache width "
            f"{kv_c_and_k_pe_cache.shape[-1]}"
        )
        block_size = attn_metadata.block_size
        # [num_blocks, block_size, d] -> [(num_blocks*block_size), d]
        kv_flat = kv_c_and_k_pe_cache.view(-1, d_qk)
        paged_kv_indices = attn_metadata.paged_kv_indices
        indptr = attn_metadata.paged_kv_indptr
        assert indptr.shape[0] >= num_tokens + 1

        # [glm5-noaiter P0] fp8 KV dequant (per-tensor scale). The write path
        # and forward_mqa view the cache as exactly current_platform.fp8_dtype().
        fp8_dtype = current_platform.fp8_dtype()
        kv_is_fp8 = kv_flat.dtype == fp8_dtype
        if kv_is_fp8 and k_scale is None:
            k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

        output = torch.empty(
            [num_tokens, q_heads, self.kv_lora_rank],
            dtype=attn_metadata.attn_out_dtype,
            device=q.device,
        )
        if num_tokens == 0:
            return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)

        # [glm5-noaiter] GLM5_DSA_FULLCTX=1 numerics-isolation rung: bypass the
        # indexer top-k selection entirely and attend each token to its FULL
        # causal context. Slot lists are derived from seq_lens + block_table +
        # query_start_loc (exact by construction: position p of token j in
        # request r is (seq_lens[r] - query_len[r]) + j; context = positions
        # 0..p through the request's block chain). Output cleans up -> the
        # indexer/kpool selection stack is the numerics bug; still gibberish
        # -> bug is elsewhere (KDA / mHC / MoE / RFI GEMM). Eager-only,
        # debug-rung cost (one slot list per token), NOT a serving mode.
        from vllm.models.glm5next.noaiter import dsa_fullctx

        fullctx = dsa_fullctx()
        if fullctx:
            if attn_metadata.seq_lens is None:
                raise RuntimeError(
                    "GLM5_DSA_FULLCTX=1 needs seq_lens in the sparse metadata"
                )
            num_reqs = attn_metadata.num_reqs
            qsl_cpu = attn_metadata.query_start_loc[: num_reqs + 1].tolist()
            seq_lens_cpu = attn_metadata.seq_lens[:num_reqs].tolist()
            req_id_cpu = attn_metadata.req_id_per_token[:num_tokens].tolist()
            block_table_cpu = attn_metadata.block_table.tolist()

            q_f = q.float()
            for t in range(num_tokens):
                r = req_id_cpu[t]
                q0, q1 = qsl_cpu[r], qsl_cpu[r + 1]
                p = (seq_lens_cpu[r] - (q1 - q0)) + (t - q0)
                chain = block_table_cpu[r]
                nfull, off = divmod(p, block_size)
                slots = [
                    b * block_size + i
                    for b in chain[:nfull]
                    for i in range(block_size)
                ]
                slots.extend(chain[nfull] * block_size + i for i in range(off + 1))
                idx = torch.tensor(slots, dtype=torch.long, device=q.device)
                kv = kv_flat.index_select(0, idx)
                if kv_is_fp8:
                    kv = kv.to(torch.float32) * k_scale
                else:
                    kv = kv.float()
                scores = torch.einsum("hd,nd->hn", q_f[t], kv) * self.scale
                probs = torch.softmax(scores, dim=-1)
                output[t] = (probs @ kv[:, : self.kv_lora_rank]).to(output.dtype)
            return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)

        # [glm5-noaiter P1] Sync-free, cudagraph-capturable decode fallback.
        # Pure-decode batches (incl. MTP SPEC>=1 verify steps) run here with
        # fixed-shape GPU ops -- the eager host loop below is capture-illegal.
        if attn_metadata.num_prefills == 0:
            self._forward_mla_torch_decode_syncfree(
                q, kv_flat, paged_kv_indices, indptr, output, k_scale, kv_is_fp8
            )
            return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)

        # Eager per-token path (prefill / mixed batches). Host-syncs by design,
        # so it never runs under FULL cudagraph capture (prefill/mixed batches
        # run eagerly or as piecewise breaks, never in a captured graph).
        indptr_cpu = indptr[: num_tokens + 1].tolist()
        q_f = q.float()
        for t in range(num_tokens):
            s = indptr_cpu[t]
            e = indptr_cpu[t + 1]
            if e <= s:
                output[t].zero_()
                continue
            # Global slots produced by the convert kernel; invalid entries
            # are never inside the [s, e) valid prefix.
            idx = paged_kv_indices[s:e].to(torch.long)
            kv = kv_flat.index_select(0, idx)  # [n, d_qk]
            if kv_is_fp8:
                kv = kv.to(torch.float32) * k_scale
            else:
                kv = kv.float()
            # MQA: all heads share the same gathered latent KV.
            scores = torch.einsum("hd,nd->hn", q_f[t], kv) * self.scale
            probs = torch.softmax(scores, dim=-1)
            # Value = latent itself truncated to kv_lora_rank (NoPE: full width).
            output[t] = (probs @ kv[:, : self.kv_lora_rank]).to(output.dtype)
        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)

    def _forward_mla_torch_decode_syncfree(
        self,
        q: torch.Tensor,
        kv_flat: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        indptr: torch.Tensor,
        output: torch.Tensor,
        k_scale: torch.Tensor | None,
        kv_is_fp8: bool,
    ) -> None:
        """[glm5-noaiter P1] Sync-free, cudagraph-capturable decode fallback.

        Replaces the per-token ``.tolist()`` host loop with fixed-shape GPU
        operations so the torch gather-attend can run inside a FULL cudagraph
        capture. All intermediate tensors have shapes fixed by
        ``(num_tokens, heads, d_qk, num_tokens * topk_tokens)`` -- no
        data-dependent shapes, no ``.tolist()``/``.item()``:

          * ``torch.searchsorted(indptr, pos + 1, right=True) - 1`` maps each
            flat CSR position to its owning token (device-side, fixed shape);
          * a validity mask ``pos < indptr[num_tokens]`` selects the packed
            prefix (the indexer packs valid slots contiguously);
          * ``scatter_reduce`` (amax / sum) performs the segmented softmax and
            the ``sum_n prob[n] * kv[n]`` reduction (expanded index -- a naked
            broadcast index collapses the trailing dims, verified in tests).

        Memory is bounded by head-blocking (HEAD_BLOCK). Decode batches are
        small, so this is cheap vs. the eager host loop; prefill never calls it.
        """
        num_tokens, q_heads, d_qk = q.shape
        dev = q.device
        f_len = paged_kv_indices.numel()  # == num_tokens * topk_tokens (fixed)
        indptr64 = indptr.to(torch.int64)

        # Flat CSR position -> owning token id + validity mask (fixed shapes).
        pos = torch.arange(f_len, device=dev, dtype=torch.int64)
        token_of_pos = torch.searchsorted(indptr64, pos, right=True) - 1
        valid = pos < indptr64[num_tokens]
        safe_token = token_of_pos.clamp(min=0, max=num_tokens - 1)

        # Gather + dequant the latent KV for every flat position (fixed shape).
        # Invalid positions clamp to slot 0 and are masked out below.
        safe_idx = paged_kv_indices.clamp(min=0, max=kv_flat.shape[0] - 1)
        kv = kv_flat.index_select(0, safe_idx.to(torch.long))  # [f_len, d_qk]
        if kv_is_fp8:
            kv = kv.to(torch.float32) * k_scale
        else:
            kv = kv.float()
        kv_v = kv[:, : self.kv_lora_rank]  # [f_len, kv_lora_rank]

        q_f = q.float()  # [num_tokens, heads, d_qk]
        acc = torch.zeros(
            (num_tokens, q_heads, self.kv_lora_rank), device=dev, dtype=torch.float32
        )
        sum_exp = torch.zeros((num_tokens, q_heads), device=dev, dtype=torch.float32)

        HEAD_BLOCK = 8
        for h0 in range(0, q_heads, HEAD_BLOCK):
            h1 = min(q_heads, h0 + HEAD_BLOCK)
            hw = h1 - h0
            q_blk = q_f[safe_token, h0:h1]  # [f_len, hw, d_qk]
            scores = torch.einsum("fhd,fd->fh", q_blk, kv) * self.scale  # [f_len, hw]
            scores = scores.masked_fill(~valid.unsqueeze(-1), float("-inf"))

            max_s = torch.full(
                (num_tokens, hw), float("-inf"), device=dev, dtype=torch.float32
            )
            max_s.scatter_reduce_(
                0,
                safe_token.unsqueeze(-1).expand(f_len, hw),
                scores,
                reduce="amax",
                include_self=True,
            )
            exp_s = (scores - max_s[safe_token]).exp()
            exp_s = exp_s.masked_fill(~valid.unsqueeze(-1), 0.0)

            sum_s = torch.zeros((num_tokens, hw), device=dev, dtype=torch.float32)
            sum_s.scatter_reduce_(
                0,
                safe_token.unsqueeze(-1).expand(f_len, hw),
                exp_s,
                reduce="sum",
                include_self=True,
            )
            sum_exp[:, h0:h1] = sum_s

            weighted = exp_s.unsqueeze(-1) * kv_v.unsqueeze(1)  # [f_len, hw, L]
            blk_acc = torch.zeros(
                (num_tokens, hw, self.kv_lora_rank), device=dev, dtype=torch.float32
            ).scatter_reduce(
                0,
                safe_token.unsqueeze(-1).unsqueeze(-1).expand(
                    f_len, hw, self.kv_lora_rank
                ),
                weighted,
                reduce="sum",
                include_self=True,
            )  # [num_tokens, hw, L]
            acc[:, h0:h1] = blk_acc

        denom = sum_exp.unsqueeze(-1).clamp_min(1e-30)  # [num_tokens, heads, 1]
        output.copy_((acc / denom).to(output.dtype))

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        fp8_attention = self.kv_cache_dtype.startswith("fp8")
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            if fp8_attention:
                q = layer._decode_concat_quant_fp8_op(  # type: ignore[attr-defined]
                    ql_nope, q_pe, layer._q_scale
                )
            else:
                q = self.q_concat_buffer[: ql_nope.shape[0]]
                # [glm5-noaiter] torch.ops._C_cache_ops.concat_mla_q IS
                # registered in the real serve (platforms/rocm.py loads
                # vllm._C_stable_libtorch at import) but may be absent in
                # minimal/foreign processes -- probe, and fall back to the
                # exact torch semantics (vllm/_custom_ops.py:2967:
                # q_out = [ql_nope | q_pe] along the last dim). GLM NoPE:
                # q_pe is 0-wide, so the concat degenerates to a plain copy.
                # The NoPE branch must come FIRST: the stable-lib kernel was
                # built for DeepSeek dims and runtime-asserts
                # `rope_dim must be 64` (cache_kernels.hip:1678), so it
                # crashes on GLM's 0-wide q_pe even though hasattr() is True.
                if q_pe.shape[-1] == 0:
                    q.copy_(ql_nope)
                elif hasattr(torch.ops._C_cache_ops, "concat_mla_q"):
                    ops.concat_mla_q(ql_nope, q_pe, q)
                else:
                    q.copy_(torch.cat([ql_nope, q_pe], dim=-1))

        num_actual_toks = attn_metadata.num_actual_tokens

        # Get topk indices
        assert self.topk_indices_buffer is not None
        # AITER sparse MLA consumes exactly index_topk entries per row. The
        # shared kpool buffer is wider because it also carries the incomplete
        # pool tail and alignment padding for CUDA backends (official
        # glm53-flash read side; no-op when the widths already match).
        topk_indices = fit_kpool_indices_to_aiter(
            self.topk_indices_buffer[:num_actual_toks], attn_metadata.topk_tokens
        )

        triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            attn_metadata.paged_kv_indptr,
            attn_metadata.paged_kv_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )

        # write the latent and rope to kv cache
        if fp8_attention:
            kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(current_platform.fp8_dtype())
            if q.dtype != current_platform.fp8_dtype():
                original_q_shape = q.shape
                q, _ = ops.scaled_fp8_quant(q.view(q.shape[0], -1), layer._q_scale)
                q = q.view(original_q_shape)
        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
        attn_out = self._forward_mla(
            layer, mla_padded_q, kv_c_and_k_pe_cache, attn_metadata
        )

        return attn_out, None
