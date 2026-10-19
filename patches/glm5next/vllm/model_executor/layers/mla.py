# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import os
import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    glm5_dsa_break_parts,
)
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.platforms import current_platform

# [glm5next-port-12] cached set of DSA decode parts forced to eager breaks
# (off by default; populated once from the GLM5_DSA_BREAK env var).
_GLM5_DSA_BREAK_PARTS: set[str] | None = None


def _glm5_dsa_break_parts() -> set[str]:
    global _GLM5_DSA_BREAK_PARTS
    if _GLM5_DSA_BREAK_PARTS is None:
        _GLM5_DSA_BREAK_PARTS = glm5_dsa_break_parts()
    return _GLM5_DSA_BREAK_PARTS


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module | None
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    g_proj: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]


# ---------------------------------------------------------------------------
# [glm5next-port-12] GLM5_DSA_BREAK support (off by default).
#
# A forced break must run its part OUTSIDE the graph at capture and at every
# replay, with its outputs at addresses that exist BEFORE the call (a fresh
# tensor returned by the part would change address each replay and break the
# downstream graph segments, which read those addresses). Two parts:
#
#   * "indexer": sparse_attn_indexer_kpool. It writes its top-k into the
#     SHARED persistent topk_indices_buffer (allocated at model init, model.py:
#     664), which the sparse-attention decode (next graph segment) reads. No
#     copy is needed -- the part already writes into a buffer that exists
#     before the call. Forcing the break is sufficient.
#
#   * "attn": the sparse-attention decode (self.mla_attn -> forward_mqa). It
#     returns a FRESH [num_tokens, num_heads, kv_lora_rank] tensor. We copy it
#     into a persistent buffer of the LARGEST shape it can produce (rows =
#     max_num_batched_tokens, the maximum over every captured batch descriptor;
#     num_heads and kv_lora_rank are fixed model dims) and pass that buffer
#     downstream (o_proj, in the next graph segment) so it reads a fixed
#     address. The buffer is allocated ONCE in __init__ (before any capture),
#     so its address is stable across captures and replays.
# ---------------------------------------------------------------------------


def _dsa_break_attn(layer, fn, num_tokens: int) -> torch.Tensor:
    """Run the DSA sparse-attention decode (fn) as a forced eager break.

    Copies the part's fresh output into layer._dsa_attn_break_buf (rows =
    num_tokens) and returns that slice, so the downstream o_proj (next graph
    segment) reads a fixed address. With no active capture (eager / no-graph
    path) it runs fn and copies into the buffer. Inside a capture it runs fn
    eagerly between the graph segments (a REAL break) and records it for
    replay.
    """
    cap = BreakableCUDAGraphCapture.current()
    buf = layer._dsa_attn_break_buf

    def _run() -> torch.Tensor:
        fresh = fn()
        buf[:num_tokens].copy_(fresh)
        return buf[:num_tokens]

    if cap is None or not cap._capturing:
        return _run()
    return cap.force_eager(_run)


@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
        non_causal_multi_token_decode: bool = False,
        allow_short_prefill_indexer_scoring_skip: bool = False,
        fuse_qkv_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.g_proj = mla_modules.g_proj

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        # When True, fuse the q_a and kv_a RMSNorms into a single kernel launch
        # (MLA layers with q-LoRA). Opt-in; default False preserves other models.
        self.fuse_qkv_rmsnorm = fuse_qkv_rmsnorm
        # qrep is active when the query projection is a DCP-group-sharded layer
        # that materializes the full group head set locally.
        q_proj_layer = self.q_b_proj if self.q_lora_rank is not None else self.q_proj
        self.dcp_q_replicate = getattr(q_proj_layer, "qrep_active", False)
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            dcp_q_replicate=self.dcp_q_replicate,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=mla_modules.topk_indices_buffer,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )
        indexer_op = getattr(self.indexer, "indexer_op", None)
        if indexer_op is not None and hasattr(
            indexer_op, "dense_mha_metadata_layer_name"
        ):
            enable_short_prefill_scoring_skip = (
                allow_short_prefill_indexer_scoring_skip
                and not self.skip_topk
                and not getattr(indexer_op, "use_pcp", False)
                and current_platform.is_cuda()
            )
            # The indexer and main MLA use independent decode thresholds and
            # may classify the same short extend differently. Bind the main
            # MLA layer name so the eager indexer op can check whether the
            # batch's top-k indices will be consumed.
            # PCP is excluded because indexer cache/scoring ownership differs
            # across ranks and the no-consumer invariant has not been
            # established there.
            indexer_op.dense_mha_metadata_layer_name = (
                self.mla_attn.layer_name if enable_short_prefill_scoring_skip else ""
            )
        self.prefix = prefix

        # [glm5next-port-12] persistent output buffer for the "attn" forced
        # break (allocated once, before any capture, so its address is stable
        # across captures and replays). Only allocated when the switch is on.
        # Rows = max_num_batched_tokens (the maximum over every captured batch
        # descriptor); cols = num_heads * v_head_dim (the inner attention
        # module's output, pre-o_proj).
        self._dsa_attn_break_buf = None
        if "attn" in glm5_dsa_break_parts():
            cfg = get_current_vllm_config()
            self._dsa_attn_break_buf = torch.empty(
                (cfg.scheduler_config.max_num_batched_tokens,
                 self.num_heads * self.v_head_dim),
                dtype=torch.bfloat16,
                device=self.o_proj.weight.device,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_proj_layer = self.q_b_proj
            if not self.fuse_qkv_rmsnorm:
                q_c = self.q_a_layernorm(q_c)
            q_proj_input = q_c
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        if self.fuse_qkv_rmsnorm and q_c is not None:
            # Fuse q_a + kv_a RMSNorm into one kernel launch (q_c is still the
            # pre-norm projection output here).
            assert self.q_a_layernorm is not None
            q_proj_input, kv_c_normed = fused_q_kv_rmsnorm(
                q_c,
                kv_c,
                self.q_a_layernorm.weight.data,
                self.kv_a_layernorm.weight.data,
                self.q_a_layernorm.variance_epsilon,
            )
            q_c = q_proj_input
        else:
            kv_c_normed = self.kv_a_layernorm(kv_c)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q = q_proj_layer(q_proj_input)[0]
        heads = self.num_heads
        if self.dcp_q_replicate:
            heads *= q_proj_layer.group_size
        q = q.view(-1, heads, self.qk_head_dim)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse and not self.skip_topk:
            _mt = os.environ.get("GLM5_MEM_TRACE", "0") == "1"
            _mt_cap = torch.cuda.is_current_stream_capturing()
            if _mt and not _mt_cap:
                torch.cuda.synchronize()
                _pk_i = torch.cuda.memory_allocated()
            if "indexer" in _glm5_dsa_break_parts():
                # Forced eager break: the indexer writes its top-k into the
                # SHARED persistent topk_indices_buffer (which the sparse
                # attention decode, in the next graph segment, reads), so no
                # copy is needed -- run it eagerly between graph segments.
                cap = BreakableCUDAGraphCapture.current()
                if cap is not None and cap._capturing:
                    cap.force_eager(
                        lambda: self.indexer(
                            hidden_states, q_c, positions, self.indexer_rope_emb
                        )
                    )
                else:
                    self.indexer(
                        hidden_states, q_c, positions, self.indexer_rope_emb
                    )
            else:
                self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
            if _mt and not _mt_cap:
                _pk_i2 = torch.cuda.memory_allocated()
                print(f"[MEM_SUB] indexer: alloc_before={_pk_i/1024**3:.3f} "
                      f"after={_pk_i2/1024**3:.3f} GiB "
                      f"delta={(_pk_i2-_pk_i)/1024**3:.3f} GiB", flush=True)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        if "attn" in _glm5_dsa_break_parts():
            _mt = os.environ.get("GLM5_MEM_TRACE", "0") == "1"
            _mt_cap = torch.cuda.is_current_stream_capturing()
            if _mt and not _mt_cap:
                torch.cuda.synchronize()
                _pk_a0 = torch.cuda.memory_stats().get(
                    "allocated_bytes.all.peak", 0)
            attn_out = _dsa_break_attn(
                self,
                lambda: self.mla_attn(
                    q,
                    kv_c_normed,
                    k_pe,
                    output_shape=(
                        hidden_states.shape[0],
                        self.num_heads * self.v_head_dim,
                    ),
                    q_dcp_replicated=q_dcp_replicated,
                ),
                q.shape[0],
            )
            if _mt and not _mt_cap:
                _pk_a1 = torch.cuda.memory_stats().get(
                    "allocated_bytes.all.peak", 0)
                print(f"[MEM_SUB] mla_attn peak_before={_pk_a0/1024**3:.3f} "
                      f"after={_pk_a1/1024**3:.3f} GiB "
                      f"delta={(_pk_a1-_pk_a0)/1024**3:.3f} GiB", flush=True)
        else:
            _mt = os.environ.get("GLM5_MEM_TRACE", "0") == "1"
            _mt_cap = torch.cuda.is_current_stream_capturing()
            if _mt and not _mt_cap:
                torch.cuda.synchronize()
                _pk_a0 = torch.cuda.memory_stats().get(
                    "allocated_bytes.all.peak", 0)
            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
                q_dcp_replicated=q_dcp_replicated,
            )
            if _mt and not _mt_cap:
                _pk_a1 = torch.cuda.memory_stats().get(
                    "allocated_bytes.all.peak", 0)
                print(f"[MEM_SUB] mla_attn peak_before={_pk_a0/1024**3:.3f} "
                      f"after={_pk_a1/1024**3:.3f} GiB "
                      f"delta={(_pk_a1-_pk_a0)/1024**3:.3f} GiB", flush=True)

        if self.g_proj is not None:
            attn_out = attn_out * self.g_proj(hidden_states)[0].sigmoid()

        return self.o_proj(attn_out)[0]
