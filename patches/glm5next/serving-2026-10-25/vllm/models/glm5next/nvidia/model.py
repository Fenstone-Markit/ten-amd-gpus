# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
from torch import nn
import os

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import (
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
    hc_contract,
    hc_expand,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from .attention import Glm5NextMLAAttention
from vllm.models.glm5next.noaiter import dsa_dense_forced
from vllm.v1.attention.ops.glm5_meta_watch import (
    meta_watch_check_c,
    meta_watch_indptr_buf,
    meta_watch_last_ntok,
)
from .kda import Glm5NextLinearAttention
from .multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    Glm5NextVisionTransformer,
)

logger = init_logger(__name__)

# glm5 overlay: sub-layer activation tap (numerics debugging only). Armed by
# Glm5NextModel.forward when GLM5_DEBUG_DUMP is set AND
# GLM5_DEBUG_SUBLAYERS="i0-i1" selects an inclusive layer range; the decoder
# layer appends (layer_idx, tag, tensor) tuples for attn_in/attn_out/mlp_out.
_GLM5_ECOUNT = os.environ.get("GLM5_EXPERT_COUNT", "0") == "1"   # [claude-ecount-02]
if _GLM5_ECOUNT:
    from vllm.v1.worker import glm5_ecount as _glm5_ec
_GLM5_SUBLAYER_DUMP: list | None = None
_GLM5_SUBLAYER_RANGE: tuple[int, int] = (-1, -1)


class Glm5NextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )

        self.swiglu_limit = swiglu_limit
        if self.swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit=self.swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextMoE(nn.Module):
    def __init__(
        self,
        config: Glm5NextConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None
        if _GLM5_ECOUNT:
            # [claude-ecount-02] counters for the read-only expert count
            _glm5_ec.init(self, config.n_routed_experts, _glm5_ec.layer_of(prefix), config.num_experts_per_token)

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        swiglu_limit = getattr(config, "swiglu_limit", None)
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            # GLM-5.3 ships n_group=1/topk_group=1: one group over all 288
            # experts, i.e. degenerate grouping == plain top-k routing. The
            # ROCm Triton MoE path asserts group args are None, so normalize
            # degenerate single-group configs to ungrouped (math unchanged).
            use_grouped_topk=(getattr(config, "n_group", 1) or 0) > 1,
            num_expert_group=(getattr(config, "n_group", 1) or 0) if (getattr(config, "n_group", 1) or 0) > 1 else None,
            topk_group=(getattr(config, "topk_group", 1) or 0) if (getattr(config, "n_group", 1) or 0) > 1 else None,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=None,
            router_logits_dtype=self.gate.out_dtype,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # The router is always external (self.gate); main's MoERunner expects
        # pre-computed router_logits, so compute them here unconditionally.
        router_logits, _ = self.gate(hidden_states)
        if _GLM5_ECOUNT:
            # [claude-ecount-02] count which experts GLM's routing rule picks; does not change the output
            _glm5_ec.observe(self, router_logits, self.gate.e_score_correction_bias)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        self.layer_kind = "kda" if config.is_kda_layer(layer_idx) else "mla"
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.is_kda_layer(layer_idx):
            self.self_attn = Glm5NextLinearAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # MLA layers require the latent head dims, which are guaranteed set
            # on MLA configs; narrow away the `int | None`.
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                # glm5 overlay: PR upstream passes quant_config=None here
                # ("MLA projections are BF16 in checkpoint") -- true for the
                # upstream FP8 checkpoint. OUR tcclaviger composite checkpoint
                # (GLM-5.3-Flash-RFA-RFI8) quantizes the DSA MLA projections
                # with RFI-8 (manifest: q_a_proj/q_b_proj/kv_a_proj_with_mqa/
                # kv_b_proj/o_proj/indexer.wk/wq_b/weights_proj -> rfi8; only
                # indexer ape/gate stay dense). The rfi_hip manifest resolves
                # each projection BY PREFIX, so pass the real quant_config and
                # let dense entries fall out as UnquantizedLinearMethod.
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=getattr(config, "mla_nope", False),
            )

        # MTP layers sit past the base model's hidden layers (layer_idx >=
        # num_hidden_layers), so they're outside mlp_layer_types; default them
        # to the last base layer's MLP type (sparse/MoE for these checkpoints).
        mlp_layer_types = config.mlp_layer_types
        mlp_type = (
            mlp_layer_types[layer_idx]
            if layer_idx < len(mlp_layer_types)
            else (mlp_layer_types[-1] if mlp_layer_types else "sparse")
        )
        if self.is_moe and self.num_experts is not None and mlp_type == "sparse":
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Cached for the hot forward path (isinstance per layer per step).
        self._mlp_is_moe = isinstance(self.mlp, Glm5NextMoE)
        # In SP, the attention output projection leaves a partial sum; the
        # decoder-layer reduce_scatter after attention completes it (DSv4 pattern).
        # MTP layers use the non-mHC path which has no sp_reduce_scatter, so
        # their o_proj must still reduce normally.
        if self.is_sequence_parallel and not is_mtp_layer:
            self.self_attn.o_proj.reduce_results = False
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc and not is_mtp_layer:
            # mhc config
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_no_norm_weight = config.mhc_no_norm_weight
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            # attn hc
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            # ffn hc
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MHCFusedPostPreOp()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # 70B or MTP layers: KDA + MoE without HC.
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            attn_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states, residual = self.post_attention_layernorm(
                attn_output, residual=residual
            )
            hidden_states = self.mlp(hidden_states)
            if self.is_mtp_layer:
                # Return the unsummed pair: the MTP caller feeds it straight
                # into shared_head's fused_add_rms_norm (one kernel instead of
                # a separate residual-add + norm). The sum itself is unchanged
                # (fp32-accumulated inside the fused kernel).
                return hidden_states, residual, None, None
            hidden_states = residual + hidden_states
            return hidden_states, residual, None, None

        # mHC start. `post`/`comb` carry the previous layer's deferred
        # hc_post inputs (its ffn-pre outputs); when present, fuse that
        # hc_post with this layer's attn hc_pre into one kernel (inter-layer
        # fusion). Layer 0 has no incoming state -> standalone hc_pre.
        x = hidden_states
        _sublayer_tap = (
            _GLM5_SUBLAYER_DUMP is not None
            and _GLM5_SUBLAYER_RANGE[0] <= self.layer_idx <= _GLM5_SUBLAYER_RANGE[1]
        )
        if post is None:
            # Standalone pre. Only the model's global layer 0 expands the
            # single-stream embed into the n-stream residual. Under PP the
            # receiving stage's first mHC layer instead receives the n-stream
            # state materialised by the sending stage (see Glm5NextModel), so
            # it already has an [N, n, H] residual and must NOT expand --
            # hence the original `if self.layer_idx == 0` guard is restored
            # (hc_expand runs only at the global layer 0, as in the overlay).
            if self.layer_idx == 0:
                x = hc_expand(x, self.n)
            residual = x
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
            residual, post, comb, x = self.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )

        # Attention needs the full token sequence; mHC above ran on the SP
        # shard. Gather for attention, scatter back afterward (DSv4 pattern).
        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        if _sublayer_tap:
            _GLM5_SUBLAYER_DUMP.append(
                (self.layer_idx, "attn_in", x.detach().float().cpu())
            )
        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )
        if _sublayer_tap:
            _GLM5_SUBLAYER_DUMP.append(
                (self.layer_idx, "attn_out", x.detach().float().cpu())
            )

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)

        # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
        residual, post, comb, x = self.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )

        # Fully Connected
        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)
        if _sublayer_tap:
            _GLM5_SUBLAYER_DUMP.append(
                (self.layer_idx, "mlp_out", x.detach().float().cpu())
            )

        # mHC end. The last mHC layer materializes its final hc_post (nothing
        # to fuse with) then contracts; every other layer defers its hc_post to
        # the next layer's fused pre, returning the state.
        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None

        return x, residual, post, comb

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        post_mix, res_mix, layer_input = self.mhc_pre_op(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        return post_mix, res_mix, layer_input

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post_op(x, residual, post, comb)

    def hc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        return self.mhc_fused_post_pre_op(
            x=x,
            residual=residual,
            post_layer_mix=post,
            comb_res_mix=comb,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            n_splits=1,
            tile_n=1,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )


class Glm5NextModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        """
        if config.index_topk is not None:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
        """
        # `index_topk` is declared on Glm5NextTextConfig with a default of None,
        # so hasattr() is True even for full-MLA configs (no kpool indexer).
        # Gate on the value being set instead.
        self.is_v32 = getattr(config, "index_topk", None) is not None
        # [glm5-noaiter R3] GLM5_DSA_DENSE=1: skip the shared top-k indices
        # buffer entirely -- every DSA layer (incl. MTP) falls back to dense
        # MLA and no Indexer/KV-cache spec is constructed. See attention.py.
        if dsa_dense_forced():
            self.is_v32 = False
        if self.is_v32:
            topk_tokens = config.index_topk
            # Reserve room for the incomplete pool tail.
            kpool = getattr(config, "index_kpool", 1) or 1
            buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
            # Sparse MLA tiles top-k in 128 columns; padded slots remain masked.
            sparse_topk_block_n = 128
            buffer_width = (
                (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
            ) * sparse_topk_block_n
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Full-MLA config (no kpool sparse indexer): no topk buffer.
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # The active slice is fixed after construction; cache it so forward
        # doesn't rebuild the slice (a fresh list) every step.
        self._active_layers = self.layers[self.start_layer : self.end_layer]

        # Part 2 identical-weights check: when GLM5_ENABLE_HC_RMS=1, report a
        # deterministic checksum (float64 sum + abs sum) of every local layer's
        # parameters, keyed by GLOBAL layer index, plus the embedding on the
        # first rank. Dummy weights are seeded per-parameter with a fixed seed
        # (weight_utils.initialize_single_dummy_weight), so the same named
        # param has the same values in every layout; these checksums prove the
        # two PP layouts load identical weights.
        if os.environ.get("GLM5_ENABLE_HC_RMS", "0") in ("1", "true", "True", "yes"):
            with torch.no_grad():
                # Named params for the identical-weights check: every local
                # decoder layer's input_layernorm weight (a clean bf16 [H]
                # RMSNorm vector seeded by dummy init) + the embedding on rank 0.
                # Logging every local layer lets me compare a named param in
                # each PP stage against the same-named param in the TP1
                # reference (which holds all layers on one card).
                for gi, layer in enumerate(self._active_layers):
                    _w = layer.input_layernorm.weight.data
                    _f = _w.detach().float().double()
                    logger.info(
                        "[GLM5_WCS] rank=%d param=layers.%d.input_layernorm "
                        "sum=%.9e abs=%.9e",
                        get_pp_group().rank_in_group,
                        self.start_layer + gi,
                        float(_f.sum().item()),
                        float(_f.abs().sum().item()),
                    )
                if get_pp_group().is_first_rank:
                    _w = self.embed_tokens.weight.data
                    _f = _w.detach().float().double()
                    logger.info(
                        "[GLM5_WCS] rank=%d param=embed_tokens.sum=%.9e abs=%.9e",
                        get_pp_group().rank_in_group,
                        float(_f.sum().item()),
                        float(_f.abs().sum().item()),
                    )

        # PP: build the stage-to-stage intermediate tensors. The model passes
        # (hidden_states, residual) across pipeline stages. With mHC the residual
        # is the n-stream multi-residual [N, n, H] (see hc_expand), NOT [N, H],
        # so the standard make_empty_intermediate_tensors_factory (which makes a
        # 2D [batch, hidden] buffer for every key) cannot size it. Use a custom
        # builder with the same contract: (batch_size, dtype, device) ->
        # IntermediateTensors. post/comb (the deferred mHC hc_post state) are not
        # propagated across stages (the receiving stage restarts from
        # hc_expand), so they are not part of the intermediate tensors.
        _n_streams = (
            config.mhc_num_residual_streams
            if getattr(config, "mhc", False)
            else 1
        )

        def _make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            # The stage-to-stage state is the n-stream multi-residual
            # [N, n, H] materialised by the sending stage's last mHC layer
            # (hc_post). One buffer, sized to exactly what is sent.
            return IntermediateTensors(
                {
                    "residual": torch.zeros(
                        (batch_size, _n_streams, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    ),
                }
            )

        self.make_empty_intermediate_tensors = _make_empty_intermediate_tensors

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # glm5 overlay: GLM5_DEBUG_DUMP=<dir> -> save embeddings + final
        # hidden of small real forwards on TP rank 0 for host-side numerics
        # A/B (tokenizer/embedding/lm_head bisection). Debug only.
        from vllm.models.glm5next.noaiter import debug_dump_dir

        _dump_dir = debug_dump_dir()
        _do_dump = False
        # The multimodal wrapper consumes input_ids upstream and calls down
        # with inputs_embeds only (input_ids=None) — gate on whichever exists.
        _seq_len = (
            input_ids.shape[0]
            if input_ids is not None
            else inputs_embeds.shape[0]
            if inputs_embeds is not None
            else 0
        )
        if _dump_dir and 0 < _seq_len <= 64:
            from vllm.distributed import get_tensor_model_parallel_rank

            _do_dump = get_tensor_model_parallel_rank() == 0
        _dump: dict = {}
        if _do_dump:
            _dump["input_ids"] = (
                input_ids.detach().cpu() if input_ids is not None else None
            )

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                if _do_dump:
                    _dump["embed_out"] = inputs_embeds.detach().float().cpu()
            else:
                hidden_states = self.embed_input_ids(input_ids)
                if _do_dump:
                    _dump["embed_out"] = hidden_states.detach().float().cpu()
            residual = None
            post = None
            comb = None
        else:
            assert intermediate_tensors is not None
            # The receiving stage gets the n-stream multi-residual [N, n, H]
            # materialised by the sending stage's last mHC layer (hc_post).
            # Feed it as the first layer's input so its standalone hc_pre
            # receives [N, n, H] directly (no expand). post/comb are None.
            residual = intermediate_tensors["residual"]
            hidden_states = residual
            post = None
            comb = None

        full_num_tokens = positions.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        # glm5 overlay: GLM5_DEBUG_LAYER_RMS=1 -> per-layer hidden/residual RMS
        # trace on TP rank 0 (numerics-corruption localization, debug only).
        from vllm.models.glm5next.noaiter import debug_layer_rms

        _dbg_rms = debug_layer_rms()
        if _dbg_rms:
            from vllm.distributed import get_tensor_model_parallel_rank

            _dbg_rms = get_tensor_model_parallel_rank() == 0
        if _do_dump:
            _dump["layers"] = []
            # Arm the sub-layer tap for this forward (GLM5_DEBUG_SUBLAYERS).
            global _GLM5_SUBLAYER_DUMP, _GLM5_SUBLAYER_RANGE
            import os as _os

            _slr = _os.environ.get("GLM5_DEBUG_SUBLAYERS", "")
            if "-" in _slr:
                try:
                    _i0, _i1 = _slr.split("-", 1)
                    _GLM5_SUBLAYER_RANGE = (int(_i0), int(_i1))
                    _GLM5_SUBLAYER_DUMP = []
                except ValueError:
                    _GLM5_SUBLAYER_DUMP = None
        aux_hidden_states: list[torch.Tensor] = []
        _hc_rms_on = os.environ.get("GLM5_ENABLE_HC_RMS", "0") in (
            "1",
            "true",
            "True",
            "yes",
        )
        if _hc_rms_on:
            from vllm.distributed import get_tensor_model_parallel_rank

            _hc_rms_on = get_tensor_model_parallel_rank() == 0
        import os as _os
        _mt = _os.environ.get("GLM5_MEM_TRACE", "0") == "1"
        _mt_base = (torch.cuda.memory_allocated() if _mt else 0)
        # [glm5next-port-23] GLM5_META_WATCH (c): at the start of every decoder
        # layer, compare paged_kv_indptr[:num_tokens+1] against the build
        # reference. Reports only the first difference per build. Off by
        # default; no-op when off (no sync when disabled).
        _mw_buf = meta_watch_indptr_buf()
        for layer_idx, layer in enumerate(self._active_layers):
            if _mt and not torch.cuda.is_current_stream_capturing():
                torch.cuda.synchronize()
            if _mw_buf is not None:
                _mw_nt = meta_watch_last_ntok()
                meta_watch_check_c(
                    "layer_start idx=%d (global %d)" % (layer_idx, self.start_layer + layer_idx),
                    _mw_buf, _mw_nt, 0,
                )
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
            if _mt and not torch.cuda.is_current_stream_capturing():
                _pk = torch.cuda.memory_stats().get("allocated_bytes.all.peak", 0)
                _gidx = self.start_layer + layer_idx
                print(f"[MEM_LAYER] layer={_gidx} cum_peak={_pk/1024**3:.3f} "
                      f"GiB  vs_base={(_pk-_mt_base)/1024**3:.3f} GiB", flush=True)
            if _hc_rms_on:
                # Boundary-comparison trace: per GLOBAL decoder layer, log the
                # hidden (ffn-pre) RMS and the n-stream residual-state RMS, with
                # the token count (prefill = prompt tokens; decode = 1). The
                # residual is the [N, n, H] mHC state, so it is the quantity
                # that crosses the PP boundary and must agree layer-by-layer.
                _gidx = self.start_layer + layer_idx
                _hf = hidden_states.float()
                _hr = (
                    residual.float().pow(2).mean().sqrt().item()
                    if isinstance(residual, torch.Tensor)
                    else float("nan")
                )
                logger.info(
                    "[GLM5_HC_RMS] layer=%d stage=%d tokens=%d hidden_rms=%.6e "
                    "residual_rms=%.6e",
                    _gidx,
                    get_pp_group().rank_in_group,
                    full_num_tokens,
                    _hf.pow(2).mean().sqrt().item(),
                    _hr,
                )
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                # DFlash taps are base hidden-size layer outputs. GLM5 mHC
                # carries expanded residual streams between layers, so
                # materialize the current deferred hc_post into a temporary
                # tensor and contract it without mutating the live state.
                if layer.mhc and not layer.is_mtp_layer and post is not None:
                    aux = hc_contract(
                        layer.hc_post(hidden_states, residual, post, comb), layer.n
                    )
                else:
                    aux = hidden_states
                if self.is_sequence_parallel:
                    aux = sp_all_gather(aux)[:full_num_tokens]
                aux_hidden_states.append(aux)
            if _do_dump:
                # Per-layer (x, residual, post, comb) for host-side stream
                # reconstruction + early-exit texture analysis.
                _dump["layers"].append(
                    (
                        hidden_states.detach().float().cpu(),
                        residual.detach().float().cpu()
                        if isinstance(residual, torch.Tensor)
                        else None,
                        post.detach().float().cpu()
                        if isinstance(post, torch.Tensor)
                        else None,
                        comb.detach().float().cpu()
                        if isinstance(comb, torch.Tensor)
                        else None,
                    )
                )
            if _dbg_rms:
                hs = hidden_states.float()
                r_rms = (
                    residual.float().pow(2).mean().sqrt().item()
                    if isinstance(residual, torch.Tensor)
                    else float("nan")
                )
                logger.info(
                    "[GLM5_RMS] layer %2d %-26s hidden rms=%.5g max=%.5g "
                    "residual rms=%.5g",
                    layer_idx,
                    type(layer.self_attn).__name__
                    if hasattr(layer, "self_attn")
                    else type(layer).__name__,
                    hs.pow(2).mean().sqrt().item(),
                    hs.abs().max().item(),
                    r_rms,
                )

        if not get_pp_group().is_last_rank:
            # Materialise the n-stream state at this stage's last mHC layer
            # (hc_post of its deferred state) and send it across the boundary.
            # This is the true model state here; port-05 instead sent
            # {hidden_states, residual} and dropped post/comb, which loses the
            # ffn-pre contribution (see audit / Part 2).
            last_layer = self._active_layers[-1]
            if post is not None:
                state = last_layer.hc_post(hidden_states, residual, post, comb)
            else:
                state = residual
            return IntermediateTensors({"residual": state})

        if self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        if _do_dump:
            _dump["pre_norm"] = hidden_states.detach().float().cpu()
            _dump["residual_final"] = (
                residual.detach().float().cpu()
                if isinstance(residual, torch.Tensor)
                else None
            )
            if _GLM5_SUBLAYER_DUMP is not None:
                _dump["sublayers"] = _GLM5_SUBLAYER_DUMP
                _GLM5_SUBLAYER_DUMP = None
        hidden_states = self.norm(hidden_states)
        if _do_dump:
            _dump["post_norm"] = hidden_states.detach().float().cpu()
            _n = getattr(self, "_glm5_dump_n", 0)
            self._glm5_dump_n = _n + 1
            torch.save(_dump, f"{_dump_dir}/glm5_dump_{_n:03d}.pt")
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA: fuse q_a_proj and kv_a_proj_with_mqa
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            # Indexer: fuse wk and weights_proj
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
            # KDA: merge q, k, v, b, f_a, g_a projections into one GEMM
            (".in_proj_qkvbfg_a", ".q_proj", 0),
            (".in_proj_qkvbfg_a", ".k_proj", 1),
            (".in_proj_qkvbfg_a", ".v_proj", 2),
            (".in_proj_qkvbfg_a", ".b_proj", 3),
            (".in_proj_qkvbfg_a", ".f_a_proj", 4),
            (".in_proj_qkvbfg_a", ".g_a_proj", 5),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # GLM-5.3-Flash NoPE checkpoints omit the RoPE rows from
        # ``kv_a_proj_with_mqa``; pad them with zeros for the model shape.
        kv_a_pad_size = 0
        if self.config.mla_nope and self.config.qk_rope_head_dim > 0:
            kv_a_pad_size = self.config.qk_rope_head_dim

        _pending_wk_fp8: dict = {}
        # [claude-mixed-01] mixed checkpoint: in the GLM5_FP8_LAYERS layers the routed experts are block FP8 (RDNA3 FP8 MoE
        # method); the checkpoint still carries their INT4 tensors, which have no parameters there and are skipped.
        import re as _re_mixed
        _mixed_fp8 = set()
        if os.environ.get("GLM5_FP8_LAYERS"):
            from vllm.model_executor.layers.quantization.fp8_rdna3 import fp8_layers as _fp8_layers
            _mixed_fp8 = _fp8_layers()
        _mixed_int4 = ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point", "weight_g_idx")

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs: dict = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if _mixed_fp8 and ".mlp.experts." in name and name.rsplit(".", 1)[-1] in _mixed_int4:
                _mm_mixed = _re_mixed.search(r"layers\.(\d+)\.", name)
                if _mm_mixed and int(_mm_mixed.group(1)) in _mixed_fp8:
                    continue
            # [claude-mixed-02] and the reverse: FP8 expert tensors of layers left on the 4-bit method
            if _mixed_fp8 and ".mlp.experts." in name and name.rsplit(".", 1)[-1] in ("weight", "weight_scale_inv"):
                _mm_mixed = _re_mixed.search(r"layers\.(\d+)\.", name)
                if _mm_mixed and int(_mm_mixed.group(1)) not in _mixed_fp8:
                    continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # Handle FP8 indexer WK: dequantize to BF16 for fusion with
            # weights_proj into wk_weights_proj.
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # FP8 checkpoint: dequantize BF16-kept MLA projections
            # (q_a_proj / kv_a_proj_with_mqa / o_proj) to BF16.
            if _try_load_fp8_attn_proj(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            ):
                continue

            # Pad kv_a_proj_with_mqa for NoPE models
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                pad = torch.zeros(
                    kv_a_pad_size,
                    *loaded_weight.shape[1:],
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                loaded_weight = torch.cat([loaded_weight, pad], dim=0)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # QKV fusion: skip if fused module doesn't exist in model
                if param_name == ".fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in enumerate(expert_params_mapping):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params


class Glm5NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsPP,
    MixtureOfExperts,
    IsHybrid,
    SupportsEagle3,
):
    # glm5 overlay: fusion table for the rfi_hip quant manifest (and stock
    # quant configs). Every fused runtime module below resolves its scheme
    # from these checkpoint-side parts, which must agree -- see
    # rfi_hip/quant_manifest.py:resolve(). Leaves match
    # Glm5NextModel.load_weights' stacked_params_mapping.
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "wk_weights_proj": ["wk", "weights_proj"],
        "in_proj_qkvbfg_a": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # PP: expose the stage-to-stage intermediate-tensors builder on the
        # wrapper too (the runner calls model.make_empty_intermediate_tensors on
        # the top-level model).
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_num_heads,
            hf_config.linear_head_dim,
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[
        MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc
    ]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3
):
    # The text model (KDA + dense-MLA + MoE) is a hybrid mamba model. The
    # multimodal wrapper must declare the same interfaces so vLLM treats it as
    # hybrid (auto-aligns mamba/attention block sizes, sizes the mamba state
    # cache); the mamba-state classmethods delegate to the text model.
    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True

    # glm5 overlay: override Glm4vForConditionalGeneration's fusion table with
    # the glm5next one (configure_quant_config reads it from THIS class; the
    # rfi_hip manifest resolves fused modules through these checkpoint-side
    # parts). hf_to_vllm_mapper is inherited from Glm4v
    # (model.language_model. -> language_model.model., lm_head. ->
    # language_model.lm_head., model.visual. -> visual.) and matches the
    # quantizer's checkpoint-side manifest keys.
    packed_modules_mapping = Glm5NextForCausalLM.packed_modules_mapping

    # NOTE: weight-prefix mapping is inherited from Glm4vForConditionalGeneration
    # (``model.visual.`` -> ``visual.``, ``model.language_model.`` ->
    # ``language_model.model.``, ``lm_head.`` -> ``language_model.lm_head.``),
    # matching the GLM-OCR / GLM-4V serialization convention. If the real
    # checkpoint's safetensors keys differ (e.g. ``language_model.model.`` with
    # no outer ``model.``), override ``hf_to_vllm_mapper`` accordingly.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm4vForConditionalGeneration, self).__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                # Read eps from the VISION sub-config, not the top-level
                # `config.rms_norm_eps`: Glm5NextConfig.__getattribute__ mirrors
                # the latter onto text_config (1e-5), silently ignoring the
                # vision tower's own (1e-6) rms_norm_eps.
                norm_eps=config.vision_config.rms_norm_eps,
                # Vision tower ships BF16 weights in this fp8 checkpoint (no
                # weight_scale_inv for visual.*), so it must NOT inherit the
                # global fp8 quant_config -- doing so incorrectly quantizes
                # the tower
                # and yields NaN image features. Mirrors the MLA/KDA proj
                # pattern (quant_config=None for BF16 submodules).
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

        # PP: expose the stage-to-stage intermediate-tensors builder on the
        # multimodal wrapper too (the runner calls
        # model.make_empty_intermediate_tensors on the top-level model). The base
        # Glm4vForConditionalGeneration.__init__ does this, but this wrapper
        # calls nn.Module.__init__ directly (super(Glm4vForConditionalGeneration,
        # self).__init__()), so alias it here from the language model.
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_encoder_cudagraph_config(self):
        # The forked vision tower (multimodal.py) has no abs-pos embeddings, so its
        # prepare_encoder_metadata does not produce "pos_embeds". Drop it from the
        # buffer_keys inherited from Glm4vForConditionalGeneration so encoder
        # CUDA-graph capture/replay does not expect a buffer that is never filled.
        config = super().get_encoder_cudagraph_config()
        config.buffer_keys = [k for k in config.buffer_keys if k != "pos_embeds"]
        return config


def get_spec_layer_idx_from_weight_name(
    config: Glm5NextConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(
                f"model.layers.{layer_idx + i}."
            ) or weight_name.startswith(f"layers.{layer_idx + i}."):
                return layer_idx + i
    return None


def _try_load_fp8_indexer_wk(name, tensor, buf, params_dict, loaded_params):
    if "indexer.wk." not in name or "wk_weights" in name:
        return False
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False
    layer_prefix = name.rsplit(".wk.", 1)[0]
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    del buf[layer_prefix]
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale_inv,
        group_shape=GroupShape(block_size, block_size),
        out_dtype=torch.bfloat16,
    )

    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


def _dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a block-FP8 (e4m3) weight with per-block scale to BF16.

    Unlike ``scaled_dequantize`` this tolerates a non-divisible (partial last
    block) shape by zero-padding to a multiple of ``block_size`` before the
    scale broadcast and trimming back afterwards (e.g. kv_a_proj_with_mqa is
    576 rows = 4*128 + 64).
    """
    out_dim, in_dim = weight_fp8.shape
    pad_out = (-out_dim) % block_size
    pad_in = (-in_dim) % block_size
    w = weight_fp8
    if pad_out or pad_in:
        w = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    # scale_inv is (ceil(out/block), ceil(in/block)); broadcast to (out, in).
    s = scale_inv.to(torch.float32)
    s_full = s.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    out = (w.to(torch.float32) * s_full).to(torch.bfloat16)
    return out[:out_dim, :in_dim].contiguous()


# FP8 checkpoint projections that the MODEL keeps in BF16, so the block-FP8
# (weight + weight_scale_inv) must be dequantized to BF16 on load.
# Maps checkpoint proj-suffix -> (buffer key, model target base, fused shard id
# or None for a direct projection, whether NoPE rope-padding applies).
_FP8_ATTN_PROJS = {
    ".q_a_proj.": ("q_a", "fused_qkv_a_proj", 0, False),
    ".kv_a_proj_with_mqa.": ("kv_a", "fused_qkv_a_proj", 1, True),
    ".q_b_proj.": ("q_b", "q_b_proj", None, False),
    ".o_proj.": ("o_proj", "o_proj", None, False),
}


def _try_load_fp8_attn_proj(
    name,
    tensor,
    buf,
    params_dict,
    loaded_params,
    kv_a_pad_size: int,
) -> bool:
    """Dequantize FP8 q_a_proj / kv_a_proj_with_mqa / o_proj to BF16 on load.

    The FP8 checkpoint stores these as block-FP8 (weight + weight_scale_inv),
    but the model holds them in BF16 (``fused_qkv_a_proj`` is always BF16 via
    DeepSeekV2FusedQkvAProjLinear; ``o_proj`` is excluded by
    modules_to_not_convert). When the model target is BF16 (no
    ``weight_scale_inv`` param) we dequantize; otherwise we return False so the
    normal stacked/direct path loads the FP8 tensor as-is.
    """
    matched = None
    for suffix, info in _FP8_ATTN_PROJS.items():
        if suffix in name:
            matched = (suffix, info)
            break
    if matched is None:
        return False
    suffix, (key, target_base, shard_id, is_kva) = matched
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False

    layer_prefix = name.rsplit(suffix, 1)[0]
    target_w = f"{layer_prefix}.{target_base}.weight"
    target_s = f"{layer_prefix}.{target_base}.weight_scale_inv"
    # If the model actually kept this projection in FP8, let the normal path
    # handle it (it has a weight_scale_inv param).
    if target_s in params_dict:
        return False

    entry = buf.setdefault(layer_prefix, {}).setdefault(key, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True

    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    buf[layer_prefix].pop(key, None)
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
    # NoPE: pad kv_a rope portion (kv_lora_rank -> kv_lora_rank + qk_rope_head_dim).
    if is_kva and kv_a_pad_size > 0:
        pad = torch.zeros(
            kv_a_pad_size,
            weight_bf16.shape[1],
            dtype=weight_bf16.dtype,
            device=weight_bf16.device,
        )
        weight_bf16 = torch.cat([weight_bf16, pad], dim=0)

    param = params_dict[target_w]
    if shard_id is None:
        param.weight_loader(param, weight_bf16)
    else:
        param.weight_loader(param, weight_bf16, shard_id)
    loaded_params.add(target_w)
    return True
