#!/usr/bin/env python3
# mixed-spec-01-edit.py <patch dir>: v1/attention/backends/mla/rocm_aiter_mla_sparse.py [claude-mixspec-01].
# Problem (found 2026-10-25 with GLM5_STAGE_TIMING): with MTP, a step that holds one request's verify rows (1 + 2 drafts) and
# another request's prompt chunk misses all three ragged-Triton gates: the prefill gate wants plain one-token decode rows, the
# decode and verify gates want no prompt rows. It falls to _forward_mla_torch, whose prompt path is a Python loop over every
# token on every sparse-attention layer: stage 1 took 2.2 s for a 2048-token mixed step against 0.6 s for the same chunk alone,
# so a 39k prompt read beside a generating request took 130 s instead of 15 s.
# Fix: a fourth gate for that mixed shape. The ragged kernel is per-token CSR (each query row reads only its own
# paged_kv_indptr[t]:[t+1] slice of paged_kv_indices, built per token in build()), and the torch fallback reads exactly the same
# indices/indptr, so verify rows inside a mixed step need no other change: the same call already serves prompt rows, plain
# decode rows and pure verify steps. Escape hatch: GLM5_SPEC_MIXED_TORCH=1 restores the torch fallback for this shape.
import sys
p = sys.argv[1] + "/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py"; s = open(p).read()
a = "def _use_rocm_sparse_spec_verify(\n"
assert s.count(a) == 1
s = s.replace(a, '''# [claude-mixspec-01] MTP verify rows (<= spec+1 per decode request) in the same step as another request's prompt chunk.
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


''' + a)
b = '''                    or (
                        not spec_verify_torch()
                        and _use_rocm_sparse_spec_verify(
                            attn_metadata,
                            q.shape[-1],
                            self.kv_lora_rank,
                            self._num_spec_tokens,
                        )
                    )
'''
assert s.count(b) == 1
s = s.replace(b, b + '''                    or _use_rocm_sparse_mixed_spec(  # [claude-mixspec-01]
                        attn_metadata,
                        q.shape[-1],
                        self.kv_lora_rank,
                        self._num_spec_tokens,
                    )
''')
assert "\nimport os\n" in s
open(p, "w").write(s); print("edited", p)
