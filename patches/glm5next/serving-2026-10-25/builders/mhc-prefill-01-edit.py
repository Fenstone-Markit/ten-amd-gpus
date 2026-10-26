#!/usr/bin/env python3
# mhc-prefill-01-edit.py <patch dir>: model_executor/layers/mhc.py [claude-mhc-01]. GLM5_MHC_FUSED_MAX_T=<tokens> (default 64,
# the overlay's _FUSED_MAX_T, so unchanged unless set): steps with at most that many tokens use the fused mHC Triton kernels
# (one launch for the post block, one for the pre block, per token), larger steps the torch reference decomposition.
# Why (claude-mt-01, 2026-10-25, cold reads of 39k/74k/116k tokens on 3x4): the torch path (fp32 pre-mix GEMM + Sinkhorn-20
# in about 80 small launches) was 18-21% of every stage's prompt time, 2.9-3.2 ms per call at about 1,150 tokens, 31 calls per
# step on a 16-layer stage. Decode already runs the fused kernels, whose header states the same math and op order as the torch
# reference (fp32 accumulation, chunked over K). Set it above the chunk size (e.g. 1000000) to use them for prompts too.
import sys
p = sys.argv[1] + "/vllm/model_executor/layers/mhc.py"; s = open(p).read()
a = '''        and _MHC_FUSED.mhc_fused_enabled()
        and _MHC_FUSED.small_t(residual)
    )
'''
assert s.count(a) == 1
s = s.replace(a, '''        and _MHC_FUSED.mhc_fused_enabled()
        and _MHC_FUSED._num_tokens(residual) <= _GLM5_MHC_FUSED_MAX_T  # [claude-mhc-01]
    )
''')
b = "def _fused_active(residual: torch.Tensor) -> bool:\n"; assert s.count(b) == 1
s = s.replace(b, '''# [claude-mhc-01] see ~/fp8-rdna3-01/mhc-prefill-01-edit.py; 64 = the overlay's _FUSED_MAX_T.
import os as _glm5_os
_GLM5_MHC_FUSED_MAX_T = int(_glm5_os.environ.get("GLM5_MHC_FUSED_MAX_T", "64") or 64)


''' + b)
open(p, "w").write(s); print("edited", p)
