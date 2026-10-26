#!/usr/bin/env python3
# mqa-triton-01-edit.py <patch dir>: [claude-mqa-01] GLM5_MQA_TRITON=1 (off unless set) sends the sparse indexer's prompt
# logits (rocm_fp8_mqa_logits without aiter) to the fused Triton kernel in v1/attention/ops/glm5_mqa_logits_rdna3.py
# (copied from ~/fp8-rdna3-01/mqa-01/mqa_logits_rdna3.py; tested against the torch fallback by mqa-test-01.py) instead of
# fp8_mqa_logits_torch. Adds one file to the patch set (24 -> 25 .py files).
import shutil, sys
d = sys.argv[1]
shutil.copy(__import__("os").path.expanduser("~/fp8-rdna3-01/mqa-01/mqa_logits_rdna3.py"), d + "/vllm/v1/attention/ops/glm5_mqa_logits_rdna3.py")
p = d + "/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"; s = open(p).read()
a = '''    else:
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
'''
assert s.count(a) == 1
s = s.replace(a, '''    else:
        if os.environ.get("GLM5_MQA_TRITON", "0") == "1":  # [claude-mqa-01]
            from vllm.v1.attention.ops.glm5_mqa_logits_rdna3 import fp8_mqa_logits_rdna3
            return fp8_mqa_logits_rdna3(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
''')
if "\nimport os\n" not in s:
    s = "import os  # [claude-mqa-01]\n" + s
open(p, "w").write(s); print("edited", p)
