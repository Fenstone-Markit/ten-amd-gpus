# Manifest

md5 of every file in patches/, and of the stock and patched vLLM files the diffs connect.

| vLLM file | Stock image | Patched (rc2) |
| --- | --- | --- |
| v1/attention/ops/triton_unified_attention.py | 006b2f296527fe98e028a953dab172fc | b184d334df3d5f88b782e5b92939d6d0 |
| v1/attention/backends/triton_attn.py | 82bb7c78e32be6471b39045275f0cd69 | 6eea73a19eafef97f5aab74da9ab2621 |
| model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py | 3b421ed1e3e6fdf9d2dd937413a40774 | 42840ce685a5cbea7046abae6021f344 |

| File | md5 |
| --- | --- |
| container/brain-rc2.sh | af20a04a65da43ff40bd389bc1f212e3 |
| container/Containerfile.rc2 | 683ffb3d5a7087f8e46283ac6f64dcaf |
| kernel/qdq_4_rdna3.cuh | b7abfa7b2f4011bba266b66094ea99bd |
| kernel/q_gemm_rdna3_det.cu | 449cbcae85f5075723924b5090c4de60 |
| README.md | cead46bf31b7a6a8c61b16c4c6b62f72 |
| vllm/rdna3_w4a16.py.diff | 3f95819300be3a2c257de6e4d17aa550 |
| vllm/triton_attn.py.diff | 91abf56fab3c54c505ef001cb030f1ee |
| vllm/triton_unified_attention.py.diff | 710442cdd1306b339124f3e44ada9c05 |
