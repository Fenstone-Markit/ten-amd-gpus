# Brain release candidate rc1, frozen 2026-09-26T07:25Z

Status: release candidate. Passed: op gate (B D X F C G), fixed-order gate (N I P R S), graph proof (fp32 op 510, installed 0, triton 0), measured on the server.
Not yet proven: multi-hour use, concurrent requests, reboot and automatic start, fair precision test on neutral text.

Measured, one request, exact token rate: 16K 67.5 tok/s, 60K 63.0 tok/s with MTP (1 draft token); without MTP 50.7 and 48.3.

Knobs: TP8; --attention-backend TRITON_ATTN; --enable-prefix-caching; KV cache auto (bf16); NCCL_ALGO=Ring; W4A16 fixed-order fp32 op (fen_fp32_op.so, patch rdna3_w4a16_fp32.py); FENSTONE_FP32_PREFILL=installed; --speculative-config mtp, 1 draft token, attention TRITON_ATTN; a fresh VLLM_CACHE_ROOT for any change of kernel knobs.
Base image: localhost/n02-vllm:snap-20260925 (ID in running-config.txt). The exact live command and environment are in running-config.txt.

Watch for: runs of exclamation marks or text turning incoherent mid-answer (speculation plus GDN align-mode cache, a known upstream risk). Fallback: the same command without --speculative-config (measured 50.7 and 48.3 tok/s).

## Files and hashes
d543b90d6ad7266db6744778a6edcba8  ./fen_fp32_op.so
5d325de15fa9b34c30f86b56bc17b5fb  ./gates/gate_det.py
a7c1d5baba4462c49339f353accae383  ./gates/gate_det_results.json
545306bac9ab9f8e8f400b66a7616d8b  ./gates/op_gate_results.json
42840ce685a5cbea7046abae6021f344  ./rdna3_w4a16_fp32.py
4f0e1f9e8045df67c1e3be3fd1cc22aa  ./results/allreduce_auto.json
271189b6e660b900061571c9f5112207  ./results/allreduce_ring.json
49c022bf06234b0377dbd3d935780fdd  ./results/allreduce_tree.json
ed6679b45bc4f7476c758868ec2fbd2c  ./results/bench_fp32det-mtp1.json
893ff165646c5dc97c8f6ef392e49ab0  ./results/bench_fp32det-ring.json
c67b2626375223568c7736430e9fe5b0  ./results/rdna_ar_test.json
c3d00913268f1ea65067145bac01a2e5  ./results/ref_fp32det-mtp1.json
b1a8337b377356970398a7be485fcf04  ./results/score_fp32det-ring.json
28abd425ee834c7903f9db409dabcbb8  ./running-config.txt
b7abfa7b2f4011bba266b66094ea99bd  ./source/qdq_4_rdna3.cuh
449cbcae85f5075723924b5090c4de60  ./source/q_gemm_rdna3_det.cu
9da6c97070b6f00e16f98e7e4f6f8613  ./tools/allreduce_bench.py
7973d62c574257adcfed948e85724998  ./tools/brain_bench.py
0ad47217283d8f5c714f4838e29be924  ./tools/compare_refs.py
657f1fe7468437f4abadac4d292623d0  ./tools/exp-fp32-tp8.sh
00bc234486691beb063c85185511eb29  ./tools/fp32_op_gate.py
33baed7649fbc82d7129258c20bf01a7  ./tools/make_patch_fp32.py
71f94f6310b623d05a555a057793e960  ./tools/ref_outputs.py
7187036a84b56d7dd343640247570a45  ./tools/score_texts.py
117c448bde58e0f589b85d61da4c696e  ./tools/spec_counts.py
a86325ef9aed9ba029c840b191b67950  ./tools/true_rate.py
