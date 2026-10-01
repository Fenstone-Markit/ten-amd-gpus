# Brain release candidate rc2, frozen 2026-09-26T11:02Z

rc2 = rc1 plus the Triton 3D decode attention at 128 segments instead of 16 (one line in triton_attn.py).
Status: release candidate. rc1's gates, plus the attention gate (attn_gate.py, preflight and full sweep, recommendation S=128: 60K decode+verify 13.85 -> 3.24 ms per token over 16 layers, error unchanged, bit-identical repeats).
Not yet proven: multi-hour use, concurrent requests, reboot and automatic start, fair precision test on neutral text.

Measured, one request, exact token rate: 16K 80.4 tok/s, 60K 76.4 tok/s (rc1: 69.8 and 65.0).
Itko's real work on rc1 (the before number): 68.0 tok/s decode at about 24K context, 85.1% speculative acceptance, 1.85 tokens per step.
Repeatability: the brain is not bit-repeatable run to run (same brain twice: 5 of 12 prompts diverge, largest logprob difference 0.20), from the installed WMMA prefill. S=16 vs S=128: 7 of 12, 0.70, top-5 agreement unchanged.

Knobs: as rc1 (see MANIFEST-rc1.md), plus NUM_PAR_SOFTMAX_SEGMENTS = 128 in the brain's triton_attn.py (file triton_attn_s128.py, applied inside the container) and compile cache /root/.cache/vllm-fp32-mtp-s128.
The exact live command and environment are in running-config.txt.

Watch for: runs of exclamation marks or text turning incoherent mid-answer (speculation plus GDN align-mode cache). Fallbacks: rc1 (put triton_attn_brain_31f3.py back, warm cache /root/.cache/vllm-fp32-mtp); or no speculation.

## Files and hashes
d543b90d6ad7266db6744778a6edcba8  ./fen_fp32_op.so
075478942b7cfac52f1f1f2aecdfcd59  ./gates/attn_gate.py
e45730d5a477cd0e3a65b518ed907c2d  ./gates/attn_gate_results.json
5d325de15fa9b34c30f86b56bc17b5fb  ./gates/gate_det.py
a7c1d5baba4462c49339f353accae383  ./gates/gate_det_results.json
545306bac9ab9f8e8f400b66a7616d8b  ./gates/op_gate_results.json
3da81f3d5673eec64ca4e69add0207ac  ./gates/triton_attn_segments.patch
8cb62a1feba3b06a3ccf44b1e3ca5eb7  ./MANIFEST-rc1.md
42840ce685a5cbea7046abae6021f344  ./rdna3_w4a16_fp32.py
4f0e1f9e8045df67c1e3be3fd1cc22aa  ./results/allreduce_auto.json
271189b6e660b900061571c9f5112207  ./results/allreduce_ring.json
49c022bf06234b0377dbd3d935780fdd  ./results/allreduce_tree.json
ed6679b45bc4f7476c758868ec2fbd2c  ./results/bench_fp32det-mtp1.json
893ff165646c5dc97c8f6ef392e49ab0  ./results/bench_fp32det-ring.json
624ec9f95a06429dc4da9cbd0ea49e67  ./results/itko_speed-20260926-0905.log
c67b2626375223568c7736430e9fe5b0  ./results/rdna_ar_test.json
c3d00913268f1ea65067145bac01a2e5  ./results/ref_fp32det-mtp1.json
16fe797dd90f7a20103f0799fedbedf1  ./results/ref_fp32det-mtp1-s128-B.json
fd12580be5fdf32ff5d9ebc4857b4a7e  ./results/ref_fp32det-mtp1-s128.json
b1a8337b377356970398a7be485fcf04  ./results/score_fp32det-ring.json
28abd425ee834c7903f9db409dabcbb8  ./running-config-rc1.txt
db51520b5c723139761a093932c26b3c  ./running-config.txt
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
6eea73a19eafef97f5aab74da9ab2621  ./triton_attn_s128.py
