# Brain release rc3c, frozen 2026-10-01T20:41Z

rc3c = rc3b plus the 4-token 3D attention path from Itko's attn-4token (commit b21b119): the dispatch gate allows up to 4 query tokens and the segment buffer holds 4 x seq_threshold_3D rows (attn4/*.patch). 108 of 108 correctness cases, the 27B crash batch (128 sequences x 4 tokens) included, broken controls fail. Served with 3 MTP draft tokens. The service's image check expects the patched triton_attn.py.

Test windows 2026-10-01, rc3c, tok/s at 16K and 60K: 1 draft 112.0 and 107.5 (same step cost as rc3b); 2 drafts 141.6 and 137.2 (2.37 tokens per step); 3 drafts 157.0 and 161.7 (2.78 to 2.91 per step). Text against rc3b: 6 of 12, 5 of 12, 9 of 12 prompts diverge; largest 0.37, 0.32, 0.29; top-5 95.9, 95.6, 94.1 percent.
Before the attention fix, 3 drafts on rc3b gave 46.4 and 15.9 (results/ref_rc3b-k3-A.json is that run's text).
Production from 2026-10-01 19:35 UTC with 3 drafts: 150.0 and 144.0 on its first check (2.67 tokens per step). Real work: 159.3 tok/s exact on an Itko writing task, average context 26,813 tokens, 3.20 tokens per step, 73.3 percent of drafts accepted (results/itko_speed-rc3c-20261001-1942.log).
Task 01 (results/task01/): R1 8 of 10, failed traps 6 (card0 not excluded) and 9 (dpm star at line end, Mhz); the first answer of 31,407 tokens decoded at 160.6 tok/s. Link width reads 16 at the root port, correct since the switch.

Run: services/ (brain-service.sh starts localhost/n02-brain:rc3c with FENSTONE_RDNA_AR=1 and 3 draft tokens). Rollback to rc3b: ~/brain/pkg/*.rc3b. Live command and environment: running-config.txt. Earlier manifests: MANIFEST-rc3b.md and, inside it, rc3a's.

## Files and hashes
1cb5146b679095cad18c21726fd21619  ./allreduce/Containerfile.rc3a
bbf33ba30a5b45a94b22c49502270390  ./allreduce/cuda_communicator.py
79ede9325e9bd35798b7b3a9b018a21c  ./allreduce/patch_comm.py
7b0c17aad5798d9c77d24365b0652500  ./allreduce/rc3a-test.sh
338a041c0ac8b406529d2b8f8c0377c1  ./allreduce/rdna_ar8.so
d3f661e06778d595361eb743f98aebd6  ./allreduce/rdna_ar_test8.py
0b06c90ad213c51386f7548e72ce5e6e  ./allreduce/rdna_custom_all_reduce.py
cf5bed6cf529a04d555a58f4fe9e75c6  ./allreduce/rdna_custom_all_reduce_tp8.cu
cfe296fa8ae3d4520806db4caeba6b12  ./attn4/Containerfile.rc3c
9ae76fc3bd0d29c092c336ab6b4c1e44  ./attn4/itko-attn-4token-README.md
8ff083b9eadeca2ca087bbc1def8295f  ./attn4/promote_rc3c.py
864a70568c59083c2015736557d48bfc  ./attn4/rc3c-spec.sh
ee77710c95fbe35444790d9fab2c1ccc  ./attn4/spec-test.sh
7c060e77e6d368cb1fcf8c5bd383b428  ./attn4/triton_attn.patch
105986eb1d88abede7df3a46a82570fb  ./attn4/triton_attn.py
b6742548b2d3683fe76b567fe909f57d  ./attn4/triton_unified_attention.patch
c84f3ca9848b4fc7e590ca5d7900ec92  ./attn4/triton_unified_attention.py
d543b90d6ad7266db6744778a6edcba8  ./fen_fp32_op.so
075478942b7cfac52f1f1f2aecdfcd59  ./gates/attn_gate.py
e45730d5a477cd0e3a65b518ed907c2d  ./gates/attn_gate_results.json
5d325de15fa9b34c30f86b56bc17b5fb  ./gates/gate_det.py
a7c1d5baba4462c49339f353accae383  ./gates/gate_det_results.json
545306bac9ab9f8e8f400b66a7616d8b  ./gates/op_gate_results.json
3da81f3d5673eec64ca4e69add0207ac  ./gates/triton_attn_segments.patch
8cb62a1feba3b06a3ccf44b1e3ca5eb7  ./MANIFEST-rc1.md
ceb0e2f0421723cc7667625a235be9b9  ./MANIFEST-rc2.md
a6a386eb4ca4fb6a04b2ce523602a937  ./MANIFEST-rc3a.md
cae72fac020bba618c95a9fba58d40f8  ./MANIFEST-rc3b.md
42840ce685a5cbea7046abae6021f344  ./rdna3_w4a16_fp32.py
4f0e1f9e8045df67c1e3be3fd1cc22aa  ./results/allreduce_auto.json
271189b6e660b900061571c9f5112207  ./results/allreduce_ring.json
49c022bf06234b0377dbd3d935780fdd  ./results/allreduce_tree.json
ed6679b45bc4f7476c758868ec2fbd2c  ./results/bench_fp32det-mtp1.json
893ff165646c5dc97c8f6ef392e49ab0  ./results/bench_fp32det-ring.json
624ec9f95a06429dc4da9cbd0ea49e67  ./results/itko_speed-20260926-0905.log
6807a2118f8fc9413d48bd81e1592623  ./results/itko_speed-rc3b-20261001-1045.log
b774924884b51c317deab30de04a5ad5  ./results/itko_speed-rc3c-20261001-1942.log
c67b2626375223568c7736430e9fe5b0  ./results/rdna_ar_test.json
46783c6458c3dc45ae6c459cdf201acc  ./results/rdna_ar_test_w4.json
ef27fee0f7238d7b3b0a5749be547c51  ./results/rdna_ar_test_w8.json
c3d00913268f1ea65067145bac01a2e5  ./results/ref_fp32det-mtp1.json
16fe797dd90f7a20103f0799fedbedf1  ./results/ref_fp32det-mtp1-s128-B.json
fd12580be5fdf32ff5d9ebc4857b4a7e  ./results/ref_fp32det-mtp1-s128.json
de5c15c9ddf3ae10687c52044a2d1a7a  ./results/ref_rc3a-A.json
5a1721898e975731a54cfced4b487ada  ./results/ref_rc3b-A.json
88008c70e5f12b7911e357492c19d047  ./results/ref_rc3b-k3-A.json
57c1dc57142754734a2102708ed40cc8  ./results/ref_rc3c-k1-A.json
ab5d08ba52d77fefaae45da04b38f135  ./results/ref_rc3c-k2-A.json
e426abcae26520b2d8374b11720a820a  ./results/ref_rc3c-k3-A.json
b1a8337b377356970398a7be485fcf04  ./results/score_fp32det-ring.json
828c4232c393675e68086b0f067773d3  ./results/task01/block-01.py
f08f86f11d1a184b393aa6a95e2e40b3  ./results/task01/metrics.jsonl
94503c865fabfe71e85e6b4ae17b1f1c  ./results/task01/once-output.json
28abd425ee834c7903f9db409dabcbb8  ./running-config-rc1.txt
db51520b5c723139761a093932c26b3c  ./running-config-rc2.txt
21c9a478eb710d2fbda020aef2162c6a  ./running-config-rc3a.txt
9c66c238547af69f61430e424a9d9de1  ./running-config-rc3b.txt
f871ebea25dda4493362080b7c2cc804  ./running-config.txt
a7389c05f36e475c2266be7b25993014  ./services/brain-service.sh
e21ca4f3423486732d3f41319feaed77  ./services/brain-warmup.sh
459201755c76727b2bc627dec7b4e691  ./services/cards.conf
5e9c62ad7d794b8af9244e1ad693333f  ./services/itko-gateway.sh
102f88adec11712545aeeafa91f1b412  ./services/n02-brain.service
9b05583b087c1e06d9e1342725713fc1  ./services/n02-itko-gateway.service
c480873551b54feb6b6f660ddf82d56b  ./services/n02-sandbox.service
c54f7cfdd80562d42b6681abd3827c6c  ./services/promote_rc3a.py
65839b88faae9e1034dd15fa6b2f5cd0  ./services-rc3b/brain-service.sh
095c2e6b2c9b624ae98a54d3625dd35a  ./services-rc3b/n02-brain.service
4c9693a24ebb9e4fa633dc48fc48e43b  ./services/sandbox-service.sh
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
e26489388e790dc53969c1f8d92614db  ./tuned/Containerfile.rc3b
7cfa693db0a3483b6cf82921b827225c  ./tuned/fen_fp32_op.so
6be1d7fc2667151841567d585060e2c5  ./tuned/fp32_op_gate_tuned.py
171c427a60e8b312f90ab2a90b55be1e  ./tuned/op_gate_run.log
c3d3bbf647835e13e2e47b81c1edf354  ./tuned/promote_rc3b.py
b7abfa7b2f4011bba266b66094ea99bd  ./tuned/qdq_4_rdna3.cuh
74f2b097ec0796c37a7a2c8e2aa9fcb3  ./tuned/q_gemm_rdna3_tuned.cu
de7caeb851f66a805bb092d724b82599  ./tuned/rc3b-test.sh
