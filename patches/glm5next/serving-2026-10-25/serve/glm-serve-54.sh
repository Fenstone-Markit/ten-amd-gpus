(
# glm-serve-54: start GLM-5.3-Flash on node02's TWELVE cards on port 8000 as models "glm", "brain" and "tutor",
# window 393,216 with MTP (two draft tokens per step), thinking level "low" unless a request asks for more,
# with port-26's patches and GLM5_PREWARM_SCAN=1 (the start-up fix, 8 of 8 clean vs 4 of 4 controls faulting).
# -08: MIOpen gets a writable database in the cache folder and FAST find mode. In -07's start MIOpen could not write
# /root/.config/miopen and the cards ran at about 3 kW for 2+ minutes after weight loading, where Itko's identical starts took 28 s.
# -09: before GLM counts as serving, long prompts (about 1.6k, 7k, 13k streamed with a tool, 27k tokens) run through it, so
# every long-prompt kernel loads during warm-up. -08 warmed with one 24-token question; Hermes's first 14,858-token prompt then
# loaded _cp_gather_indexer_quant_cache_kernel for the first time and a worker hit an illegal memory access 2 s later.
# Itko's clean port-26 runs had loaded it during warm-up. A workaround: the runtime root cause is port-27's.
# -10: empty caches every start, as Itko's tested runs did (order19.py empties /tmp/ov_cache and /tmp/ov_triton before each start).
# -05 to -09 reused ~/glm-serve/cache-port26 (1,544 Triton files by -09, written by runs that were killed mid-start); -08 then
# crashed and -09 hung on their first long prompt. Each warm-up request now times out after 300 s instead of 900 s.
# -11: warm-up starts with two tiny prompts (about 40 and 60 tokens), as Itko's clean runs did (24 and 26 tokens first). In -09 and -10
# the first request was 1,600 tokens: the 13 KDA kernels first loaded inside it and a worker faulted 3 s after the last load
# (-09: 'repeats can not be negative' on PP2; -10: illegal memory access on PP3_TP1). -10 had fresh caches, so the cache is ruled out.
# -14 (claude-canary-01): patches from ~/glm-serve/claude-patches-01 = port-26's 16 files with gpu_worker.py adding GLM5_CANARY:
# four marked 512 KiB blocks filling the first 2 MiB of the small pool right after the prewarm, checked at every step start.
# A hit logs the overwritten bytes. Same capped-clock reproducer as -11/-12 (both faulted on the second tiny prompt).
# -15 (claude-canary-02): the canary never writes to its blocks (-14's restore wrote into whatever shared the memory and
# raised 'Inplace update to inference tensor'); it is also checked before and after each start-up stage.
# -20 (hipmem-shim-02): -15 exactly, plus an LD_PRELOAD shim. When PyTorch's allocator (libc10_hip.so) releases a physical handle,
# the shim keeps it in a pool keyed by size and allocation properties (card id included) and hands it back on PyTorch's next
# hipMemCreate with the same key, so ROCm never frees and re-hands-out that memory. On one card this took the one-card test from
# 20 and 21 hits to 0 in 40 rounds with no out-of-memory; -15's canary was overwritten on the six middle-stage cards in every start.
# -21: -20 with a 1,024-token prefill chunk (BT=1024, was 2048). -20's start died in memory profiling: hipBLASLt ran out of memory
# on the six middle-stage cards (6 x hipblaslt.cpp:164) and a worker exited, because the memory PyTorch released stays in the shim's
# pool and libraries load their GPU code outside PyTorch (about 200 MiB for the first bf16 GEMMs, hipBLASLt or rocBLAS alike).
# A smaller chunk lowers PyTorch's profiling peak. Also: the wait stops as soon as an engine dies (-20 waited on a dead engine).
# -23: -20's settings again (window 131,072, chunk 2,048) with -21's dead-engine stop, and patches from ~/glm-serve/claude-patches-04
# = claude-patches-02 with gpu_worker.py starting glm5_memsnap (claude-memsnap-01) on rank 4 (PP2_TP0, a middle-stage worker):
# PyTorch's allocation history with Python stacks, a snapshot on every allocation retry and every 5 s, in $CACHE/memsnap.
# Purpose: see what makes the 24.4 GiB PyTorch peak on the middle-stage cards that -20 to -22 ran out of memory at.
# -24: -23 plus --language-model-only (text only; Hermes and the tutor send text). -23's memory snapshot on rank 4 (PP2_TP0) showed
# 18.4 GiB of weights and about 5 GiB more at the peak, nearly all from vLLM profiling the vision encoder with a dummy video:
# activation temporaries 2.9 GiB, one 1.16 GiB GEMM output, the patch-embedding conv 365 MiB, the dummy video 332 MiB, encoder 289 MiB.
# -25: -24 with claude-patches-05: gpu_worker.py memsnap-02 puts @torch.inference_mode() and @with_gpu_sync_check back on
# execute_model. canary-01/-02, noempty-01 and memsnap-01 had inserted _glm5_canary_check between those decorators and execute_model,
# so execute_model ran outside inference mode; -24 served its first request into 'Inplace update to inference tensor outside
# InferenceMode' on PP1 and hung (also in -14 and -17). Decorators checked method by method against port-26's gpu_worker.py.
# The memory recorder stays in the tree but is off (no GLM5_MEMSNAP_RANK).
# -26: -25 with the shim gate counting matches, not lines: -25 came up clean (prewarm 10/10, canary armed 10/10, no hits) and was
# stopped by 'fewer than 10 processes report the shim loaded' because workers' start lines merge in the log when written at once.
# -27: -26 with claude-patches-06 = claude-patches-05 plus claude-ecount-02: a read-only count of which routed experts GLM's
# routing picks, per MoE layer (GLM5_EXPERT_COUNT=1, counters saved to $CACHE/expert-count on trigger files; the counting rule
# matched vLLM's topk_sigmoid kernel on 20,000 of 20,000 test tokens). Also three fixes from the checked soak reviews: the
# warm-up must be 6 of 6 correct, a canary hit stops GLM, and an unreadable card temperature counts as hot, not 0 C.
# -28: -27 on twelve cards: Frankenstein's G11 and G12 (RX 7900 XTX) join node02's ten. Six pipeline stages of two (PP6 x TP2),
# layers 10,7,7,7,7,7: by the checkpoint's bytes (stage-bytes-01.py) the heaviest card holds about 14.6 GiB of weights against 17.0
# on ten cards (-27's middle stages, 18.4 GiB measured), so every card has about 2 to 3 GiB more room. All INT4 (wtdcode), same
# patches and gates as -27, every per-card gate now expects 12. The expert counter is off (GLM5_EXPERT_COUNT=0) until
# claude-ecount-03 adds its NaN guard. Refuses unless node02 shows exactly twelve cards including G11 and G12.
# -29: G11 and G12 sit in the board's CPU slots 4 and 5, not behind the PCIe switch, so bus order could pair each with a switch card
# and every tensor-parallel exchange (two per layer per token) would cross switch, CPU and slot. The card order is now set
# explicitly (ROCR_VISIBLE_DEVICES by unique id, as glm-fp8-test-07/08): G11 and G12 first, one pair (stage 1, layers 0-9),
# then the ten switch cards in bus order, so they keep -27's pairs. The order is printed and kept in $OUT/card-order.txt.
# -30: the heat stop is 107 C junction (was 100 C), the operator's choice: the cards throttle themselves at 110 C.
# -31: layers 10,7,7,8,8,5 (was 10,7,7,7,7,7). -30 died at KV-cache set-up on stage 4 (layers 24-30): 'Some layers are not correctly
# initialized'. GLM's KV layout (kv_cache_utils, glm5n branch) gives a stage one cache tensor per full-attention layer, each shared
# with at most one layer of each of the 4 linear-attention (KDA) groups (KDA layers in order, k-th -> group k % 4). Stage 24-30 has one
# full-attention layer (27) and six KDA layers, two of them in the same groups, so two got no cache. Every stage of 10,7,7,8,8,5
# passes that rule (pp-split-01.py checked all 6-stage splits); heaviest card about 15.0 GiB of weights (two stages), 17.0 on ten cards.
# -32: PCIe link watch. -31's run hung node02 at about 19:53 UTC: the BMC logged 127 PCI parity errors on G12's slot-3 root port
# (40:03.3) and 22 on G12's own port, starting at 19:34 while idle, the last at 19:52:59; the switch cards logged none. Now: before
# the start, 30 s idle must add no PCIe error on any card's chain; during start-up and warm-up any non-fatal or fatal error, or 10+
# new corrected errors on one device, stops GLM (read every 3 s with the heat check, from sysfs AER counters; $OUT/pcie.log).
# -33: our FP8 kernels in serving. The mixed checkpoint ~/models/mixed/glm53-mixed-e3 (wtdcode's AWQ 4-bit files linked, plus the
# official block-FP8 routed experts of every third MoE layer) with GLM5_FP8_LAYERS=6,12,15,18,21,27,36,42: in those 8 layers the
# routed experts run on libfp8rdna3.so (our decode kernel up to 32 tokens, our WMMA GEMM for prompts; fp8_rdna3 v04); every other
# layer is unchanged (AWQ 4-bit on the RDNA3 HIP kernel). 8 layers = one or two per stage, about 1.64 GiB more per card each, so the
# fullest card should land near 21 GiB of 24. Patches: claude-patches-08 = -06 + the 4 FP8 files + model.py skipping the 4-bit expert
# tensors of FP8 layers (patches-08-build-01.sh). Gates added: the log must show our FP8 MoE method on all 8 layers x 2 cards (16
# lines), and every card's memory is printed per stage after start-up. Everything else as -32.
# -34: claude-patches-09 (patches-09-build-01.sh): -33 died loading weights with KeyError 'layers.3.mlp.experts.routed_experts.w2_weight'.
# The mixed checkpoint holds FP8 experts for 14 layers, the start enabled 8, and layer 3 (4-bit method) received its FP8 tensors.
# model.py now also skips FP8 expert tensors of layers not in GLM5_FP8_LAYERS (only when that is set). Tested on 2 cards, 8 layers,
# FP8 on layer 6 with layer 3 left on 4-bit (glm-fp8-test-10). Nothing else changed.
# -35: the FP8 gate fixed. -34 came up clean (weights 31 s, KV 462,723 tokens, no PCIe errors, canary clean) and its gate stopped it:
# it expected one set-up line per FP8 layer per card (16), but fp8_rdna3 prints it once per worker process. -35 requires the line
# from each of the 12 workers by name (PP0..PP5 x TP0/TP1); every stage holds at least one FP8 layer in this list. Not shown by
# that line: that each FP8 layer received its FP8 experts (vLLM 0.27 does not catch unfilled parameters of post-processed quantised
# modules). The proof here is the warm-up, 6 of 6 exact answers; a per-layer checksum line is owed in the next patch set.
# -36 (step B): window 393,216 (was 131,072) and GLM5_WS_FIT=1. KV cache unchanged (1 GiB per card = 462,723 tokens), so one
# request can use up to 393,216; a second request at the same time can still run, and vLLM pauses one if the cache fills.
# Read before changing: the sparse indexer reserves a K-gather workspace of 40 x window x 132 bytes per card at start
# (indexer.py get_max_prefill_buffer_size): 692 MB at 131,072, 2.08 GB at 393,216, which would not fit (PP3 and PP4 had
# 0.58 GiB measured free in -35). GLM5_WS_FIT=1 (the port-13 switch) sizes it to window x 132 bytes = 52 MB: one prompt's
# context per chunk is never more than the window. Other buffers that grow with the window: decode indexer scores
# (heads x 2 x window x 4 B, about 0.1 GB) and block tables (small); MLA's prefill workspace is capped at 65,536 tokens.
# Expected: about 0.5 GB per card more room than -35. Untested before this start: the window and WS_FIT on real weights at
# this length (port-14 ran WS_FIT on dummy weights at 65,536). Check after start: ~/ringer/lc-check-01.py.
# -37 (step C, MTP): GLM's own draft layer (layer 45, "nextn") guesses the next token; the full model checks the guess in the
# same pass, so a right guess gives two tokens for about the price of one. Changes from -36:
#  - checkpoint ~/models/mixed/glm53-mixed-e4 (= e3 plus layer 45's routed experts in block FP8 from the official checkpoint;
#    e3 had them in BF16, 14.5 GB, too big for the last stage). 45 is added to the FP8 layer list, so the draft layer runs on
#    our FP8 kernels too.
#  - FP8 layers 3,6,12,18,27,36,42,45: one or two per stage for the new split (same count of 8 target layers as before).
#  - split 8,8,8,8,8,5 (was 10,7,7,8,8,5). The draft layer is one more attention layer, so the glm5n KV layout makes 3 KDA groups
#    (34 KDA over 12 attention layers) instead of 4, and 10,7,7,8,8,5 then breaks the layout rule; 8,8,8,8,8,5 passes
#    (pp-split-mtp-01.py, all 6-stage splits checked).
#  - --speculative-config method mtp, 1 draft token; GLM5_NEXT_MTP_PROPOSER=1 picks the overlay's GLM proposer (per-group draft
#    KV plumbing; the stock EAGLE proposer fails on this model per the overlay README). vLLM puts the drafter on the last stage.
#  - claude-patches-10 = -09 plus the overlay's mtp.py declaring SupportsPP (37-01 stopped: 'Pipeline parallelism is not
#    supported for this model'; the drafter is built only on the last stage; Glm4MoeLiteMTP in the image does the same).
#  - --no-async-scheduling: 37-03 loaded, profiled and set up its KV cache, then died on the first decode step with
#    'PP+async expects sampled_token_ids to have shape [num_reqs, 1]': vLLM 0.27's pipeline + async path sends one token per
#    request between stages; with a draft token a step can accept two. Plain (non-async) scheduling sends tokens via the scheduler.
#  - claude-patches-10 build-05: one batch in flight with a draft model (37-04/05/06 hung: the engine fetched the draft before
#    recording the sampled token, so stages disagreed on a step). Stack dumps every 90 s stay on (GLM5_TB_SIGNAL=1).
#  - cudagraph sizes [2,4]: with one draft token every decode step is 2 tokens per request (1 or 2 requests).
# Untested before this start: MTP with pipeline parallelism on this model, the drafter on the FP8 method, the new split.
# -38: the serving start chosen on 2026-10-25 from the glm-serve-37 tests: window 393,216, MTP with 2 draft tokens, async
# scheduling off, claude-patches-10 build-10. Debug switches off (no stack dumps every 90 s, no step timing, no profiler).
# Measured on -37 with these settings (37-20): warm-up 6 of 6; probe count 50.2 / code 40.2 / prose 32.9 tok/s (glm-serve-36
# about 26.7 on all three); long-window check 3 of 3 at 152k/248k/384k tokens, decode 28.5 / 30.8 / 33.1 tok/s, first token
# 50 / 93 / 174 s. Full ringer on 37-14 (1 draft): tool 20/20, code 11/12, needles 15/16, maths 297/300.
# -54: TEST (vision). -51 with claude-patches-18 (= patches-17 + claude-vis-01, ~/fp8-rdna3-01/vision-stage1-01-edit.py) and a
#   switch VIS=1 (default 0 = text only exactly as -51): drops --language-model-only, allows 1 image and no video per prompt
#   (--limit-mm-per-prompt), caps images at VISTOK tokens (default 1024; the processor's own cap is 8000), and sets
#   GLM5_VISION_FIRST_STAGE_ONLY=1 so only stage 1 holds the 1.05 GiB tower. Untested as of 2026-10-25.
# -51: PRODUCTION candidate, with a long-read gate after the warm-up (step 4b below). -50 (claude-patches-16) with both prompt-reading fixes on by default: MHCT=1000000 (fused mHC
#   kernels for every step) and MQA=1 (Triton indexer logits). Everything else as -46 (3x4, MTP 2, caching on, BT 2048).
#   Tested on -50: cold reads 39k/74k/116k 11.1/21.1/32.7 s (-46: 15.3/31.2/54.0); ringer 20/20 12/12 15/16 296/300; probe
#   count/code/prose 60.2/50.0/41.4 tok/s (-46 same session: 57.1/48.5/38.6); needle-set-01 see the 2026-10-25 work log.
# -50: -49 with claude-patches-16 = patches-15 plus claude-mqa-01 (~/fp8-rdna3-01/mqa-triton-01-edit.py): switch MQA=1
#   (default 0) = GLM5_MQA_TRITON=1, the sparse indexer's prompt logits on a fused Triton kernel (glm5_mqa_logits_rdna3.py)
#   instead of the torch fallback. mqa-test-01/02 on one card: 5x to 38x faster, matches exact fp32 maths (top-512 100%;
#   the torch fallback 99.81%, it rounds scores to bf16). 25 patch files.
# -49 (tested): MHCT=1000000 read 39k/74k/116k in 12.5/26.1/45.6 s (-46: 15.3/31.2/54.0); ringer 20/20 11/12 14/16 297/300.
# -49: -48 with claude-patches-15 = patches-14 plus mhc.py claude-mhc-01 (~/fp8-rdna3-01/mhc-prefill-01-edit.py): switch
#   MHCT=<tokens> (default 64 = as before) = GLM5_MHC_FUSED_MAX_T, the largest step that uses the fused mHC Triton kernels.
#   claude-mt-01 put the torch mHC path at 18-21% of prompt time. Test start.
# -48: -47 with claude-patches-14 = patches-12 plus gpu_worker.py claude-mt-01 (~/fp8-rdna3-01/module-timing-01-edit.py):
#   switch MT=1 (default 0) = GLM5_MODULE_TIMING=1, GPU time per layer part during prompt steps, written per stage to
#   module-timing/mt-pp<N>.txt in the run cache. Test start only.
# -47 (tested 2026-10-25): BT 4608 read 39k/74k/116k in 16.2/31.1/53.8 s vs 15.3/31.2/54.0 at 2048, peak VRAM +1.4 GiB: dropped.
# -47: -46 with BT a test switch (default 2048 as before) = --max-num-batched-tokens, the largest prompt chunk per step.
#   With caching on, chunks are whole 1,152-token blocks: BT 2048 gives 1,152-token chunks, 2304 gives 2,304, 4608 gives 4,608.
# -46: PRODUCTION from 2026-10-25 21:00 UTC. -44 (claude-patches-12) with prefix caching ON by default (PFX=1); every
#   other default as -40 (3x4, split 16,16,13, MTP 2, window 393,216, KV 1.8 GiB per card). Debug switches stay, default off.
#   Tests: clean ringer with caching (-41) 20/20 12/12 15/16 295/300; prefix-stress 41 of 42 (one neighbour-record miss);
#   -44 mixed-check-02 5/5; -44 ringer under a constant counting load 20/20 11/12 15/16 298/300, 32 counts all exact.
#   (-45 / claude-patches-13 = the FAIR switch, not used: Itko is the only user.)
# -44: -43 with claude-patches-12 = patches-11 plus rocm_aiter_mla_sparse.py claude-mixspec-01 (~/fp8-rdna3-01/mixed-spec-01-edit.py):
#   with MTP, a step mixing one request's verify rows with another's prompt chunk now takes the ragged Triton sparse-MLA call
#   like every other shape, not the torch fallback's per-token Python loop (43 STT: stage 1 2.2 s per 2048-token mixed step
#   vs 0.6 s alone; a 39k prompt beside a generating request 130 s vs 15 s). GLM5_SPEC_MIXED_TORCH=1 = old behaviour.
# -43: -42 with claude-patches-11 = patches-10 plus gpu_worker.py GLM5_STAGE_TIMING (claude-stt-01, built by
#   ~/fp8-rdna3-01/stage-timing-01-edit.py; off unless STT=1): each stage writes its per-call times to the run cache
#   stage-timing/st-pp<N>.txt. Test switch STT=1 (default 0); everything else as -42.
# -42: -41 plus ITER=1 (test switch) = --enable-logging-iteration-details: the engine logs every step (prompt and generated
#   tokens in it, and the time it took), to see why a long prompt read alongside a generating request runs about 8x slower.
# -41: -40 plus PFX=1 (test switch) = --enable-prefix-caching. vLLM keeps prefix caching opt-in for hybrid models; with it on,
#   the KDA (linear-attention) state cache runs in 'align' mode (states kept at block boundaries) and the kpool tail opts out
#   by design (kv_cache_interface.py). The upstream RDNA4 port serves with --enable-prefix-caching. Untested here with PP,
#   our glm5n KV grouping patch and MTP together; check with ~/ringer/prefix-check-01.py.
# -40: the production start chosen by the 2026-10-25 layout bake-off: PP3xTP4 (split 16,16,13), MTP 2 drafts, window
# 393,216 with a 1.8 GiB KV cache per card (442,368 tokens), FP8 layers 3,6,12,18,27,36,42 + draft 45. Defaults below; the
# switches stay for tests. Bake-off (131k window): 3x4 probe 59.6/49.5/41.8 tok/s, ringer 36.8/34.1/33.8, 20/20 11/12 15/16
# 297/300; 6x2 49.3/41.2/35.5, 30.7/27.5/26.8; 5x2 (no MTP) 26.4/26.5/26.6, 27.0/23.6/23.5.
# -39: layout test start. Same as -38 (patches-10 build-10, checkpoint e4, WS_FIT, shim, gates) with the layout as switches:
#   TP, PP, PART (VLLM_PP_LAYER_PARTITION), CARDS (12 = G11+G12 first then the 10 switch cards; 10 = the switch cards only),
#   F8L (FP8 expert layers; every stage must hold one, the gate checks each worker), WIN, SPEC, ASYNC, KVB,
#   ROCPROF (empty = off; "DELAY:SECONDS" runs vLLM under rocprofv3 kernel trace for that window, output in the run's cache).
# Layouts tested 2026-10-25: 6x2 (= -38), 3x4 split 16,16,13 (MTP keeps 3 KDA groups; passes the KV rule), 5x2 on the switch
# cards split 10,9,9,9,8 without MTP (with MTP the KV rule forces stages in multiples of 4 and the last stage would not fit).
# node02. Changes: starts one container (n02-glm), logs under ~/glm-serve. Touches nothing of Itko's or Hermes.
N=n02-glm; PORT=8000; IMG=localhost/n02-sandbox:hermes-10
SHIM=~/glm-serve/vmm-repro-tool/probe/hipmem-shim-02.so
W=~/sandbox/workspace; OV=$W/glm5next-port-01/overlay; PT=~/glm-serve/claude-patches-18
SNAPREL=mixed/glm53-mixed-e4
TP=${TP:-4}; PP=${PP:-3}; PART=${PART:-16,16,13}; CARDS=${CARDS:-12}; NW=$((TP*PP)); ROCPROF=${ROCPROF:-}
F8L=${F8L:-3,6,12,18,27,36,42,45}; F8N=$(echo $F8L | tr , '\n' | wc -l); FLIB=~/fp8-rdna3-01/libfp8rdna3.so
G=~/glm-serve; OUT=$G/run-$(date -u +%Y%m%d-%H%M%S); CACHE=$OUT/cache
WIN=${WIN:-393216}; KVB=${KVB:-1932735283}; BT=${BT:-2048}
# Test switches (defaults = the MTP start): SPEC=0 drops the draft model (same checkpoint, split and patches) to measure what
# MTP itself costs; ASYNC=1 keeps vLLM's async scheduling on (only valid with SPEC=0 until the pipeline+async MTP path exists).
SPEC=${SPEC:-2}; ASYNC=${ASYNC:-0}; PFX=${PFX:-1}; PFXOPT=""; [ "$PFX" = 1 ] && PFXOPT="--enable-prefix-caching"
STT=${STT:-0}; STTENV=""; [ "$STT" = 1 ] && STTENV="-e GLM5_STAGE_TIMING=1"; MHCT=${MHCT:-1000000}; STTENV="$STTENV -e GLM5_MHC_FUSED_MAX_T=$MHCT"; MQA=${MQA:-1}; STTENV="$STTENV -e GLM5_MQA_TRITON=$MQA"; VIS=${VIS:-0}; VISTOK=${VISTOK:-1024}; if [ "$VIS" = 1 ]; then LMO="--limit-mm-per-prompt '{\"image\":1,\"video\":0}' --mm-processor-kwargs '{\"max_image_tokens\":$VISTOK}'"; STTENV="$STTENV -e GLM5_VISION_FIRST_STAGE_ONLY=1"; else LMO=--language-model-only; fi; MT=${MT:-0}; [ "$MT" = 1 ] && STTENV="$STTENV -e GLM5_MODULE_TIMING=1"; ITER=${ITER:-0}; ITEROPT=""; [ "$ITER" = 1 ] && ITEROPT="--enable-logging-iteration-details"
if [ "$SPEC" = 0 ]; then SPECOPT=""; CGS="1,2"; else SPECOPT="--speculative-config='{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC}'"; CGS="$((SPEC+1)),$((2*SPEC+2))"; fi
if [ "$ASYNC" = 1 ]; then ASYNCOPT=""; else ASYNCOPT="--no-async-scheduling"; fi
PROFCMD=""; [ -n "$ROCPROF" ] && PROFCMD="/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/bin/rocprofv3 --kernel-trace --stats --collection-period $ROCPROF -d /cache/rocprof -o %pid% --output-format csv --"
echo "test switches: SPEC=$SPEC ASYNC=$ASYNC PFX=$PFX ITER=$ITER STT=$STT MT=$MT VIS=$VIS VISTOK=$VISTOK MHCT=$MHCT MQA=$MQA BT=$BT (speculative option: ${SPECOPT:-none}; cudagraph sizes $CGS)"
no() { echo "REFUSED: $*; nothing started"; exit 1; }
# ---- PCIe link watch (glm-serve-32) ----
# Every card's chain: the GPU function and each device above it that keeps AER counters (the card's own ports, the switch
# or root port), read from sysfs (no sudo). A non-fatal or fatal count rising stops GLM; so do 10 or more new corrected errors
# on one device. Before the start, 30 s idle must add no error at all (on 2026-10-24 G12's link erred while idle, then hung node02).
SYSP=${SYSP:-/sys/bus/pci/devices}
pcie_devs() {
  for d in $SYSP/*; do
    [ "$(cat $d/vendor 2>/dev/null)" = 0x1002 ] && [ "$(cat $d/device 2>/dev/null)" = 0x744c ] || continue
    p=$(readlink -f $d)
    while [ -n "$p" ] && [ "$p" != / ]; do
      case $(basename $p) in pci[0-9]*) break;; esac
      [ -f $p/aer_dev_correctable ] && echo $p
      p=$(dirname $p)
    done
  done | sort -u
}
pcie_read() {
  for p in $PDEVS; do
    c=$(sed -n 's/^TOTAL_ERR_COR //p' $p/aer_dev_correctable 2>/dev/null); nf=$(sed -n 's/^TOTAL_ERR_NONFATAL //p' $p/aer_dev_nonfatal 2>/dev/null); f=$(sed -n 's/^TOTAL_ERR_FATAL //p' $p/aer_dev_fatal 2>/dev/null)
    echo "$(basename $p) ${c:-0} ${nf:-0} ${f:-0}"
  done
}
# pcie_diff BASE NOW -> one line per device whose counts rose: "dev +cor +nonfatal +fatal"
pcie_diff() { awk 'NR==FNR{c[$1]=$2;n[$1]=$3;f[$1]=$4;next} {dc=$2-c[$1]; dn=$3-n[$1]; df=$4-f[$1]; if(dc>0||dn>0||df>0) printf "%s +%d +%d +%d\n",$1,dc,dn,df}' <(echo "$1") <(echo "$2"); }
pcie_bad() { echo "$1" | awk '$3!="+0"||$4!="+0"||substr($2,2)+0>=10' | head -5; }
linkcheck() {
  local now d bad
  now=$(pcie_read); d=$(pcie_diff "$PBASE" "$now")
  [ -n "$d" ] && [ "$d" != "$PLAST" ] && { echo "$(date -u +%H:%M:%S) $(echo $d)" >> $OUT/pcie.log; PLAST="$d"; }
  bad=$(pcie_bad "$d")
  if [ -n "$bad" ]; then
    [ -n "$WP" ] && kill $WP 2>/dev/null
    echo "STOPPED: PCIe link errors since the start (device +corrected +non-fatal +fatal):"; echo "$bad" | sed 's/^/  /'
    echo "  all rises so far:"; echo "$d" | sed 's/^/    /'
    podman logs $N > $OUT/server.log 2>&1; podman rm -f $N >/dev/null 2>&1
    echo "GLM stopped and removed; PCIe history in $OUT/pcie.log, server log in $OUT/server.log"; exit 1
  fi
}
# ---- end PCIe link watch ----

echo "######## 1. checks (read-only)"
[ -f ~/models/$SNAPREL/config.json ] || no "the mixed checkpoint is not under ~/models/$SNAPREL"
[ "$(md5sum < $FLIB | cut -c1-12)" = 8461c0fceb57 ] || no "$FLIB is not the tested kernel library (md5 8461c0fceb57)"
for L in ${F8L//,/ }; do [ -f ~/models/$SNAPREL/fp8-experts-layer$(printf %02d $L).safetensors ] || no "no FP8 experts for layer $L in the mixed checkpoint"; done
[ -d $OV/app/vllm/vllm/models/glm5next ] || no "the overlay is not at $OV"
podman image exists $IMG || no "image $IMG is missing"
[ "$(md5sum < $SHIM 2>/dev/null | cut -c1-32)" = f2ac26766046cf4cc26912713c0eea5e ] || no "the shim at $SHIM is missing or not the one tested"
podman container exists $N && no "a container named $N exists (GLM may already be serving; podman rm -f $N clears it)"
cat > /tmp/glm-serve-34.md5 <<'EOF'
6d7480f352ca9f0e56295fca16b79140  vllm/compilation/breakable_cudagraph.py
7771dd069b343f0fba3410a19a50db35  vllm/model_executor/layers/mhc.py
55a99c3bf42467ee39bbcdc1db188d70  vllm/model_executor/layers/mla.py
cc9fa47c910497bd3e6504caa9b56a97  vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
427a60b200ef2c192bd00968cd66386a  vllm/model_executor/layers/quantization/fp8.py
22df21a6d0ab282df2ddfb556408167b  vllm/model_executor/layers/quantization/fp8_rdna3.py
a36da34ea8e7230bc546b8b0a1293ac1  vllm/model_executor/layers/quantization/glm5_fp8_rdna3.py
6ca7e8cef09f34dc6bee6ce2aa6c5fd5  vllm/model_executor/layers/sparse_attn_indexer_kpool.py
c6c7a41463c0ecf6277edb99a49f4fce  vllm/models/glm5next/nvidia/model.py
43f254ea6cc1cdbc497b4bcf4d10b9e4  vllm/platforms/rocm.py
a409f32989342ec8505f0e03fde4427f  vllm/third_party/flash_linear_attention/ops/index.py
e34c083fb75c7a6100d290c9939a73bb  vllm/v1/attention/backends/mla/indexer.py
f66b272c98ab388d8b3b2fe2cd15d622  vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
d12343153d948749c04232ff48d26bd8  vllm/v1/attention/backends/registry.py
2e8e9c9ffb3cfade9a7db5b6e70dbbce  vllm/v1/attention/ops/glm5_meta_watch.py
6cd1504b94245a0fa273b2d65e164ee8  vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
da41a6062905bdc449a7f7704099e60e  vllm/v1/attention/ops/glm5_mqa_logits_rdna3.py
b95956c3c6a0450a8d4da9ad352aac4f  vllm/v1/core/kv_cache_utils.py
db867e968036b981365fb4c3f9494c6f  vllm/v1/core/sched/scheduler.py
4f98d5b104b339508ba7f88b17dcaeca  vllm/v1/worker/glm5_ecount.py
32ddc3f705e3b546adb1c903a8cf0136  vllm/v1/worker/glm5_memsnap.py
a557cd5d10463ad1fc4d33c3208b10fa  vllm/v1/worker/gpu_model_runner.py
afc8e8520b0da998d57ed47d59b926c4  vllm/v1/worker/gpu_worker.py
5f200b7e1dbf533b5e7973cfca3b1536  vllm/models/glm5next/nvidia/mtp.py
c18f165d77a194cdffe0a13e54ccc2f7  vllm/v1/engine/core.py
EOF
(cd $PT && md5sum -c --quiet /tmp/glm-serve-34.md5) || no "claude-patches-18 files are not the expected ones (module-timing-02-edit.py + vision-stage1-01-edit.py + patches-10-build-10.sh + stage-timing-01-edit.py + mixed-spec-01-edit.py + module-timing-01-edit.py + mhc-prefill-01-edit.py + mqa-triton-01-edit.py)"
[ "$(find $PT/vllm -name '*.py' | wc -l)" = 25 ] || no "claude-patches-10 has $(find $PT/vllm -name '*.py' | wc -l) patch files, not 22 (patches-06's 18 + 4 FP8 files)"
T=""; for d in $W/glm5next-port-[0-9]*/; do d=$(basename $d); case $d in *-aborted*|*-start[0-9]*) continue;; esac; [ -n "$(git -C $W ls-files -- $d | head -1)" ] || T="$T $d"; done
[ -z "$T" ] || no "Itko task folder(s) with nothing committed:$T, so a task may be running"
curl -s -m 3 -o /dev/null localhost:$PORT/health && no "something already answers on port $PORT"
curl -s -m 3 -o /dev/null localhost:8020/health && no "something answers on port 8020"
[ "$(systemctl --user is-active n02-brain.service)" = active ] && no "n02-brain.service is active on node02"
NC=0; BUSY=""
for d in /sys/bus/pci/devices/*/unique_id; do D=$(dirname $d); [ -f $D/mem_info_vram_used ] || continue; NC=$((NC+1)); U=$(( $(cat $D/mem_info_vram_used 2>/dev/null || echo 0)/1048576 )); [ $U -lt 1024 ] || BUSY="$BUSY $(cut -c1-16 $d)=${U}MiB"; done
[ $NC = 12 ] || no "node02 shows $NC cards, not 12 (are G11 and G12 in? n02-wake, then run this again)"
for g in b2b6d6c8246dfb99 be6dc925c0f4732f; do grep -qx $g /sys/bus/pci/devices/*/unique_id 2>/dev/null || no "card $g (G11 or G12) is not in node02"; done
[ -z "$BUSY" ] || no "these cards already hold memory:$BUSY"
ORDER="GPU-b2b6d6c8246dfb99,GPU-be6dc925c0f4732f"; NSW=0
for D in $(ls -d /sys/bus/pci/devices/* | sort); do [ -f $D/mem_info_vram_used ] || continue; U=$(cat $D/unique_id)
  case $U in b2b6d6c8246dfb99|be6dc925c0f4732f) continue;; esac; ORDER="$ORDER,GPU-$U"; NSW=$((NSW+1)); done
[ $NSW = 10 ] || no "found $NSW cards besides G11 and G12, not 10"
[ $CARDS = 10 ] && ORDER=${ORDER#GPU-b2b6d6c8246dfb99,GPU-be6dc925c0f4732f,}
[ $(echo $ORDER | tr , '\n' | wc -l) = $NW ] || no "layout TP=$TP x PP=$PP needs $NW cards, CARDS=$CARDS gives $(echo $ORDER | tr , '\n' | wc -l)"
[ $(echo $PART | tr , '\n' | wc -l) = $PP ] || no "PART=$PART does not have $PP stages"
[ $(echo $PART | tr , '\n' | paste -sd+ | bc) = 45 ] || no "PART=$PART does not add up to 45 layers"
echo "layout: TP=$TP PP=$PP partition $PART on $NW cards; FP8 layers $F8L"
echo "checks passed: 12 cards free (G11 and G12 among them), ports 8000 and 8020 quiet, no Itko task, patches as audited"

echo "######## 2. starting $N (window $WIN, KV cache $KVB bytes per card, two requests at a time); fresh empty caches in $CACHE"
[ -e $CACHE ] && no "$CACHE already exists"
mkdir -p $OUT $CACHE/vllm $CACHE/triton $CACHE/miopen $CACHE/memsnap $CACHE/expert-count
echo "$ORDER" | tr ',' '\n' | awk -v tp=$TP '{printf "device %d stage %d: %s\n", NR-1, int((NR-1)/tp)+1, $0}' | tee $OUT/card-order.txt
PDEVS=$(pcie_devs); NPD=$(echo $PDEVS | wc -w)
[ $NPD -ge 24 ] || no "the PCIe watch found only $NPD devices with error counters on the cards' chains (expected at least 2 per card)"
PBASE=$(pcie_read); echo "PCIe watch: $NPD devices on the cards' chains; 30 s idle check (no error may appear)"; sleep 30
D0=$(pcie_diff "$PBASE" "$(pcie_read)"); [ -z "$D0" ] || { echo "$D0" > $OUT/pcie-idle.txt; no "PCIe errors while idle in 30 s (device +corrected +non-fatal +fatal): $(echo $D0)"; }
PBASE=$(pcie_read); PLAST=; echo "idle check clean"
podman run -d --name $N --restart=no --entrypoint bash --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ulimit memlock=-1:-1 --security-opt seccomp=unconfined --pids-limit=-1 --ipc=host --network=host \
  -e ROCR_VISIBLE_DEVICES=$ORDER -e HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NW-1))) \
  -e VLLM_ROCM_USE_AITER=0 -e GPU_MAX_HW_QUEUES=1 -e HSA_ENABLE_INTERRUPT=1 -e HSA_ENABLE_MWAITX=1 -e TRUST_REMOTE_CODE=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 -e VLLM_CLAV_CONV1D=1 -e OMP_NUM_THREADS=8 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_RPC_TIMEOUT=3600000 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 -e VLLM_PP_LAYER_PARTITION=$PART \
  -e GLM5_DSA_BREAK=indexer -e GLM5_PROFILE_ALWAYS=1 -e GLM5_KDA_NOCACHE=1 -e GLM5_PREWARM_SCAN=1 -e GLM5_CANARY=1 -e LD_PRELOAD=/shim/hipmem-shim-02.so -e SHIM_MODE=recycle -e GLM5_EXPERT_COUNT=0 -e GLM5_WS_FIT=1 -e GLM5_NEXT_MTP_PROPOSER=1 -e GLM5_FP8_LAYERS=$F8L -e GLM5_FP8_LIB=/fp8/libfp8rdna3.so -e GLM5_ECOUNT_DIR=/cache/expert-count $STTENV \
  -e PYTHONPATH=/tmp/ov -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton \
  -e MIOPEN_USER_DB_PATH=/cache/miopen -e MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen -e MIOPEN_FIND_MODE=FAST \
  -v $OV:/overlay:ro -v $PT:/patches:ro -v $HOME/models:/models:ro -v $CACHE:/cache -v $SHIM:/shim/hipmem-shim-02.so:ro -v $FLIB:/fp8/libfp8rdna3.so:ro \
  $IMG -c "
S=/opt/python/lib/python3.14/site-packages/vllm; B=/tmp/ov/vllm
[ -d \$S ] || { echo 'no vllm at '\$S; exit 1; }
mkdir -p /tmp/ov && cp -a \$S/. \$B/ && cp -a /overlay/app/vllm/vllm/. \$B/ && (cd /patches/vllm && find . -name '*.py' -exec cp --parents {} \$B/ \;)
{ (cd /overlay/app/vllm/vllm && find . -name '*.py' -exec dirname {} \;); (cd /patches/vllm && find . -name '*.py' -exec dirname {} \;); } | sort -u | while read -r d; do rm -rf \$B/\$d/__pycache__; done
echo \"[glm-serve-27] tree built: \$(find /patches/vllm -name '*.py' | wc -l) patched files over the overlay\"
cd /tmp/ov && exec $PROFCMD python3 -m vllm.entrypoints.cli.main serve /models/$SNAPREL --served-model-name glm brain tutor --host 0.0.0.0 --port $PORT \
  --trust-remote-code $LMO --tensor-parallel-size $TP --pipeline-parallel-size $PP --distributed-executor-backend mp \
  --gpu-memory-utilization 0.97 --max-num-seqs 2 --max-model-len $WIN --max-num-batched-tokens $BT \
  --kv-cache-memory-bytes $KVB --kv-cache-dtype auto \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --default-chat-template-kwargs '{\"reasoning_effort\":\"low\"}' \
  $ASYNCOPT $SPECOPT $PFXOPT $ITEROPT \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$CGS]}'
" > /dev/null || no "podman could not start $N"

echo "######## 3. waiting for it (up to 45 minutes); every card checked every 3 s, stop at 107 C; log in $OUT/thermal.log"
cardline() { for d in /sys/bus/pci/devices/*/unique_id; do D=$(dirname $d); [ -f $D/mem_info_vram_used ] || continue; U=$(cat $d); case $U in d580007fa41ec467) n=G7;; fec7cca764f40561) n=G9;; b2b6d6c8246dfb99) n=G11;; be6dc925c0f4732f) n=G12;; *) n=${U:0:12};; esac; H=$(ls -d $D/hwmon/hwmon* | head -1); j=$(( $(cat $H/temp2_input 2>/dev/null || echo 999000)/1000 )); f=$(cat $H/fan1_input 2>/dev/null || echo -); p=$(( $(cat $H/power1_average 2>/dev/null || echo 0)/1000000 )); printf "%s j=%s fan=%s p=%s | " $n $j $f $p; done; echo; }
heat() {
  L="$(date -u +%H:%M:%S) $(cardline)"; echo "$L" >> $OUT/thermal.log
  HOT=0; HOTN=-; for x in $(echo "$L" | grep -oE '[^ |]+ j=[0-9]+' | tr ' ' '_'); do t=${x##*j=}; [ $t -gt $HOT ] && { HOT=$t; HOTN=${x%%_j=*}; }; done
  if [ $HOT -ge 107 ]; then
    [ -n "$WP" ] && kill $WP 2>/dev/null
    echo "STOPPED: $HOTN reached ${HOT} C. Its last 10 readings (junction, fan rpm, power W):"
    tail -10 $OUT/thermal.log | while read -r ln; do echo "  ${ln%% *}  $(echo "$ln" | grep -oE "$HOTN j=[0-9]+ fan=[^ ]+ p=[0-9]+")"; done
    echo "every card at the stop:"; tail -1 $OUT/thermal.log | cut -c1-400
    podman logs $N > $OUT/server.log 2>&1; podman rm -f $N >/dev/null 2>&1
    echo "MIOpen 'Unable to create' warnings in this start: $(grep -c 'MIOpen: Warning.*Unable to create' $OUT/server.log); last step logged: $(grep -oE 'Loading weights took|memory profile ran|Graph capturing finished|init engine' $OUT/server.log | tail -1)"; exit 1
  fi
  linkcheck
}
shimline() { echo "shim: $(grep -oE 'hipmem-shim-02\] pid [0-9]+ start' $OUT/server.log | sort -u | wc -l) processes loaded it; $(grep -c 'REAL FUNCTIONS MISSING' $OUT/server.log) missing real functions; pooled reuses (largest count per process): $(grep -oE 'hipmem-shim-02\] pid [0-9]+ [a-z]+.*from pool [0-9]+' $OUT/server.log | awk '{p=$3; n=$NF+0; if (n>m[p]) m[p]=n} END {for (p in m) printf "%s ", m[p]}')"; }
UP=no; HOT=0; HOTN=-; WP=
for i in $(seq 900); do
  sleep 3; heat
  [ "$(podman inspect --format '{{.State.Running}}' $N 2>/dev/null)" = true ] || break
  curl -sf -m 3 -o /dev/null localhost:$PORT/health && { UP=yes; break; }
  [ $((i % 5)) = 0 ] && podman logs --tail 120 $N 2>&1 | grep -qE 'died unexpectedly|EngineDeadError|RuntimeError: cancelled' && { echo "  an engine died; stopping the wait"; break; }
  [ $((i % 40)) = 0 ] && echo "  $(date -u +%H:%MZ) still starting, hottest card $HOTN ${HOT} C"
done
podman logs $N > $OUT/server.log 2>&1
if [ $UP != yes ]; then
  echo "memory snapshots on rank 4: $(ls $CACHE/memsnap/*.pickle 2>/dev/null | wc -l); recorder started: $(grep -c 'GLM5_MEMSNAP\] recording' $OUT/server.log)"
  echo "######## GLM DID NOT COME UP. hipBLASLt out-of-memory lines: $(grep -c 'hipblaslt.cpp:164' $OUT/server.log); canary hits: $(grep -c 'GLM5_CANARY\] HIT' $OUT/server.log). First errors:"; grep -nE 'out of memory|OutOfMemory|died unexpectedly|Error' $OUT/server.log | grep -v shm_broadcast | head -6 | cut -c1-200; shimline; echo "The last lines:"; grep -v '^\s*$' $OUT/server.log | tail -12 | cut -c1-200
  podman rm -f $N >/dev/null 2>&1; echo "container removed; log in $OUT/server.log"; exit 1
fi
echo "up after about $((i*3)) s; hottest card during start-up: $(grep -oE '[^ |]+ j=[0-9]+' $OUT/thermal.log | awk -F'j=' '$2+0 > m {m = $2+0; l = $0} END {print l}')"
echo "hipBLASLt out-of-memory lines: $(grep -c 'hipblaslt.cpp:164' $OUT/server.log) (-20: 6, then a worker died)"
echo "MIOpen 'Unable to create' warnings: $(grep -c 'MIOpen: Warning.*Unable to create' $OUT/server.log) (0 expected); waits over 60 s: $(grep -c 'No available shared memory broadcast' $OUT/server.log)"
grep -oE "Loading weights took [0-9.]+ seconds|init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s" $OUT/server.log | sort | uniq -c | head -2
grep -E "GPU KV cache size|Maximum concurrency" $OUT/server.log | sed -E 's/^.*\] //' | sort | uniq -c | head -3
PW=$(grep -c "GLM5_PREWARM_SCAN\] scan kernels loaded" $OUT/server.log); echo "prewarm lines: $PW of $NW"
[ "$PW" = $NW ] || { echo "WARNING: the prewarm did not run on every worker; stopping GLM to be safe"; podman rm -f $N >/dev/null; exit 1; }
echo "canary armed on: $(grep -c 'GLM5_CANARY\] armed' $OUT/server.log) of $NW workers; first: $(grep -m1 -oE 'GLM5_CANARY\] armed.*' $OUT/server.log | cut -c1-230)"
echo "canary hits so far: $(grep -c 'GLM5_CANARY\] HIT' $OUT/server.log) (-15: on the six middle-stage cards in every start)"
[ "$(grep -c 'GLM5_CANARY\] HIT' $OUT/server.log)" = 0 ] || { echo "WARNING: canary hits during start-up; stopping GLM"; podman rm -f $N >/dev/null; exit 1; }
shimline
[ "$(grep -oE 'hipmem-shim-02\] pid [0-9]+ start \(real functions found\)' $OUT/server.log | sort -u | wc -l)" -ge $NW ] || { echo "WARNING: fewer than 12 processes report the shim loaded; stopping GLM"; podman rm -f $N >/dev/null; exit 1; }
grep -qi "illegal memory access" $OUT/server.log && { echo "WARNING: an illegal memory access is in the log; stopping GLM"; podman rm -f $N >/dev/null; exit 1; }

F8W=$(grep -oE "\(Worker_PP[0-9]+_TP[0-9]+ pid=[0-9]+\)[^(]*\[claude-fp8-rdna3-01\] FP8 MoE on RDNA3 kernels" $OUT/server.log | grep -oE "Worker_PP[0-9]+_TP[0-9]+" | sort -u | wc -l)
echo "our FP8 MoE kernels set up on $F8W of $NW workers (one line per worker; $F8N FP8 layers: $F8L); first: $(grep -m1 -oE '\[claude-fp8-rdna3-01\] FP8 MoE on RDNA3 kernels.*' $OUT/server.log | cut -c1-140)"
[ "$F8W" = $NW ] || { echo "WARNING: not every worker set up our FP8 kernels; stopping GLM"; podman rm -f $N >/dev/null; exit 1; }
echo "memory per card after start-up (device, stage, card, used of total GiB):"
k=0; for g in ${ORDER//,/ }; do u=${g#GPU-}; D=$(dirname $(grep -lx $u /sys/bus/pci/devices/*/unique_id)); printf "  device %2d stage %d  %s  %5.1f of %4.1f\n" $k $((k / TP + 1)) $u $(echo "$(cat $D/mem_info_vram_used)/1073741824" | bc -l) $(echo "$(cat $D/mem_info_vram_total)/1073741824" | bc -l); k=$((k + 1)); done | tee $OUT/vram-after-start.txt
echo "######## 4. warm-up: two tiny prompts, then four long ones so every long-prompt kernel loads now, not under real traffic (heat stop still on)"
cat > $OUT/warmup.py <<'PYEOF'
# glm-warmup (in glm-serve-09): warm GLM's long-prompt kernels before it serves. Prompts of about 1.8k, 8k, 15k (streamed, with a tool,
# like Hermes) and 29k tokens, each a "find the code" question with a known answer. Writes one line per request.
import json, re, sys, time, urllib.request
BASE = sys.argv[1]; OUT = sys.argv[2]
def lines(n):
    return "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1))
def ask(n, target, stream=False, tools=False, maxtok=1024):
    msg = lines(n) + f"\n\nWhat code does gate {target} use? Answer with the code only, for example K123."
    body = {"model": "glm", "messages": [{"role": "user", "content": msg}], "max_tokens": maxtok, "temperature": 0}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace",
                          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    if stream:
        body["stream"] = True; body["stream_options"] = {"include_usage": True}
    want = f"K{(target * 7919) % 9973}"
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    content, ptok, done = "", None, False
    with urllib.request.urlopen(req, timeout=300) as r:
        if stream:
            for raw in r:
                s = raw.decode().strip()
                if not s.startswith("data:"): continue
                s = s[5:].strip()
                if s == "[DONE]": done = True; break
                d = json.loads(s)
                if d.get("usage"): ptok = d["usage"].get("prompt_tokens")
                for c in d.get("choices", []):
                    content += (c.get("delta", {}).get("content") or "")
        else:
            d = json.loads(r.read()); done = True
            content = d["choices"][0]["message"].get("content") or ""; ptok = d["usage"]["prompt_tokens"]
    ok = done and re.search(rf'\b{want}\b', content) is not None
    return f"{'PASS' if ok else 'WRONG'} prompt {ptok} tokens{' streamed+tool' if stream else ''} in {time.time() - t0:.0f} s: want {want}, got {content.strip()[:40]!r}"
plan = [(3, 2, False, False), (5, 4, False, False), (130, 77, False, False), (570, 401, False, False), (1060, 733, True, True), (2070, 1500, False, False)]
with open(OUT, "w") as f:
    for n, t, s, tl in plan:
        try: line = ask(n, t, s, tl)
        except Exception as e: line = f"ERROR at {n} records: {repr(e)[:200]}"
        print(line, flush=True); f.write(line + "\n"); f.flush()
        if line.startswith("ERROR"): break
    f.write("WARMUP END\n")
PYEOF
python3 $OUT/warmup.py http://127.0.0.1:$PORT $OUT/warmup.txt > /dev/null 2>&1 &
WP=$!
while kill -0 $WP 2>/dev/null; do
  sleep 3; heat
  [ "$(podman inspect --format '{{.State.Running}}' $N 2>/dev/null)" = true ] || { kill $WP 2>/dev/null; break; }
done
wait $WP 2>/dev/null; WP=
cat $OUT/warmup.txt 2>/dev/null | cut -c1-160
podman logs $N > $OUT/server.log 2>&1
echo "######## canary hits: $(grep -c 'GLM5_CANARY\] HIT' $OUT/server.log)"; shimline; grep -oE 'GLM5_CANARY\] HIT.*' $OUT/server.log | head -6 | cut -c1-260
if [ "$(podman inspect --format '{{.State.Running}}' $N 2>/dev/null)" != true ]; then
  echo "######## GLM DIED DURING WARM-UP (nothing else was using it). Fault lines:"
  grep -nE "illegal memory access|Memory access fault|died unexpectedly|EngineDeadError" $OUT/server.log | head -5 | cut -c1-200
  echo "the container is left (exited) for its logs; full log: $OUT/server.log. Remove it with: podman rm n02-glm"; exit 1
fi
grep -qi "illegal memory access" $OUT/server.log && { echo "WARNING: an illegal memory access is in the log; stopping GLM"; podman rm -f $N >/dev/null; exit 1; }
echo "warm-up answers correct: $(grep -c '^PASS' $OUT/warmup.txt 2>/dev/null) of 6"
[ "$(grep -c '^PASS' $OUT/warmup.txt 2>/dev/null)" = 6 ] || { echo "WARNING: the warm-up was not 6 of 6 correct; stopping GLM"; podman rm -f $N >/dev/null; exit 1; }
[ "$(grep -c 'GLM5_CANARY\] HIT' $OUT/server.log)" = 0 ] || { echo "WARNING: canary hits during the warm-up; stopping GLM"; podman rm -f $N >/dev/null; exit 1; }
echo "expert-count layers set up: $(grep -c 'claude-ecount' $OUT/server.log) log lines; counts go to $CACHE/expert-count"
grep -q '^ERROR' $OUT/warmup.txt && { echo "WARNING: a warm-up request failed (above); stopping GLM"; podman rm -f $N >/dev/null; exit 1; }
######## [glm-serve-51] long-read gate. On 2026-10-25 the FIRST long cold read after a fresh start (74k tokens) came back
# garbage twice in about ten starts (empty answers; with prefix caching the bad read is stored and every follow-up on it fails).
# Later reads on the same server were always right. Cause not found yet. So before GLM counts as serving: cold reads of about
# 74k and 116k tokens under a private cache salt (never reused by real traffic), answers checked; up to 3 attempts.
echo "######## 4b. long-read gate: 74k and 116k cold reads, answers checked (up to 3 attempts)"
cat > $OUT/longread.py <<'PYEOF'
import json, sys, time, urllib.request
BASE, OUT, A = sys.argv[1], sys.argv[2], sys.argv[3]
recs = lambda n: "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1))
res = []
for n, g in [(4500, 4001), (6900, 6789)]:
    body = {"model": "glm", "messages": [{"role": "user", "content": recs(n) + f"\n\nWhat code does gate {g} use? Answer with the code only, for example K123."}],
            "max_tokens": 64, "temperature": 0, "cache_salt": f"glm-serve-longread-{time.time_ns()}-{A}"}
    t = time.time()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read())
        c = r["choices"][0]["message"].get("content") or ""; want = f"K{(g * 7919) % 9973}"
        res.append(f"{'PASS' if want in c else 'WRONG'} attempt {A}: {r['usage']['prompt_tokens']} tokens in {time.time() - t:.0f} s, want {want}, got {c.strip()[:30]!r}")
    except Exception as e:
        res.append(f"ERROR attempt {A}: {repr(e)[:160]}")
open(OUT, "a").write("\n".join(res) + "\n")
PYEOF
LRP=0
for A in 1 2 3; do
  python3 $OUT/longread.py http://127.0.0.1:$PORT $OUT/longread.txt $A > /dev/null 2>&1 &
  WP=$!
  while kill -0 $WP 2>/dev/null; do sleep 3; heat; [ "$(podman inspect --format '{{.State.Running}}' $N 2>/dev/null)" = true ] || { kill $WP 2>/dev/null; break; }; done
  wait $WP 2>/dev/null; WP=
  [ "$(podman inspect --format '{{.State.Running}}' $N 2>/dev/null)" = true ] || { echo "######## GLM DIED DURING THE LONG-READ GATE; log: $OUT/server.log"; podman logs $N > $OUT/server.log 2>&1; exit 1; }
  if [ "$(grep -c "^PASS attempt $A:" $OUT/longread.txt)" = 2 ]; then LRP=$A; break; fi
done
cat $OUT/longread.txt | cut -c1-160
[ "$LRP" = 0 ] && { echo "WARNING: the long-read gate failed 3 attempts; stopping GLM"; podman logs $N > $OUT/server.log 2>&1; podman rm -f $N >/dev/null; exit 1; }
[ "$LRP" = 1 ] || echo "NOTE: the first long read(s) after start were WRONG and later ones right (the 2026-10-25 start-up fault); serving after attempt $LRP"
podman logs $N > $OUT/server.log 2>&1
echo "long-prompt kernels loaded during warm-up:"
for k in _cp_gather_indexer_quant_cache_kernel BuildPrefillChunkMetadataKernel; do
  T=$(grep "compilation during inference: $k" $OUT/server.log | head -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1); echo "  $k: ${T:-NOT SEEN}"; done
echo "kernel first-loads logged since start: $(grep -c 'compilation during inference' $OUT/server.log)"
curl -s -m 10 localhost:$PORT/v1/models | python3 -c "import json,sys; print('models:', ' '.join(m['id'] for m in json.load(sys.stdin)['data']))" || echo "could not list models"
echo "######## GLM IS SERVING: node02 port $PORT, models glm/brain/tutor, window $WIN. Logs: $OUT"
echo "Itko's gate will now refuse GPU tasks (port 8000 answers). This script does not change where Hermes or the tutor point."
echo "To stop GLM: podman rm -f $N"
)
