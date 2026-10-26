# Chapter 9: What we were given, and what we built

*GLM-5.3-Flash on twelve RX 7900 XTX: our own FP8 kernels, MTP across pipeline stages, prefix caching, and three slow paths found and fixed. Measured from 2026-10-24 to 2026-10-26.*

GLM-5.3-Flash is Z.ai's 320-billion-parameter mixture-of-experts model, released on 2026-08-26, with about 18 billion parameters active per token ([model card](https://huggingface.co/zai-org/GLM-5.3-Flash)). Its 45 layers are 34 linear-attention layers (KDA) and 11 sparse-attention layers (DSA, where a learned indexer picks the 2,048 earlier tokens each token looks at), with four residual streams mixed between layers (mHC) and one extra layer that drafts the next token (MTP). Chapters 5 to 8 got it serving on ten cards. This chapter is about making it the model my agent actually works with: faster to answer, able to hold long conversations, and checked at every step for whether it still answers right.

It is also the chapter where I set down plainly what we were given and what we had to build. Every claim about what exists elsewhere below was checked online on 2026-10-25, with links; where I found nothing, it says "found no public report", not "the first".

The work in this chapter was written and run on the machine by Claude, the assistant that writes my agent's briefs, over SSH, mostly overnight. I chose the order of work, set the rules it worked under, corrected it when it went the wrong way (the layout, below), and did the hardware.

## Findings

1. **Follow-up turns start in 1.4 seconds instead of 18.** My agent resends its whole conversation every turn. With prefix caching on, a follow-up at 48,000 tokens of history gets its first word in 1.4 s; before, every turn re-read everything. Answers from the cache matched answers read fresh.
2. **New text is read 1.6 times faster than this morning.** Cold reads of 39k / 74k / 116k tokens take 11.1 / 21.1 / 32.7 s, about 3,550 tokens per second at every length (before: 15.3 / 31.2 / 54.0 s). The gain came from finding three places where prompt-sized steps fell back to slow code left in the port, not from any setting.
3. **Our own FP8 kernels serve.** RDNA3 has no FP8 matrix instructions and AMD's GEMM library offers no FP8 for it. Seven of the 42 expert layers and the draft layer run from the official FP8 checkpoint on kernels written for this card. In decode they cost the same per layer as the 4-bit ones.
4. **MTP works across pipeline stages, and two draft tokens pay.** Making it work took five fixes in how vLLM 0.27 passes speculative results between stages. One of them was a real correctness bug: the earlier stages' linear-attention layers never learned how many drafts were accepted. Decode on the probe: 59.5 / 46.9 / 41.0 tok/s for counting, code and prose.
5. **Four cards per stage beat two.** A bake-off of 3x4, 6x2 and 5x2 made 3x4 (three pipeline stages of four cards) the layout: about 25 % faster decode than 6x2, at the cost of slower long-prompt reading. 5x2 cannot hold the draft layer at all.
6. **A faster kernel turned out to be the more accurate one.** Our Triton kernel for the indexer's prompt scores picks the same top 512 tokens as exact maths 100 % of the time; the path it replaced picked 99.81 %, because it rounded scores to 16 bits.
7. **Accuracy held.** The full test set after every change above: 20/20 tool calls, 12/12 code tasks, 14/16 needles (both misses a neighbouring record), 298/300 maths.
8. **Vision fits.** Built on the first stage only, the image encoder costs about 0.4 GiB on four cards and nothing on the tightest ones; it read every test image right.
9. **One fault is guarded, not explained.** Three times a long read came back wrong or empty, twice as the first long read after a start. With prefix caching, a bad read is reused by every question that follows it. Every start now checks two long reads before it serves.

## What was available, and what we built

### What the vendors and upstream provide

| Piece | What exists publicly (checked 2026-10-25) | Source |
| --- | --- | --- |
| vLLM's GLM-5.3-Flash recipe | AMD section covers gfx950 (MI350X, MI355X) only, through AITER (`VLLM_ROCM_USE_AITER=1`, the AITER sparse-MLA backend); MTP validated on 4x MI355X | [recipes.vllm.ai](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) |
| AMD's ROCm vLLM guide | The sparse-attention backend for DSA models (DeepSeek-V3.2, GLM-5): "Instinct MI300X/MI325X/MI350X/MI355X only" | [ROCm docs](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html) |
| AITER on RDNA | README lists gfx1100 and gfx1201 as "Experimental"; vLLM's own gate turns AITER off on every RDNA card (`get_cdna_version() > 2`), with a separate Triton-only path for gfx12, not gfx11 | [ROCm/aiter](https://github.com/ROCm/aiter), [vLLM `_aiter_ops.py`](https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py) |
| GLM-5.3-Flash in vLLM main | Present, but its sparse indexer raises "only supported on AITER" on ROCm, so it cannot start on gfx1100 | [vLLM `sparse_indexer.py`](https://github.com/vllm-project/vllm/blob/main/vllm/models/glm5next/amd/sparse_indexer.py) |
| Sparse MLA on gfx11 | An AMD engineer's PR routes DSA models to a Triton path on gfx11; open, not merged, tested on Strix Halo (gfx1151) | [vLLM #59628](https://github.com/vllm-project/vllm/pull/59628) |
| vLLM on the RX 7900 XTX | Listed as supported in the install docs; nothing about FP8 or attention backends for it | [vLLM docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) |
| FP8 matrix instructions | RDNA3's WMMA covers f16, bf16, i8 and i4; the FP8 WMMA builtins are gfx12 only | [GPUOpen](https://gpuopen.com/learn/wmma_on_rdna3/), [LLVM](https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td) |
| FP8 GEMM library | hipBLASLt's FP8 types are for gfx942, gfx950 and gfx12; gfx1100 is not listed | [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.2.4/reference/data-type-support.html) |
| MTP with pipeline stages | Supported upstream on vLLM's newer (V2) model runner since 2026-09-11; the V1 runner is reported broken for it; my image (vLLM 0.27) runs V1 | [vLLM #46994](https://github.com/vllm-project/vllm/pull/46994), [#49355](https://github.com/vllm-project/vllm/issues/49355) |
| Prefix caching for hybrid models | Supported ("align" mode: linear-attention state kept at block boundaries) | [vLLM docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.md) |

### What the community had done

| Who | What | Hardware |
| --- | --- | --- |
| djdeniro, [GLM-5.3-Flash-rocm-r9700](https://huggingface.co/djdeniro/GLM-5.3-Flash-rocm-r9700) (MIT) | A vLLM overlay that runs the model without AITER: a no-AITER sparse-attention path, the indexer's pooled-key op on ROCm, fused mHC kernels, an MTP proposer; reports 34-39 tok/s, 82-88 with MTP | 8x Radeon AI PRO R9700 (RDNA4), one tensor-parallel group of 8 |
| OpenMOSE, [Strata-GLM-AMD](https://github.com/OpenMOSE/Strata-GLM-AMD) | Its own HIP engine for this model on RDNA3, with the indexer, KDA, mHC and an MTP draft, 3-bit GGUF, 30-34 tok/s | 2x Radeon Pro W7900 (gfx1100) |
| mw00, [project-maya](https://github.com/mw00/project-maya) | An engine rewritten from Strata; a user report of 53-58 tok/s at 2-bit on two RX 7900 XTX plus an integrated Radeon 8060S ([#458](https://github.com/mw00/project-maya/issues/458)) | RX 7900 XT/XTX, marked experimental |
| llama.cpp | GLM-5.3-Flash in master, with the indexer, KDA, mHC and the MTP draft ([glm5-next.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/glm5-next.cpp)) | any; on these cards it read prompts about 15x slower than vLLM (Chapter 5) |

So this model already runs on RDNA3 elsewhere, on one or two cards with low-bit weights and engines written for it. What I found no public report of, as of 2026-10-25, is GLM-5.3-Flash under vLLM on RDNA3, on more than a couple of cards, with tensor and pipeline parallelism across twelve, FP8 experts, MTP and prefix caching together. Everything below is what it took to get there from the pieces above.

### What we built, layer by layer

| Layer | Given | Built here (chapter) |
| --- | --- | --- |
| Image | AMD's `rocm/vllm` 0.27.0 (vLLM 0.27.1, ROCm 10) | one class name so any compressed-tensors 4-bit MoE loads on gfx1100 (5) |
| Model port | djdeniro's RDNA4 overlay (72 files) | 14 files to start it on gfx1100 with pipeline stages: backend names, a decorator, a scheduler from the wrong vLLM, the multi-stream residual across stages, the cache grouped per stage, a capture-illegal copy, a wrong-output kernel switched off (5) |
| Long windows | | the linear-attention chunk-count fault behind every window above 32,768 (5) |
| Memory | | the ROCm overwrite narrowed to a one-card test, a library that keeps released memory in the process, 5 GiB of vision profiling removed (6, 8) |
| FP8 experts | an FP8 checkpoint from Z.ai, no FP8 compute on the card | decode GEMV and MoE kernels, a WMMA prefill GEMM, the vLLM method, a checkpoint that mixes 4-bit and FP8 layers (8, 9) |
| Twelve cards | | the layer-split rule GLM's cache layout imposes, a PCIe link watch after a slot hung the machine (9) |
| MTP | the overlay's proposer, vLLM 0.27's V1 runner | five fixes for pipeline stages, one of them a correctness bug (9) |
| Prefix caching | upstream's align mode | validated under stress and mixed load (9) |
| Prompt speed | the overlay's fallbacks for shapes it did not expect | a dispatch gate for mixed MTP steps, the fused mHC kernels for prompt steps, a new Triton kernel for the indexer's prompt scores (9) |
| Vision | the overlay's vision tower | built on the first stage only (9) |
| Guards | | warm-up answers checked, a memory canary, the PCIe watch, a 107 C stop, a long-read gate (6 to 9) |

### Where each part of the model runs, and whose code it is

| Part | What runs on these cards | Came from | Our part |
| --- | --- | --- | --- |
| Experts, 35 of 42 layers | RDNA3 4-bit HIP kernel (`CompressedTensorsWNA16RDNA3MoEMethod`) | AMD's vLLM image | the load fix; timing |
| Experts, 7 layers and the draft layer | `libfp8rdna3`: GEMV and MoE for decode, WMMA GEMM for prompts | us | all of it |
| Sparse attention | ragged Triton sparse-MLA kernel | the overlay (from the official ROCm GLM image, its notes say) | a capture fix; the gate that sends mixed MTP steps to it |
| Indexer scores, prompts | fused Triton kernel `glm5_mqa_logits_rdna3` | us | all of it (the overlay used a torch fallback) |
| Indexer scores, decode; top-k; pooled keys | Triton and vLLM ops | the overlay and vLLM | the indexer moved outside the decode graph; its workspace sized to the window |
| Linear attention (KDA) | flash-linear-attention Triton kernels | vLLM, via the overlay | chunk indices built on the host (the long-window fault) |
| Residual mixing (mHC) | fused Triton kernels | the overlay | the image's tilelang kernel switched off (wrong output on gfx1100); the fused kernels used for prompts too |
| All-reduce | RCCL, direct peer-to-peer | AMD | measured |
| Draft step | vLLM spec decode and the overlay's proposer | vLLM and the overlay | five pipeline fixes |

Counting files: the serving patch set is 25 files over the overlay. Of those, the FP8 path (four files and the kernel library), the indexer kernel, and the MTP, mixed-step and diagnostic changes are new this fortnight. Every change sits behind a switch, off unless the start sets it.

### The flags and switches that had to be set

| Setting | Value here | Why |
| --- | --- | --- |
| `--tensor-parallel-size`, `--pipeline-parallel-size`, `VLLM_PP_LAYER_PARTITION` | 4, 3, 16,16,13 | the bake-off; the split must satisfy the cache-layout rule below |
| `--max-model-len`, `--kv-cache-memory-bytes` | 393,216; 1.8 GiB per card | one long conversation; a fixed cache, since vLLM's estimate leaves none |
| `GLM5_WS_FIT=1` | on | the indexer's gather workspace was 40 x window x 132 bytes per card; sized to one window it fell from 2.08 GB to 52 MB at 393k |
| `--speculative-config` | MTP, 2 draft tokens | 3 drafts were slower |
| `--no-async-scheduling` | on | with MTP, async scheduling worked after the fixes but was slower here (26 against 31 tok/s) |
| `--enable-prefix-caching` | on | off by default for hybrid models |
| `--language-model-only` | on (off with images on) | the vision encoder's dummy-video profiling took 5 GiB on middle cards |
| `--compilation-config` | `FULL_DECODE_ONLY`, capture sizes 3 and 6 | graphs for decode steps with drafts |
| `--max-num-batched-tokens` | 2,048 | 4,608 gave no speed and 1.4 GiB more peak memory |
| `--max-num-seqs`, `--gpu-memory-utilization` | 2, 0.97 | one agent, plus room for a second request |
| `--default-chat-template-kwargs` | `reasoning_effort` low | the model ignores `enable_thinking` |
| `VLLM_ROCM_USE_AITER` | 0 | AITER does not run here |
| `GLM5_DSA_BREAK=indexer`, `GLM5_PROFILE_ALWAYS=1`, `GLM5_KDA_NOCACHE=1` | on | Chapter 5's three: decode graphs, the fixed cache, the long-window fault |
| `GLM5_MHC_FUSED_MAX_T` | 1,000,000 | fused mHC kernels for prompt steps too (the overlay stopped at 64 tokens) |
| `GLM5_MQA_TRITON=1` | on | our indexer kernel |
| `GLM5_FP8_LAYERS`, `GLM5_FP8_LIB` | 3,6,12,18,27,36,42,45 | which experts run on our FP8 kernels |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `LD_PRELOAD` of the shim | on | Chapter 6 |
| `MIOPEN_USER_DB_PATH`, `MIOPEN_FIND_MODE=FAST` | a writable cache | Chapter 7's three kilowatts |
| `ROCR_VISIBLE_DEVICES` | cards by unique id, in stage order | bus numbers changed twice in one evening |

## Our FP8 kernels

The official checkpoint stores the experts in FP8 with 128 x 128 block scales. The cards cannot multiply FP8, so the kernels convert it on the way in: an e4m3 byte becomes a 16-bit float with two masks and a shift per pair, exact for all 256 values, with the exponent offset folded into the scale. Decode is a matrix-vector product limited by memory: our GEMV reached 898 GB/s on an 18,432 x 7,168 layer (94 % of the card's 960 GB/s), and the MoE decode kernel 845 GB/s at GLM's shape, about 2x the same kernels with 16-bit weights. Prompts use the RDNA3 matrix instructions (WMMA) on the converted values: about 51 TFLOP/s on a 64 x 4,096 x 4,096 tile. Measured against real layer-3 experts, decode and prefill agree with a reference to 2.3e-3 and 4.0e-3.

In vLLM they sit behind an FP8 method for this card, and a mixed checkpoint links the 4-bit files and adds FP8 experts layer by layer, so the memory each card has decides how many layers can be FP8. On twelve cards that is 7 of 42 plus the draft layer. Timed layer by layer in decode, an FP8 expert layer costs the same as a 4-bit one, about 1.3 ms per step: the FP8 layers cost memory, not speed. Whether they buy accuracy is the A/B still owed.

## Twelve cards and the layout

Two cards from the second machine joined the ten on 2026-10-24, on a bifurcation adapter beside the switch. The first slot I used threw parity errors while idle and hung the machine; the second has run clean since. Every start now watches every link's error counters and refuses if a link errs while idle.

GLM's cache layout has a rule that pipeline splits must respect: each stage gets one cache tensor per sparse-attention layer, shared with at most one linear-attention layer from each group. With the draft layer counted, the groups go from 4 to 3, and only some splits are legal. Claude first chose 6 stages of 2 cards, because it fit; I had asked for tensor-parallel groups of 2 or 4, and said spreading the model over every stage was bound to be slow. The bake-off:

| Layout | Decode probe (count / code / prose) | Ringer decode (short / 8k / 32k) | Long prompts | Fits MTP |
| --- | --- | --- | --- | --- |
| 3 stages x 4 cards | 59.6 / 49.5 / 41.8 | 36.8 / 34.1 / 33.8 | 1.5x slower than 6x2 | yes |
| 6 x 2 | 49.3 / 41.2 / 35.5 | 30.7 / 27.5 / 26.8 | fastest | yes |
| 5 x 2 (ten cards) | 26.4 / 26.5 / 26.6 | 27.0 / 23.6 / 23.5 | | no |

3x4 serves. Its long-prompt reading was the weak side, which is what the prompt fixes below went after.

## MTP across pipeline stages

The draft layer guesses the next tokens and the model checks them in the same step, so an accepted guess is a token for free. The draft layer was not quantised in the 4-bit checkpoint (14.5 GB of 16-bit experts), so it was rebuilt from the official FP8 checkpoint (6.75 GiB) and runs on our kernels. Then, one start at a time:

| Start | What stopped it | Fix |
| --- | --- | --- |
| 1 | "Pipeline parallelism is not supported for this model" | the draft class declares it, as vLLM's own GLM-4 draft class does |
| 2 | the earlier stages died: "no attribute 'drafter'" | six places that use the drafter check that their stage has one |
| 3 to 6 | a hang on the first decode steps | stack dumps every 90 s showed it: the engine fetched a step's draft before recording the step's token, so the stages disagreed. One step at a time while decoding |
| 7 | 11 to 18 % of drafts accepted | the drafter never received the word embedding: vLLM shares it only without pipeline stages, and GLM's draft layer has no copy |
| 11 | async output garbled | earlier stages never trimmed the scheduler's optimistic placeholders |

And one that no start stopped on: vLLM 0.27 sent the earlier stages only one sampled token per request, so their linear-attention layers always assumed one accepted token and kept the wrong state whenever a draft was accepted. The last stage now sends every stage the next token, the accepted count and the drafts. Upstream vLLM has MTP under pipeline stages on its newer model runner (merged 2026-09-11, [#46994](https://github.com/vllm-project/vllm/pull/46994)); this is the same capability on the older runner this image has, which upstream reports broken for it.

| Drafts | count | code | prose (tok/s, 6x2) |
| --- | --- | --- | --- |
| 0 | 26.7 | 26.7 | 26.7 |
| 1 | 38.3 | 36.3 | 32.5 |
| 2 | 49.2 | 41.3 | 33.3 |
| 3 | 44.4 | 41.4 | 31.5 |

Acceptance follows the text: nearly every guess on counting, 86-88 % of first guesses on code, 50-65 % on prose. My agent's work is mostly code and tool calls.

## Prefix caching

With caching on, vLLM keeps the linear-attention state at block boundaries (1,152 tokens here) and resumes a conversation from the last one it holds. Tested before it served: 24 of 24 cold-extend-reuse rounds under fresh cache keys; 17 of 18 on a ladder of long prompts with another request generating beside it (the miss a neighbouring record); cached and fresh answers to the same prompt identical. At 48k history, the same question again answers in 1.35 s, a follow-up turn in 1.42 s.

## Three slow paths for prompt-sized steps

The overlay was written for decode first, with "correctness-first" fallbacks for shapes it did not expect. Three of them were where the prompt time went.

**A prompt read beside a generating request.** With two requests, a step can hold one request's draft-check tokens and another's prompt chunk. That shape matched none of the overlay's three gates into the fast sparse-attention kernel and fell to a Python loop over every token on every sparse layer: 2.2 s per step on the first stage, against 0.6 s for the same chunk alone. A 39k prompt read beside a generating request took 130 s instead of 15. A fourth gate sends that shape to the same kernel, which already reads each token's own slice of keys: 37-40 s, answers exact. What remains is the one-step-at-a-time rule MTP needed, which only matters with two users.

**Where the rest of the time went.** The kernel profilers produced no data on these cards (one recorded no GPU events; the other writes only at a clean exit, which vLLM's shutdown never reaches), so Claude put GPU timing events on every module of every layer and read the share of each part during cold reads of 230,773 tokens:

| Part | Before (stage 1 / 2 / 3) | After both fixes |
| --- | --- | --- |
| experts | 27 / 34 / 27 % | 41 / 53 / 41 % |
| residual mixing (mHC) | 19 / 21 / 18 % | 5 / 5 / 4 % |
| sparse indexer | 25 / 21 / 17 % | 10 / 3 / 2 % |
| linear attention (KDA) | 12 / 12 / 9 % | 19 / 19 / 14 % |
| sparse attention itself | 5 / 5 / 4 % | 8 / 8 / 7 % |
| stage time | 98 / 100 / 92 s | 63 / 63 / 59 s |

**mHC.** Mixing four residual streams should be cheap, but for steps over 64 tokens the overlay used a torch decomposition of about 80 small launches per call; its own fused kernels did the same maths in one launch and were already used for decode. Using them for prompts too took mHC from 18.9 s to 3.0 s per stage.

**The indexer's prompt scores.** For each new token and each earlier pooled key the indexer computes a weighted sum over 32 heads of a scaled, rectified dot product. Without AITER the overlay ran that in torch, one head at a time, with about seven full passes over a large fp32 table per head. The new Triton kernel loads each block of keys once, reuses it for all 32 heads, and keeps the sums in fp32:

| Rows x pooled keys | torch fallback | Triton kernel |
| --- | --- | --- |
| 1,152 x 2,500 | 4.9 ms | 0.2 ms |
| 1,152 x 10,000 | 16.4 ms | 0.7 ms |
| 1,152 x 29,100 (a 116k prompt) | 75.3 ms | 2.0 ms |

Against exact maths it chose the same top 512 keys 100 % of the time; the torch path, which rounds each head's scores to 16 bits, 99.81 %.

| Cold read | Before | After |
| --- | --- | --- |
| 39k tokens | 15.3 s | 11.1 s |
| 74k | 31.2 s | 21.1 s |
| 116k | 54.0 s | 32.7 s |

Two things tried did not help enough to keep: bigger prompt chunks (no faster, 1.4 GiB more memory), and four RCCL rings instead of two (6-7 % faster, accuracy inside the noise but not shown equal; parked). RCCL already talks card to card directly.

## Vision on one stage

The vision tower is 1.05 GiB, and the model built it on every stage although vLLM runs images through the first stage only. Built on the first stage alone, with one image per prompt and at most 1,024 image tokens, it added 0.4 GiB to the first stage's four cards and nothing to the third, the tightest. It read a code drawn in an image, a line of text exactly, a border colour, and one label out of a grid of 32 in a 1,600 x 1,200 image (1.5 s). Text answers were unchanged on the full test set. It is built and tested; whether it serves is my call, and my agent does not send images yet.

## The fault that is not explained

Three times a long read was wrong: twice on the first read past 48,000 tokens after a start (on the first caching run, five empty long answers; on a fresh start of the production server, empty answers to every question on a 74k prompt), and once as a wrong code on the same kind of 74k prompt later in a server's life. With prefix caching that matters more, because every follow-up question reuses the bad read. Most fresh starts were clean. The cause is not shown. Every start now reads 74k and 116k cold under a private cache key and checks both answers, up to three tries, before it serves; at the start that serves tonight the first try was wrong and the second right. Suspects, none shown: the first time a working buffer larger than the warm-up's is needed, the library that recycles released memory, or a first-use path at that size.

## Withdrawn

| Claim | Why withdrawn |
| --- | --- |
| The five empty answers on the first caching run came from a test running alongside | the server log shows one request running at the time |
| On the last stage, the time outside the layers is the draft layer | it is the last stage waiting for the second (13 layers against 16) |
| Prefix caching causes the slow mixed step | the same slowdown with caching off |
| The start-up fault happens only on the first long read after a start | the same wrong answer appeared on a later read of the same prompt |

## Left open

- **The fault above.** Find its cause; the gate is not a fix.
- **The experts** are now half of every stage's prompt time, at about 32 tokens per expert per chunk, well below what the matrix units can do.
- **Decode.** The first stage takes 16.7 ms of a step against 11 to 12 on the others; the draft layer runs without a CUDA graph; the third draft guess was never accepted at 3 drafts.
- **FP8 against 4-bit on quality**, layer for layer, before spending memory on more FP8 layers or on images.
- **Power.** Summed peaks of 3,330 to 3,652 W in the test runs, above the 3,500 W I set.
- **RDNA4.** The FP8 prefill kernel uses gfx1100 matrix builtins; the R9700s need their own build, and they have FP8 instructions of their own.

The patches, the kernels, the serving script and the test tools are in [patches/glm5next/serving-2026-10-25/](../patches/glm5next/serving-2026-10-25/README.md).
