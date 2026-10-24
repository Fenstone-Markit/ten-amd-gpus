# Chapter 8: Serving it

*How GLM-5.3-Flash came to serve on ten RX 7900 XTX after Chapter 6 said it could not: 5 GiB of memory it did not need, a backstop for the fault underneath, a soak of 1,065 requests, and a first count of which experts do the work. Measured on 2026-10-23 and 24.*

Chapter 6 ended with the overwrite narrowed to memory the ROCm runtime gives out again while a live buffer still owns it, and with GLM failing on its first real requests because its three middle pipeline stages run at the edge of full. Nothing I could set on ten cards avoided it. This chapter is what did, and what the model did once it stayed up.

## Findings

1. **GLM-5.3-Flash serves on the ten cards, and my agent runs on it.** Window 131,072 tokens. A 116,399-token prompt was read and answered correctly in 41.2 s, about 2,800 tokens per second. Since 2026-10-24 my agent's gateway points at it instead of the 27B.
2. **The middle cards were full because of a vision encoder nothing here uses.** At start-up vLLM measures the model's peak memory by running it on dummy inputs, including a dummy video through the vision encoder. On a middle-stage card that took about 5 GiB on top of 18.4 GiB of weights, nearly all of it for the vision part. My agent and the tutor send only text. Serving text only gives that memory back, and PyTorch then never has to release memory on any card, so Chapter 6's fault never gets its trigger.
3. **A library that keeps released memory in the process fixes the one-card test, but could not start GLM on its own.** It took the test from 20 and 21 hits to 0, with no out-of-memory. On GLM, with the vision encoder still profiled, the memory it held in its pool was not available to libraries that load their GPU code outside PyTorch, and the six middle cards ran out of memory at start-up. With text-only serving it stays loaded as a backstop; in serving it has not had to recycle anything, so it is untested there.
4. **A 2-hour 18-minute soak: 1,042 of 1,065 requests right, no crash, no error.** Every one of 204 tool calls was right, streamed or not. The tutor work confirmed 530 maths problems by a blind re-solve. Two requests at a time throughout, all ten cards, hottest reading 99 C, peak 2,891 W.
5. **Recall falls with length.** Finding one line in a list: 90 of 92 below 10,000 tokens, 22 of 29 from 10,000 to 50,000, 12 of 18 above 50,000. Every miss was a well-formed answer from the wrong line, mostly a line within 40 of the right one.
6. **Every expert gets used, and kinds of work route differently.** A count of which of the 288 experts each token picks, across seven kinds of work, found no expert unused in any layer over the whole mix, and between 0.2 and 2.4 per layer unused within one kind. Code and agent work route most alike; Arabic and Urdu, and long synthetic lists, least like anything else. This is a count only: nothing was removed.
7. **One of my own patches had a mistake in it through every start from the first canary run on.** A check I added sat between two decorators of vLLM's model-step function, so model steps ran outside PyTorch's inference mode. The first text-only start found it: its first request failed with "Inplace update to inference tensor outside InferenceMode".

## Where the memory went

Chapter 6 had already shown that the six middle-stage cards were the ones being overwritten, and that vLLM reported them at about zero headroom after profiling. The first attempts with the backstop library (below) ran out of memory on exactly those cards, so I recorded PyTorch's allocation history on one of them (pipeline stage 2, first card) through a start-up ([its log](../bench/glm-soak-2026-10-24/memsnap-23-rank4-log.txt)).

| On one middle-stage card, at the profiling peak | Memory |
| --- | --- |
| Weights | 18.4 GiB |
| Activation temporaries of the vision encoder's dummy run | 2.9 GiB |
| One matrix-multiply output in it | 1.16 GiB |
| The patch-embedding convolution | 365 MiB |
| The dummy video itself | 332 MiB |
| The encoder | 289 MiB |
| The card | 24 GiB, with PyTorch's first two out-of-memory retries at the peak and 0 bytes free on the card |

GLM-5.3-Flash takes images and video as well as text, and vLLM sizes everything for the largest input the model accepts. `--language-model-only` serves text only. With it, the middle cards finish profiling with about a third of a GiB to spare, PyTorch never releases memory, and the canary from Chapter 6 has been clean in every start since.

That is avoiding the trigger, not fixing it. The runtime still hands out memory a live buffer owns whenever PyTorch releases on a full card; nothing in this chapter changes that.

## The backstop

The library sits in front of two of PyTorch's virtual-memory calls, create and release. When PyTorch releases a physical piece of memory, the library keeps it in a pool keyed by its size and properties (the card included) and hands it back on PyTorch's next request with the same key. The runtime never sees the release, so it never gives that memory to anyone else.

| One-card test, 40 rounds, 150 MiB free | Overwrites | Out of memory |
| --- | --- | --- |
| Library passing everything through | 20 | 0 |
| Library recycling | 0 | 0 |
| Two cards in one process, recycling | 0 | 0 |

Its limit showed on GLM. With the vision encoder still profiled, all four starts with the library loaded died at start-up: hipBLASLt, the matrix-multiply library, ran out of memory on the six middle cards. Memory PyTorch released was sitting in the library's pool, and libraries that load GPU code outside PyTorch (about 200 MiB for the first bf16 multiplies) could not get it. Neither a smaller prefill chunk nor a 65,536-token window helped. The fix was not to need the memory in the first place.

The library and its tests are in [tools/vmm-repro/narrowing/](../tools/vmm-repro/narrowing/) (`hipmem-shim-02.c`).

## The recipe

| Setting | Value |
| --- | --- |
| Model | `wtdcode/GLM-5.3-Flash-AWQ-W4A16`, revision `abd7b077` |
| Image | a local image built on AMD's `rocm/vllm` 0.27.0 (ROCm 10.0, PyTorch 2.12), with my agent's port of the model (its port-26 patch set) mounted over it |
| Layout | five pipeline stages of two cards, layers split 10, 9, 9, 9, 8 |
| Text only | `--language-model-only` |
| Allocator | PyTorch's expandable segments, with the backstop library preloaded |
| Window and cache | 131,072 tokens; 1 GiB of cache per card, 307,820 tokens in all; two requests at a time |
| Prefill chunk | 2,048 tokens |
| Graphs | decode only, for one or two requests |
| MIOpen | a writable database and fast find mode (Chapter 7) |
| Chapter 5 and 6 switches | the CPU-built chunk counts, the sparse indexer outside the decode graph, the forced profile, the start-up prewarm |
| Warm-up | six prompts from 73 to 32,092 tokens; the server counts as up only when all six are right |

It starts in about 2 minutes 15 seconds, 30 seconds of it loading weights. The serving script and the patch set are not in this repository yet: they carry this machine's paths, and the switches from Chapters 5 and 6 should come out one at a time first, now that the trigger is gone.

## The soak

Two lanes of requests for 2 hours 18 minutes, all ten cards: tutor maths generated and then solved blind, tool calls the way my agent makes them, needle-in-a-list prompts from 2,000 to 116,000 tokens, and reviews of six of my own scripts. The driver and every request's result are in [bench/glm-soak-2026-10-24/](../bench/glm-soak-2026-10-24/).

| Kind | Requests | Right | Wrong | Errors | Median time (s) | Largest prompt (tokens) |
| --- | --- | --- | --- | --- | --- | --- |
| Tool call, streamed or not | 102 | 102 | 0 | 0 | 1.5 | 185 |
| Tool result, streamed or not | 102 | 102 | 0 | 0 | 1.4 | 238 |
| Tutor problems, generate | 162 | 161 | 1 | 0 | 18.3 | 151 |
| Tutor problems, solve blind | 537 | 530 | 7 | 0 | 2.3 | 121 |
| Needle, short | 92 | 90 | 2 | 0 | 3.8 | 7,382 |
| Needle, mid | 23 | 18 | 5 | 0 | 29.4 | 39,283 |
| Needle, long | 24 | 16 | 8 | 0 | 79.8 | 116,400 |
| Script review | 23 | 23 written | checked below | 0 | 179.4 | 8,203 |

Each needle prompt is a list of lines like "Record 518: gate 518 uses code K1173" and asks for one gate's code. Of the 15 misses, 12 gave the code of a line within 40 of the right one and 3 gave back the gate number itself, or one 2 away. None was garbage, and the canary saw nothing. By length:

| Prompt length | Right |
| --- | --- |
| below 10,000 tokens | 90 of 92 |
| 10,000 to 50,000 | 22 of 29 |
| 50,000 to 90,000 | 3 of 4 |
| above 90,000 | 9 of 14 |

That is the model at 4 bits on a synthetic list, not proof of a fault. It is also not good enough to lean on for exact recall at 100,000 tokens.

The 23 reviews were each checked against the script they reviewed. About 20 distinct findings were real, mostly failure paths that carried on instead of stopping, and about 10 were wrong. None of the real ones had been hit in a run. The fixes went into new versions of those scripts before any of them ran again, and the publishing script behind this chapter is one of them.

## Which experts do the work

Each of GLM's 42 mixture-of-experts layers has 288 experts and picks 8 for every token (plus one shared expert every token uses). I counted the picks: a few lines in the model's expert layer add each token's eight choices to counters on the card, using the same rule vLLM's routing kernel uses (it picked the same eight as that kernel for all 20,000 random test inputs, on the older vLLM build; not re-checked on this one). After each kind of work the counters were saved, so each kind gets its own count.

| Kind of work | Tokens counted | Effective experts per layer (of 288) | Experts covering half the picks | Unused per layer |
| --- | --- | --- | --- | --- |
| Agent tool calls | 23,877 | 181 | 52 | 2.4 |
| Code | 19,794 | 225 | 76 | 0.2 |
| Tutor maths | 18,030 | 172 | 49 | 1.1 |
| Long synthetic lists | 394,078 | 99 | 24 | 1.9 |
| Arabic and Urdu (Quran verses) | 26,784 | 116 | 28 | 0.6 |
| General knowledge | 16,740 | 187 | 58 | 0.9 |
| Medicine (lay questions) | 12,222 | 150 | 42 | 2.2 |

Effective experts is the exponential of the entropy of the pick distribution: 288 if every expert were picked equally often. The counts are clean: the counters saw within 206 tokens of what vLLM itself reported for each kind, and no traffic from anything else.

How alike two kinds route, as the cosine similarity of their pick counts averaged over the 42 layers (1.00 is identical):

| Kind | Agent | Code | Maths | Lists | Arabic, Urdu | General | Medicine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agent | 1.00 | 0.70 | 0.52 | 0.26 | 0.32 | 0.52 | 0.43 |
| Code | 0.70 | 1.00 | 0.54 | 0.32 | 0.36 | 0.59 | 0.49 |
| Maths | 0.52 | 0.54 | 1.00 | 0.24 | 0.34 | 0.58 | 0.46 |
| Lists | 0.26 | 0.32 | 0.24 | 1.00 | 0.16 | 0.28 | 0.25 |
| Arabic, Urdu | 0.32 | 0.36 | 0.34 | 0.16 | 1.00 | 0.34 | 0.27 |
| General | 0.52 | 0.59 | 0.58 | 0.28 | 0.34 | 1.00 | 0.68 |
| Medicine | 0.43 | 0.49 | 0.46 | 0.25 | 0.27 | 0.68 | 1.00 |

Across layers, the first ones spread work widely in every kind (165 to 275 effective experts at layers 3 to 6), and routing narrows from about layer 12 on. Code is the most spread kind in all 42 layers; the most concentrated is the long lists in 30 layers and Arabic and Urdu in the other 12.

The hooks, the driver, the table script and the raw counts are in [tools/expert-count/](../tools/expert-count/).

What this does not show: whether any expert can go. A rarely picked expert can still matter when it is picked, and medicine routing close to general knowledge says nothing about what medical questions lose without its experts. That needs a quality test with experts masked, which is pruning work and not part of this chapter. The mix is also small and mostly mine: 6 to 60 prompts per kind, some of them repeated, English except for the Arabic and Urdu, and the long-context kind is synthetic. The routing weights were also meant to be summed, but their totals went NaN during start-up on four of the five stages, so this count uses picks only.

## What is shown, and what is not

Shown: GLM-5.3-Flash serves 131,072 tokens on ten RX 7900 XTX through a 2-hour 18-minute soak of real kinds of work with no crash and no error, once its vision encoder is not profiled. Not shown: that the fault underneath is gone. It is avoided because the middle cards no longer run out of headroom; a longer window, a larger cache, or images would bring them back to the edge. Decode speed was not measured again with this recipe (Chapter 5 measured 35 tok/s, one request at a time).

## Left open

- **The fault itself.** The runtime still gives out memory a live buffer owns after a release on a full card. The upstream report, with the one-card test, is still to be filed.
- **Long-context recall.** 9 of 14 above 90,000 tokens is not enough for exact recall. Whether the 8-bit weights do better is the next question.
- **8-bit weights and multi-token prediction.** The official FP8 checkpoint is downloaded. These cards have no FP8 hardware; kernels that convert FP8 on the fly are in progress, and twelve cards fit about a third of the expert layers in FP8 plus the prediction layer.
- **Pruning,** if it is ever done, with the count above as the starting point and a quality test as the judge.
- **The serving script and patches in this repository,** after the Chapter 5 and 6 switches come out one at a time.
