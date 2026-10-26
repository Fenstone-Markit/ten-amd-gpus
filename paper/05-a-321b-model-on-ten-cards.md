# Chapter 5: A 321B model on ten consumer cards

*GLM-5.3-Flash on ten RX 7900 XTX, through a community RDNA4 port, an agent doing most of the debugging, and one bug that hid behind the number 32,768. Measured from 2026-10-14 to 2026-10-19.*

GLM-5.3-Flash is Z.ai's 321-billion-parameter mixture-of-experts model, about 18 billion active per token. Of its 45 layers, 34 are linear attention (KDA, Kimi Delta Attention) and 11 are sparse attention (DSA, a learned indexer that picks which earlier tokens each token looks at), three linear to every sparse one. The official vLLM recipe asks for vLLM 0.29 or later, and its AMD section covers one chip, the MI355X, through AMD's AITER library. AITER does not run on RDNA3. So there was no official way to run this model on my cards, and I wanted it as the brain my agent thinks with.

This chapter is how it got from "no official path" to a 131,072-token window at 35 tokens per second, what was wrong along the way, and what is still not known.

## Findings

1. **It runs, at 131,072 tokens, with correct answers.** Ten cards, five pipeline stages of two cards each, 4-bit weights (177 GiB on disk). A 119,517-token prompt was read in 43.9 seconds, about 2,700 tokens per second, and answered correctly. Decode is 35 tokens per second. The cache holds 307,820 tokens, 2.35 full windows. Tool calls work.
2. **Every window above 32,768 failed, each in a different way.** At 33,792 the server started and gave a wrong answer. At 36,864 it died while starting. At 40,960 it raised an error on the second request. At 65,536 it crashed the GPU on the first. Four symptoms, one cause: a small tensor of per-request chunk counts in the linear-attention layers held garbage.
3. **The fix put back what upstream already does.** The community port builds those chunk counts on the GPU, to avoid a pause while the CPU computes them. Upstream vLLM computes them on the CPU. Switching back (`GLM5_KDA_NOCACHE=1`) made every window up to 131,072 work and cost nothing I could measure: a 28,000-token prompt read in 9.7 seconds against 9.5 before.
4. **The failure follows the GPU version, not its cache.** Keeping the result cache but building on the CPU passes everything; dropping the cache but building on the GPU still crashes, one request later. What overwrites the GPU version's memory is still not named, so the long window counts as working on every test run, not as proven safe.
5. **Two of my explanations were wrong, and my agent's logs said so.** One blamed the page-size alignment between the two attention kinds; the log prints the same alignment at every window, failing or not. The other blamed the decode graph writing into the prompt's memory; eager mode, which has no graphs, failed the same way. Both are recorded as withdrawn below.
6. **Prompt reading is where ten cards earn their keep.** The same model on llama.cpp reads about 176 tokens per second (44,004 tokens in 250 seconds) and decodes at about 33. vLLM reads about 2,700 to 3,000. For an agent that re-reads long conversations, that is the difference between waiting seconds and waiting minutes.
7. **Some doors on RDNA3 are simply shut.** AITER does not run, so its long-prompt top-k path is out. There is no dense MLA prefill backend, so the model cannot fall back to dense attention to rule its sparse path in or out. FP8 is a storage format here, not something the cards compute in.

## The window ladder

All on ten cards, real weights, graph mode, five stages of two cards, utilisation 0.97, two requests at most at a time.

| Window | Without the fix | With `GLM5_KDA_NOCACHE=1` |
| --- | --- | --- |
| 8,192 | serves; 33.6 tok/s short (2026-10-16) | |
| 32,768 | serves; 34.6 to 35.1 tok/s; a 28,134-token prompt in 9.5 s, correct | passes the same checks |
| 33,792 | starts; the 28,134-token answer is wrong | |
| 36,864 | dies during start-up | |
| 40,960 | "repeats can not be negative" on the second request | correct: 35.1 tok/s, the 28k prompt in 9.7 s |
| 65,536 | GPU fault on the first request | correct: 28k in 9.7 s, 59k in 19.3 s, tool call passes |
| 131,072 | | correct: 119,517 tokens in 43.9 s, 35.1 tok/s decode |

The cache for the 131,072 run was set to 1 GiB per card (`--kv-cache-memory-bytes 1073741824`); the others used 0.5 GiB. Every long answer is checked against a known line in a generated log, so "correct" means the model found the right needle, not that it produced text.

## The three knobs that made it work

The port is the community RDNA4 overlay (djdeniro's GLM-5.3-Flash-rocm-r9700, MIT), 72 whole-file replacements over vLLM, laid over AMD's own vLLM 0.27 image, plus 14 patch files of mine and one load fix in the image. All of them, with the exact serve command, are in [patches/glm5next/](../patches/glm5next/README.md). Every patch added while chasing the long-window fault sits behind a switch that is off by default. Three switches matter for serving:

| Switch | Off | On |
| --- | --- | --- |
| `GLM5_DSA_BREAK=indexer` | The first decode step replayed from a graph faults on the first stage's cards (a read of unmapped memory) | The sparse-attention indexer runs outside the captured graph; graphs work, decode goes from 19 to 34 tok/s against eager |
| `GLM5_PROFILE_ALWAYS=1` | Giving vLLM a fixed cache size skips its start-up memory profile | The profile runs anyway, then the fixed size is used. Needed because the profile's own estimate leaves no room for a cache above about 32k |
| `GLM5_KDA_NOCACHE=1` | Chunk counts built on the GPU behind an identity-keyed cache; every window above 32,768 fails | Built on the CPU, as upstream does; everything up to 131,072 works |

The rest of the patches make the port start at all on this image: a backend name the overlay expects, a doubled decorator, a scheduler from the wrong vLLM version, pipeline-stage hand-offs for the model's multi-stream residual, the cache grouped per stage, a host-to-device copy that is illegal during graph capture, and a fused kernel the image ships that gives wrong answers on gfx1100.

## How the bug was found

Most of this was done by my agent, Itko, working through seventeen numbered tasks on two cards and then all ten. I wrote the briefs and checked every result from its logs before the next brief, not from its summaries. That rule caught both wrong explanations.

The trail, in order:

1. The failing line is `torch.repeat_interleave` in `prepare_chunk_indices`, in the vendored flash-linear-attention code. At 40,960 it says "repeats can not be negative"; at 65,536 the same kernel faults in hardware.
2. Dummy weights fail too, so the weights are not the cause.
3. Eager mode passed at 40,960 on short requests, and failed at 65,536 on the first one.
4. A dump taken right at the failing call showed the request's boundaries correct (`[0, 29]` for a 29-token request) and the chunk count, a single number, reading 0, then 1, then negative within the same function call. A calculation cannot do that. Something else was writing that memory.
5. Clamping the count to zero did not help: the next line still read a negative number.
6. Serializing every kernel and every memory copy did not remove it either, so it is not two operations racing.
7. The sparse indexer's gather kernel, the other suspect, was checked bound by bound over 286 launches at 65,536: nothing out of range.
8. Building the counts on the CPU, the way upstream vLLM does, removed every failure.
9. Splitting that fix in two settled which half mattered: the CPU build with the cache passes; the GPU build without the cache fails on the next long prompt.

## Withdrawn explanations

| What was claimed | Why it was withdrawn |
| --- | --- |
| The page-size alignment between the linear and sparse layers changes with the window and builds the bad chunk counts | The log prints the same alignment (2,176-token blocks, 2.64 % padding) at 32,768 and at every failing window |
| The captured decode graph writes into memory the eager prompt step is using | No decode was running during the failing request, and eager mode, which has no graphs, fails the same way at 65,536 |
| The illegal memory access at 65,536 comes from the indexer's gather kernel | It was named only because it compiled just before the crash; a bounds check over every launch found nothing out of range |

## Moving the brain

GLM needs all ten cards, and every test on all ten meant stopping the 27B brain my agent thinks with, so the agent could not test anything itself. The brain moved to the second machine, a Threadripper with two RX 7900 XTX (the eleventh and twelfth cards):

| | Eight cards (before) | Two cards, no peer link | Two cards, Resizable BAR on |
| --- | --- | --- | --- |
| Window | 262,144 | 229,376 | 196,608 |
| Decode | 130 to 160 tok/s | 57 to 110 tok/s | 61 to 119 tok/s |
| A 120,000-token prompt, first token | | 333.7 s | 329.1 s |

Turning on Resizable BAR (CSM off, Above 4G on) gave the two cards a direct link, so the RDNA all-reduce from rc3 engaged at two cards and decode rose about 10 %. Prompt reading did not change: two cards simply have a quarter of the compute. The window shrank, most likely because the card driving the screen holds about 250 MiB for the console once CSM is off (the other card holds 26 MiB idle), and the cache is sized to the card with the least room. Not yet confirmed by moving the screen.

## Power

Every card has run at its vendor minimum power limit since 2026-10-17 (294 W or 261 W), set at every boot. It cost no measurable decode speed.

*Correction, 2026-10-26: the boot-time service that sets those limits reported success at every boot from 2026-10-17 while capping only 4 of the 10 cards; it was found and replaced on 2026-10-24. So during these runs some cards were most likely at their stock limit, and which ones on which day is not recorded. The per-card and summed figures below are as measured; "every card at its vendor minimum" was not true for them.* During the 131,072 run my agent's telemetry caught single readings up to 515 W on a card whose maximum board power is 291 W: a spike lasting less than a sample, not draw the card can sustain, because the limit governs average power over a window, as Chapter 4 found for one card. What matters for the circuit is the ten-card sum under the heaviest load, so my agent measured it: two 119,517-token prompts at once at the 131,072 window, sampled every half second.

| Phase | Ten-card peak | Ten-card mean |
| --- | --- | --- |
| Start-up, compiling and tuning | 2,807 W | 310 W |
| Reading two 120,000-token prompts | 3,060 W | 2,370 W |
| Answering two short prompts | 1,736 W | 1,429 W |

Per card, the driver's averaged reading went 14 to 44 % over the cap for moments, while the phase mean stayed under it, about 237 W per card. The peak sum plus the host's roughly 400 W stays under the circuit's 3,840 W. The two long prompts were read one after the other, not side by side: the second answer started 102 seconds after both were sent, both correct.

## Left open

- **The writer.** Whatever put garbage in the chunk counts is not named. The fix avoids it; it does not explain it.
- **A soak.** One clean run per window is not production. GLM becomes the brain only after hours of mixed requests and a real agent session.
- **The upstream path.** vLLM 0.31 supports GLM-5.3-Flash natively but turns pipeline parallelism off for it, and this layout needs five stages. When an official build runs this model on RDNA3, the port retires and its switches become the checklist to test it against.
- **A clock ceiling and an undervolt,** to cut the short spikes at their source and lower heat, card by card.
- **Undefined padding in the output kernel.** The KDA output kernel this model uses loads the last, partial chunk of a prompt without asking for zeros past the end, which leaves those lanes undefined. Prompts ending 24, 29 and 38 tokens into a chunk all answered correctly, so it is a hardening item, not a fault: one argument (`padding_option="zero"`) on four loads. The same class of problem is open upstream as [vLLM #59404](https://github.com/vllm-project/vllm/issues/59404) for a different file.
- **Decode with two requests at once.** Every decode number here is one request at a time.
- **Speculative decoding (MTP).** llama.cpp gets about 60 % of two draft tokens accepted on this model; in vLLM the draft head does not fit beside a long window yet.
