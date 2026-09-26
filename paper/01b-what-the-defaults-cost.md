# Chapter 1.2: What the defaults cost

*Node02: ten RX 7900 XTX (gfx1100), EPYC 7663, vLLM 0.23.1.dev1 on ROCm 7.14.1, image
`rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0`. The model is
Qwen3.8-27B AWQ INT4, the one my agent actually runs on. Every number here was measured on this
machine, two runs per point, and the two runs agreed to within 0.5 % unless I say otherwise.*

I am not a kernel engineer. I run logistics for a living, and I built this machine to do real work.
I did this with an AI assistant and a local agent running on the same box, and I checked every
number against the machine rather than against anyone's expectations, including my own.

In its first week my agent went from quick to painful. This chapter is every wall I hit getting it
back, and the knob that moved each one. None of them needed new hardware. Every one of them was
set wrong, by default, for this card.

## The short version

| Context | Where I started | Where I am now | Change |
| --- | --- | --- | --- |
| 2,000 tokens | 21.5 tok/s | **53.7 tok/s** | 2.5× |
| 60,000 tokens | 4.59 tok/s | **40.7 tok/s** | 8.9× |
| Wait for the first word, 60K conversation | 26.6 s | **0.2 s** | 130× |

Same cards, same model, same weights.

> **Correction, 26 September 2026.** The 0.2 s wait in the last row is not supported by this
> chapter's own measurements: Knob 1 below measured 0.51 s at 60K. The current build measures 0.57 s.

## Findings

1. **The machine got slower the longer it worked, and nothing said why.** My agent started at about
   12 tok/s early in a session and slid to 4.5, then 3, as its conversations grew toward 60,000
   tokens. Every reply began with a 26-second silence. No error, no warning, a clean dashboard.
2. **Four defaults caused it, and fixing them took 60K decode from 4.59 to 20.68 tok/s (4.5×)
   without changing a line of code.** The attention backend, prefix caching, the KV cache format,
   and a PCIe switch quietly routing card-to-card traffic through the CPU.
3. **Spreading the model across eight cards instead of four took it to 27.9 tok/s.**
4. **Even then, about 25 ms of every 30.7 ms token had no measured owner.** Reading the weights
   explains 2.7 ms. Launching the kernels explains about 2.4. Two confident explanations for the
   rest were proposed and withdrawn.
5. **Fifteen of those milliseconds were one kernel.** vLLM runs this model's quantized layers
   through a slow Triton kernel. A native RDNA3 kernel, written for this exact card, ships in the
   same image and is skipped because of one entry in one list. Selecting it: **53.7 tok/s at 2K and
   40.7 at 60K, 1.49 to 1.65× faster.**
6. **About 10 ms per token are still unaccounted for.** That is the next chapter, not a solved
   problem.

The silicon was never the limit. The tuning was.

## The symptom

My agent, Itko, runs on this model. On its first day it read about 12 tok/s at the start of a
session. Over the following days it got slower, reaching 4.5 tok/s, then 3. It never crashed. It
just crawled, and each turn opened with a long wait before anything appeared.

I thought it was degrading day by day. It was not. It was the same curve every night, climbed a
little further each time, as the agent's own display showed:

| How far into a conversation | What the agent showed |
| --- | --- |
| About 9,000 tokens | about 12 tok/s |
| 18,000 tokens | 9 tok/s |
| About 27,000 tokens | 8 tok/s |
| 32,000 tokens | 6 tok/s |

Speed depended on how much conversation sat behind each token. Every task started fresh and climbed
the same slope.

Measured directly against the server, with no agent in the loop, the model on its defaults looked
like this at four-way tensor parallel:

| Context | ms per token | tok/s | Of which, attention |
| --- | --- | --- | --- |
| 2,000 | 46.6 | 21.47 | about 3 ms |
| 8,000 | 63.2 | 15.83 | about 20 ms |
| 16,000 | 86.8 | 11.53 | about 44 ms |
| 32,000 | 134.5 | 7.44 | about 92 ms |
| 60,000 | 217.8 | 4.59 | about 175 ms |

Attention grew at a steady 2.9 ms per thousand tokens of context, which made the split easy to read.
**At 60,000 tokens, 80 % of every token was attention.** The agent was not misreporting. The model
really was that slow, and it was slow for a reason that could be fixed.

## The knobs

Each knob below says what it does, what happens if you turn it, what happens if you turn it the
other way, and how to confirm it took. They are in the order I found them.

### Knob 1: prefix caching

`--enable-prefix-caching`

This was the first thing I changed, and it did not do what I hoped.

| Direction | What you get |
| --- | --- |
| Off, which is how the brain first ran | Every turn reprocesses the whole conversation before writing a word: **26.6 s of silence at 60K, 12.0 s at 16K.** The agent's own speed display counts that wait, so it looks slower than the model really is |
| On, for agents and chat | **The first word at 60K arrives in 0.51 s, at 16K in 0.24 s.** Only the new part of each turn is processed |
| Off on purpose, for benchmarking | Honest numbers. With it on, published throughput can be inflated by up to 1.74× (Chapter 1, §5.2) |

**What it did not do:** decode stayed at 4.58 tok/s at 60K. I expected this knob to cure the crawl.
It cured the wait. The crawl was something else, and the measurement said so the same night: speed
roughly halved every time the context doubled, which is the signature of attention. That is Knob 2.

**Caveat:** on hybrid models like this one, vLLM runs its state cache in an `align` mode it calls
experimental. If output ever looks wrong, turn this off first.

### Knob 2: the attention backend

`--attention-backend TRITON_ATTN`

| Direction | What you get |
| --- | --- |
| Left on the default | Fine up to about 2,000 tokens. Beyond that, each doubling of context roughly halves your speed |
| Turned to TRITON_ATTN | **60K decode goes from 4.59 to 20.68 tok/s, 4.5×.** The context curve flattens: 42 to 48 ms per token from 2K all the way to 60K |

Profiled on a smaller sibling model on one card, the default path ran attention at 1 to 2 % of the
card's measured memory bandwidth. It was starving the GPU of work, a few workgroups on a card with 96
compute units. The Triton backend splits long sequences across the card and brought the same work to
55 to 62 % of the ceiling. The 4.5× above is measured on the 27B itself. Nothing warns you
which one you are on.

**Confirm it took:** the server log must say `Using TRITON_ATTN backend (selected via
--attention-backend)`. If it says `Overriding with ROCM_ATTN`, the flag did not apply.

### Knob 3: the KV cache format

Remove `--kv-cache-dtype fp8`

| Direction | What you get |
| --- | --- |
| fp8, the tempting one | Looks like free memory. **It disqualifies AMD's fast attention kernel**, and at 16K it ran 6.1× slower |
| Default (`auto`) | The fast kernel stays eligible. 44.58 tok/s at 16K where fp8 gave 7.25 |

Measured on MiniMax-M2.7 AWQ at eight-way tensor parallel, same cards, same context:

| Prompt tokens | fp8 KV | Default KV, fast kernel |
| --- | --- | --- |
| 512 | 37.80 | 46.68 |
| 2,048 | 26.86 | 46.95 |
| 8,192 | 12.45 | 46.14 |
| 16,384 | 7.25 | **44.58** |

The memory it saved was never needed. The cache held 217,552 tokens without it. Turn fp8 on only
when you are truly out of cache, and expect to pay for it in speed.

### Knob 4: the PCIe switch

A host setting, not a flag. All ten cards sit behind one PEX880xx Gen4 switch, bought so cards could
talk to each other directly.

| Direction | What you get |
| --- | --- |
| Left as it boots | ACS redirect is on (`ACSCtl` reads `001d`). Every card-to-card write goes up to the CPU and back. Prefill is 32 % slower, and two busy servers slow each other a further 30 % |
| Redirect cleared | Time to first token at 4,096 tokens: 0.72 s becomes 0.49 s. Concurrent servers stop interfering entirely |

Measured with a 9B model across three servers at once, which is the case the switch was bought for.

RCCL reports peer-to-peer transport either way, because from its point of view the cards are doing
direct transfers. It cannot see the route. Decode barely moves (1 to 2 %), because decode messages
are small.

**Make it stick:** clearing the bit is a `setpci` write per bridge, it resets at every boot, and idle
bridges quietly set it again. Here it is a systemd service at boot plus a ten-minute timer.

### Knob 5: how many cards per model

`--tensor-parallel-size`

| Cards | ms per token at 2K / 16K / 60K | tok/s at 60K |
| --- | --- | --- |
| 2 | 71.1 / 72.8 / 78.3 | 12.8 |
| 4 | 40.1 / 41.4 / 45.5 | 22.0 |
| 8 | **30.0 / 31.4 / 35.9** | **27.9** |

For one user on a model that needs several cards, the highest degree that fits won. For many users,
Chapter 1 found the opposite: more servers on fewer cards each. **Measure the workload you have.**

### Knob 6: the collective algorithm

`NCCL_ALGO=Tree`

Worth about 1 ms per token. The all-reduces here move about 10 KB each, where the cost is the number
of hops, not bandwidth, and a tree has fewer hops than a ring.

> **Correction, 26 September 2026.** This was measured at four cards and deployed at eight. At eight
> cards the tree is slower at every message size, by 3.2 ms per token at decode. The brain now runs
> the ring. See [Chapter 1.3, Knob 8](01c-what-the-rest-of-the-stack-costs.md).

### Knob 7: the kernel that runs the quantized layers

This is the big one, and it is not a flag. It is three small edits to one Python file.

At eight cards, a token cost 30.7 ms, and about 25 ms of it had no measured owner. It turned out
that 18.1 ms of every token went to the model's 255 quantized matrix multiplies, all running through
one Triton kernel. That figure comes from timing production's exact kernel, at production's exact
shapes, in isolation. That kernel has two problems on this card:

- **It walks each matrix alone.** Every program steps through the full inner dimension in sequence,
  up to 160 steps, with no way to split the work. At this model's shapes most of the GPU sits idle.
- **It keeps the wrong tile.** Its settings were tuned for RDNA 3.5 (a 40-compute-unit part). When
  vLLM compiles the model, the setting meant for large batches gets frozen in, so every single-user
  token is computed as if it were a batch of 128. Correcting only that would save 5.7 ms per token.

Meanwhile, the same image ships a native RDNA3 kernel for exactly this job. vLLM skips it for this
model because the checkpoint stores its zero points explicitly (`uint4`), and the native kernel's
list of accepted formats only has the symmetric kind (`uint4b8`). The fix accepts `uint4`, converts
the zero points the way the Triton path already does, and tells the kernel not to apply an old GPTQ
offset to them.

| Context | Triton (default) | Native RDNA3 kernel | Change |
| --- | --- | --- | --- |
| 2,000 | 30.7 ms, 32.5 tok/s | **18.6 ms, 53.7 tok/s** | 1.65× |
| 16,000 | 32.1 ms, 31.1 tok/s | **20.0 ms, 49.9 tok/s** | 1.61× |
| 60,000 | 36.4 ms, 27.4 tok/s | **24.5 ms, 40.7 tok/s** | 1.49× |

Measured on the production model at production settings, in a separate container, so the running
server was never touched. I predicted 19 to 23 ms at 2K before the run, from the kernel measurements
alone, and wrote it down. It came in at 18.6.

| Direction | What you get |
| --- | --- |
| Default | Deterministic output, and the speed above on the left |
| Native kernel | 1.49 to 1.65× faster. **It costs 0.2 % in perplexity and the output is no longer bit-for-bit repeatable.** Chapter 2.1 measures both, and one sharp case |

**Two traps, both of which I hit:**

- **The compile cache will lie to you.** After changing which kernel vLLM selects, it reused the
  graph it had compiled for the old kernel and fed it weights prepared for the new one. The server
  started in three minutes, reported healthy, and scored the reference text at a perplexity of
  1.8 million, where the real model scores 1.61.
  Point `VLLM_CACHE_ROOT` at an empty folder, and then check the compiled graph for the kernel you
  expect.
- **Check the kernel, not the flag.** The only proof the change took was counting kernel calls in the
  compiled graph: 255 native, zero Triton.

I am finishing a version of the native kernel that adds its partial sums in fp32, which should
remove the precision cost while keeping most of the speed. Until then, this knob is a choice, not a
free win.

> **Update, 26 September 2026.** Done. The fixed-order version is faster than this one (41.4 tok/s at
> 60K), at the precision floor, and bit-for-bit repeatable. See
> [Chapter 2.2](02b-the-speed-without-the-cost.md).

## Knobs that did not work

Worth listing, because each one looks promising and each one cost time.

| Knob | Result |
| --- | --- |
| Speculative decoding | +40 to 54 % on a 9B model on one card. **9 % slower on the 27B** at four cards, even after fixing a crash it exposed. The draft head costs more than it saves at this size |
| `NCCL_MIN_NCHANNELS=8` | No effect |
| `VLLM_USE_NCCL_SYMM_MEM=1` | No effect |
| vLLM's custom all-reduce, QUICK_REDUCE | Cannot be enabled on this card by any flag. The check lists only MI300-class architectures, even though every card pair here has peer access |
| Smaller tiles in the native kernel | 22 % faster on split shapes in isolation. On the real merged shapes, only 14 % at batch 1 and slightly slower at batch 4. Not a knob yet |
| One card for the 27B | Does not fit with any usable cache |

> **Update, 26 September 2026.** Speculative decoding is now 35 % faster at 60K, on eight cards with
> the fixed kernel. The four-card result above stands for that configuration. See
> [Chapter 1.3, Knob 9](01c-what-the-rest-of-the-stack-costs.md).

## Where the time goes now

A 2,000-token decode step at eight cards, in milliseconds:

| Part | Before | After |
| --- | --- | --- |
| Reading the weights (the physical floor) | 2.7 | 2.7 |
| Launching roughly 800 kernels | about 2.4 | about 2.4 |
| Quantized matrix multiplies, beyond the floor | about 15 | **about 3.6** |
| Attention, linear attention, norms, 128 all-reduces | about 10 | about 10 |
| **Total** | **30.7** | **18.6** |

The last row is the same size before and after the change, which is what you would expect if the
kernel swap touched only what it was meant to. Those 10 ms are the next wall. The candidates are
the 48 linear-attention layers, and the 128 all-reduces for which this card is refused the fast
path.

## What this does not prove

One machine, one model, one software build. The kernel measurements predicted the server this time
because they measured production's own kernels at production's own shapes. The first attempt at
this work measured a smaller model on fewer cards and extrapolated, three times, and never held up.
The precision cost of the native kernel is measured on twelve prompts, which is enough to rule out a
bug and not enough to price the cost precisely.

## To AMD, and to vLLM's ROCm maintainers

The RX 7900 XTX did everything its specification says. The native kernel that produced most of this
chapter's gain was already written, by someone who clearly cared about this card. What is missing is
the last mile:

- **A format list with one entry missing**, which sends every asymmetric 4-bit checkpoint to a
  slower path.
- **A tile table tuned for a 40-compute-unit part**, frozen in at compile time on a 96-unit card.
- **A fast all-reduce switched off by architecture name**, on cards that pass the hardware check it
  actually needs.
- **No RDNA3 continuous-integration runner**, which is how all three shipped without anyone
  noticing.

Most of these are small changes for someone with commit access. Together with the defaults above,
they are the difference between 4.59 and 40.7 tok/s on hardware people already own.

## Reproduce it

Everything is in `tools/`: the decode benchmark (`brain_bench.py`), the output recorder and scorer
used as the quality gate (`ref_outputs.py`, `compare_refs.py`, `score_texts.py`), the patch builder
(`make_patch.py`) and the launcher for the separate test container (`exp-tp8.sh`). The kernel
measurements behind Knob 7 are in `tools/kernel/`, each with its correctness gate and a broken
control it has to reject.
