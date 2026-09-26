# Chapter 1.3: What the rest of the stack costs

*Node02: ten RX 7900 XTX (gfx1100), EPYC 7663, vLLM 0.23.1.dev1 on ROCm 7.14.1, in a snapshot of the
production container built from `rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0`.
The model is Qwen3.8-27B AWQ INT4 at eight-way tensor parallel, the one my agent runs on. Speeds are
one request, decode only, with tokens counted exactly, at the context stated. The first run after each
restart includes warm-up and is not used; where two warm runs exist they agree within 0.5 %. I quote
16K and 60K because that is where my agent actually works.*

I am not an engineer or a researcher. I set out to show that the RX 7900 XTX has far more to offer
than the software lets it deliver, and this chapter is more of the same evidence: every gain below
came from settings and one kernel, on cards people already own.

Chapter 1.2 ended at 40.7 tok/s at 60K and said about 10 ms of every token had no measured owner.
Chapter 2.1 then showed that the kernel behind that number was not free: it cost precision and
repeatability. Chapter 2.2 fixes the kernel, and the fixed version is what this chapter starts from:
**41.4 tok/s at 60K, at the precision floor, bit-for-bit repeatable.**

This chapter is the rest of the stack. Three more knobs, a profile of where a token's time goes
now, and a check that the benchmark predicts real work.

## The short version

| Context | Start of this chapter | Now | Change |
| --- | --- | --- | --- |
| 16,000 tokens | 50.7 tok/s | **80.4 tok/s** | 1.59× |
| 60,000 tokens | 41.4 tok/s | **76.4 tok/s** | 1.85× |
| My agent's real work, about 24,000 tokens | not measured | **68.0 tok/s** | |

Same cards, same model, same weights. From the very first measurement in Chapter 1.2, 4.59 tok/s at
60K, that is 16.6×.

## Findings

1. **A knob from Chapter 1.2 reverses with the number of cards.** At four cards a tree all-reduce was
   the faster choice. At eight cards the ring is faster at every message size, and at one decode
   token it is 40.6 against 65.4 microseconds. There are 128 of them per token: 3.2 ms. 41.4 became
   48.3 tok/s at 60K.
2. **Speculative decoding went from a loss to a 35 % gain at 60K.** Chapter 1.2 rejected it as 9 %
   slower. On eight cards with the fixed kernel, one draft token lifts 60K from 48.3 to 65.0 tok/s and
   16K from 61.7 to 69.8, and 85 % of drafts are accepted on my agent's real traffic.
3. **One constant was starving attention.** The long-context attention path splits each request into
   16 segments. At eight cards each card holds one KV head, so a single request ran on 16 workgroups
   of a 96-unit card. At 128 segments, attention at 60K fell from 7.2 to 1.6 ms per token and the
   brain reached 76.4 tok/s. It is one line.
4. **The benchmark predicts real work.** My agent's real traffic, read from the server's own
   counters, ran at 68.0 tok/s at an average of 24,000 tokens, between the benchmark's 16K and 60K
   figures on the same build.
5. **The 10 ms now have owners.** A profile of every kernel in a decode step shows 1,664 kernels,
   all-reduce as the largest single cost, and about a fifth of the step spent with the GPU waiting
   between kernels, a few microseconds at a time.

## The knobs

The same format as Chapter 1.2: what each knob does, both directions, and how to confirm it took.
Numbering continues from there.

### Knob 8: the collective algorithm, measured at the card count you deploy

`NCCL_ALGO=Ring`, in place of Chapter 1.2's `NCCL_ALGO=Tree`

Chapter 1.2's Knob 6 was measured at four cards and deployed at eight. I measured the all-reduce
itself at eight cards, inside CUDA graphs, slowest card reported, sums checked for exactness:

| Message | Tree | Ring | RCCL's own choice |
| --- | --- | --- | --- |
| 1 token, 10 KB | 65.4 us | **40.6 us** | 41.1 us |
| 2 tokens, 20 KB | 79.5 us | **72.8 us** | 74.5 us |
| 128 tokens, 1.3 MB | 391 us | **214 us** | 214 us |
| 2,048 tokens, 21 MB | 3,485 us | **2,754 us** | 2,750 us |

| Direction | What you get |
| --- | --- |
| Tree, the right choice at four cards | 3.2 ms per token lost at eight cards, and slower prefill |
| Ring, or no setting at all | 60K decode from 41.4 to **48.3 tok/s**, 16K from 50.7 to 61.7 |

RCCL's own choice matched the ring at every size here. Forcing an algorithm was the mistake, not
the algorithm.

The large messages also show where PCIe width matters. At 21 MB the ring moves 13.3 GB/s, close to
what a Gen4 x8 link carries, so prefill is bound by the link. At 10 KB it moves 0.44 GB/s: decode is
bound by latency, and a wider link would not change it.

**Confirm it took:** read the variable from the server process itself
(`/proc/<pid>/environ`), not from the script that was meant to set it.

### Knob 9: speculative decoding

`--speculative-config '{"method":"mtp","num_speculative_tokens":1,"attention_backend":"TRITON_ATTN"}'`

The model ships a small draft head (MTP) that guesses the next token. The main model then checks two
tokens in one step instead of producing one.

| Direction | What you get |
| --- | --- |
| Off | One token per step. 20.7 ms per token at 60K |
| On, one draft token | A step costs more (28.8 ms at 60K) and yields 1.83 tokens: **65.0 tok/s at 60K (35 % faster), 69.8 at 16K (13 % faster)** |

Acceptance of the draft, three ways:

| Text | Accepted |
| --- | --- |
| The benchmark's repeated sentence | 76 % |
| Twelve varied prompts: code, SQL, prose, arithmetic | 89 % |
| My agent's real traffic, from the server's counters | **85.1 %** |

Chapter 1.2 found it 9 % slower on four cards with the Triton kernel. I did not re-measure that
case. Between the two measurements the quantized kernel, the all-reduce and the card count all
changed, and checking two tokens now costs little more than checking one: the fixed kernel takes
6.4 ms per token at one token and 7.7 at four.

**The trap:** my benchmark counted streamed chunks and called each one a token. With speculation a
chunk carries one or two tokens, so the first measurement showed a regression. Counting the tokens
inside each chunk showed the gain. **With speculation on, count tokens, not chunks.**

**The other thing to confirm:** the draft head chooses its attention backend separately, so it goes
inside the speculative config. Without it the draft runs on the slow default path.

**The caveat:** speculation plus this hybrid model's `align` cache mode is the combination upstream
users have reported ([vLLM #57925](https://github.com/vllm-project/vllm/pull/57925)) producing runs of exclamation marks after long use. I have not seen it. If
output ever turns into punctuation, this knob goes off first.

### Knob 10: the attention segment count

`NUM_PAR_SOFTMAX_SEGMENTS = 128` in `vllm/v1/attention/backends/triton_attn.py`, in place of 16

This is the knob from Chapter 1.2 in reverse. TRITON_ATTN fixed long-context attention by splitting
each request across the card, into 16 segments. At eight cards each card holds one of the model's
four KV heads, and a single request launches a grid of 1 × 1 × 16: **16 workgroups on a card with 96
compute units.** At 60K each workgroup walked 235 tiles in sequence while most of the card waited.

My agent measured every segment count the kernel accepts (it must be a power of two) on one card, at
the brain's exact shapes and cache layout, against an exact float64 reference, with a broken
control it had to reject:

| 16 layers, one request | 16 segments | 128 segments |
| --- | --- | --- |
| 60K, one token | 6.64 ms | **1.61 ms** |
| 60K, a speculative check of two tokens | 7.21 ms | **1.62 ms** |
| 100K, one token | 11.05 ms | **2.52 ms** |
| 16K, one token | 2.14 ms | **0.72 ms** |
| Error against float64 | 0.0016 | 0.0016 |

At 60K, 128 segments read the cache at about 620 GB/s, about 80 % of what the card can move. The error is
unchanged, and repeated runs are bit-identical.

| Direction | What you get |
| --- | --- |
| 16, the default | A single long request uses a sixth of the card for attention |
| 128 | **60K from 65.0 to 76.4 tok/s, 16K from 69.8 to 80.4** |

Four concurrent requests gain less (their attention time falls to 0.70 of what it was, against 0.22 for one request), because they already launch four times the workgroups.
The default is not wrong for a busy server. It is wrong for one person with a long conversation.

**Confirm it took:** the value is read once at startup, when the buffers are allocated and captured
into CUDA graphs. Changing the file does nothing until the server restarts.

## Knobs that did not work

| Knob | Result |
| --- | --- |
| vLLM's custom all-reduce on this card | Not compiled into the image at all. There is nothing to switch on |
| QuickReduce, AMD's MI300 all-reduce | Compiled in. Forced onto these cards, it loads, runs, and **returns a wrong value in every element**, at 10 KB and at 2.6 MB, above its own minimum size. Its architecture check is correct to refuse it here. Chapter 2.2 has the detail |
| Routing prefill through the fixed kernel too | Runs out of memory on a 2,000-token prompt: its scratch space grows with the number of tokens |
| A bigger PCIe link for decode | Not measured directly, but at 10 KB the ring moves 0.44 GB/s. Decode is bound by latency, not width |

## Where the time goes now

One decode step at 60K takes 24.0 ms and yields 1.83 tokens. I profiled 16 steady steps with every
kernel recorded. With CUDA graphs and stack capture off, the profiled kernel times matched my
separate measurements within about 10 % (attention 1.5 against 1.6 ms), though the profiler
stretched the step itself by about a fifth.

| Part | ms per step | Kernels per step |
| --- | --- | --- |
| All-reduce | 7.0 | 130 |
| Quantized matrix multiplies | 5.4 | 255 |
| The fixed kernel's separate reduction pass | 0.7 | 255 |
| Copies and concatenation | 1.5 | about 360 |
| Attention | 1.5 | 32 |
| Small fused kernels: norms, activations, reshapes | 1.8 | about 400 |
| bf16 matrix multiplies: output head, draft head | 0.6 | 50 |
| Linear attention (48 layers) | 0.4 | 96 |
| **The GPU waiting between kernels** | **about 5.2** | |

Two things stand out.

**All-reduce is now the largest single cost.** 128 of them per token, each 40 to 73 microseconds on
RCCL, where a purpose-built RDNA3 all-reduce ([vLLM #57767](https://github.com/vllm-project/vllm/pull/57767)) measured 10.4 microseconds
on two of these cards, exact against RCCL. It has no eight-card path yet.

**The waiting is not one stall. It is 1,664 small gaps.** Holes of 100 microseconds or more add up to
about 1 ms per step. The rest is the few microseconds between one kernel finishing and the next
starting, about 3.1 microseconds each, which matches the launch minimum measured in Chapter 1.2.
The lever there is fewer kernels, not a smarter scheduler. The fixed kernel alone launches 255 extra
reduction kernels per token, and folding them into the main kernel is the next thing my agent is
building.

## Is the benchmark real?

A benchmark is a repeated sentence at a fixed length. My agent's work is code, tool calls and JSON at
whatever length the conversation has reached. So for an hour I read the server's own per-request
timings while the agent worked, and ignored the benchmark:

| Measure | Result |
| --- | --- |
| Decode, finished requests | **68.0 tok/s** (15,801 tokens over 232 s of decode) |
| Average prompt | 24,377 tokens |
| Draft tokens accepted | 85.1 % |
| Prefill per request | 1.20 s |

68.0 sits between the benchmark's 69.8 at 16K and 65.0 at 60K on the same build. The benchmark is a
fair predictor at this context. The agent's own speed display read 57, about 16 % under the server:
it counts its own overhead.

## What this does not prove

Every speed here is one request at a time. The concurrency curve, many users at once, is not
measured, and every change in this chapter behaves differently under load: the segment count gains
less with four requests, and the all-reduce messages grow. The real-traffic hour was one task at
about 24K of context, not a long session at 60K. The profile is one run. The speculative-decoding
risk in the caveat above has not been exercised by long use.

## To AMD, and to vLLM's ROCm maintainers

- **A collective setting that has to be measured per card count.** Tree was right at four cards and
  wrong at eight. RCCL's automatic choice was right at eight. A note in the ROCm tuning guide would
  have saved a day.
- **A segment count of 16 for everyone.** On a one-KV-head-per-card layout it leaves most of a
  96-unit card idle at long context. Computing it from the compute-unit count and the KV heads per
  card would carry across card counts.
- **The MI300 all-reduce compiled into a consumer image, where it returns wrong values.** The
  architecture check is doing its job. The custom all-reduce that would suit decode is not compiled
  in at all.
- **An RDNA3 all-reduce exists** ([vLLM #57767](https://github.com/vllm-project/vllm/pull/57767)), exact and about 1.6× faster than RCCL at two
  cards. An eight-card path is the missing piece for this machine.

None of it needs new silicon. With Chapter 1.2's list, it is the difference between 4.59 and 76.4
tok/s at 60K on cards people already own. The performance was in the card the whole time.

## Reproduce it

In `tools/`: `allreduce_bench.py` (the collective, three algorithms, sums verified),
`true_rate.py` (decode counted in tokens rather than chunks), `spec_counts.py` and
`itko_speed.py` (acceptance and real-traffic speed from the server's own counters), and
`attn_gate.py` (the segment-count gate, with its exact reference and decoy-block control). The
patched files, the Containerfile that builds this configuration and the launcher are in
`patches/`, each with its hash.
