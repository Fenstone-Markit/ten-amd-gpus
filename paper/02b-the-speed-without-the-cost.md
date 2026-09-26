# Chapter 2.2: The speed without the cost

*The companion to Chapter 1.3, and the fix Chapter 2.1 promised. Same machine, model and image
(`rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0`). Kernel measurements are on one card at the brain's exact shapes, against an exact
float64 reference. Brain measurements are at eight-way tensor parallel.*

Chapter 2.1 ended with a trade. The native kernel from Chapter 1.2 made the model 1.49 to 1.65×
faster, and it cost precision and repeatability, because it rounded to 16 bits after every partial
addition. I said the fix was to keep the sum in 32 bits, and that until it passed its test, the
speed was a trade and I was treating it as one.

I am not an engineer or a researcher. The claim running through these chapters is that the RX 7900
XTX has a lot being left on the table, and that claim is only worth making if the speed I found does
not come with a hidden cost. So I do not keep any speed I have not checked.

This chapter is that fix. It is also a correction to how Chapter 2.1 priced the cost, and one more
kernel that loads, runs, and returns wrong answers.

## Findings

1. **The fixed kernel is faster than the one it replaces, at the precision floor, and repeatable.**
   6.43 ms per token for the 255 quantized multiplies, against 6.66 for the native kernel and 17.97
   for production's Triton kernel, all timed the same way. Its error sits at the limit of the 16-bit
   format on every shape, and its output is bit-identical over 20 runs.
2. **Getting there took two attempts.** Keeping the sum in 32 bits with atomic additions fixed the
   precision but not the repeatability: the order of the additions still varied. Giving every part
   of the work its own slot and adding the slots in a fixed order fixed both.
3. **Chapter 2.1 priced the cost with a biased ruler.** The twelve texts it scored were written by
   production. Scoring any other server on production's exact words penalises it at every near-tie
   where it would have chosen differently. The 0.2 % perplexity cost in Chapter 2.1 is withdrawn as a
   measure of precision.
4. **The brain as a whole is still not bit-repeatable.** The decode kernel is. The prefill kernel,
   which reads each prompt, still adds in an unpredictable order. With prefill routed through the
   fixed kernel too, two runs of all twelve prompts were identical. That configuration is not usable:
   it runs out of memory on long prompts.
5. **Someone upstream built the same fix independently.** An open vLLM pull request ([vLLM #54706](https://github.com/vllm-project/vllm/pull/54706)), approved
   by an AMD maintainer, uses the same design for both the decode and the prefill kernels.
6. **A second kernel that looks healthy and is wrong.** AMD's QuickReduce all-reduce is compiled into
   this image. Forced onto these cards, it initialises, runs, and returns a wrong value in every
   element. The check that keeps it switched off on consumer cards is correct.

## The fix, in two attempts

| Kernel | ms per token, 255 calls, one token | Error against float64 | Same output twice |
| --- | --- | --- | --- |
| Triton, production's default | 17.97 | about the 16-bit floor | Yes |
| Native, 16-bit atomic additions (Chapter 1.2, Knob 7) | 6.66 | about 0.8 % run-to-run noise on every output | No |
| Native, 32-bit atomic additions (first attempt) | 7.09 | at the floor | No: rare one-step flips |
| **Native, fixed-order 32-bit sum (the fix)** | **6.43** | **at the floor** | **Yes, over 20 runs** |

**Why the first attempt was not enough.** Atomic additions are applied in whatever order the hardware
reaches them. In 32 bits the order barely matters, but it matters exactly at a near-tie in the final
rounding, which is rare and real. I had asked for repeatability, not just precision, so the first
attempt failed its own gate.

**What the fix does.** The matrix multiply is split into slices along its long dimension, as before.
Each slice now writes its 32-bit partial result into a slot of its own instead of adding into a
shared output. A second, small kernel then adds the slots in order, first to last, and rounds to 16
bits once. The same inputs produce the same additions in the same order, every time.

**Why it is also faster.** Atomic additions on this card have no native 16-bit form and are emulated
with a compare-and-swap loop that retries under contention. Plain writes to separate slots do not
contend at all. The extra pass that adds the slots costs less than the contention it removes.

On the brain, the fixed kernel replaced the native one at every context measured:

| Context | Native kernel (Chapter 1.2) | Fixed kernel |
| --- | --- | --- |
| 2,000 | 18.6 ms, 53.7 tok/s | **18.3 ms, 54.6 tok/s** |
| 16,000 | 20.0 ms, 49.9 tok/s | **19.7 ms, 50.7 tok/s** |
| 60,000 | 24.5 ms, 40.7 tok/s | **24.1 ms, 41.4 tok/s** |

The speed from Chapter 1.2 is kept, and slightly improved, without its cost. This is the starting
point of Chapter 1.3.

Its gate required, before a single line was written: error within 10 % of the 16-bit floor on every
shape and within 25 % of production's kernel; bit-identical output over 20 runs on every shape,
batch size and seed; at least 8 ms per token faster than production's kernel; and a deliberately
broken version, with wrong zero points, that the gate had to reject.

## The correction: a ruler that favoured production

Chapter 2.1's Check 3 scored a fixed text on two servers. The idea is sound: nothing is generated, so
nothing can diverge. The text was not neutral. It was the twelve answers production itself had
written.

A model's own output is, by construction, the sequence of tokens it found most likely. At every
near-tie, production's text contains production's choice. Any other server, even a more exact one,
pays for disagreeing at those tokens. Breaking the scores down by token showed it: in one text, a
JSON answer, the five most expensive tokens accounted for 59 to 117 % of each alternative server's
entire perplexity gap.

The clearest sign came from the most exact configuration I have run, with every multiply through the
fixed kernel. It scored 1.6139, **worse** than the less exact configuration's 1.6124, on production's
text. A more exact server scoring worse is what a biased ruler predicts.

So:

- **Chapter 2.1's 0.2 % perplexity cost is withdrawn** as a measure of the native kernel's precision.
  Its other findings stand: the loss of repeatability, the confidence drop on the digit, and the
  cause.
- **Perplexity comparisons need neutral text,** written by people, not by any of the servers being
  compared. That test is not run yet. Until it is, the evidence for the fixed kernel's quality is the
  kernel-level gate against an exact reference, which is the stronger test anyway.
- **The digit test Chapter 2.1 set** came from the same twelve texts, so it inherits the same bias. It
  moves to the neutral set.

## Repeatability of the whole brain

Chapter 2.1's Check 1 found production perfectly repeatable: the same twelve prompts twice, identical
to the last decimal. The fixed kernel restores that property to the decode kernel. The brain as a
whole does not have it:

| Comparison | Prompts that diverge, of 12 | Largest difference in log-probability | Top five choices agree |
| --- | --- | --- | --- |
| The current brain against itself, run twice | 5 | 0.20 | 95.0 % |
| Attention at 16 segments against 128 (Chapter 1.3, Knob 10) | 7 | 0.70 | 94.6 % |

The first row is the baseline every later comparison has to be judged against, and it is not zero.
The cause is the prefill kernel, which reads each prompt with 16-bit atomic additions, the same
pattern this chapter removed from decode. The proof was a test configuration that routed prefill
through the fixed kernel as well: the same twelve prompts twice, **none diverged**, every top-five
set identical.

That configuration is not deployable. The fixed kernel's slots grow with the number of tokens, and a
2,000-token prompt asked for 666 MiB it did not have. The real fix is a fixed-order version of the
prefill kernel, which is the second half of the upstream pull request below.

The second row, the attention change, sits close to the first. At the kernel level its error against
float64 is identical to five decimal places. The difference at the brain level is a different
summation order tipping a few more near-ties, and the neutral-text scoring will price it properly.

## Someone else built the same fix

While this work ran, a vLLM pull request ([vLLM #54706](https://github.com/vllm-project/vllm/pull/54706)) arrived at the same design for the same kernels:
32-bit partial results, reduced in a fixed ascending order, rounded once. It covers both the decode
kernel and the prefill kernel. At the time of writing it is open, approved by an AMD maintainer and
labelled ready. Two independent routes to one design is a reasonable sign it is the right one.

## A kernel that is wrong in every element

vLLM ships a fast all-reduce for AMD's datacenter cards, QuickReduce, and switches it off on consumer
cards by architecture name. The image I run compiles it for exactly these consumer cards, so I
forced it on, on two cards, outside the brain, with a check it could not talk its way past: card one
adds 1, card two adds 2, and every element must come back as 3.

| Message | Elements returned wrong |
| --- | --- |
| 10 KB | 5,120 of 5,120 |
| 2.6 MB, above its own minimum size | 1,310,720 of 1,310,720 |

It set up cleanly, exchanged its buffers, ran, and did not hang. It simply returned no correct
values. A developer upstream ([vLLM #57925](https://github.com/vllm-project/vllm/pull/57925)) has since traced it to one constant: a memory-descriptor setting
written in the older datacenter encoding, which these cards read differently, so every load returns
zero. The architecture check that keeps it off is protecting users. It should stay until that
constant is fixed.

The all-reduce written for these cards ([vLLM #57767](https://github.com/vllm-project/vllm/pull/57767)) passed the same check, and matched
RCCL bit for bit on random data at every size tested, from 1 to 24 tokens.

## The checks, continued

Chapter 2.1 listed five checks. This chapter added four.

### Check 6: judge a kernel against an exact reference

Compute the answer in float64 from the same quantized inputs, and score every kernel against that.

| Result | What it tells you |
| --- | --- |
| Near the 16-bit floor | The kernel is as exact as its output format allows |
| Near another kernel | Nothing, if the two share their arithmetic. This is how Chapter 2.1's first kernel checks missed the precision cost |

### Check 7: require identical bits against the kernel you are replacing

For a change that must not alter any arithmetic, such as merging two kernels into one, "close" is not
the bar. Every output element must be bit-identical to the kernel being replaced, on every shape,
batch size and seed, including when replayed inside a CUDA graph.

### Check 8: make the broken control break what matters

A broken control proves a check can fail. It only proves that if it breaks the thing the check is
for. My first attention check shifted each request's cache blocks by one position, and at long
context the check barely noticed: 7 % error, below its own bar. Attention treats the keys as a set,
so shuffling a request's own blocks hardly changes the answer. The control that works points at
different blocks with different contents, and produces an error of 100 %.

**A control that fails to fail is itself the finding.** The check refused to run, which was correct,
and the control was redesigned before anything was measured.

### Check 9: prove the code under test is the code in production

My agent's test environment had carried the stock attention files, without production's patches,
for days. Nothing had noticed, because every earlier check ran against whatever was installed. The
attention check recorded the fingerprint of production's files and refused to run anywhere else. It
refused, and that refusal found the gap.

| Result | What it tells you |
| --- | --- |
| Fingerprints match | The result applies to production |
| They differ | Nothing measured there applies to production. Stop, and fix the environment first |

## What this does not prove

The fixed kernel's precision is proven at the kernel level, against an exact reference, on the
brain's own shapes. Its effect on the model's answers is not yet measured on neutral text, because
the only text I had was biased. The whole brain is not repeatable until the prefill kernel is fixed
as well. Twelve prompts are still twelve prompts.

## To AMD, and to vLLM's ROCm maintainers

- **[vLLM #54706](https://github.com/vllm-project/vllm/pull/54706) is the fix for Chapter 2.1's precision cost,** and it covers the prefill
  kernel this chapter could not fix in time. It has an AMD approval. Merging it would make the
  fastest path on this card also the exact and repeatable one.
- **QuickReduce is compiled into the consumer image and returns wrong values there.** Either compile
  it out of consumer builds or fix the descriptor constant. The architecture check is the only thing
  standing between users and silent corruption, and anyone who removes it to chase speed gets
  exactly that.

The fixed kernel was a small change to a kernel AMD's contributors had already written well. The
precision and the repeatability were available on this card all along.

## Reproduce it

The kernel gates are in `tools/kernel/`: `gate_det.py` (the fixed-order gate: exact reference,
20-run repeatability, the wrong-zeros control) and `fp32_op_gate.py` (packages the kernel as a
PyTorch operation and proves the compiled server calls it). The all-reduce checks are in `tools/`:
`qr_test.py` (QuickReduce, exact-sum and bit-for-bit against RCCL) and `rdna_ar_test.py` (the RDNA3
all-reduce, the same checks, through CUDA graph replays). The fixed kernel's source, its build and
the patched layer file are in `patches/`, each with its hash.
