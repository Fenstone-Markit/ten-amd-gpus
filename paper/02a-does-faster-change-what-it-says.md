# Chapter 2.1: Does making it faster change what it says?

*The companion to Chapter 1.2. Every speed change there was checked against the model's own output
before I kept it. This chapter is how, and the three times the checking mattered.*

A faster server is only worth having if it still gives the same answers. That sounds obvious. It is
also the part of performance work that is easiest to skip, because a wrong kernel does not crash. It
returns numbers of the right shape, at the right speed, and the model built on top of it keeps
writing fluent text. You find out weeks later, if you find out at all.

I cannot read a GPU kernel and tell you whether its arithmetic is right. So I set one rule: **every
speed change has to pass a check that is capable of failing, and I have to have seen it fail.**

## Findings

1. **The four default fixes changed nothing in the output.** The attention backend, prefix caching
   and the attention launch tuning all produced byte-identical text before and after. The switch
   setting touches no arithmetic at all.
2. **A check that looks at one request at a time missed a real memory bug.** It certified a kernel
   that wrote outside its buffer whenever more than 64 requests ran together. Only a check that
   builds batches found it: a crash at 65, a hang at 128.
3. **A server can report healthy and be producing nonsense.** After a kernel change, vLLM reused a
   stale compiled graph. It started in three minutes, answered every request, and scored the
   reference text at a perplexity of 1.8 million, where the real model scores 1.61. Scoring caught
   it in one minute.
4. **The native kernel from Chapter 1.2 is correct, and it is not free.** It moves perplexity by
   0.2 %, the output is no longer bit-for-bit repeatable, and on one computed digit in structured
   output its confidence fell from 99 % to between 6 and 15 %.
5. **The cause is how the kernel adds its partial results, and it is fixable.** It rounds to a
   16-bit format after every partial addition, where the slower kernel rounds once. Keeping the sum
   in 32 bits should remove the cost and keep most of the speed.

## The checks, as knobs

The same idea as the speed knobs: what each check does, and what it catches. Use them together.

### Check 1: run the same request twice on the unchanged server

At temperature 0, with a fixed seed, a deterministic server gives identical output every time. Run
your reference set twice before changing anything.

| Result | What it tells you |
| --- | --- |
| Identical | Any difference after your change comes from your change. Here: 12 prompts, 1,649 tokens, identical both times, down to the last decimal of every probability |
| Not identical | Your server has its own run-to-run drift. Measure it first, then judge every later difference against it, not against perfection |

### Check 2: compare the generated text

Record the tokens and the model's top five choices at every step, then compare runs.

| Result | What it tells you |
| --- | --- |
| Byte-identical | Nothing changed in the numbers that decide the output |
| Diverges | Something changed. By itself this does not tell you whether it matters: see Check 4 |

### Check 3: score a fixed text on both servers

Feed both servers the exact same text and read back how likely each one finds every token of it.
Nothing is generated, so nothing can diverge. Any difference is purely how differently the two
servers judge identical text. This is the check that separates a small numerical change from a
broken model, because a broken model scores everything as unlikely.

| Result | What it tells you |
| --- | --- |
| Perplexity within a fraction of a percent | The model is intact |
| Perplexity far higher, on every text | Something systematic is wrong, whatever the server says about its health |

### Check 4: look at where the text diverged

When two runs split apart, look at the unchanged server's own margin between its first and second
choice at that token.

| Margin | What it means |
| --- | --- |
| Tiny | A coin flip. Any rounding difference upstream can tip it. Not a quality change |
| Large | The model's judgement moved. Worth finding out why |

### Check 5: prove the change actually happened

Flags can be ignored and caches can be reused. After any change to which kernel runs, look inside
what the server actually compiled and count the kernel calls you expect.

## The defaults: nothing changed

| Change from Chapter 1.2 | Speed effect | Output |
| --- | --- | --- |
| `--attention-backend TRITON_ATTN` | 4.59 to 20.68 tok/s at 60K | Byte-identical at 512, 8,192 and 16,384 tokens |
| `--enable-prefix-caching` | First word 26.61 s to 0.51 s | Byte-identical |
| Attention launch tuning (`num_warps` 8 for head size 256), tested on the default attention path | +16.7 % end to end at 8K | Byte-identical at every length tested |
| Clearing ACS redirect on the PCIe switch | +32 % prefill | Not applicable: it changes the route traffic takes, not any arithmetic |

Each was measured twice per point, and the text was compared as bytes rather than read.

## The bug a single request could not see

Speculative decoding needed a small change to one attention path, so that a two-token verification
step could use the fast route. I built a check for it, `gate.py`, which compared the patched
attention against a reference across many shapes. It passed.

It was wrong. The fast route sizes its scratch buffers for 128 query tokens, but indexes them by the
position of each token across the whole batch. With one request, that position never runs past the
buffer. With more than 64 requests doing two-token steps at once, it does:

| Requests in the batch | What happened |
| --- | --- |
| Up to 64 | Correct |
| 65 | `hipErrorIllegalAddress`: a write outside the buffer |
| 128 | The GPU hung |

Reproduced on two different model shapes. My check could not see it, because my check only ever sent
one request. The fix sizes the buffers for what the kernel actually indexes. The batched check went
from 32 of 40 cases passing to 40 of 40, all on the fast path.

It is a bug in upstream vLLM, and it stays fixed in the running server even though speculative
decoding itself was rejected (it made the 27B slower). The rule it left:

**A check for any change to attention must build batches, and must include a deliberately broken
version that it has to reject. A check that has never failed has never been tested.**

## The server that said it was fine

After patching which kernel runs the quantized layers (Chapter 1.2, Knob 7), I started the patched
server in a separate container, built from a snapshot of production. It came up in 190 seconds,
reported healthy, and answered requests.

It should have taken about twelve minutes. vLLM had found the graph it compiled that morning for the old
kernel and reused it, then handed it weights laid out for the new one. The tensors happened to be
the same total size, so nothing crashed and nothing read out of bounds. The kernel simply read the
numbers in the wrong order.

| Server | Mean log-probability | Perplexity |
| --- | --- | --- |
| Production | −0.476 | 1.61 |
| Patched, stale compiled graph | −14.41 | **about 1.8 million** |

For scale, guessing uniformly across this model's vocabulary would score about −12.4. The patched
server was worse than random, on all twelve texts. Scoring it took one minute.

The compiled graph confirmed it: 255 calls to the old kernel, zero to the new one. Pointing the
compile cache at an empty folder forced a genuine rebuild of about twelve minutes, and the graph then showed
255 native calls and zero old ones.

This was also the proof that Check 3 works. Before this I had only ever seen it pass. Now it had
failed, on a real fault, in the real system, loudly.

## The native kernel: correct, and not free

With the compile cache fixed, the patched server ran the kernel it was meant to.

**The model is intact:**

| Measure | Production | Native kernel |
| --- | --- | --- |
| Perplexity on 1,649 fixed tokens | 1.6093 | 1.6125 (+0.2 %) |
| Largest change in any text's average score | baseline | 0.011 |
| Same server run twice | Identical | Differs slightly |

A real bug, such as getting the zero points wrong, would move every weight by a full quantization
step and shift perplexity by far more than 0.2 %, the way the stale graph did.

I had written down, before the run, that differences should look like the patched server's own
run-to-run noise. They were about six times larger. That criterion was wrong, and I should have
seen why in advance: comparing the patched server with itself only measures the order its additions
happen in, while comparing it with production also includes a fixed difference in how it rounds.
I am recording the criterion and the miss rather than quietly moving the line.

**Where the generated text split:** six of twelve prompts diverged within 128 tokens. Five of them
were coin flips. Production's top two choices were one or two steps apart in the 16-bit format the
model computes in, as close as two choices can be, and any rounding difference tips them.

**The sixth was not a coin flip.** The model was writing a second entry in a JSON list, right after
"10 cards ... 32GB each", and the next tokens were the digits of a number in that entry:

| Token | Production | Native kernel, run A | Native kernel, run B |
| --- | --- | --- | --- |
| `'3'` | 99 % | 15 % | 6 % |
| `'2'` | 99.8 % | 45 % | 29 % |

Both patched runs sit far from production, on the same side. That is not noise scattered around the
right answer. It is a consistent loss of confidence on a number.

**Why:** the native kernel splits each matrix across many parts of the GPU and adds the partial
results into the output one at a time, rounding to 16 bits after each addition. The slower kernel
keeps its running sum in 32 bits and rounds once. That extra rounding adds a small error to every
one of the 255 matrix multiplies in a token. Most tokens do not care. A token whose value depends on
precise internal state, like a digit in structured output, can.

My kernel-level checks never showed this, because they compared variants of the native kernel
against the native kernel, which carries the same rounding. I never compared against an exact
reference. That is the lesson I am taking from it.

**What it costs in practice:**

- **On average, very little.** 0.2 % less confidence per token, a small fraction of what 4-bit
  quantization itself already costs.
- **In exact structured output, possibly more.** Numbers in JSON and tool calls are where near-ties
  and computed values live. My agent's work is mostly tool calls.
- **Repeatability.** Production gives identical answers on reruns. The native kernel does not, which
  matters for evaluations and for debugging.

## The fix, and how it will be judged

Keep the native kernel, but have it add its partial results in 32 bits and round to 16 once at the
end. It costs roughly one extra small step per matrix multiply, an estimated 0.8 ms per token, which
would keep most of the 12 ms gain. Measuring that estimate is the first job.

It has a ready-made test: the digit at token 131. Production is 99 % sure of it. A fixed kernel
should bring the patched server back close to that, with perplexity back at 1.609 and Check 3's
differences down near production's own. Until it passes, the speed in Chapter 1.2's Knob 7 is a
trade, and I am treating it as one.

## What this does not prove

Twelve prompts are enough to rule out a broken model and to find one sharp case. They are not enough
to price the precision cost exactly, and they are easy text that the model wrote itself. A larger set
of real text, with code, long documents and numbers, and the frozen rubric from Chapter 2 scored on
both kernels, would settle it.

## Reproduce it

`tools/ref_outputs.py` records a reference set, `tools/compare_refs.py` compares two runs, and
`tools/score_texts.py` scores fixed text and compares two servers. All three refuse to run if the
server is busy, so every number is taken alone. The batched attention check and its broken control
are in `tools/gate/`.
