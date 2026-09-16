# Ten AMD GPUs as Inference Infrastructure

Ten RX 7900 XTX on one EPYC board, in a garage in Edmonton. No Infinity Fabric,
no NVLink, no vendor support contract. Every card talks to every other card over
PCIe Gen4 x8 and nothing else.

This repository is what has been learned running it. It is not a finished paper.
It is a record that gets longer as the machine teaches us something, published as
it goes, including the parts that turned out to be wrong.

## Why publish it

Almost nothing exists on running serious local inference on consumer AMD
hardware at this scale. There is excellent work on one or two cards, and there
are vendor numbers from Instinct parts nobody reading this owns. Between those
sits a gap: what actually happens when you put ten gaming cards in a box and try
to serve real models on them.

Most of what is here was found by hitting it. Some of it contradicts what we
believed a month ago. That is recorded too, in place, rather than edited out.

## What has been found so far

**The container image is the largest single variable in throughput.** Two
AMD-published images built five days apart differ by up to 53 percent in decode
throughput at 16k context, on identical hardware and weights. At 512 tokens they
are indistinguishable, which is why a short-prompt benchmark would call them
equivalent. [Chapter 1.1](paper/01a-what-the-image-costs.md)

**Model size does not predict output quality.** Six models scored against a
frozen ten-trap rubric. Five landed at 6 or 7 out of 10 regardless of parameter
count, active parameters, vendor or quantization. A 229B model did not beat a
35B one. [Chapter 2](paper/02-output-quality.md)

**Carefully scoped 4-bit quantization costs nothing measurable.** Two controlled
pairs, same weights in bf16 and AWQ INT4, identical first-output scores down to
which specific traps failed. The caveat is that this is a property of the
uploader's module selection as much as of the method.
[Chapter 2](paper/02-output-quality.md)

**No model fabricated anything.** Across six models and roughly thirty
correction turns, not one invented a sysfs path or claimed an unavailable tool to
cover being told it was wrong. [Chapter 2](paper/02-output-quality.md)

**Two defects in AMD-published images, found by running them.** A hardcoded
gated-activation factor that breaks non-gated MoE models, and a class-name list
missing its RDNA3 entry that prevents any compressed-tensors W4A16 MoE from
loading at all. Both shipped because there is no gfx1100 CI runner.
[Chapter 1.1](paper/01a-what-the-image-costs.md),
[vLLM #56790](https://github.com/vllm-project/vllm/issues/56790)

## Chapters

| | |
|---|---|
| [01 — Throughput and power](paper/01-throughput-and-power.md) | What ten consumer AMD cards deliver, and what they draw doing it. |
| [01a — What the image costs](paper/01a-what-the-image-costs.md) | The software stack is not a constant. Two images, four configurations, up to 53 percent apart at long context. |
| [02 — Output quality](paper/02-output-quality.md) | Six models against a frozen rubric. Size did not predict quality, and quantization cost nothing. |

## Tools

Everything used to produce the numbers is here, so the numbers can be checked.

| | |
|---|---|
| [`bench/n02-bench`](bench/n02-bench) | Throughput harness. Exact prompt lengths verified against the server's tokenizer, pinned output length, discarded warm-up, median of three, decode measured separately from prefill. |
| [`tasks/task-01-sysfs-collector.md`](tasks/task-01-sysfs-collector.md) | The frozen evaluation rubric. Ten binary traps, all mechanically checkable, all derived from defects hit during a real build. |
| [`tasks/scores.jsonl`](tasks/scores.jsonl) | Every score, one row per model per run, with configuration and caveats attached. |
| [`bench/`](bench/) | Raw JSONL output behind every figure in the chapters. |

## The machine

| | |
|---|---|
| GPUs | 10× RX 7900 XTX, gfx1100, 240 GB aggregate VRAM |
| CPU | EPYC 7663, 56 core Milan |
| Board | ASRock Rack ROMED8-2T/BCM |
| Memory | 512 GB DDR4-3200 ECC, 8 channels |
| Interconnect | PCIe Gen4 x8 to every card, no bridges, no fabric |
| Power | Multiple supplies on a 20 A 240 V dedicated circuit |
| Cooling | Open bench, garage |

Every measurement in this repository names the container image it was taken on.
Given the finding in Chapter 1.1, a throughput number without an image tag is not
a measurement.

## Coming

**Kernel work.** Decode Context Parallelism exists upstream and shards the KV
cache along the sequence dimension rather than by attention head, which is the
direct fix for the head-count floor. It is unavailable on gfx1100 because no
attention backend on this architecture returns the softmax log-sum-exp during
decode. The state needed to produce it is already computed and already exported
by the Triton path. There is precedent for the work on two other architectures
and none on this one.

**A harder task.** Task 01 establishes a floor and cannot rank above it. The next
one needs traps that separate rather than gate.

**Tool access.** Every defect recorded in Chapter 2 was discoverable by one shell
command. If defect counts collapse when the model can run commands, the gap is
grounding rather than capability, which is worth more than any score in the
table.

## Corrections

Wanted, particularly on anything here that is wrong. Several claims in earlier
drafts were withdrawn after being disproved by the machine itself, and those
withdrawals are recorded in the chapters rather than deleted.

Open an issue.

## License

MIT. Fenstone Markit is the trading name of SovereignAI Solutions Inc.,
Alberta, Canada.
