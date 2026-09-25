# Ten AMD GPUs as Inference Infrastructure

Ten RX 7900 XTX on one EPYC board, in a garage in Edmonton. No Infinity Fabric, no NVLink, no
vendor support contract. Every card talks to every other card over PCIe Gen4 x8, through one PCIe
switch, and nothing else.

I am not a researcher. I run logistics for a living, I built this machine to do real work, and I
wanted it faster. This repository is every wall I hit and how I got past it, measured on the machine
and published as it goes, including the parts that turned out to be wrong.

## The headline

| On one 27B model, the one my agent runs on | Where I started | Where it is now |
| --- | --- | --- |
| Decode at 60,000 tokens of context | 4.59 tok/s | **40.7 tok/s** |
| Decode at 2,000 tokens | 21.5 tok/s | **53.7 tok/s** |
| Wait for the first word in a long conversation | 26.6 s | **0.2 s** |

Same cards, same model, same weights. **The silicon was never the limit. The tuning was.** Every
knob that moved these numbers, and what happens if you turn it either way, is in
[Chapter 1.2](paper/01b-what-the-defaults-cost.md).

## What has been found so far

**Four defaults cost 4.5× at long context, and none of them warn you.** The attention backend,
prefix caching, the KV cache format, and a PCIe switch quietly routing card-to-card traffic through
the CPU. No error, no log line, a clean dashboard.
[Chapter 1.2](paper/01b-what-the-defaults-cost.md)

**A native RDNA3 kernel ships in the image and is never used.** vLLM runs this model's quantized
layers through a slower Triton path because one format is missing from one list. Selecting the
native kernel is 1.49 to 1.65× faster at every context length measured.
[Chapter 1.2](paper/01b-what-the-defaults-cost.md)

**Faster can change what the model says, and a server can look healthy while talking nonsense.** A
stale compiled graph produced a perplexity of 1.8 million on a server reporting healthy. The native
kernel is correct but costs 0.2 % in perplexity and bit-for-bit repeatability, with one sharp miss on
a digit in structured output. The fix is in progress.
[Chapter 2.1](paper/02a-does-faster-change-what-it-says.md)

**The container image is the largest single variable in throughput.** Two AMD-published images
built five days apart differ by up to 53 % in decode at 16K context, on identical hardware and
weights, and are indistinguishable at 512 tokens.
[Chapter 1.1](paper/01a-what-the-image-costs.md)

**Fewer cards per model beats more, for throughput.** Two four-card servers delivered 1.71× the
throughput of one eight-card server, for 1.19× the power. Ten cards serve a 357B model at 23.3
tok/s. [Chapter 1](paper/01-throughput-and-power.md)

**Model size does not predict output quality, and carefully scoped 4-bit costs nothing
measurable.** Five models from 35B to 120B landed at 6 or 7 out of 10 on a frozen rubric. No model
fabricated anything across roughly thirty correction turns.
[Chapter 2](paper/02-output-quality.md)

**Defects in shipped software, found by running it.** A class-name list that stops any
compressed-tensors W4A16 mixture-of-experts from loading on gfx1100
([vLLM #56790](https://github.com/vllm-project/vllm/issues/56790)). An attention buffer sized for
128 tokens that writes out of bounds above 64 concurrent requests. A kernel tile tuned for a
40-compute-unit part and frozen in at compile time. A fast all-reduce switched off by architecture
name on cards that pass the check it actually needs. Like the first, they shipped without a
gfx1100 test runner in the loop to catch them. [Chapter 1.1](paper/01a-what-the-image-costs.md),
[Chapter 1.2](paper/01b-what-the-defaults-cost.md),
[Chapter 2.1](paper/02a-does-faster-change-what-it-says.md)

## Chapters

| Chapter | What it covers |
| --- | --- |
| [1: Throughput and power](paper/01-throughput-and-power.md) | What ten consumer AMD cards deliver, what they draw doing it, and where the platform stops |
| [1.1: What the image costs](paper/01a-what-the-image-costs.md) | The software stack is not a constant. Two images, four configurations, up to 53 % apart at long context |
| [1.2: What the defaults cost](paper/01b-what-the-defaults-cost.md) | Seven knobs, each with both directions. 4.59 to 40.7 tok/s at 60K on the same hardware |
| [2: Output quality](paper/02-output-quality.md) | Six models against a frozen rubric. Size did not predict quality, and quantization cost nothing |
| [2.1: Does faster change what it says?](paper/02a-does-faster-change-what-it-says.md) | Five checks for any speed change, the bug a single request could not see, and the honest cost of the fastest kernel |

## Tools

Everything used to produce the numbers is here, so the numbers can be checked.

| Tool | What it does |
| --- | --- |
| [`bench/n02-bench`](bench/n02-bench) | Throughput harness. Exact prompt lengths verified against the server's tokenizer, pinned output length, discarded warm-up, median of three, decode measured separately from prefill |
| [`tools/`](tools/) | The decode benchmark and quality gate behind Chapters 1.2 and 2.1: speed at 2K, 16K and 60K, reference recording, text comparison, fixed-text scoring, the kernel patch builder and the test-container launcher |
| [`tools/kernel/`](tools/kernel/) | Isolated kernel measurements, each with a correctness gate and a broken control it has to reject |
| [`tasks/task-01-sysfs-collector.md`](tasks/task-01-sysfs-collector.md) | The frozen evaluation rubric. Ten binary traps, all mechanically checkable, all derived from defects hit during a real build |
| [`tasks/scores.jsonl`](tasks/scores.jsonl) | Every score, one row per model per run, with configuration and caveats attached |
| [`bench/`](bench/) | Raw JSONL output behind every figure in the chapters |

## The machine

| Component | Specification |
| --- | --- |
| GPUs | 10× RX 7900 XTX, gfx1100, 240 GB aggregate VRAM, four manufacturers, bought secondhand |
| CPU | EPYC 7663, 56 core Milan |
| Board | ASRock Rack ROMED8-2T/BCM |
| Memory | 512 GB DDR4-3200 ECC, 8 channels |
| Interconnect | PCIe Gen4 x8 to every card, through one PEX880xx Gen4 switch. No fabric |
| Power | Multiple supplies on a 20 A 240 V dedicated circuit |
| Cooling | Open bench, garage |

Every measurement in this repository names the container image it was taken on. Given the finding
in Chapter 1.1, a throughput number without an image tag is not a measurement.

## Coming

**The native kernel without its cost.** A version that adds its partial results in 32 bits, which
should keep most of Chapter 1.2's speed and remove Chapter 2.1's precision cost. It already has its
test: one digit that production is 99 % sure of.

**The last 10 ms.** About 10 ms of every token still have no measured owner: the 48 linear-attention
layers, the norms, and 128 all-reduces. An open pull request adds a fast all-reduce for RDNA3
(vLLM #57767), and it gets tested here next.

**A harder task.** Task 01 establishes a floor and cannot rank above it. The next one needs traps
that separate rather than gate.

**Tool access.** Every defect recorded in Chapter 2 was discoverable by one shell command. If defect
counts collapse when the model can run commands, the gap is grounding rather than capability.

## Corrections

Wanted, particularly on anything here that is wrong. Several claims in earlier drafts were withdrawn
after being disproved by the machine itself, and those withdrawals are recorded in the chapters
rather than deleted.

Open an issue.

## License

MIT. Fenstone Markit is the trading name of SovereignAI Solutions Inc., Alberta, Canada.
