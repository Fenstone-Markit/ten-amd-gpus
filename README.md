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
| Decode at 60,000 tokens of context | 4.59 tok/s | **145 to 162 tok/s** |
| Decode at 16,000 tokens | 11.5 tok/s | **150 to 165 tok/s** |
| Wait for the first word in a long conversation | 26.6 s | **0.6 s** |
| My agent's real work, from the server's own counters, at an average of 104,000 tokens of context | not measured | **134 tok/s** |
| The frozen quality rubric (Task 01) | | **8 of 10, passed in 2 turns, the best on record** |

Same cards, same model, same weights, with the quantized multiplies at the precision floor. The speed
range is speculative decoding: the more predictable the text, the more of its three draft tokens are
accepted per step. **The silicon was never the limit. The tuning was.**

## To AMD: what you are leaving on the table

Every row below is a software change, measured on this machine. Most are a few lines.

| The fix | What it was worth here | Proof |
| --- | --- | --- |
| Accept asymmetric `uint4` in the RDNA3 W4A16 kernel's format list | 1.49 to 1.65× faster decode, from one missing list entry | [1.2](paper/01b-what-the-defaults-cost.md) |
| Merge the fixed-order version of that kernel ([vLLM #54706](https://github.com/vllm-project/vllm/pull/54706)) | The same speed, at the precision floor, bit-for-bit repeatable | [2.2](paper/02b-the-speed-without-the-cost.md) |
| Size long-context attention to the card instead of a fixed 16 segments | Attention at 60K from 7.2 to 1.6 ms per token | [1.3](paper/01c-what-the-rest-of-the-stack-costs.md) |
| An eight-card RDNA3 all-reduce: I extended [vLLM #57767](https://github.com/vllm-project/vllm/pull/57767) from two and four cards to eight | 3.8× faster than RCCL at decode size, exact on every card; 38 % faster decode end to end | [rc3](patches/rc3/README.md) |
| Let the long-context attention path verify four tokens at once, not two (two lines) | Three draft tokens become nearly free: decode at 60K goes from 110 tok/s to between 145 and 162 | [rc3](patches/rc3/README.md) |
| Fix QuickReduce for consumer cards, or compile it out of their image | Today it returns a wrong value in every element on this card | [2.2](paper/02b-the-speed-without-the-cost.md) |
| An RDNA3 continuous-integration runner | Every defect in this repository shipped without one | all |

With the defaults and settings in Chapters 1.2 and 1.3 and the rc3 patches, these took one model from
4.59 to about 160 tok/s at long context, on cards people already own. **The RX 7900 XTX already has the silicon. The software is
leaving it on the table.**

## What else I found

- **Four defaults cost 4.5× at long context, and none of them warn you.** [Chapter 1.2](paper/01b-what-the-defaults-cost.md)
- **A setting can reverse with the number of cards.** The faster all-reduce at four cards is the slower one at eight. [Chapter 1.3](paper/01c-what-the-rest-of-the-stack-costs.md)
- **The collective was the largest cost, and replacing it was the largest single fix.** Extending an RDNA3 all-reduce to eight cards made each call 3.8× faster than RCCL and decode 39 % faster end to end. [rc3](patches/rc3/README.md)
- **A server can report healthy while talking nonsense.** A stale compiled graph scored a perplexity of 1.8 million. [Chapter 2.1](paper/02a-does-faster-change-what-it-says.md)
- **The container image is the largest single variable.** Two AMD images five days apart differ by up to 53 % at 16K. [Chapter 1.1](paper/01a-what-the-image-costs.md)
- **Fewer cards per model beats more, for throughput.** Ten cards serve a 357B model at 23.3 tok/s. [Chapter 1](paper/01-throughput-and-power.md)
- **Model size does not predict output quality**, and carefully scoped 4-bit costs nothing measurable. [Chapter 2](paper/02-output-quality.md)
- **Draft tokens are nearly free once attention can verify several at once.** Each extra one costs about 0.7 ms per step. [rc3](patches/rc3/README.md)
- **Speed did not cost quality.** After every change above, the agent's model scored 8 of 10 on the frozen rubric and fixed both defects from evidence alone. [tasks/](tasks/)

## Chapters

| Chapter | What it covers |
| --- | --- |
| [1: Throughput and power](paper/01-throughput-and-power.md) | What ten consumer AMD cards deliver, what they draw doing it, and where the platform stops |
| [1.1: What the image costs](paper/01a-what-the-image-costs.md) | The software stack is not a constant. Two images, four configurations, up to 53 % apart at long context |
| [1.2: What the defaults cost](paper/01b-what-the-defaults-cost.md) | Seven knobs, each with both directions. 4.59 to 40.7 tok/s at 60K on the same hardware |
| [1.3: What the rest of the stack costs](paper/01c-what-the-rest-of-the-stack-costs.md) | Three more knobs, a profile of every kernel in a token, and a check against real work. 41.4 to 76.4 tok/s at 60K |
| [2: Output quality](paper/02-output-quality.md) | Six models against a frozen rubric. Size did not predict quality, and quantization cost nothing |
| [2.1: Does faster change what it says?](paper/02a-does-faster-change-what-it-says.md) | Five checks for any speed change, the bug a single request could not see, and the honest cost of the fastest kernel |
| [2.2: The speed without the cost](paper/02b-the-speed-without-the-cost.md) | The fixed kernel, a correction to how 2.1 priced the cost, and a kernel that is wrong in every element |
| [rc3: the all-reduce and the draft tokens](patches/rc3/README.md) | The eight-card all-reduce, a tuned kernel, and the attention change that made three draft tokens work. 76.4 to about 160 tok/s at 60K |

## Tools

| Folder | What is in it |
| --- | --- |
| [`patches/`](patches/) | Everything that turns the stock image into the configuration in Chapters 1.3 and 2.2, and [`patches/rc3/`](patches/rc3/) for the rest: diffs, kernels, Containerfiles, tests and gates, all hashed |
| [`tools/`](tools/) | Every script behind every figure, each gate with a broken control it has to reject |
| [`bench/`](bench/) | Raw output behind every figure, including `n02-bench`, the throughput harness; [`bench/rc3/`](bench/rc3/) for the rc3 releases |
| [`tasks/`](tasks/) | The frozen evaluation rubric and every score |

## The machine

| Component | Specification |
| --- | --- |
| GPUs | 10× RX 7900 XTX, gfx1100, 240 GB aggregate VRAM, four manufacturers, bought secondhand |
| CPU and board | EPYC 7663 (56-core Milan) on an ASRock Rack ROMED8-2T/BCM |
| Memory | 512 GB DDR4-3200 ECC, 8 channels |
| Interconnect | PCIe Gen4 x8 to every card, through one PEX880xx Gen4 switch. No fabric |
| Power and cooling | Multiple supplies on a 20 A 240 V circuit, open bench, garage |
| The agent's model | `cyankiwi/Qwen3.8-27B-AWQ-INT4`, revision `6e134bae` |

Every measurement names the container image it was taken on. After Chapter 1.1, a throughput number
without an image tag is not a measurement.

## Left open

Measured far enough that anyone can pick them up:

- **Serving more than one user at once.** Every number here is one request at a time.
- **Startup.** A warm restart takes about eight minutes, most of it re-timing kernels that were timed last time.
- **The all-reduce fused with the norm**, and **an 8-bit conversation cache**: each worth single-digit percent, measured far enough to start.
- **One kernel per multiply.** A fused version was correct but slower at two rows and up; the separate reduction pass stays.
- **Precision on neutral text**, replacing the figure Chapter 2.2 withdrew.
- **A harder evaluation task**, with tool access, that separates models above the floor Task 01 sets.

## Corrections

Wanted, particularly on anything here that is wrong. Withdrawals are recorded in place, never deleted:

| Where | What was wrong | Settled in |
| --- | --- | --- |
| Chapter 1.2, Knob 6 | The tree all-reduce was measured at four cards; at eight it is 3.2 ms per token slower | Chapter 1.3 |
| Chapter 1.2, the short version | A 0.2 s first word that the chapter's own Knob 1 measured as 0.51 s | Chapter 1.2, in place |
| Chapter 1.2, knobs that did not work (and Chapter 2.1, which repeats it) | Speculative decoding: slower at four cards with the Triton kernel, 35 % faster at eight with the fixed kernel | Chapter 1.3 |
| Chapter 2.1, Finding 4 | The 0.2 % perplexity cost was measured on production's own text, which favours production | Chapter 2.2 |
| This README's earlier headline | 40.7 tok/s carried the precision cost of Chapter 2.1 without saying so | Chapter 2.2 |

Open an issue.

## License

MIT, except files that carry their own license header. Files derived from vLLM keep their Apache-2.0
headers; the license text is in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). Fenstone Markit is the trading
name of SovereignAI Solutions Inc., Alberta, Canada.
