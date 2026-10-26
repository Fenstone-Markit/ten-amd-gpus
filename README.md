# Ten AMD GPUs as Inference Infrastructure

Twelve RX 7900 XTX on one EPYC board (ten until 2026-10-24), in a garage in Edmonton. No Infinity Fabric, no NVLink, no
vendor support contract. Every card talks to the others over PCIe Gen4 x8: ten through one PCIe switch, two on a bifurcated
slot beside it, and nothing else.

I am not a researcher. I run logistics for a living, I built this machine to do real work, and I
wanted it faster. This repository is every wall I hit and how I got past it, measured on the machine
and published as it goes, including the parts that turned out to be wrong.

## Now: a 320B model runs my agent, on twelve cards

| GLM-5.3-Flash, 320B mixture of experts, my agent's brain | Measured (2026-10-25) |
| --- | --- |
| Context window | **393,216 tokens** |
| A follow-up turn at 48,000 tokens of history, to the first word | **1.4 s** (about 18 s before prefix caching) |
| Reading a new 116,400-token prompt | **32.7 s**, about 3,550 tok/s (54.0 s the same morning) |
| Decode: counting / code / prose, two draft tokens | **59.5 / 46.9 / 41.0 tok/s** |
| Tool calls, code tasks, needles, maths | **20/20, 12/12, 14/16, 298/300** |
| Experts on FP8 kernels written for this card | 7 of 42 layers and the draft layer; RDNA3 has no FP8 matrix instructions |
| Official support for these cards | none: every official GLM-5.x path is AMD Instinct through AITER |

It runs through a community RDNA4 port, the patches and kernels my assistant and my agent wrote on top of it, and several things found the hard way. On a nearly full card, ROCm hands out memory a live buffer still owns; the model's start-up profiling of a vision encoder filled the middle cards to that edge until it served text only. Since then: our own FP8 kernels, multi-token prediction across pipeline stages (five fixes, one of them a correctness bug), prefix caching, and three slow fallbacks for prompt-sized steps found and replaced, one by a new kernel that is also the more exact one. What was available and what we built is set out side by side in [Chapter 9](paper/09-what-we-were-given-and-what-we-built.md). [Chapter 5](paper/05-a-321b-model-on-ten-cards.md), [Chapter 6](paper/06-the-memory-underneath.md), [Chapter 8](paper/08-serving-it.md)

## Where it started: one 27B model

| On one 27B model, my agent's brain until 2026-10-24 | Where I started | Where it got to |
| --- | --- | --- |
| Decode at 60,000 tokens of context | 4.59 tok/s | **145 to 162 tok/s** |
| Decode at 16,000 tokens | 11.5 tok/s | **150 to 165 tok/s** |
| Wait for the first word in a long conversation | 26.6 s | **0.6 s** |
| My agent's real work, from the server's own counters, at an average of 104,000 tokens of context | not measured | **134 tok/s** |
| The frozen quality rubric (Task 01) | no baseline | **8 of 10, passed in 2 turns, the best on record** |

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
| A GLM-5.3-Flash path for RDNA, not only the MI355X (vLLM main's indexer refuses to run without AITER; the gfx11 sparse-attention PR [#59628](https://github.com/vllm-project/vllm/pull/59628) is open) | A 320B model serving 393,216 tokens on twelve consumer cards, through a community port | [5](paper/05-a-321b-model-on-ten-cards.md), [8](paper/08-serving-it.md), [9](paper/09-what-we-were-given-and-what-we-built.md) |
| Stop handing out memory that a live, mapped virtual-memory handle still owns | Live buffers overwritten in 2 MiB pages, with no error, whenever PyTorch releases memory on a nearly full card; a one-card test shows it in seconds | [6](paper/06-the-memory-underneath.md), [the test](tools/vmm-repro/README.md) |
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
- **A small model's missing knowledge cannot be prompted away, or searched away.** A family tutor on one card invented a game's ending under instructions not to, and again from search snippets; what worked was letting the small model decide, fetching full text, and having the 27B write. [Chapter 3](paper/03-a-tutor-on-one-card.md)
- **A consumer NVMe drive can stop answering with its link healthy, and the model keeps serving from memory.** Chapter 4 blamed a sleep state; withdrawn: it hung again with every power-saving state off and a clean health log. [Chapter 4](paper/04-running-it.md), [Chapter 7](paper/07-heat-a-hang-and-a-repaste.md)
- **Every RX 7900 XTX reports x16, wherever it sits.** The real link is at the card's own upstream port: x8 on the switch, and x8 in an x16 slot on a damaged refurbished card. [Chapter 4](paper/04-running-it.md)
- **A 321B model serves 131,072 tokens on ten consumer cards, and survives a 2-hour soak.** GLM-5.3-Flash read a 120,000-token prompt in 44 s and decoded at 35 tok/s in Chapter 5's tests; llama.cpp reads the same model about 15 times slower. It failed in service until Chapter 8. [Chapter 5](paper/05-a-321b-model-on-ten-cards.md), [Chapter 8](paper/08-serving-it.md)
- **One speed shortcut broke every window above 32,768, four different ways.** A wrong answer, a dead start-up, an error, a GPU crash: all from chunk counts built on the GPU. Building them on the CPU, as upstream does, fixed all four at no measurable cost. [Chapter 5](paper/05-a-321b-model-on-ten-cards.md)
- **On a nearly full card, memory handed back to ROCm can be given out again while a live buffer still owns it.** With PyTorch's expandable segments on ROCm 10, a buffer nothing writes to was overwritten in whole 2 MiB pages; a plain `hipMalloc` made after the release landed on it. Keeping released memory inside the process removes it on a one-card test. ROCm 7.2.4 looks clean only because released memory never comes back there. [Chapter 6](paper/06-the-memory-underneath.md), [the tests](tools/vmm-repro/README.md)
- **A cache folder that cannot be written cost three kilowatts.** A model start ran ten cards at about 2,990 W for over two minutes without progress because AMD's convolution library could not save its tuning. [Chapter 7](paper/07-heat-a-hang-and-a-repaste.md)
- **One card's heat was its paste.** 110 C within seconds before a PTM7950 pad, 83 C after, at the same clock setting and power. A clock cap is a target, not a ceiling. [Chapter 7](paper/07-heat-a-hang-and-a-repaste.md)
- **About 5 GiB on each middle card went to profiling a vision encoder nothing here uses.** vLLM sizes memory for the largest input a model accepts, a dummy video for GLM. Serving text only gave it back, and that is what made GLM serve. [Chapter 8](paper/08-serving-it.md)
- **Exact recall falls with length, even when nothing is broken.** 90 of 92 needles found below 10,000 tokens, 9 of 14 above 90,000, every miss a nearby line. [Chapter 8](paper/08-serving-it.md)
- **Every expert gets used; kinds of work route differently.** Across seven kinds of work no expert went unused in any layer; medicine routes most like general knowledge, code most like agent work, Arabic and Urdu least like anything. Counted, not pruned. [Chapter 8](paper/08-serving-it.md), [the count](tools/expert-count/README.md)
- **FP8 runs on cards with no FP8 hardware, at the cost of memory, not speed.** Kernels that convert e4m3 on the way in reach 898 GB/s in decode (94 % of the card's bandwidth); an FP8 expert layer costs the same per decode step as a 4-bit one. [Chapter 9](paper/09-what-we-were-given-and-what-we-built.md)
- **Draft tokens across pipeline stages needed five fixes, and one was a wrong answer waiting to happen.** The earlier stages' linear-attention layers never learned how many drafts were accepted. [Chapter 9](paper/09-what-we-were-given-and-what-we-built.md)
- **The port's slow paths were fallbacks for shapes it did not expect.** A prompt read beside a generating request ran 130 s instead of 37; two more fallbacks were 40 % of prompt time. Replacing them took a 116k read from 54 s to 33 s. [Chapter 9](paper/09-what-we-were-given-and-what-we-built.md)
- **The stack carries to other models.** Seven models, from 2 cards to 8, ran on the same stack, and the all-reduce engaged in every run. 4-bit bought speed only on dense models. [Models](MODELS.md)

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
| [3: A tutor on one refurbished card](paper/03-a-tutor-on-one-card.md) | A homework tutor for teenagers on one 24 GB card, over WhatsApp: what prompts could not fix, what search could not fix, and the split that did |
| [4: Running it](paper/04-running-it.md) | A console built from real hardware, a drive that stopped answering (its sleep-state explanation withdrawn in Chapter 7), a card 18 degrees hotter than its neighbour, and a retraction |
| [5: A 321B model on ten consumer cards](paper/05-a-321b-model-on-ten-cards.md) | GLM-5.3-Flash with no official path: the window ladder, the bug behind 32,768, two withdrawn explanations, and moving the brain to two cards |
| [6: The memory underneath](paper/06-the-memory-underneath.md) | Why GLM failed in service: a GPU code object inside the model's memory, a marked block overwritten only on full cards, a one-card test with no model, and what narrowed it down |
| [7: Heat, a hang and a repaste](paper/07-heat-a-hang-and-a-repaste.md) | Clock caps that are not ceilings, a 27-degree repaste, a hang a power drain fixed, a start-up that burned three kilowatts doing nothing, and the system drive's fourth hang |
| [8: Serving it](paper/08-serving-it.md) | How GLM came to serve: the vision encoder's 5 GiB, a backstop for the fault underneath, a 2-hour soak, and which experts each kind of work picks |
| [9: What we were given, and what we built](paper/09-what-we-were-given-and-what-we-built.md) | Twelve cards, our FP8 kernels, MTP across pipeline stages, prefix caching and three slow paths replaced, set against what AMD, upstream and the community provide |

## Tools

| Folder | What is in it |
| --- | --- |
| [`patches/`](patches/) | Everything that turns the stock image into the configuration in Chapters 1.3 and 2.2, and [`patches/rc3/`](patches/rc3/) for the rest: diffs, kernels, Containerfiles, tests and gates, all hashed |
| [`tools/`](tools/) | Every script behind every figure, each gate with a broken control it has to reject |
| [`MODELS.md`](MODELS.md) | Every model run on the current stack: speed, cards, what each number means, and the raw record of each run |
| [`bench/`](bench/) | Raw output behind every figure, including `n02-bench`, the throughput harness; [`bench/rc3/`](bench/rc3/) for the rc3 releases |
| [`tasks/`](tasks/) | The frozen evaluation rubric and every score |

## The machine

| Component | Specification |
| --- | --- |
| GPUs | 12× RX 7900 XTX, gfx1100, 288 GB aggregate VRAM, six makers (10 until 2026-10-24) |
| CPU and board | EPYC 7663 (56-core Milan) on an ASRock Rack ROMED8-2T/BCM |
| Memory | 512 GB DDR4-3200 ECC, 8 channels |
| Interconnect | PCIe Gen4 x8 to every card: ten through one PEX880xx Gen4 switch, two on a bifurcated x16 slot. No fabric |
| Power and cooling | Multiple supplies on a 20 A 240 V circuit, open bench, garage |
| The agent's model | GLM-5.3-Flash: `wtdcode/GLM-5.3-Flash-AWQ-W4A16` (revision `abd7b077`) with the official FP8 experts of 7 layers and the draft layer, on all twelve cards (three pipeline stages of four). Before 2026-10-24 `cyankiwi/Qwen3.8-27B-AWQ-INT4`, revision `6e134bae` |

Every measurement names the container image it was taken on. After Chapter 1.1, a throughput number
without an image tag is not a measurement.

## Left open

Measured far enough that anyone can pick them up:

- **Serving more than one user at once.** Every speed number here is one request at a time, except Chapter 9's mixed-load tests.
- **Startup.** A warm restart takes about eight minutes, most of it re-timing kernels that were timed last time.
- **The all-reduce fused with the norm**, and **an 8-bit conversation cache**: each worth single-digit percent, measured far enough to start.
- **One kernel per multiply.** A fused version was correct but slower at two rows and up; the separate reduction pass stays.
- **Precision on neutral text**, replacing the figure Chapter 2.2 withdrew.
- **A harder evaluation task**, with tool access, that separates models above the floor Task 01 sets.
- **The memory fault itself.** GLM serves because the trigger is avoided; ROCm still hands out memory a live buffer owns after a release on a full card. [Chapter 6](paper/06-the-memory-underneath.md)
- **A long read that sometimes comes back wrong.** Three times, twice as the first long read after a start; every start now checks two long reads before it serves. Not explained. [Chapter 9](paper/09-what-we-were-given-and-what-we-built.md)
- **FP8 against 4-bit on quality, layer for layer**, before more memory goes to FP8 layers or to images. [Chapter 9](paper/09-what-we-were-given-and-what-we-built.md)
- **Exact recall above 90,000 tokens** (9 of 14), and decode speed with the serving recipe, not measured again since Chapter 5. [Chapter 8](paper/08-serving-it.md)
- **The tutor's open items:** a real safety check on every answer, real homework photos, documents, and the 35B on two cards. [Chapter 3](paper/03-a-tutor-on-one-card.md)

## Corrections

Wanted, particularly on anything here that is wrong. Withdrawals are recorded in place, never deleted:

| Where | What was wrong | Settled in |
| --- | --- | --- |
| Chapter 1.2, Knob 6 | The tree all-reduce was measured at four cards; at eight it is 3.2 ms per token slower | Chapter 1.3 |
| Chapter 1.2, the short version | A 0.2 s first word that the chapter's own Knob 1 measured as 0.51 s | Chapter 1.2, in place |
| Chapter 1.2, knobs that did not work (and Chapter 2.1, which repeats it) | Speculative decoding: slower at four cards with the Triton kernel, 35 % faster at eight with the fixed kernel | Chapter 1.3 |
| Chapter 2.1, Finding 4 | The 0.2 % perplexity cost was measured on production's own text, which favours production | Chapter 2.2 |
| This README's earlier headline | 40.7 tok/s carried the precision cost of Chapter 2.1 without saying so | Chapter 2.2 |
| Chapter 4, the drive | A sleep state blamed for the system drive's hang; it hangs with every power-saving state off | Chapter 7 |
| Chapter 5, "serves 131,072 tokens" | Held in tests, not in service; the CPU-built chunk counts hid an outside overwrite rather than fixing it | Chapter 6 |
| Chapter 6, findings 5 to 7 | Which step triggers the overwrite, what ROCm 7.2 shows, and whether anything avoids it | Chapter 6, in place |
| Chapter 5, Power | "Every card at its vendor minimum since 2026-10-17": the boot service capped only 4 of 10 cards until 2026-10-24 | Chapter 5, in place |

Open an issue.

## License

MIT, except files that carry their own license header. Files derived from vLLM keep their Apache-2.0
headers; the license text is in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). Fenstone Markit is the trading
name of SovereignAI Solutions Inc., Alberta, Canada.
