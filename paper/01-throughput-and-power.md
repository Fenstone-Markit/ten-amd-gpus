# Ten AMD Consumer GPUs as Inference Infrastructure

### Chapter 1: Throughput, power and the limits of the platform

*A love letter to AMD, and a frustrated one*

Almost everything published about running large language models locally
assumes one of two things: a single graphics card, or a datacenter. This paper
describes the space between. Ten consumer gaming GPUs, 240 GB of combined
memory, on one machine, in a garage. It works, and better than I expected.

## Findings

1. **Ten RX 7900 XTX serve a 357-billion-parameter model at 23.3 tokens per
   second**, only 25 % slower than a 72 B model on the same cards, and on less
   power. §6
2. **Fewer cards per model beats more.** Two servers on four cards each
   delivered 1.71× the throughput of one server on all eight, for 1.19× the
   power. Five two-card servers reached 1,965 tokens per second, 95 % of linear
   scaling. The usual advice to maximise tensor parallelism is wrong on this
   interconnect. §5.5
3. **Time to first token is meaningless for reasoning models.** It read 48 ms
   while the user waited 19.9 seconds for the first visible word, a 415× gap
   that every standard tool misses. §5.7
4. **Prefix caching inflates published throughput by up to 1.74×.** What looked
   like a collapse at high concurrency was the cache draining. §5.2
5. **4-bit quantization costs nothing on mixture-of-experts and about 2× on a
   dense model**, and the evidence points at which kernel each path loads
   rather than at 4-bit arithmetic itself. A 36 B mixture-of-experts runs at 84.6 tokens per
   second on four cards drawing 821 W. §4.3
6. **FP8 mixture-of-experts does not load at all.** The cause is a kernel library
   scoped to datacenter parts, not the silicon. §4.1, §8.3
7. **The electrical circuit binds before the hardware does.** Peak draw is
   3,108 W. The previous 15 A / 110 V circuit was at its limit with four
   cards. §7.1
8. **Standard tools report wrong values on this platform**: link width, power,
   memory use and link speed, each shown wrong and each with a working
   substitute. §3.1, §3.3, §7.5
9. **An undocumented container limit caps you at two model servers** until you
   raise it. §7.3

**This chapter measures rate, not whether the output is any good.** Chapter 2
takes that up. The sections below walk through how each finding was reached,
starting with what the results do not establish.

## 1. Scope and limitations

This paper reports throughput, latency, memory, thermal and power measurements
on one machine. Readers should be clear about what it does not establish.

**I did not measure output quality.** Every figure here is a rate. Whether
any configuration degraded correctness relative to a bf16 baseline is
untested in this chapter. This is the most significant limitation and I state
it first. Chapter 2 addresses it directly.

**I did not measure production or agentic workloads.** All benchmarks use
fixed-length synthetic prompts. Real work has growing context and tool-call
latency. My figures should not be read as predictions of it.

**Single machine, single operator.** No replication across hardware. Where a
finding could be confirmed against public sources I say so. Where it could
not, I say that too.

**Vendor-heterogeneous fleet.** Ten cards from four manufacturers, with three
firmware revisions among one vendor's five. This is representative of what a
secondhand-constrained buyer actually assembles. It introduces variance I
have characterised but not eliminated.

## 2. Platform

Ten RX 7900 XTX cards, gfx1100, 24 GB each for 240 GB total. Five are XFX
across three firmware revisions, two MSI, two AMD reference, one ASRock.

The host is an AMD EPYC 7663 with 56 cores on an ASRock Rack ROMED8-2T/BCM
board, 512 GB of DDR4 ECC across eight channels. Every root port runs PCIe
Gen4 ×8, which is not what the cards report and is the subject of §3.1. Power
comes from a dedicated 20 A / 240 V circuit.

Every figure in this chapter was measured with each card on its own root port.
The ten cards have since been moved behind a single PCIe Gen4 switch; Chapter 1.2
covers what that changed, including one switch default that cost 32 % of prefill.

Software is vLLM 0.23.1 on ROCm 7.14.1, torch 2.11.0, under rootless podman
5.7.0.

Board-level power caps differ by manufacturer: 327 W on XFX and ASRock, 291 W
on MSI and reference designs. `power1_cap_max` reads 350 W on all ten, so
these are vendor defaults rather than hardware ceilings.

## 3. Interconnect

### 3.1 Link width is not what the GPU reports

Every card reports `current_link_width` = ×16. That value describes a link
internal to the card's own bridge. It does not describe the slot. The root
port, three levels up the sysfs tree, reports ×8.

| Measurement | Value | of ×8 | of ×16 |
|---|---|---|---|
| Gen4 ×8 theoretical | 15.75 GB/s | | |
| Device-to-device, 512 MB | 13.04 GB/s | 83 % | 41 % |
| Host-to-device, pinned | 12.92 GB/s | 82 % | 41 % |
| RCCL topology estimate | 12.0 GB/s | 76 % | 38 % |

Device-to-device and host-to-device agree within 1 %. The constraint is the
lane, not the peer path. RCCL debug output confirms peer-to-peer transport is
active.

I identified the discrepancy by arithmetic before topology. 41 % is not a
plausible efficiency for a well-formed ×16 link. 83 % is unremarkable for ×8.

### 3.2 Collective bandwidth

Ring all-reduce, 256 MB payload, 20 iterations:

| Ranks | GB/s | of 13.04 achievable |
|---|---|---|
| 2 | 8.48 | 65 % |
| 4 | 5.84 | 45 % |
| 8 | 5.07 | 39 % |

`NCCL_MIN_NCHANNELS` at 2, 8 and 16 produced 8.32, 8.47 and 8.48 GB/s.
Channel count is not the limiting factor. I report the negative result to
save someone else the experiment.

vLLM's optimised custom all-reduce is disabled by the platform, not by
configuration: the enabling check lists only the MI300-class architectures
(`gfx94`, `gfx95`), even where every card pair reports peer access. All
collectives run through RCCL. An open pull request adds an RDNA3 path
(vLLM #57767); it has not been tested here.

### 3.3 Measurement hazard: suspended links

Idle cards report Gen1 ×1 on every root port while the fleet moves 13 GB/s.
That is fifty times what the reading permits. Wake the devices before taking
any link or thermal measurement. I built a finding on this artifact before
catching it.

## 4. Weight formats

| Format | Dense | Mixture-of-experts |
|---|---|---|
| bf16 | works, native | works, native |
| FP8 | not tested | **does not load** |
| AWQ INT4 | works, ≈2× slower (§4.3) | works, no measured penalty |

### 4.1 FP8 mixture-of-experts does not load

```
NotImplementedError: No FP8 MoE backend supports the deployment configuration.
```

Forcing a specific backend produces a message that names the cause:

```
ValueError: FP8 MoE backend MARLIN does not support the deployment
            configuration since kernel does not support current device rocm.
```

| Stack | TP | Flags | Result |
|---|---|---|---|
| vLLM 0.23.1 / ROCm 7.14.1 | 8 | default | block-size error |
| vLLM 0.23.1 / ROCm 7.14.1 | 4 | default | no backend |
| vLLM 0.23.1 / ROCm 7.14.1 | 4 | AITER off, eager | no backend |
| vLLM 0.27.0 / ROCm 10.0.0 | 4 | default | no backend |
| vLLM 0.23.1 / ROCm 7.14.1 | 4 | force Marlin | *kernel does not support rocm* |

The message names ROCm rather than gfx1100. Marlin, CUTLASS, DeepGEMM and
FlashInfer are CUDA-only at source level. AITER, which is AMD's own kernel
library, announces its own absence at every startup on this platform.

**This is corroborated publicly and is a software gap, not a silicon one.**
The SGLang project tracks a closely related fault in which a CDNA binary
force-loaded on gfx1100 produces corrupt expert indices, and documents that
AITER is CDNA-only while being imported eagerly across dozens of files [R1].
That framing matters because it implies a fixable path.

**Method contribution.** vLLM's backend selector logs each rejection at debug
level, but those lines do not reach the log. Forcing a specific backend
converts a silent "no backend supports this" into a specific rejection with a
stated reason. I recommend the technique generally.

### 4.2 Block quantization constrains parallelism

```
ValueError: Weight input_size_per_partition = 64 is not divisible by
            weight quantization block_k = 128.
```

Block-quantized weights constrain partition width. Fewer ranks give wider
partitions: eight ranks gives 64, four gives 128. This is the real parallelism
constraint on weight format. It is easily confused with attention head counts,
which behave differently (§9).

### 4.3 The AWQ penalty depends on architecture

Neither model needed a separate backend, unlike FP8. Which kernel executed the
quantized layers was not recorded at the time, and an earlier version of this
section named it wrongly; see the correction below.

Dense model, Qwen2.5-72B, 8-way tensor parallel:

| Measure | bf16 | AWQ INT4 |
|---|---|---|
| tok/s, 1 request | **30.9** | 15.0 |
| Watts | 2,139 | 2,574 |

Mixture-of-experts, Qwen3.6-35B-A3B, 4-way:

| Measure | bf16 | AWQ INT4 |
|---|---|---|
| tok/s, 1 request | 82.2 | **84.6** |
| tok/s, 8 concurrent | 272.7 | **414.3** |
| Watts, 8 concurrent | 1,236 | **1,066** |
| KV cache available | 2.63 GiB | **12.88 GiB** |

Both comparisons are controlled. Same model, same parallel degree,
quantization the only variable.

**Correction.** An earlier version of this section attributed the dense penalty
to Marlin unpacking INT4 before each multiply. Marlin does not run on ROCm at
all (§4.1), so that explanation cannot be right, and it is withdrawn.

Chapter 1.2 found a mechanism of the right size in a closely related path. For
compressed-tensors W4A16 checkpoints on this stack, vLLM runs every dense
quantized layer through a Triton kernel that walks the full reduction dimension
serially and, once the model is compiled, keeps a tile sized for large batches
even at batch 1. A native RDNA3 kernel exists in the same image and is skipped.
Switching to it made single-stream decode on a dense 27 B model 1.49 to 1.65×
faster, depending on context length. The mixture-of-experts path on this stack,
by contrast, loads a native RDNA3 MoE kernel (Chapter 1.1 records the log line),
which fits the MoE model showing no penalty. Whether the 72 B AWQ checkpoint took
the same Triton path was not recorded, so this is consistent with the result
above rather than a demonstration of its cause.

The concurrent-request gap is amplified by the bf16 variant being
cache-constrained at 27× concurrency where AWQ had 134× available.

## 5. Parallelism

### 5.1 Instrument warning

Two benchmark harnesses were used and **they are not comparable.** On one
identical configuration, a synthetic-dataset harness reported 301 tok/s where
a chat-endpoint harness with a real prompt reported 432. A 43 % difference
attributable entirely to the instrument. All comparisons below hold the
harness constant.

### 5.2 Prefix caching inflates published numbers

Qwen2.5-7B, identical sweeps:

| Concurrency | caching on | caching off | inflation |
|---|---|---|---|
| 1 | 98.9 | 77.6 | 1.27× |
| 32 | 1,328.0 | 1,017.4 | 1.31× |
| 64 | 2,137.6 | 1,231.1 | **1.74×** |
| 96 | 1,304.3 | 1,297.5 | 1.00× |

Prefix cache hit rate fell from 63.2 % to 10.3 % within a single sweep as
prompt count scaled. What appeared to be a throughput collapse at high
concurrency was the cache draining. KV cache peaked at 23.3 % with no requests
queued.

**Any published benchmark should disable prefix caching or report hit rate.**
I nearly published the artifact as a hardware finding.

### 5.3 Dense scaling

Qwen2.5-7B, 2-way tensor parallel, caching off:

| Concurrency | tok/s | ms/token | first token | W |
|---|---|---|---|---|
| 1 | 77.6 | 10.0 | 747 ms | 788 |
| 32 | 1,017.4 | 27.4 | 1,055 ms | 1,014 |
| 96 | 1,297.5 | 67.8 | 1,590 ms | 1,061 |

16.7× throughput for 6.8× latency.

Qwen2.5-72B, 8-way:

| Concurrency | tok/s | ms/token | first token | W |
|---|---|---|---|---|
| 1 | 28.9 | 30.6 | 1,053 ms | 2,326 |
| 8 | 100.7 | 65.0 | 3,750 ms | 2,709 |
| 32 | 167.6 | 160.8 | 7,826 ms | 2,738 |

5.8× throughput for 5.3× latency. Batching returns little at this model size.

### 5.4 A prediction that held

From measured ring latency and approximately 80 all-reduce operations per
token at eight ranks, I predicted a communication-bound ceiling near 30 tok/s
before running the benchmark. Measured: 28.9 on the synthetic harness, 30.9 on
the chat harness.

The same prediction **failed** on a 28-layer 7 B model, where 4-way tensor
parallel was faster single-stream than 2-way, 107.4 against 77.6 tok/s,
because the collective is too cheap to bind at that depth. I report the
failure because it establishes where the model applies.

### 5.5 Fewer cards per model, more models

Identical model, harness and card count:

| Configuration | tok/s @ 8 concurrent | W | tok/s per kW |
|---|---|---|---|
| 35B MoE, one 8-way server | 432.3 | 1,542 | 280 |
| 35B MoE, **two 4-way servers** | **739.7** | 1,831 | **404** |

1.71× throughput for 1.19× power, at a cost of 2.7 ms per token.

Extended to a smaller model, Qwen3.5-9B at 8 concurrent requests per server:

| Configuration | Cards | tok/s | W | tok/s per card |
|---|---|---|---|---|
| one 2-way server | 2 | 412 | 734 | 206 |
| two 4-way servers | 8 | 996 | 1,852 | 124 |
| five 2-way servers | 10 | **1,965** | 2,620 | 197 |

Five servers reached 4.77× a single server, which is 95 % of linear, with a
1 % spread across the five instances.

**On this interconnect, tensor parallelism costs roughly 40 % of per-card
throughput at these model sizes, while additional independent servers cost
almost nothing.** I attributed this to ×8 root ports and RCCL-only
collectives. Chapter 1.2 found a second contributor: at the smaller per-card
matrix shapes that high tensor parallelism produces, the quantized matrix
kernels run far below memory bandwidth, so each card's work shrinks much less
than the division suggests. The result may not transfer to systems with wider
links, working custom all-reduce, or different kernels.

### 5.6 Pipeline parallelism

Qwen2.5-72B on eight cards, same harness:

| Measure | 8-way tensor | 2-stage pipeline × 4-way tensor |
|---|---|---|
| tok/s, 1 request | **30.9** | 19.9 |
| tok/s, 8 concurrent | **158.6** | 120.6 |
| first token, 8 concurrent | **249 ms** | 327 ms |
| KV cache | larger | 4.99 GiB |
| Watts | 2,266 | **1,909** |

Pipeline parallelism loses on every performance axis for a model that fits
under tensor parallelism. Its lower power reflects idle silicon rather than
efficiency.

It scales better with concurrency, 6.05× against 3.48× from 1 to 8 requests,
but starts too far behind to overtake.

**This reverses when a model does not fit.** See §6.

### 5.7 Reasoning models and the first-token metric

Qwen3.6-35B-A3B, identical prompt:

| Measure | reasoning on | reasoning off |
|---|---|---|
| tok/s | 88.9 | 95.8 |
| tokens per request | 1,998 (**90 % trace**) | 1,100 |
| time to first token | 48 ms | 51 ms |
| **time to first visible token** | **19,914 ms** | 51 ms |

Throughput is unchanged. The useful fraction of it is not. On one
representative prompt: 1,859 tokens in 20.6 s with reasoning, 762 tokens in
7.7 s without, equivalent answers.

Every standard benchmarking tool reports the 48 ms figure. I suggest
instrumenting both, and report both throughout.

**Implementation note.** vLLM's reasoning stream field is named `reasoning`,
not `reasoning_content`. Reading the wrong field counts only visible tokens
while the clock runs through the trace, producing figures that are inverted
rather than merely inaccurate. I made this error and discarded a run.

## 6. The largest model I served

GLM-4.6-AWQ: 357 B parameters, 160 experts, 176 GB of weights.

| Configuration | Cards | Outcome |
|---|---|---|
| 8-way tensor, util 0.97 | 8 | out of memory, allocator |
| 8-way tensor + expert parallel, 0.97 | 8 | out of memory, hipBLASLt workspace |
| 8-way tensor, 0.96 / 0.94 / 0.90 | 8 | no memory for cache blocks |
| **5-stage pipeline × 2-way tensor + expert parallel, 0.90** | **10** | **serves** |

Eight cards hold the weights at 22 GB of 24 GB each. They cannot also hold
activation buffers, library workspace and KV cache across 92 layers. No
memory-utilization setting resolved it. Three settings failed identically.

| Concurrency | tok/s | ms/token | first token | W |
|---|---|---|---|---|
| 1 | 23.3 | 42.6 | 305 ms | 1,743 |
| 4 | 64.0 | 58.2 | 589 ms | 1,842 |

KV cache: 3.98 GiB, 43,232 tokens, 10.55× maximum concurrency at 4,096 context.

Placed in context:

| Model | Parameters | tok/s, 1 request | W |
|---|---|---|---|
| Qwen3.5-9B | 9 B | 412 (at 8 concurrent) | 734 |
| Qwen3.6-35B MoE | 36 B | 84.6 | 821 |
| Qwen2.5-72B | 72 B | 30.9 | 2,139 |
| **GLM-4.6** | **357 B** | **23.3** | **1,743** |

A 357 B model runs 25 % slower than a 72 B on less power, because sparse
activation and INT4-on-MoE compound favourably. I measured both effects
independently before attempting this configuration.

**This is the ceiling of the platform, not a recommended operating point.**
4,096 tokens of context and ten concurrent requests demonstrates capability.
It does not constitute a deployment.

## 7. Physical and infrastructure findings

### 7.1 Electrical

Peak measured draw: 3,108 W under matrix multiplication, ten cards. Idle:
183 W.

On the prior 15 A / 110 V circuit, four cards drew approximately 1,330 W
against a roughly 1,320 W continuous rating. **At consumer scale the circuit,
not the power supply, is the binding constraint**, and I found no build guide
that mentions it.

### 7.2 Thermal

Eight cards without auxiliary airflow reached 110 °C and aborted a soak at 68
seconds. Ten cards with four 120 mm fans peak at 89 °C junction. A later and
heavier run, eight cards at 100 % busy and 2,428 W fleet draw, plateaued at
93 °C for seventeen minutes: an equilibrium, not a climb.

By vendor cap group, under identical load:

| Group | Mean W | matmul/s |
|---|---|---|
| 327 W boards | 242–247 | 88–90 |
| 291 W boards | 217–218 | 82–86 |

29 W buys 6 to 7 %. No card draws its cap. Matrix multiplication is not
power-limited at any driver-accepted setting here.

**One card has a thermal-interface fault.** The signature identifies it on its
own: highest junction temperature, lowest memory temperature, a 32 °C
core-to-memory spread against 5 to 14 °C fleet-wide, coolest at idle, fastest
to cool, and unbounded temperature rise under sustained load at every cap
setting.

Slot position was eliminated. A known-good card ran cooler in the suspect
slot. Power was eliminated. It drew 253 W against a 300 W cap.

I record the signature because it is diagnostic and I could not find it
documented.

### 7.3 A container limit that caps you at two servers

Rootless podman defaults to `pids.max = 2048`. Two vLLM servers consume
approximately 1,726. A third fails with:

```
libgomp: Thread creation failed: Resource temporarily unavailable
```

This presents as a GPU or network fault and is neither. I eliminated port
collision and ulimit hypotheses before identifying it.

Resolution: `--pids-limit=-1` on the container and `OMP_NUM_THREADS=8` per
server. OpenMP otherwise spawns one thread per core per process. With the cap
applied, **five servers consume 1,397 processes where two previously consumed
1,726.**

### 7.4 Rootless container GPU access

The standard Docker idiom `--group-add video --group-add render` produces a
container that starts cleanly and enumerates **zero GPUs** under rootless
podman, which requires `--group-add keep-groups`.

### 7.5 Instrumentation that reports incorrect values

On this platform: `rocm-smi` power reporting fails. `rocm-smi --showmemuse`
reports 0 % on cards holding weights. Suspended cards report 0 °C and 0 W.
Suspended links report Gen1 ×1. The sysfs values `mem_info_vram_used` and
`power1_average` are correct where the tooling is not.

## 8. Discussion

### 8.1 What the platform delivered

Before the gaps, the capability, because the capability is the larger fact and
it is easy to lose under a list of workarounds.

**Ten consumer cards from four manufacturers, with three firmware revisions
among one vendor's five, cooperate without incident.** I did not curate a
matched set. I bought what was available at prices I could pay, and the
fleet works.

**The interconnect performs to specification.** 13.04 GB/s device-to-device
against a 15.75 GB/s theoretical ceiling is 83 % efficiency on a Gen4 ×8 link,
and device-to-device matched host-to-device within 1 %. Nothing is quietly
broken underneath.

**The hardware is predictable.** I derived an expected throughput ceiling
from measured collective latency, before running the benchmark, and the
serving result landed within 4 %. A platform whose behaviour can be predicted
from first principles is a platform you can engineer on.

**It is clean under load.** Ten cards running matrix multiplication for a full
minute produced one system event log entry, which was the clearing event
itself. Sessions running eighteen hours and longer completed without
intervention.

**The efficiency is good where it matters.** A 36 B mixture-of-experts in INT4
on four cards delivers 414 tokens per second at eight concurrent requests on
1,066 W, which is 389 tokens per second per kilowatt. Two four-way servers
reach 404. These are not consolation numbers.

**And 176 GB of weights fit.** A 357 B model, in a garage, on gaming cards
bought secondhand across a market with no enterprise supply. That was not an
obvious outcome when I started and it is the reason this paper exists.

### 8.2 On deployment form factor

The measurements suggest the most useful configurations on this platform are
the smaller ones. A 36 B mixture-of-experts on four cards at 84 tok/s and
821 W is a working tool. A 357 B model on ten cards at 23 tok/s and 1,743 W
demonstrates a ceiling.

I built the larger machine to establish where that ceiling is. I would not
recommend it as a deployment. The trajectory that matters for local inference
is toward compact, low-power systems capable of what this arrangement does
today, and the gap between those two things is a statement about 2026 rather
than about architecture.

### 8.3 On AMD, and what would help

I want to be direct about my position, because a list of workarounds can
read as a complaint and this is not one.

**The RX 7900 XTX is a good card and it earned its place here.** 24 GB per
board at secondhand prices no current-generation part approaches. Ten of them
held a 357 B model. The compute is real, the memory bandwidth is real, and
across every measurement in this paper the silicon did what the specification
says it does. When something failed, it was never the card.

**RDNA3 consumer parts are officially supported.** ROCm 7.2 lists the RX 7900
XTX (gfx1100) among supported consumer GPUs alongside RDNA4 [R2]. I say that
plainly because my findings could be misread as evidence of abandonment, and
they are not that.

The gap I measured is narrower and more specific. As one public assessment
puts it, official support on the consumer side is not the same as tested,
optimised and production-ready. Most framework authors test on MI-series
datacenter cards, and consumer RDNA support is real but secondary [R3].

That matches my experience precisely. Everything I could not do traces to
kernel coverage rather than to hardware:

- **AITER is scoped to CDNA** and imported eagerly across framework code, so
  consumer systems fall back or fail to load [R1]. This is the direct cause of
  my FP8 mixture-of-experts result.
- **No RDNA continuous-integration runner exists** in at least one major
  serving framework. AMD CI targets MI-series, without regression coverage
  for consumer architectures [R1]. Support that is not tested regresses
  silently, and I observed that pattern.
- **No tuned kernel configuration ships for this device.** vLLM told me
  explicitly that no tuning file exists for the RX 7900 XTX and that it was
  running defaults. Whatever performance I measured, it was not the tuned
  path.
- **Fallbacks are taken silently.** ROCm's optimised paged-attention kernel
  declined my configuration and fell back to a portable implementation. It
  worked correctly. Nothing in a benchmark result would tell you it happened.
- **A native kernel that ships and is not used.** The image contains an RDNA3
  quantized matrix kernel written for this exact GPU. vLLM skips it for
  asymmetric checkpoints and runs a slower Triton path instead, with no message.
  Selecting it made decode 1.49 to 1.65× faster on a dense 27 B model
  (Chapter 1.2).

Read together, these say something encouraging. **The numbers in this paper
are a floor, not a ceiling.** I measured untuned kernels on fallback paths
and still served a 357 B model. The headroom that exists is in software that
already runs, and it belongs to whoever writes the configurations.

Three things would change the picture materially, in ascending cost:

1. **One RDNA CI runner** in the major serving frameworks. Support that is
   tested does not regress silently, and this is the cheapest durable fix on
   the list.
2. **Published tuned kernel configurations** for common gfx1100 shapes, or a
   documented path for users to generate and contribute their own. I would
   contribute mine.
3. **A portable fallback for AITER-gated paths**, so consumer architectures
   degrade in performance rather than failing to load. Slow is a result. Not
   loading is not.

None of these require new silicon. All of them extend the useful life of
hardware already in users' hands, and the people running it are exactly the
population that produces the community knowledge, the bug reports and the
tutorials that a platform needs.

I built this because I thought the hardware deserved it. The measurements
say I was right. I would like to keep going.

## 9. Recommendations for practitioners

**On model selection.** Check `num_key_value_heads` before downloading, but
not for the reason commonly assumed. vLLM accepts tensor parallel degree in
either direction: it splits KV heads across ranks when the head count is at
least the parallel degree, and replicates them when the degree exceeds the
count. A model with two KV heads runs at TP=4. The genuine divisibility
constraint on this platform is block quantization, §4.2, which limits
partition width rather than head count.

**On parallelism.** For throughput under many users, run more servers at
lower parallel degree rather than fewer at higher. On a ×8 interconnect the
collective cost dominates before the compute does. For a single user on a
model that needs several cards, the answer can reverse: the 27 B model in
Chapter 1.2 decodes fastest at the highest degree that fits. Measure both.

**On measurement.** Disable prefix caching before publishing any throughput
figure. Hold the harness constant across comparisons. Treat every instrument
as suspect until its output agrees with arithmetic.

**On reasoning models.** Instrument time to first visible token separately
from time to first token. The standard metric will mislead you by two orders
of magnitude.

**On the building.** Measure the electrical circuit before selecting the
hardware. The breaker binds before the power supplies do.

## 10. Reproducibility

All benchmark tooling, fleet configurations and the complete measurement log
are available. Every table in this paper traces to a logged row with recorded
conditions: model, quantization, parallel degree, concurrency, prefix-caching
state, power caps, ambient temperature and thermal telemetry captured
alongside each run.

Runs where any card exceeded 100 °C junction are flagged in the log and
excluded from performance claims.

## References

- **[R1]** SGLang issue #30599, "Officially support consumer Radeon
  RDNA3/RDNA4, tracking and enablement plan," 2026. Documents AITER as
  CDNA-only, the absence of RDNA CI runners, and a related fused-MoE fault on
  gfx1100.
- **[R2]** DEV Community, "AMD ROCm on Consumer GPUs," March 2026.
- **[R3]** IDFS AI, "AMD GPUs for AI Inference in 2026: What Works, What's
  Broken, and What to Actually Buy," April 2026.

## Appendix: open questions carried into Chapter 2

- Output quality across quantization formats, measured against a bf16
  reference. The most significant gap in this chapter and the subject of the
  next.
- Sustained agentic or production workload measurement.
- Whether the INT4 dense-versus-MoE penalty difference is architectural.
  Partly answered: it tracks which kernel each path loads (§4.3, Chapter 1.2).
- Speculative decoding. Measured since: +40 to 54 % on a 9 B model on one card,
  9 % slower on the 27 B at four-way tensor parallel. See Chapter 1.2.
- A floor-capped power series for cross-platform comparison.
