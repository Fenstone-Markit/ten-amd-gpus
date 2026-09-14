# ten-amd-gpus 
Ten RX 7900 XTX cards, 240 GB of combined VRAM, one EPYC board, a garage in
Canada. This repository holds the measurements, the tooling that produced
them, and an honest account of where the platform stops.

Almost everything published about local inference assumes a single card or a
datacenter. This is the space between.

## Headline results

The largest model we served was GLM-4.6 at 357 billion parameters and 176 GB
of weights, running at 23.3 tokens per second across all ten cards.

The most efficient configuration was Qwen3.6-35B mixture-of-experts in INT4 on
four cards, reaching 414 tokens per second at eight concurrent requests while
drawing 1,066 W.

The highest aggregate throughput came from five independent servers spread
across ten cards, reaching 1,965 tokens per second, which is 95 % of linear
scaling.

Peak draw under matrix multiplication was 3,108 W. Idle draw was 183 W.

Two results we did not expect and could not find documented elsewhere:

**Splitting a model across fewer cards is faster than splitting it across
more.** Two servers on four cards each delivered 1.71× the throughput of one
server on all eight, same hardware, for 1.19× the power. The usual advice is
to maximise tensor parallelism. On a ×8 interconnect that advice is wrong.

**Reasoning models make the standard latency metric meaningless.** Time to
first token read 48 ms while the user waited 19.9 seconds for the first word
they could see. A 415× gap that no standard benchmarking tool reports.

## The paper

**[Chapter 1: Throughput, power and the limits of the platform](paper/chapter-1.md)**

45 logged benchmark configurations plus roughly 30 targeted measurements,
across five model families from 9 B to 357 B parameters. Interconnect,
weight formats, parallelism strategy, thermal and electrical limits, and a
specific account of what AMD could change.

**Chapter 2: Output quality, in progress. Chapter 1 measures rate and says
so in its first limitation. Chapter 2 measures whether the output is any good,
using a frozen task with a known-correct answer and a mechanical rubric.

## What is here

```
paper/        the chapters
tasks/        frozen evaluation tasks and scores
tools/        the tooling that produced the measurements
data/         raw benchmark logs
```

### Tooling

**`n02-fleet`** is a declarative multi-server vLLM launcher. Configs name GPUs
by `unique_id` rather than by index, and every launch is verified against
sysfs before it is reported as successful.

**`n02-ask`** is a logging conversation driver. It counts tokens before
sending, refuses rather than letting a reply truncate silently, and records
wall time, throughput and time to first token for every turn.

**`n02-modelscan`** searches the Hugging Face Hub and filters on the
constraints that actually decide whether a model will run: whether vLLM
registers the architecture, what the weight format is, and whether the
arithmetic fits against measured bytes per parameter.

**`fenstone-monitor`** is fleet telemetry. The collector runs on the inference
host and the server and dashboard run elsewhere, because software running on a
machine cannot report that machine being down.

## Findings that cost us the most time

Recorded here because each one presents as something it is not.

**Link width is not what the GPU reports.** Every card says ×16. That describes
a link inside the card's own bridge. The root port, three levels up the sysfs
tree, says ×8. We caught it by arithmetic: 41 % efficiency is not plausible for
a well-formed ×16 link, 83 % is unremarkable for ×8.

**A container process limit caps you at two model servers.** Rootless podman
defaults to `pids.max = 2048`. Two vLLM servers consume about 1,726. The third
fails with an OpenMP thread creation error that looks like a GPU or network
fault and is neither.

**Prefix caching inflates published throughput by up to 1.74×.** Hit rate fell
from 63.2 % to 10.3 % within a single sweep. What looked like a throughput
collapse at high concurrency was the cache draining. We nearly published the
artifact as a hardware finding.

**Several standard tools report wrong values on this platform.** `rocm-smi`
power reporting fails. `--showmemuse` reports 0 % on cards holding weights.
Suspended cards report 0 °C, 0 W and Gen1 ×1. The sysfs values are correct
where the tooling is not.

**One card has a thermal-interface fault identifiable by signature alone:**
highest junction temperature, *lowest* memory temperature, a 32 °C
core-to-memory spread against 5 to 14 °C fleet-wide, coolest at idle, fastest
to cool. Slot position and power were both eliminated. We record the signature
because it is diagnostic and we could not find it documented.

## Upstream contributions

- **vLLM [#56790](https://github.com/vllm-project/vllm/issues/56790)**: the RDNA3
  fused-MoE path hardcodes a 2× gated-activation factor, which breaks
  non-gated (ReLU²) models such as NVIDIA's Nemotron 3. Reported with a root
  cause and a verified one-line fix, tested on this hardware.

## Scope

One machine, one operator, no replication across hardware. Ten cards from four
manufacturers with three firmware revisions among one vendor's five, which is
representative of what a secondhand-constrained buyer actually assembles and
introduces variance we have characterised but not eliminated.

Where a finding could be confirmed against public sources we say so. Where it
could not, we say that too.

## Reproducibility

Every table in the paper traces to a logged row with recorded conditions:
model, quantization, parallel degree, concurrency, prefix-caching state, power
caps, ambient temperature and thermal telemetry captured alongside each run.
Runs where any card exceeded 100 °C junction are flagged and excluded from
performance claims.

## Contact

Issues and corrections welcome. If you are running RDNA3 multi-GPU and hit
something in here, or something that contradicts it, open an issue.

SovereignAI Solutions Inc., trading as Fenstone Markit. Canada.
