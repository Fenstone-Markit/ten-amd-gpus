# Chapter 2: Output Quality

Chapter 1 measured how fast ten consumer AMD cards serve tokens and what that
costs in watts. It said nothing about whether the tokens were any good.

This chapter is the quality axis. It is narrower than Chapter 1 and the result is
a null one, which is itself the finding.

## Why not use published benchmarks

Leaderboard scores are produced on hardware nobody reading this owns, on tasks
that do not resemble the work, with prompts short enough to hide the behaviour
that matters.

Three mismatches motivated building something local. The tasks are not the work:
a model that scores well on competitive programming may still be unable to read
an undocumented filesystem, which is most of what infrastructure code does. The
prompts are short, so the throughput measured at turn one is not the throughput
you get at turn five. And the quantization is not yours, because a published
score is for the original weights, not the 4-bit repack you can actually fit in
24 GB.

## Task 01

Write `n02-collector.py`, a telemetry collector that reads GPU state from sysfs
across ten RX 7900 XTX and posts it to an ingest endpoint. The model receives the
real specification, roughly 4,600 tokens, as a system prompt. The user message is
one line asking for the complete file with no placeholders and no explanation.

The task was derived from a real build. The reference implementation runs in
production and is committed. Every trap is a defect that was actually hit.

Ten binary checks, mechanically verifiable, no human judgement.

| Group | Count | What it covers |
|---|---|---|
| Constraint compliance | 4 | No forbidden tools, both card globs present, no fabricated identity on read failure, absent sensors produce null rather than zero |
| Host traps | 5 | Glob root, non-GPU device exclusion, hwmon nesting depth, PCIe walk direction, dpm token scanning |
| Execution | 1 | Runs unedited and emits ten cards with non-null junction temperatures |

The five host traps are the interesting ones. `card0` is an ASPEED BMC
controller, not a GPU. The hwmon directory nests two levels below the device
directory. The PCIe link walk has to go up to the root port, because the bridge
immediately above the card reports 16 while the root port reports 8. The dpm
frequency files put a state label before the frequency, and one of those labels
is not a number.

None of these are hard. All of them are invisible from training data.

R1 is the first emitted file, scored out of ten, no correction turns. R2 is turns
to a working collector, with the model given evidence rather than fixes: the
observed wrong behaviour and the sysfs reading that contradicts it, never the
corrected code. A hint level from 0 to 4 records how much help each fix required,
because turn count alone conflates a model that fixes on a nudge with one that
needs the answer handed over.

## Results

Six runs. All served locally, temperature 0.6, top_p 0.95, prefix caching off.

| Model | Total | Active | Quant | Cards | R1 | Turns | Regressions | Fabrications |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-Next | 80B | 3B | bf16 | 10 | 7/10 | 2 | 0 | 0 |
| Qwen3-Coder-Next | 80B | 3B | AWQ INT4 | 4 | 7/10 | 6 | 1 | 0 |
| Qwen3.6-35B-A3B | 35B | 3B | AWQ INT4 | 4 | 6/10 | 4 | 0 | 0 |
| Qwen3.6-35B-A3B | 35B | 3B | bf16 | 4 | 6/10 | 5 | 3 | 0 |
| Nemotron-3-Super | 120B | 12B | AWQ INT4 | 4 | 6/10 | 5 | 2 | 0 |
| MiniMax-M2.7 | 229B | 10B | AWQ INT4 | 8 | 3/10 | 5 | 0 | 0 |

The MiniMax row carries a caveat that should be repeated wherever it is quoted.
It was the reference run. The rubric was written from its transcript afterwards,
so the traps it failed are partly the traps it defined. Its score is honest but
not independent, and it is not evidence that MiniMax is worse at this work.

## Quantization does not cost quality

Two controlled pairs, each differing only in weight format.

On Qwen3.6-35B-A3B: identical R1, and the AWQ build converged in four turns
against bf16's five, with zero regressions against three.

On Qwen3-Coder-Next: identical R1 of 7/10, failing the same three traps, with the
same silent hang on trap 10. The bf16 build converged in two turns, the AWQ in
six.

The turn counts diverge in opposite directions across the two pairs, which is
what sampling noise looks like. The R1 scores do not diverge at all. First-output
quality was indistinguishable in both pairs, down to which specific traps failed.

The claim needs scoping to the build rather than the bit width. Inspecting the
quantization configs of the checkpoints used here shows a careful quantizer
leaves a substantial set of modules in full precision. One ignores 241 modules:
every Mamba projection, every shared-expert projection, both latent projections
per layer, and the output head, quantizing only the routed experts. Another, from
the same uploader on a different architecture, ignores 502. Since routed experts
hold the overwhelming majority of parameters in a sparse MoE, that yields nearly
all the size reduction while leaving the paths every token traverses untouched.

So: a carefully scoped 4-bit quantization costs nothing measurable on this task.
A naive one that quantizes everything is a different artifact and has not been
measured here. Anyone reading a score off a Hub repack is reading a property of
that uploader's module selection as much as of the method.

## The task does not discriminate above a low bar

Five independently scored runs landed at 6 or 7 out of 10. Across that set the
total parameter count ranges from 35B to 120B, active parameters from 3B to 12B,
two vendors, two weight formats. None of those variables moved the score.

Task 01 measures a floor. It separates a model that can follow a specification
and traverse a filesystem from one that cannot, and every current model of
reasonable size clears it. It does not rank models above it.

The alternative reading, that these models are equally good, is not supported.
The task is the limitation.

## Nothing was fabricated

Across six models and roughly thirty correction turns, no model invented a sysfs
path, claimed an unavailable tool, or asserted a fact about the host to cover
being told it was wrong.

Models were repeatedly told their output was broken. The failure mode that would
matter most for unattended work did not occur once.

One case sat close to the line. Told a disk free-space field was null, a model
produced a value matching a different filesystem. It was reading the device the
specification named, and the specification was stale. The reading was correct.

## Defects cluster, and they are grounding failures

Every file compiled. None drifted or degraded as length grew. Four hundred lines
held together with consistent signatures.

The largest class is path resolution. Stop at the first hwmon directory rather
than descending into the numbered one inside it. Return the first PCIe device
walking up rather than the last one carrying a link width. Take the first numeric
token on a line rather than scanning for the frequency. These are not
low-confidence guesses. They are high-confidence continuations that happen to be
wrong about this machine, because in almost every codebase in training data the
first match is the right one.

Second is naming: field names half-remembered from convention rather than read
from the specification in front of the model. `load_1min` where the spec says
`load_1m`.

A third class appears only under iteration. Asked to emit the corrected file in
full, models rewrite four hundred lines from scratch with no diff and
occasionally lose something already correct. One dropped an import it had used
correctly the previous turn. One collapsed card discovery from ten to one while
restructuring an unrelated path. One wrote a corrected function alongside the
broken one and left both in the file. This is the most operationally annoying of
the three, because it converts a one-defect turn into a two-defect turn.

An earlier draft claimed every defect was a name or path assumed rather than
verified. The rewrite regressions are a different failure and that claim is
withdrawn.

Every defect recorded here was discoverable by one shell command. These models
did not lack knowledge, they lacked the ability to look. That points at tool
access rather than better weights, and it is the subject of the next task.

## Decode speed does not follow active parameters

Nemotron-3-Super at 12B active was the fastest model measured on this rig, ahead
of two 3B-active models.

The explanation is in the config. Its 88 layers break down as 40 Mamba, 40 MoE
and 8 attention. Fewer than one layer in ten carries attention, so the cost that
normally grows with conversation length applies to a small fraction of the
forward pass. It also carries a multi-token prediction layer. The routing is
unusual in the same direction: 512 routed experts with 22 active per token,
projected through a 1024-dimension latent space, plus one always-on shared
expert. A very wide, very sparse layer is cheap per token regardless of total
size.

This is one model, so it is not a rule. What it retires is the idea that active
parameter count predicts decode speed. Layer composition dominates, and layer
composition is readable from `config.json` before downloading anything.

Throughput measurements are in Chapter 1.1, which uses a purpose-built harness.
Figures reported in an earlier draft of this chapter were taken inside evaluation
conversations, where turn lengths varied and warm-up was included, and have been
removed.

## Limits

One task, one domain. It measures specification compliance and filesystem
traversal, not algorithmic ability, refactoring, test writing, or debugging code
the model did not write.

It rewards a spec-following disposition. A model that ignores the specification
and writes good general code scores badly. Deliberate, and still a bias.

One host. Ten RX 7900 XTX on an EPYC board. The traps are real but they are this
machine's traps.

Five independent data points. A single additional model could move any finding
here.

Publishing this task means later models may train on it. Model release dates are
recorded alongside scores for that reason.

## Next

Task 01 has reached its limit. It establishes a floor and cannot rank above it.

A harder task, with traps that separate rather than gate. Candidates are
debugging code the model did not write, and working against a specification
containing an internal contradiction the model has to notice.

A tool-access variant of this task, where the model can run commands on the host.
If defect counts collapse under tool access, the gap is grounding rather than
capability, and that is worth more than any score in the table above.
