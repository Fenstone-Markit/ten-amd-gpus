# Task 01 — sysfs Telemetry Collector

**Status: FROZEN 2026-09-11. Do not edit the prompt or the rubric.**

Editing either invalidates every score already recorded against it. If the
task needs to change, create Task 02.

---

## Why this task

Derived from a real build run on 2026-09-11 using MiniMax-M2.7-AWQ. It is not
synthetic. The reference implementation exists, works, and is committed at
`~/monitor/n02-collector/n02-collector.py` (git a7fb291).

It tests the thing that actually breaks on real work: not whether a model can
write Python, but whether it can resolve undocumented filesystem structure
correctly and whether it obeys explicit constraints that contradict its
training priors.

Every model that has ever written GPU monitoring code was trained on
`nvidia-smi` and `rocm-smi`. This task forbids both and tells the model
exactly why. Compliance is measurable.

---

## Inputs — give the model exactly these, nothing more

1. `~/specs/node02-monitor-SPEC.md` as the system message
2. This user message, verbatim:

```
Output n02-collector.py only. Complete file, no placeholders. Do not write
any other file yet. Do not explain the code after the block.
```

Sampling: temperature 0.6, top_p 0.95. Record if different.

**No follow-up turns for the R1 score.** R1 measures the first output. The
iteration score R2 is separate and does allow correction turns.

---

## R1 — first-output rubric, 10 binary points

Each is mechanically checkable. No human judgement. Run against the model's
first emitted file only.

### Constraint compliance (4 points)

| # | Check | Pass condition |
|---|---|---|
| 1 | No forbidden tools | `grep -E 'rocm-smi\|nvidia-smi'` finds no invocation (a comment naming them as forbidden is allowed) |
| 2 | Both globs present | Source contains both a `card[0-9]` and a `card[0-9][0-9]` pattern |
| 3 | No default identity | No code path assigns a fabricated value to `unique_id` on read failure |
| 4 | Nulls not zeros | Absent or unreadable sensors produce `None`, never `0` |

### The five traps (5 points)

These are the defects the reference run actually hit. Each is a real property
of this host that is not documented anywhere the model could have read.

| # | Trap | Pass condition |
|---|---|---|
| 5 | Glob root | Globs from a fixed root (`Path('/sys/class/drm').glob('card[0-9]/device')`) rather than treating the pattern as a literal path. **Reference run failed this.** |
| 6 | Non-GPU device | `card0` is an ASPEED BMC controller with driver `ast`. Excluded by requiring a readable `unique_id`, not by card number or driver name. **Reference run failed this.** |
| 7 | hwmon depth | Resolves `<carddev>/hwmon/hwmon<N>/`, two levels below the device dir. **Reference run failed this.** |
| 8 | Link direction | Walks **up** to the root port, not down into a bridge child. **Reference run failed this.** |
| 9 | dpm token | Finds the frequency by scanning tokens on the starred line, not by index, and tolerates a non-numeric state label (`S: 0Mhz *`). **Reference run failed this.** |

### Execution (1 point)

| # | Check | Pass condition |
|---|---|---|
| 10 | Runs | `python3 <file> --once` (or equivalent) emits ten cards with non-null junction temps, without editing |

**R1 reference score: MiniMax-M2.7-AWQ scored 4/10** — passed 1, 2, 4 and
partially 3; failed 5, 6, 7, 8, 9, 10.

Trap 3 is scored a pass only if no fabricated identity appears anywhere. The
reference run assigned `card_dir.name` as a fallback and therefore **fails**
trap 3. Corrected reference score: **3/10**.

---

## Hint level — how much help each fix required

Turn count alone conflates three different behaviours. A model told "the walk
goes up, not down" that then fixes it is not the same as one that needs
"add .resolve()", even when both take two turns. And the operator's patience
contaminates the count: escalating to a specific instruction because a turn
is taking too long makes a model look better than it is.

**Scale, restated 2026-09-14.** The original definitions were lost; these
are reconstructed from how every run to date was actually conducted. Scores
recorded before this date used the original wording and may differ at the
margins.

| Level | What the operator supplies |
|---|---|
| 0 | Nothing. The model found and fixed the defect unprompted, or it never occurred |
| 1 | Symptom only. What was observed to be wrong, with no cause named |
| 2 | Cause named, not corrected. The wrong assumption is identified; the fix is not |
| 3 | Answer supplied. The correct path, name, or value is handed over |
| 4 | Corrected code pasted. Scored as a failure to converge, not a fix |

Level 4 is recorded rather than used. Pasting corrected code ends the
measurement, because what is being measured from that point is the operator.

Operator discipline: start at level 1, escalate only after a failed turn,
record the level per defect rather than per turn.

---

## R2 — iteration score

Continue the conversation, correcting one defect set per turn. Feed the model
**evidence, not fixes**: the observed wrong behaviour and the sysfs reading
that contradicts it. Never paste the corrected code.

Record:

- **turns_to_pass** — turns until `--once` emits ten cards with non-null
  junction, memory, edge, power, fan, link width 8, and clocks
- **total_generation_s** — sum of wall time across those turns
- **total_output_tokens**
- **regressions** — count of turns that broke a previously passing check
- **fabrications** — count of claims about the host that were false and
  unprompted (asserting a sysfs path exists when it does not, inventing a
  tool). This is the score that matters most; weight it separately.

**R2 reference: MiniMax-M2.7-AWQ, 5 turns, ~1,920s generation, 0 regressions,
0 fabrications.**

Zero fabrications across five turns is the notable result. Record it for every
model — a model that scores higher on R1 but fabricates is worse for this
work, not better.

---

## Context ceiling

Record the input token count at every turn. The reference run degraded from
14.75 tok/s at 4,240 input tokens to 6.74 tok/s at 15,341 — a 2.2× throughput
loss across one build. This is measured with `--no-enable-prefix-caching` on
the server, so it is reprocessing cost, not attention cost.

DNF is recorded ONLY when the context fills before the task passes. Record
the turn at which it ran out. That is a real failure mode for agentic work and
should not be scored as a pass.

**DNF is not for operator fatigue.** A model that takes eight turns took eight
turns; that is the result. Stopping early because the run is going badly
converts a measurement into an opinion, and the number that would have been
most informative — how much iteration this model actually needs — is the one
thrown away. Keep turning until it passes or the context wall is hit.

---

## Recording

One row per model per run in `~/tasks/scores.jsonl`:

```json
{
  "task": "01-sysfs-collector",
  "model": "et0dev/MiniMax-M2.7-AWQ-4bit",
  "quant": "AWQ-INT4",
  "tp": 8,
  "cards": 8,
  "kv_cache_dtype": "fp8",
  "max_model_len": 32768,
  "prefix_caching": false,
  "temperature": 0.6,
  "top_p": 0.95,
  "date": "2026-09-11",
  "r1_score": 3,
  "r1_failed": [3, 5, 6, 7, 8, 9, 10],
  "turns_to_pass": 5,
  "total_generation_s": 1920,
  "total_output_tokens": 23000,
  "regressions": 0,
  "fabrications": 0,
  "context_at_final_turn": 15341,
  "dnf": false,
  "notes": "reference run; established the rubric"
}
```

The reference row's token and timing figures are approximate — reconstructed
from `~/asks/monitor-build/metrics.jsonl`, which has the exact values for
turns 1, 3 and 4. Turn 2 was not separately logged. Mark any figure that is
reconstructed rather than measured.

---

## Known weaknesses of this task

State these whenever a score from it is published.

- **One task, one domain.** It measures sysfs traversal and constraint
  compliance. It does not measure algorithmic ability, refactoring, test
  writing, or debugging code the model did not write.
- **It rewards a spec-following disposition.** A model that ignores the spec
  and writes good general code scores badly here. That is intentional but it
  is a bias.
- **Reference-run contamination risk.** If this task or the reference
  implementation is ever published, later models may have trained on it.
  Record the model's release date alongside the score.
- **The host is singular.** Ten RX 7900 XTX on an EPYC board. The traps are
  real but they are this machine's traps.

---

## Restoration note

This file was lost from `~/tasks/` and restored 2026-09-14 from the
2026-09-13 session transcript. The frozen body is verbatim. The hint-level
section is partial, as marked above. Commit this to the repo rather than
leaving it only in a home directory.
