#!/usr/bin/env python3
"""Compare two recorded runs from ref_outputs.py, prompt by prompt.

Usage: python3 compare_refs.py LABEL_A LABEL_B     e.g.  compare_refs.py prod-A prod-B

For each prompt: the first token position where the two runs differ (or 'same' if all 128 match),
and, over the shared prefix, the largest difference in the chosen token's logprob and how often
the two runs agree on the top-5 set. Compare prod-A vs prod-B first: that is production's own
run-to-run noise. The patched kernel passes if prod-A vs patched looks like that, not worse.
"""
import json, os, sys

def load(label):
    return json.load(open(os.path.expanduser(f"~/handtest/ref_{label}.json")))

A, B = load(sys.argv[1]), load(sys.argv[2])
if len(A["outputs"]) != len(B["outputs"]):
    sys.exit("ABORT: the two runs have different prompt counts")
print(f"{A['label']}  vs  {B['label']}")
print(f"{'#':>2s} {'first diff':>10s} {'max |dlogprob|':>15s} {'top5 same':>10s}  text A")
diverged, worst, same_sets, steps = 0, 0.0, 0, 0
for i, (a, b) in enumerate(zip(A["outputs"], B["outputs"])):
    if a["prompt"] != b["prompt"]:
        sys.exit(f"ABORT: prompt {i} differs between the runs")
    n = min(len(a["tokens"]), len(b["tokens"]))
    first = next((k for k in range(n) if a["tokens"][k] != b["tokens"][k]), None)
    if first is None and len(a["tokens"]) != len(b["tokens"]):
        first = n
    shared = n if first is None else first
    dmax = max((abs(a["token_logprobs"][k] - b["token_logprobs"][k]) for k in range(shared)
                if a["token_logprobs"][k] is not None and b["token_logprobs"][k] is not None), default=0.0)
    ss = sum(1 for k in range(shared) if set(a["top_logprobs"][k]) == set(b["top_logprobs"][k]))
    diverged += first is not None
    worst = max(worst, dmax); same_sets += ss; steps += shared
    print(f"{i:2d} {'same' if first is None else first:>10} {dmax:15.4f} {ss:>4d}/{shared:<5d}  {a['text'][:50]!r}")
print(f"\nprompts that diverge: {diverged} of {len(A['outputs'])}   largest logprob difference: {worst:.4f}   "
      f"top-5 sets identical at {same_sets} of {steps} shared steps ({100 * same_sets / max(steps, 1):.1f}%)")
