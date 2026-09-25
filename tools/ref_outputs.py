#!/usr/bin/env python3
"""Record the brain's greedy output on a fixed prompt set: tokens, their logprobs, and the top 5
alternatives at every step. Used to gate the patched kernel against production's own output.

Usage: python3 ref_outputs.py PORT LABEL      e.g.  python3 ref_outputs.py 8000 prod-A
Writes ~/handtest/ref_LABEL.json. Stdlib only; runs on the host.
Record production twice (prod-A, prod-B) BEFORE the experiment: that pair is the run-to-run
baseline every later difference is judged against.
"""
import json, os, sys, time, urllib.request

port, label = int(sys.argv[1]), sys.argv[2]
base = f"http://localhost:{port}"
out = os.path.expanduser(f"~/handtest/ref_{label}.json")
GEN = 128

PROMPTS = [
    "The three laws of thermodynamics can be summarised as follows:",
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
    "Question: A train leaves at 09:40 and arrives at 13:15. How long is the journey?\nAnswer:",
    "Translate into French: 'The warehouse ships three hundred orders before noon.'\nFrench:",
    "Write a short, formal email declining a meeting invitation because of a scheduling conflict.\n\n",
    "In PCIe, a posted write differs from a non-posted read in that",
    "SELECT customer_id, SUM(amount) AS total\nFROM orders\nWHERE",
    "The capital of Alberta is",
    "List five prime numbers greater than 100, one per line:\n",
    "Explain to a ten-year-old why the sky is blue.\n\n",
    "{\"name\": \"Node02\", \"cards\": 10, \"vram_gb\": 240, \"description\": \"",
    "Once upon a time, in a garage full of graphics cards,",
]

def get(path):
    return json.load(urllib.request.urlopen(base + path, timeout=5))

served = [m["id"] for m in get("/v1/models")["data"]]
if served != ["brain"]:
    sys.exit(f"ABORT: port {port} serves {served}, expected ['brain']")
busy = [l for l in urllib.request.urlopen(base + "/metrics", timeout=5).read().decode().splitlines()
        if l.startswith("vllm:num_requests_running")]
if any(float(l.split()[-1]) > 0 for l in busy):
    sys.exit(f"ABORT: the server is busy ({busy}); stop Itko or other clients first")

res = {"label": label, "port": port, "gen": GEN,
       "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "outputs": []}
for i, p in enumerate(PROMPTS):
    b = json.dumps({"model": "brain", "prompt": p, "max_tokens": GEN, "temperature": 0,
                    "seed": 1234, "logprobs": 5, "stream": False}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        base + "/v1/completions", b, {"Content-Type": "application/json"}), timeout=300))
    ch = r["choices"][0]
    lp = ch["logprobs"]
    res["outputs"].append({"prompt": p, "text": ch["text"], "finish": ch.get("finish_reason"),
                           "tokens": lp["tokens"], "token_logprobs": lp["token_logprobs"],
                           "top_logprobs": lp["top_logprobs"]})
    print(f"{i:2d}  {len(lp['tokens']):3d} tokens  {ch['text'][:60]!r}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
print(f"saved -> {out}")
