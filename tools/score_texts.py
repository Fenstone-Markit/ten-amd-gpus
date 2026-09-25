#!/usr/bin/env python3
"""Teacher-forced scoring: feed the SAME fixed texts to a server and record the logprob it assigns
to every token. Unlike greedy generation, nothing can diverge, so any difference between two servers
is purely how differently they judge identical text. A systematic kernel error (for example a zero
point off by one) shows up as a clear drop in mean logprob across all texts.

The fixed texts are production's own recorded answers: prompt + output from ref_prod-A.json.

Usage:
  python3 score_texts.py PORT LABEL            score on a server  -> ~/handtest/score_LABEL.json
  python3 score_texts.py --compare LABEL_A LABEL_B
Stdlib only; runs on the host.
"""
import json, math, os, sys, time, urllib.request, urllib.error

HT = os.path.expanduser("~/handtest")

def score(port, label):
    base = f"http://localhost:{port}"
    served = [m["id"] for m in json.load(urllib.request.urlopen(base + "/v1/models", timeout=5))["data"]]
    if served != ["brain"]:
        sys.exit(f"ABORT: port {port} serves {served}, expected ['brain']")
    busy = [l for l in urllib.request.urlopen(base + "/metrics", timeout=5).read().decode().splitlines()
            if l.startswith("vllm:num_requests_running")]
    if any(float(l.split()[-1]) > 0 for l in busy):
        sys.exit(f"ABORT: the server is busy ({busy}); stop Itko or other clients first")
    ref = json.load(open(f"{HT}/ref_prod-A.json"))
    res = {"label": label, "port": port, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "texts": []}
    for i, o in enumerate(ref["outputs"]):
        text = o["prompt"] + o["text"]
        b = json.dumps({"model": "brain", "prompt": text, "max_tokens": 1, "temperature": 0,
                        "echo": True, "logprobs": 1}).encode()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                base + "/v1/completions", b, {"Content-Type": "application/json"}), timeout=300))
        except urllib.error.HTTPError as e:
            sys.exit(f"ABORT: server refused the scoring request ({e.code}): {e.read().decode()[:400]}")
        lp = r["choices"][0]["logprobs"]
        # echo returns the input tokens followed by the 1 generated token; keep only the input
        n_in = len(lp["tokens"]) - 1
        toks, lps = lp["tokens"][:n_in], lp["token_logprobs"][:n_in]
        vals = [x for x in lps[1:] if x is not None]
        if len(vals) != n_in - 1 or not all(math.isfinite(x) for x in vals):
            sys.exit(f"ABORT: text {i}: missing or non-finite logprobs; scoring is not trustworthy on this server")
        res["texts"].append({"tokens": toks, "logprobs": lps})
        print(f"{i:2d}  {n_in:4d} tokens  mean logprob {sum(vals) / len(vals):8.4f}", flush=True)
        json.dump(res, open(f"{HT}/score_{label}.json", "w"), indent=1)
    print(f"saved -> {HT}/score_{label}.json")

def compare(la, lb):
    A = json.load(open(f"{HT}/score_{la}.json")); B = json.load(open(f"{HT}/score_{lb}.json"))
    diffs, sa, sb = [], [], []
    print(f"{la}  vs  {lb}")
    print(f"{'#':>2s} {'tokens':>6s} {'mean A':>9s} {'mean B':>9s} {'mean |d|':>9s} {'max |d|':>8s}")
    for i, (a, b) in enumerate(zip(A["texts"], B["texts"])):
        if a["tokens"] != b["tokens"]:
            sys.exit(f"ABORT: text {i} tokenised differently on the two servers; not comparable")
        d = [abs(x - y) for x, y in zip(a["logprobs"][1:], b["logprobs"][1:])]
        diffs += d; sa += a["logprobs"][1:]; sb += b["logprobs"][1:]
        print(f"{i:2d} {len(d):6d} {sum(a['logprobs'][1:]) / len(d):9.4f} {sum(b['logprobs'][1:]) / len(d):9.4f} "
              f"{sum(d) / len(d):9.5f} {max(d):8.4f}")
    diffs.sort()
    ma, mb = sum(sa) / len(sa), sum(sb) / len(sb)
    print(f"\nall {len(diffs)} tokens: mean logprob {ma:.5f} vs {mb:.5f} (perplexity {math.exp(-ma):.4f} vs {math.exp(-mb):.4f}); "
          f"|d| mean {sum(diffs) / len(diffs):.5f}, 99th pct {diffs[int(0.99 * (len(diffs) - 1))]:.4f}, max {diffs[-1]:.4f}")

if sys.argv[1] == "--compare":
    compare(sys.argv[2], sys.argv[3])
else:
    score(int(sys.argv[1]), sys.argv[2])
