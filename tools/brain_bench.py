#!/usr/bin/env python3
"""The brain's decode measurement, exactly as documented in 03_SCRIPTS.md
("The measurement script that produced all of this"), saved as a file.

Method unchanged: repeated-sentence prompt of about ctx tokens, 64 generated tokens forced with
min_tokens + ignore_eos, temperature 0, streamed; ms/token is timed from the first streamed
token to the last. Two runs per length (the second is warm when prefix caching is on).
The launch script's header numbers (30.0 / 31.4 / 35.9 ms at 2K / 16K / 60K) came from this method.

Usage: python3 brain_bench.py PORT LABEL      e.g.  python3 brain_bench.py 8000 prod-before
Writes ~/handtest/bench_LABEL.json. Stdlib only; runs on the host.
"""
import json, os, sys, time, urllib.request

port, label = int(sys.argv[1]), sys.argv[2]
base = f"http://localhost:{port}"
out = os.path.expanduser(f"~/handtest/bench_{label}.json")

def get(path):
    return json.load(urllib.request.urlopen(base + path, timeout=5))

served = [m["id"] for m in get("/v1/models")["data"]]
if served != ["brain"]:
    sys.exit(f"ABORT: port {port} serves {served}, expected ['brain']")
busy = [l for l in urllib.request.urlopen(base + "/metrics", timeout=5).read().decode().splitlines()
        if l.startswith("vllm:num_requests_running")]
if any(float(l.split()[-1]) > 0 for l in busy):
    sys.exit(f"ABORT: the server is busy ({busy}); stop Itko or other clients first, numbers must be taken alone")

def run(ctx, gen=64):
    p = "The quick brown fox jumps over the lazy dog. " * (ctx // 10)
    b = json.dumps({"model": "brain", "prompt": p, "max_tokens": gen, "min_tokens": gen,
                    "temperature": 0, "ignore_eos": True, "stream": True}).encode()
    r = urllib.request.Request(base + "/v1/completions", b, {"Content-Type": "application/json"})
    t0 = time.time(); f = None; n = 0
    for line in urllib.request.urlopen(r):
        if line.startswith(b"data: ") and b"[DONE]" not in line:
            if f is None:
                f = time.time() - t0
            n += 1
    tot = time.time() - t0
    ms = (tot - f) / (n - 1) * 1000 if n > 1 else 0
    print(f"ctx~{ctx:>6}  first token {f:6.2f}s  {ms:6.1f} ms/token  ({(n - 1) / (tot - f):5.2f} tok/s)  tokens {n}", flush=True)
    return {"ctx": ctx, "first_token_s": f, "ms_per_token": ms, "tokens": n}

print(f"[{time.strftime('%H:%M:%S')}] {label} on :{port}", flush=True)
res = {"label": label, "port": port, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
for c in (2000, 16000, 60000):
    res["runs"].append(run(c)); res["runs"].append(run(c))
    json.dump(res, open(out, "w"), indent=1)
print(f"saved -> {out}")
