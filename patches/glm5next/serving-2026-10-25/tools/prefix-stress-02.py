#!/usr/bin/env python3
# prefix-stress-02.py: as -01, but closer to the ringer run that failed. Each round (fresh cache_salt) walks the long end of the needle
# ladder: 2500 -> 3000 -> 4500 -> 4500 -> 6000 -> 6900 records, each one reusing the previous one from the cache.
# With CONC=1 a second thread keeps a short counting request running at the same time (max-num-seqs is 2), so cached prefills are
# mixed with another request's decode, as happened during the failing run. Failures are saved to prefix-stress-02-fail-*.json.
import json, os, re, sys, threading, time, urllib.request
BASE = "http://127.0.0.1:8000"; ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3; CONC = os.environ.get("CONC", "1") == "1"; TAG = time.strftime("%H%M%S")
recs = lambda n: "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1))
def post(body): return json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
def ask(n, gate, salt):
    t = time.time(); r = post({"model": "glm", "messages": [{"role": "user", "content": recs(n) + f"\n\nWhat code does gate {gate} use? Answer with the code only, for example K123."}],
                               "max_tokens": 1024, "temperature": 0, "cache_salt": salt})
    m = r["choices"][0]["message"]; return m.get("content") or "", m.get("reasoning_content") or m.get("reasoning") or "", r["usage"], round(time.time() - t, 1)
stop = threading.Event(); side = {"n": 0, "bad": 0}
def background():
    k = 0
    while not stop.is_set():
        a = 1 + k % 50; b = a + 120; k += 1
        r = post({"model": "glm", "messages": [{"role": "user", "content": f"Count from {a} to {b}, numbers separated by commas, nothing else."}], "max_tokens": 700, "temperature": 0})
        nums = [int(x) for x in re.findall(r"\d+", r["choices"][0]["message"].get("content") or "")]
        side["n"] += 1; side["bad"] += nums[: b - a + 1] != list(range(a, b + 1))
th = threading.Thread(target=background, daemon=True)
if CONC: th.start()
fails = 0
for k in range(ROUNDS):
    salt = f"stress2-{TAG}-{k}"; line = []
    for step, (n, g) in enumerate([(2500, 2017 - k), (3000, 2950 - k), (4500, 4001 + k), (4500, 1234 + k), (6000, 5555 - k), (6900, 6789 - k)], 1):
        c, rz, u, s = ask(n, g, salt); want = f"K{(g * 7919) % 9973}"; ok = want in c
        line.append(f"{n}/{g} {'ok' if ok else 'WRONG'} {u['completion_tokens']}t {s}s")
        if not ok:
            fails += 1; json.dump({"round": k, "step": step, "records": n, "gate": g, "want": want, "content": c[:2000], "reasoning": rz[:4000], "usage": u, "secs": s},
                                  open(__import__("os").path.expanduser("~/ringer/") + f"prefix-stress-02-fail-{TAG}-{k}-{step}.json", "w"), indent=1)
    print(f"round {k + 1}: " + " | ".join(line), flush=True)
stop.set(); time.sleep(0.1)
print(f"failures: {fails}; background counts {side['n']} done, {side['bad']} wrong"); sys.exit(1 if fails or side["bad"] else 0)
