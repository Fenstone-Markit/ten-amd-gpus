#!/usr/bin/env python3
# mixed-check-02.py: as -01 (speed of a long prompt read beside a generating request), plus every answer checked exactly:
# the needle's code, and the background count (every number 1..400 in order). ROUNDS rounds, each with a fresh salt and
# another gate and prompt size (2500, 3000, 4500 records: about 39k, 48k, 74k tokens).
import json, re, sys, threading, time, urllib.request
BASE = "http://127.0.0.1:8000"; TAG = time.strftime("%H%M%S"); ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
recs = lambda n: "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1))
def stream(body, stamps, text):
    body = dict(body, stream=True); t0 = time.time(); first = None
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for raw in urllib.request.urlopen(req, timeout=1800):
        s = raw.decode().strip()
        if not s.startswith("data:") or s[5:].strip() == "[DONE]": continue
        for c in json.loads(s[5:]).get("choices", []):
            txt = (c.get("delta") or {}).get("content") or ""
            if txt:
                if first is None: first = time.time() - t0
                stamps.append((time.time(), txt.count(","))); text.append(txt)
    return first
bad = 0
for r, (n, g) in enumerate([(2500, 2017), (3000, 2950), (4500, 4001), (2500, 777), (3000, 1500)][:ROUNDS]):
    bg, bgt = [], []
    th = threading.Thread(target=lambda: stream({"model": "glm", "messages": [{"role": "user", "content": "Count from 1 to 400, numbers separated by commas, nothing else."}],
                                                 "max_tokens": 1800, "temperature": 0}, bg, bgt)); th.start()
    time.sleep(5); p0 = time.time(); nt = []
    f = stream({"model": "glm", "messages": [{"role": "user", "content": recs(n) + f"\n\nWhat code does gate {g} use? Answer with the code only, for example K123."}],
                "max_tokens": 32, "temperature": 0, "cache_salt": f"mixed2-{TAG}-{r}"}, [], nt)
    p1 = p0 + f; th.join()
    want = f"K{(g * 7919) % 9973}"; nok = want in "".join(nt)
    nums = [int(x) for x in re.findall(r"\d+", "".join(bgt))]; cok = nums[:400] == list(range(1, 401))
    dur = sum(k for t, k in bg if p0 <= t < p1) / max(p1 - p0, 1e-9)
    bad += (not nok) + (not cok)
    print(f"round {r + 1}: {n} records, first token {f:.1f} s, needle {'ok' if nok else 'WRONG ' + repr(''.join(nt)[:30])}; "
          f"background count {'ok' if cok else 'WRONG'}, {dur:.1f} numbers/s during the read", flush=True)
print("failures:", bad); sys.exit(1 if bad else 0)
