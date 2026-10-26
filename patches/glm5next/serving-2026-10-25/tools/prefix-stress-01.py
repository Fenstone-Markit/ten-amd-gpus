#!/usr/bin/env python3
# prefix-stress-01.py: hunts the one-off bad answers seen in ringer pfx-3x4 (long needles answered with 1024 tokens and no content).
# Each round uses a fresh cache_salt, so nothing is shared with earlier rounds: (1) 3000 records, cold (about 48k tokens);
# (2) 4500 records (about 74k): reuses the 48k from step 1 and computes the other 26k, the step that failed once;
# (3) the same 4500 records with another gate: a full cache hit. Every answer is checked exactly. On a failure the content and the
# reasoning text are written to ~/ringer/prefix-stress-01-fail-<round>-<step>.json so we can see what the model produced.
import json, sys, time, urllib.request
BASE = "http://127.0.0.1:8000"; ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8; TAG = time.strftime("%H%M%S")
recs = lambda n: "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1))
def ask(n, gate, salt):
    body = {"model": "glm", "messages": [{"role": "user", "content": recs(n) + f"\n\nWhat code does gate {gate} use? Answer with the code only, for example K123."}],
            "max_tokens": 1024, "temperature": 0, "cache_salt": salt}
    t = time.time(); r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    m = r["choices"][0]["message"]; return m.get("content") or "", m.get("reasoning_content") or m.get("reasoning") or "", r["usage"], round(time.time() - t, 1)
fails = 0
for k in range(ROUNDS):
    salt = f"stress-{TAG}-{k}"; line = []
    for step, (n, g) in enumerate([(3000, 2950 - k), (4500, 4001 + k), (4500, 1234 + k)], 1):
        c, rz, u, s = ask(n, g, salt); want = f"K{(g * 7919) % 9973}"; ok = want in c
        line.append(f"{n}/{g} {'ok' if ok else 'WRONG'} {u['completion_tokens']}tok {s}s")
        if not ok:
            fails += 1; json.dump({"round": k, "step": step, "records": n, "gate": g, "want": want, "content": c[:2000], "reasoning": rz[:4000], "usage": u, "secs": s},
                                  open(__import__("os").path.expanduser("~/ringer/") + f"prefix-stress-01-fail-{TAG}-{k}-{step}.json", "w"), indent=1)
    print(f"round {k + 1}: " + " | ".join(line), flush=True)
print("failures:", fails); sys.exit(1 if fails else 0)
