#!/usr/bin/env python3
# first-read-01.py <label>: right after a GLM start, the first long cold read: 4500 records (74k tokens), fresh salt, then the
# same question again from the cache, then a second fresh 74k read. Saves content and reasoning text of every answer to
# ~/ringer/first-read-01-<label>.json so a bad read can be looked at.
import json, sys, time, urllib.request
L = sys.argv[1] if len(sys.argv) > 1 else "x"; T = time.strftime("%H%M%S")
recs = lambda n: "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1))
out = []
for step, (salt, g) in enumerate([("a", 4001), ("a", 1234), ("b", 4001)], 1):
    body = {"model": "glm", "messages": [{"role": "user", "content": recs(4500) + f"\n\nWhat code does gate {g} use? Answer with the code only, for example K123."}],
            "max_tokens": 300, "temperature": 0, "cache_salt": f"fr-{L}-{T}-{salt}"}
    t = time.time(); r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    m = r["choices"][0]["message"]; c = m.get("content") or ""; rz = m.get("reasoning_content") or m.get("reasoning") or ""
    want = f"K{(g * 7919) % 9973}"
    out.append({"step": step, "salt": salt, "gate": g, "want": want, "ok": want in c, "content": c[:600], "reasoning": rz[:1500], "usage": r["usage"], "secs": round(time.time() - t, 1)})
    print(f"step {step} (salt {salt}, gate {g}): {'ok' if want in c else 'WRONG'}, {r['usage']['completion_tokens']} tokens, {time.time() - t:.1f} s, content {c[:40]!r}, reasoning {rz[:80]!r}", flush=True)
json.dump(out, open(__import__("os").path.expanduser("~/ringer/") + f"first-read-01-{L}.json", "w"), indent=1)
