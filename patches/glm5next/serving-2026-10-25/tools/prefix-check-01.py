#!/usr/bin/env python3
# prefix-check-01.py: does prefix caching work and stay correct? About 48k tokens of records, temperature 0.
#  1 cold: records + "gate 1111?"            -> first-token time, answer
#  2 same prompt again                         -> must give the identical answer, much faster first token
#  3 follow-up turn (prompt 1 + its answer + "gate 2222?") -> right code, fast first token (agent-style turn)
#  4 different records (cold, no shared prefix) -> right code, normal first-token time (no false cache hit)
import json, re, sys, time, urllib.request
BASE = "http://127.0.0.1:8000"
def chat(msgs, n=64):
    body = {"model": "glm", "messages": msgs, "max_tokens": n, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data:"): continue
            s = s[5:].strip()
            if s == "[DONE]": break
            d = json.loads(s)
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if ttft is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")): ttft = time.time() - t0
                text += dl.get("content") or ""
    return text.strip(), usage.get("prompt_tokens"), ttft, time.time() - t0
def recs(n, off=0): return "\n".join(f"Record {i}: gate {i} uses code K{((i + off) * 7919) % 9973}." for i in range(1, n + 1))
def want(g, off=0): return f"K{((g + off) * 7919) % 9973}"
Q = lambda g: f"\n\nWhat code does gate {g} use? Answer with the code only, for example K123."
P1 = recs(3000) + Q(1111)
rows = []
a1, pt, t1, _ = chat([{"role": "user", "content": P1}]); rows.append(("1 cold", pt, t1, a1, want(1111) in a1))
a2, pt, t2, _ = chat([{"role": "user", "content": P1}]); rows.append(("2 same again", pt, t2, a2, a2 == a1 and want(1111) in a2))
a3, pt, t3, _ = chat([{"role": "user", "content": P1}, {"role": "assistant", "content": a1}, {"role": "user", "content": "What code does gate 2222 use? Answer with the code only."}])
rows.append(("3 follow-up turn", pt, t3, a3, want(2222) in a3))
P4 = recs(3000, off=17) + Q(1500)
a4, pt, t4, _ = chat([{"role": "user", "content": P4}]); rows.append(("4 new records", pt, t4, a4, want(1500, 17) in a4))
print("| Step | Prompt tokens | First token (s) | Answer | Right |")
print("| --- | --- | --- | --- | --- |")
for name, pt, t, a, ok in rows: print(f"| {name} | {pt} | {t:.2f} | {a[:30]!r} | {'yes' if ok else 'NO'} |")
sys.exit(0 if all(r[4] for r in rows) else 1)
