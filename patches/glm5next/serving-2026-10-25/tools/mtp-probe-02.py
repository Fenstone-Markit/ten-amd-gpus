#!/usr/bin/env python3
# mtp-probe-02.py: layout bake-off probe. Three fixed prompts at temperature 0 (counting, code, prose; tok/s and the draft
# acceptance per position if MTP is on) plus one find-the-code prompt of about 50k tokens (working context) asked to write
# at least 256 tokens: found or not, first-token time, decode tok/s at that depth. Prints one summary line per item.
import json, re, subprocess, sys, time, urllib.request
BASE = "http://127.0.0.1:8000"
def chat(msgs, n, min_tokens=None, stream=False):
    body = {"model": "glm", "messages": msgs, "max_tokens": n, "temperature": 0}
    if min_tokens: body["min_tokens"] = min_tokens
    if stream: body["stream"] = True; body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=1200) as r:
        if not stream:
            d = json.loads(r.read()); return d["choices"][0]["message"].get("content") or "", d["usage"], time.time() - t0, None
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
    return text, usage, time.time() - t0, ttft
def accept():
    out = subprocess.run("podman logs --since 15s n02-glm 2>&1 | grep SpecDecoding | tail -1", shell=True, capture_output=True, text=True).stdout
    m = re.search(r"Per-position acceptance rate: ([0-9., ]+)", out); return m.group(1).strip(" ,") if m else "-"
res = {}
for name, p in [("count", "Count from 1 to 150, numbers separated by commas, nothing else."),
                ("code", "Write a Python function that returns the nth Fibonacci number, with a docstring and three doctest examples."),
                ("prose", "Explain in plain English how a refrigerator keeps food cold.")]:
    c, u, dt, _ = chat([{"role": "user", "content": p}], 300); time.sleep(11)
    n = u["completion_tokens"]; res[name] = round(n / dt, 1)
    print(f"{name}: {n} tokens in {dt:.1f} s = {n/dt:.1f} tok/s; drafts accepted per position: {accept()}", flush=True)
recs = "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, 3001))
c, u, dt, ttft = chat([{"role": "user", "content": recs + "\n\nWhat code does gate 2222 use? Answer with the code only, for example K123."}], 512, min_tokens=256, stream=True)
want = f"K{(2222 * 7919) % 9973}"; ct = u.get("completion_tokens", 0)
dec = (ct - 1) / (dt - ttft) if ttft and ct > 1 else 0
print(f"50k: {u.get('prompt_tokens')} prompt tokens; found {want}: {want in c} (answer starts {c.strip()[:24]!r}); first token {ttft:.1f} s; decode {dec:.1f} tok/s at that depth", flush=True)
