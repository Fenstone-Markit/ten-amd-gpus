#!/usr/bin/env python3
# prefill-speed-01.py: how fast GLM reads a new prompt on its own. Cold reads (fresh cache_salt, so nothing is reused) of
# about 39k, 74k and 116k tokens; time to first token, tokens per second, answer checked; and the most VRAM any card used
# during the reads (sysfs mem_info_vram_used, sampled every 0.25 s), per card, as GiB.
import glob, json, sys, threading, time, urllib.request
TAG = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%H%M%S")
recs = lambda n: "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1))
cards = sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")); peak = {c: 0 for c in cards}; stop = threading.Event()
def sample():
    while not stop.is_set():
        for c in cards:
            try: peak[c] = max(peak[c], int(open(c).read()))
            except Exception: pass
        time.sleep(0.25)
th = threading.Thread(target=sample, daemon=True); th.start()
for n, g in [(2500, 2017), (4500, 4001), (6900, 6789)]:
    body = {"model": "glm", "messages": [{"role": "user", "content": recs(n) + f"\n\nWhat code does gate {g} use? Answer with the code only, for example K123."}],
            "max_tokens": 32, "temperature": 0, "cache_salt": f"pf-{TAG}-{n}", "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time(); first = None; text = ""; pt = None
    for raw in urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800):
        s = raw.decode().strip()
        if not s.startswith("data:") or s[5:].strip() == "[DONE]": continue
        d = json.loads(s[5:])
        if d.get("usage"): pt = d["usage"]["prompt_tokens"]
        for c in d.get("choices", []):
            t = (c.get("delta") or {}).get("content") or ""
            if t and first is None: first = time.time() - t0
            text += t
    want = f"K{(g * 7919) % 9973}"
    print(f"{pt} tokens: first token {first:.1f} s = {pt / first:.0f} tokens/s, answer {'ok' if want in text else 'WRONG ' + repr(text[:30])}", flush=True)
stop.set(); time.sleep(0.3)
print("peak VRAM per card (GiB, sysfs order): " + " ".join(f"{peak[c] / 2**30:.1f}" for c in cards))
