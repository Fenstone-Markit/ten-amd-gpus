#!/usr/bin/env python3
# lc-check-01.py: long-window check for GLM after glm-serve-36 (step B). Three find-the-code prompts at about 150k, 250k and
# 384k tokens (same record format as ringer-02's needle part), each asked to write at least 128 tokens so decode speed at that
# depth is measured too. One request at a time. Stops on a junction of MAXJ (default 107 C) on any card, or any new non-fatal or
# fatal PCIe error on the cards' chains. Writes ~/ringer/lc-<UTC time>/results.jsonl and summary.txt.
# Use: python3 ~/ringer/lc-check-01.py            (DRY=1 builds the prompts and prints sizes without sending anything)
import glob, json, os, re, sys, threading, time, urllib.request
BASE = "http://127.0.0.1:8000"; MAXJ = int(os.environ.get("MAXJ", "107")); DRY = os.environ.get("DRY") == "1"
NEEDLES = [(8950, 4321), (14900, 11111), (23400, 20000)]          # (records, gate): 152,543, 247,987, 384,038 tokens (GLM tokenizer)
def records(n): return "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1))
def want(t): return f"K{(t * 7919) % 9973}"
CARDS = [p for p in sorted(glob.glob("/sys/bus/pci/devices/*")) if os.path.exists(p + "/mem_info_vram_used")]
def chain(p):
    out = []; q = os.path.realpath(p)
    while q != "/" and not os.path.basename(q).startswith("pci"):
        if os.path.exists(q + "/aer_dev_correctable"): out.append(q)
        q = os.path.dirname(q)
    return out
DEVS = sorted({q for p in CARDS for q in chain(p)})
def tot(path, key):
    try:
        for l in open(path):
            if l.startswith(key): return int(l.split()[1])
    except OSError: pass
    return 0
def pcie(): return {q: (tot(q + "/aer_dev_nonfatal", "TOTAL_ERR_NONFATAL"), tot(q + "/aer_dev_fatal", "TOTAL_ERR_FATAL")) for q in DEVS}
def junctions():
    out = []
    for p in CARDS:
        for h in glob.glob(p + "/hwmon/hwmon*/temp2_input"):
            try: out.append(int(open(h).read()) // 1000)
            except OSError: pass
    return out
stop = threading.Event(); why = []; live = [None]; peak = [0]
def guard(base):
    while not stop.is_set():
        j = junctions(); peak[0] = max([peak[0]] + j)
        if j and max(j) >= MAXJ: why.append(f"junction {max(j)} C"); stop.set()
        now = pcie(); d = [q for q in DEVS if now[q] != base[q]]
        if d: why.append("PCIe non-fatal/fatal error on " + ", ".join(os.path.basename(q) for q in d)); stop.set()
        if stop.is_set() and live[0] is not None:
            try: live[0].close()
            except Exception: pass
        time.sleep(2)
def ask(prompt):
    body = {"model": "glm", "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "min_tokens": 128,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time(); ttft = None; text = ""; usage = {}; done = False
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2400) as r:
        live[0] = r
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data:"): continue
            s = s[5:].strip()
            if s == "[DONE]": done = True; break
            d = json.loads(s)
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if ttft is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")): ttft = time.time() - t0
                text += dl.get("content") or ""
    live[0] = None
    return {"text": text, "usage": usage, "ttft": ttft, "latency": time.time() - t0, "done": done}
if DRY:
    for n, t in NEEDLES:
        p = records(n); print(f"{n} records, gate {t}, want {want(t)}: {len(p):,} characters ")
    sys.exit(0)
D = os.path.expanduser("~/ringer/lc-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime())); os.makedirs(D)
base = pcie(); threading.Thread(target=guard, args=(base,), daemon=True).start()
rows = []
for n, t in NEEDLES:
    if stop.is_set(): break
    prompt = records(n) + f"\n\nWhat code does gate {t} use? Answer with the code only, for example K123."
    print(f"asking: {n} records, gate {t} ...", flush=True)
    try: out = ask(prompt); err = None
    except Exception as e: out = None; err = repr(e)[:200]
    u = (out or {}).get("usage", {}); pt = u.get("prompt_tokens"); ct = u.get("completion_tokens")
    ok = bool(out and out["done"] and re.search(r"\b" + want(t) + r"\b", out["text"]))
    dec = round((ct - 1) / (out["latency"] - out["ttft"]), 2) if out and out["ttft"] and ct and ct > 1 else None
    row = {"records": n, "gate": t, "want": want(t), "ok": ok, "prompt_tokens": pt, "completion_tokens": ct,
           "ttft_s": round(out["ttft"], 1) if out and out["ttft"] else None, "decode_tok_s": dec,
           "got": (out["text"].strip()[:40] if out else None), "error": err, "peak_junction": peak[0]}
    rows.append(row); open(D + "/results.jsonl", "a").write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
stop.set(); time.sleep(0.1)
end = pcie(); newerr = [os.path.basename(q) for q in DEVS if end[q] != base[q]]
L = [f"lc-check-01 {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
     f"found: {sum(r['ok'] for r in rows)} of {len(NEEDLES)} (asked {len(rows)})"]
L += [f"  ~{r['prompt_tokens']} tokens: {'OK' if r['ok'] else 'MISS'} want {r['want']} got {r['got']!r}, first token {r['ttft_s']} s, decode {r['decode_tok_s']} tok/s{', error ' + r['error'] if r['error'] else ''}" for r in rows]
L += [f"peak junction {peak[0]} C; PCIe non-fatal/fatal changes: {', '.join(newerr) or 'none'}; stopped early: {'; '.join(why) or 'no'}"]
open(D + "/summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
sys.exit(0 if rows and all(r["ok"] for r in rows) and len(rows) == len(NEEDLES) and not why else 1)
