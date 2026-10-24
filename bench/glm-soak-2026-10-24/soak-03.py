# soak-03: soak-02 with reviews at low reasoning (the server default): at medium, a 2 KB script's review spent 16,384 tokens
# second-guessing itself and wrote no answer.
# soak-02: soak-01 with reviews at medium reasoning and up to 16,384 tokens (soak-01's high reasoning spent all 8,192 thinking),
# tutor sets at temperature 0.8 with a fresh seed and duplicate questions dropped (temperature 0 repeated sets), and a last status write.
# soak-01: a long run of useful, checked work for GLM-5.3-Flash on node02 (port 8000 only; never Frankenstein's brain).
# Two lanes at once (GLM serves two requests at a time):
#   lane A: script reviews (reasoning high), long find-the-code prompts (about 50k to 117k tokens), mid ones, Hermes-style tool calls
#   lane B: tutor math problem sets (arithmetic checked in code), blind re-solves of those problems, short prompts, tool calls
# Only lane A sends long prompts, so two long prefills never overlap.
# Watchdog (every 2 s): any card at SOAK_MAXJ C (default 100) or the ten cards at SOAK_MAXW W (default 3500) stops the load, saves GLM's
# log and removes the GLM container (SOAK_HEAT_ACTION=none keeps it, for tests). Every 30 s GLM's log since the start is checked for
# a canary hit, illegal memory access, inference-tensor error, dead worker or engine, or hipBLASLt out-of-memory: any one stops the
# load and leaves the container for diagnosis. Three request errors in a row in one lane also stop it.
# Args: OUT_DIR HOURS REVIEW_DIR. Writes results.jsonl, thermal.jsonl, status.txt, summary.md, reviews/, tutor/.
import ast, glob, json, math, os, random, re, statistics, subprocess, sys, threading, time, urllib.error, urllib.request
BASE = "http://127.0.0.1:8000"; D = sys.argv[1]; HOURS = float(sys.argv[2]); RV = sys.argv[3]
MAXJ = int(os.environ.get("SOAK_MAXJ", "100")); MAXW = int(os.environ.get("SOAK_MAXW", "3500")); HEAT_ACTION = os.environ.get("SOAK_HEAT_ACTION", "rm")
T0 = time.time(); END = T0 + HOURS * 3600; START_ISO = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0 - 5))
os.makedirs(D + "/reviews", exist_ok=True); os.makedirs(D + "/tutor", exist_ok=True)
stop = threading.Event(); why = []; lk = threading.Lock(); maxima = {"j": 0, "j_card": "", "w": 0}
rng = random.Random(20261024)
def put(path, rec):
    with lk:
        with open(path, "a") as f: f.write(json.dumps(rec) + "\n")
def halt(reason):
    with lk:
        if not why: why.append(reason)
    stop.set()
# ---------- GLM calls ----------
def call(messages, max_tokens, tools=None, stream=False, effort=None, timeout=900, temperature=0, seed=None):
    body = {"model": "glm", "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if seed is not None: body["seed"] = seed
    if tools: body["tools"] = tools
    if effort: body["chat_template_kwargs"] = {"reasoning_effort": effort}
    if stream: body["stream"] = True; body["stream_options"] = {"include_usage": True}
    t0 = time.time(); ttft = None; content = ""; reasoning = ""; tcs = {}; usage = {}; done = False
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            d = json.loads(r.read()); m = d["choices"][0]["message"]; done = True; usage = d.get("usage", {})
            content = m.get("content") or ""; reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
            for i, tc in enumerate(m.get("tool_calls") or []): tcs[i] = {"name": tc["function"]["name"], "args": tc["function"].get("arguments") or ""}
        else:
            for raw in r:
                s = raw.decode().strip()
                if not s.startswith("data:"): continue
                s = s[5:].strip()
                if s == "[DONE]": done = True; break
                d = json.loads(s)
                if d.get("usage"): usage = d["usage"]
                for c in d.get("choices", []):
                    dl = c.get("delta", {})
                    if ttft is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or dl.get("tool_calls")): ttft = time.time() - t0
                    content += dl.get("content") or ""; reasoning += dl.get("reasoning_content") or dl.get("reasoning") or ""
                    for tc in dl.get("tool_calls") or []:
                        e = tcs.setdefault(tc.get("index", 0), {"name": "", "args": ""}); f = tc.get("function") or {}
                        e["name"] += f.get("name") or ""; e["args"] += f.get("arguments") or ""
    return {"content": content, "reasoning": reasoning, "tool_calls": [tcs[k] for k in sorted(tcs)], "usage": usage,
            "latency": time.time() - t0, "ttft": ttft, "done": done}
def record(lane, kind, out, ok, **extra):
    u = out.get("usage", {}) if out else {}
    put(D + "/results.jsonl", {"t": round(time.time() - T0, 1), "lane": lane, "kind": kind, "ok": ok,
        "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
        "latency": round(out["latency"], 2) if out else None, "ttft": round(out["ttft"], 2) if out and out.get("ttft") else None, **extra})
# ---------- jobs ----------
def needle(lane, n, kind):
    t = rng.randint(1, n); want = f"K{(t * 7919) % 9973}"
    msg = "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1)) + f"\n\nWhat code does gate {t} use? Answer with the code only, for example K123."
    out = call([{"role": "user", "content": msg}], 1024, stream=rng.random() < 0.5)
    ok = out["done"] and re.search(r"\b" + want + r"\b", out["content"]) is not None
    record(lane, kind, out, ok, records=n, want=want, got=out["content"].strip()[:40])
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
ITEMS = ["Rotate the backup drive on the T340", "Order the second PTM7950 pad", "Check G6 fan curve", "Renew the domain", "Clean the intake filters"]
def toolcall(lane):
    stream = rng.random() < 0.5; name = f"todo-{rng.randint(100, 999)}.md"; path = f"/workspace/notes/{name}"
    items = rng.sample(ITEMS, 3); msgs = [{"role": "user", "content": f"Use the read_file tool to read {path}, then tell me the first item on it, word for word."}]
    out = call(msgs, 1024, tools=TOOLS, stream=stream)
    tc = out["tool_calls"][0] if out["tool_calls"] else None
    try: args = json.loads(tc["args"]) if tc else {}
    except Exception: args = {}
    ok1 = bool(tc) and tc["name"] == "read_file" and str(args.get("path", "")).endswith(name)
    record(lane, "tool-call" + ("-stream" if stream else ""), out, ok1, step=1, got=(tc or {}).get("name", "") + " " + str(args)[:60])
    if not ok1: return
    msgs += [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps(args)}}]},
             {"role": "tool", "tool_call_id": "call_1", "content": "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items))}]
    out2 = call(msgs, 1024, tools=TOOLS, stream=stream)
    record(lane, "tool-result" + ("-stream" if stream else ""), out2, items[0].lower() in out2["content"].lower(), step=2, want=items[0], got=out2["content"].strip()[:60])
REVIEW_PROMPT = ("You are reviewing a file from our GPU server tooling (node02: ten RX 7900 XTX cards, rootless podman, vLLM). "
    "Report real problems only: wrong logic, quoting or escaping mistakes, checks that can pass when they should fail, unsafe or destructive commands, "
    "race conditions, resource leaks, and places where the code does not do what its header comment says. For each finding give the line number, "
    "what is wrong, why it matters and a concrete fix. Do not restate the code, do not suggest style changes. If you find nothing real, answer NO FINDINGS. "
    "Keep your thinking focused and always finish with the list of findings.\n\n")
def review(lane, path, n):
    src = open(path).read(); numbered = "\n".join(f"{i + 1:4d}  {l}" for i, l in enumerate(src.split("\n")))
    out = call([{"role": "user", "content": REVIEW_PROMPT + f"File: {os.path.basename(path)}\n```\n{numbered}\n```"}], 16384, effort="low", stream=True, timeout=1800)
    base = f"{D}/reviews/{os.path.basename(path)}-review-{n:02d}"
    with open(base + ".md", "w") as f: f.write(f"# Review {n} of {os.path.basename(path)} by GLM-5.3-Flash (unverified)\n\n" + out["content"] + "\n")
    with open(base + "-reasoning.txt", "w") as f: f.write(out["reasoning"])
    record(lane, "review", out, out["done"] and len(out["content"].strip()) > 0, file=os.path.basename(path), n=n)
TOPICS = ["linear equations", "percentages and discounts", "ratios and proportions", "simple interest", "compound interest", "area and perimeter",
          "circles (use pi)", "the Pythagorean theorem", "probability with dice and coins", "unit conversion", "speed, distance and time",
          "exponents and powers", "systems of two linear equations", "quadratic equations (give the positive root)", "mean and median", "volume of prisms and cylinders (use pi)"]
GEN_PROMPT = ("Write 5 practice problems for a math tutor used by teenagers aged 14 and up. Topic: {topic}. Mix easy, medium and hard. "
    "Each problem must have exactly one numeric answer. Reply with only a JSON array, no prose, no code fences. Each item is an object with: "
    "\"question\" (string), \"answer_value\" (a number), \"answer_text\" (the answer with units if any), and \"check\" (a Python arithmetic expression using only "
    "numbers from the question, + - * / ** % ( ), and if needed pi or sqrt(), that evaluates to answer_value; for example \"(240-60)/3\").")
OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
       ast.Mod: lambda a, b: a % b, ast.FloorDiv: lambda a, b: a // b, ast.Pow: lambda a, b: a ** b if abs(b) <= 64 and abs(a) < 1e9 else math.nan}
def safe_eval(expr):
    expr = expr.replace("math.", "").replace("^", "**")
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)): v = ev(n.operand); return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Name) and n.id in ("pi", "e"): return math.pi if n.id == "pi" else math.e
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("sqrt", "abs", "round") and 1 <= len(n.args) <= 2 and not n.keywords:
            a = [ev(x) for x in n.args]; return math.sqrt(a[0]) if n.func.id == "sqrt" else (abs(a[0]) if n.func.id == "abs" else round(*[a[0]] + [int(x) for x in a[1:]]))
        raise ValueError("not allowed: " + type(n).__name__)
    return ev(ast.parse(expr, mode="eval"))
def close(a, b, rel, absol): return abs(a - b) <= max(absol, rel * max(1.0, abs(b)))
solve_q = []; qid = [0]; seen_q = set()
def tutor_gen(lane):
    topic = rng.choice(TOPICS); out = call([{"role": "user", "content": GEN_PROMPT.format(topic=topic)}], 4096, temperature=0.8, seed=rng.randint(1, 10**9))
    txt = out["content"].strip(); txt = re.sub(r"^```(json)?|```$", "", txt).strip(); good = 0; items = []
    try: items = json.loads(txt[txt.find("["): txt.rfind("]") + 1])
    except Exception as e: put(D + "/tutor/rejected.jsonl", {"topic": topic, "why": "not JSON: " + repr(e)[:80], "raw": txt[:2000]})
    for it in items if isinstance(items, list) else []:
        try:
            v = float(it["answer_value"]); c = safe_eval(str(it["check"])); qn = re.sub(r"\W+", " ", str(it["question"]).lower()).strip()
            with lk: dup = qn in seen_q; seen_q.add(qn)
            if dup: raise ValueError("duplicate question")
            if not close(c, v, 1e-3, 0.01): raise ValueError(f"check gives {c}, answer_value {v}")
            with lk: qid[0] += 1; it = {"id": f"q{qid[0]:04d}", "topic": topic, **it}
            put(D + "/tutor/bank-candidates.jsonl", it); good += 1
            with lk: solve_q.append(it)
        except Exception as e: put(D + "/tutor/rejected.jsonl", {"topic": topic, "why": repr(e)[:120], "item": it})
    record(lane, "tutor-generate", out, good > 0, topic=topic, items=len(items) if isinstance(items, list) else 0, arithmetic_ok=good)
def tutor_solve(lane, it):
    out = call([{"role": "user", "content": "Solve this problem. Show brief working, then end with a final line exactly in the form 'ANSWER: <number>' "
                 "(the number only, no units).\n\n" + it["question"]}], 2048)
    m = re.findall(r"ANSWER:\s*([-+0-9.,/ eE*()pi]+)", out["content"]); got = None
    try: got = safe_eval(m[-1].replace(",", "").strip()) if m else None
    except Exception: got = None
    ok = got is not None and close(got, float(it["answer_value"]), 1e-2, 0.01)
    put(D + "/tutor/" + ("bank-verified.jsonl" if ok else "disagreements.jsonl"), {**it, "blind_answer": got, "blind_text": out["content"][-400:]})
    record(lane, "tutor-solve", out, ok, id=it["id"], want=it["answer_value"], got=got)
# ---------- lanes ----------
FILES = sorted(glob.glob(RV + "/*")); rev_n = {}
def lane_a():
    errs = 0; plan = ["review", "needle-long", "tool", "needle-mid", "needle-long", "review", "tool", "needle-mid"]; i = 0; rc = 0
    while not stop.is_set() and time.time() < END:
        k = plan[i % len(plan)]; i += 1
        try:
            if k == "review":
                f = FILES[rc % len(FILES)]; rc += 1; rev_n[f] = rev_n.get(f, 0) + 1; n = rev_n[f]
                review("A", f, n)
            elif k == "needle-long": needle("A", rng.choice([3000, 4500, 6000, 6900]), "needle-long")
            elif k == "needle-mid": needle("A", rng.choice([800, 1500, 2500]), "needle-mid")
            else: toolcall("A")
            errs = 0
        except Exception as e:
            errs += 1; put(D + "/results.jsonl", {"t": round(time.time() - T0, 1), "lane": "A", "kind": k, "ok": False, "error": repr(e)[:200]})
            if errs >= 3: halt("lane A: three request errors in a row, last " + repr(e)[:120])
            time.sleep(5)
def lane_b():
    errs = 0
    while not stop.is_set() and time.time() < END:
        with lk: pend = solve_q.pop(0) if solve_q else None
        r = rng.random(); k = "solve" if pend and r < 0.6 else ("gen" if r < 0.8 or not pend else ("needle" if r < 0.9 else "tool"))
        if pend and k != "solve":
            with lk: solve_q.insert(0, pend)
        try:
            if k == "solve": tutor_solve("B", pend)
            elif k == "gen": tutor_gen("B")
            elif k == "needle": needle("B", rng.choice([50, 200, 500]), "needle-short")
            else: toolcall("B")
            errs = 0
        except Exception as e:
            errs += 1; put(D + "/results.jsonl", {"t": round(time.time() - T0, 1), "lane": "B", "kind": k, "ok": False, "error": repr(e)[:200]})
            if errs >= 3: halt("lane B: three request errors in a row, last " + repr(e)[:120])
            time.sleep(5)
# ---------- watchdog ----------
CARDS = [p for p in sorted(glob.glob("/sys/bus/pci/devices/*")) if os.path.exists(p + "/mem_info_vram_used")]
NAMES = {"d580007fa41ec467": "G7", "fec7cca764f40561": "G9"}
FAULTS = {"canary hit": r"GLM5_CANARY\] HIT", "illegal memory access": r"(?i)illegal memory access", "inference-tensor error": r"Inplace update to inference tensor",
          "worker died": r"died unexpectedly", "engine dead": r"EngineDeadError", "hipBLASLt out of memory": r"hipblaslt\.cpp:164", "memory access fault": r"Memory access fault"}
def save_log(tag):
    with open(f"{D}/glm-log-{tag}.txt", "w") as f: subprocess.run(["podman", "logs", "--since", START_ISO, "n02-glm"], stdout=f, stderr=subprocess.STDOUT, timeout=60)
def watchdog():
    last_t = 0; last_log = 0
    while not stop.is_set() and time.time() < END + 1800:
        js = []; w = 0
        for p in CARDS:
            h = glob.glob(p + "/hwmon/hwmon*")[0]; j = int(open(h + "/temp2_input").read()) // 1000; pw = int(open(h + "/power1_average").read()) // 1000000
            js.append((j, NAMES.get(open(p + "/unique_id").read().strip(), os.path.basename(p)[5:10]))); w += pw
        mj, mc = max(js)
        with lk:
            if mj > maxima["j"]: maxima["j"], maxima["j_card"] = mj, mc
            maxima["w"] = max(maxima["w"], w)
        if time.time() - last_t >= 10: put(D + "/thermal.jsonl", {"t": round(time.time() - T0), "max_j": mj, "card": mc, "sum_w": w, "j": [x[0] for x in js]}); last_t = time.time()
        if mj >= MAXJ or w >= MAXW:
            halt(f"watchdog: {mc} at {mj} C, ten cards at {w} W (limits {MAXJ} C, {MAXW} W)"); save_log("at-heat-stop")
            if HEAT_ACTION == "rm": subprocess.run(["podman", "rm", "-f", "n02-glm"], capture_output=True, timeout=120)
            return
        if time.time() - last_log >= 30:
            last_log = time.time()
            try:
                run = subprocess.run(["podman", "inspect", "--format", "{{.State.Running}}", "n02-glm"], capture_output=True, text=True, timeout=30).stdout.strip()
                if run != "true": halt("GLM container is not running"); return
                L = subprocess.run(["podman", "logs", "--since", START_ISO, "n02-glm"], capture_output=True, text=True, timeout=60)
                L = L.stdout + L.stderr
                for name, pat in FAULTS.items():
                    if re.search(pat, L): halt("GLM log: " + name); save_log("at-fault"); return
            except Exception as e: put(D + "/results.jsonl", {"t": round(time.time() - T0, 1), "lane": "watchdog", "kind": "log-check", "ok": False, "error": repr(e)[:200]})
        time.sleep(2)
def status():
    while not stop.is_set() and time.time() < END + 1800:
        try:
            R = [json.loads(l) for l in open(D + "/results.jsonl")] if os.path.exists(D + "/results.jsonl") else []
            with open(D + "/status.txt", "w") as f:
                f.write(f"elapsed {(time.time() - T0) / 60:.0f} of {HOURS * 60:.0f} min; requests {len(R)}, ok {sum(1 for r in R if r.get('ok'))}, "
                        f"not ok {sum(1 for r in R if not r.get('ok'))}; hottest {maxima['j']} C ({maxima['j_card']}); peak {maxima['w']} W; stop: {why[0] if why else 'none'}\n")
        except Exception: pass
        time.sleep(30)
        if all(not t.is_alive() for t in lanes): break
# ---------- run ----------
put(D + "/results.jsonl", {"t": 0, "lane": "-", "kind": "start", "ok": True, "hours": HOURS, "files": [os.path.basename(f) for f in FILES], "maxj": MAXJ, "maxw": MAXW})
lanes = [threading.Thread(target=lane_a, name="A"), threading.Thread(target=lane_b, name="B")]
wd = threading.Thread(target=watchdog, daemon=True); st = threading.Thread(target=status, daemon=True)
wd.start(); [t.start() for t in lanes]; st.start(); [t.join() for t in lanes]
if not why: why.append("time up" if time.time() >= END else "lanes ended")
stop.set(); time.sleep(3)
# ---------- summary ----------
R = [json.loads(l) for l in open(D + "/results.jsonl")][1:]
def tok(r): return f"{r['prompt_tokens'] / r['latency']:.0f}" if r.get("prompt_tokens") and r.get("latency") else "-"
by = {}
for r in R: by.setdefault(r["kind"], []).append(r)
lines = [f"# GLM soak {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(T0))}, {(time.time() - T0) / 60:.0f} minutes", "",
         f"Ended: {why[0]}. Hottest card {maxima['j']} C ({maxima['j_card']}); highest ten-card power {maxima['w']} W.", "",
         "| Kind | Requests | Correct or complete | Not | Errors | Median latency (s) | Largest prompt (tokens) |", "| --- | --- | --- | --- | --- | --- | --- |"]
for k, rs in sorted(by.items()):
    lat = [r["latency"] for r in rs if r.get("latency")]; pt = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens")]
    lines.append(f"| {k} | {len(rs)} | {sum(1 for r in rs if r.get('ok'))} | {sum(1 for r in rs if not r.get('ok') and 'error' not in r)} | {sum(1 for r in rs if 'error' in r)} | "
                 f"{statistics.median(lat):.1f} | {max(pt) if pt else '-'} |" if lat else f"| {k} | {len(rs)} | {sum(1 for r in rs if r.get('ok'))} | - | {sum(1 for r in rs if 'error' in r)} | - | - |")
nl = [r for r in R if r["kind"] == "needle-long" and r.get("ok")]
if nl: lines += ["", "Long prompts read: " + ", ".join(f"{r['prompt_tokens']} tokens in {r['latency']:.0f} s" for r in nl[:12])]
cnt = lambda p: sum(1 for _ in open(p)) if os.path.exists(p) else 0
lines += ["", f"Tutor bank: {cnt(D + '/tutor/bank-candidates.jsonl')} problems passed the arithmetic check, {cnt(D + '/tutor/bank-verified.jsonl')} confirmed by a blind re-solve, "
          f"{cnt(D + '/tutor/disagreements.jsonl')} disagreements, {cnt(D + '/tutor/rejected.jsonl')} rejected.", f"Reviews written: {len(glob.glob(D + '/reviews/*-review-*.md'))} (unverified; read before acting on any)."]
errs = [r for r in R if "error" in r]
if errs: lines += ["", "Errors:"] + [f"- t={r['t']} s lane {r['lane']} {r['kind']}: {r['error']}" for r in errs[:15]]
open(D + "/summary.md", "w").write("\n".join(lines) + "\n")
open(D + "/status.txt", "w").write(f"finished after {(time.time() - T0) / 60:.0f} min; {len(R)} requests, {sum(1 for r in R if r.get('ok'))} ok; hottest {maxima['j']} C ({maxima['j_card']}); peak {maxima['w']} W; ended: {why[0]}\n")
try: save_log("at-end")
except Exception: pass
print("\n".join(lines))
