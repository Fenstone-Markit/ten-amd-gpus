# ecount-calib-02: drive GLM (node02 port 8000) through seven categories of work, one after another, and save a snapshot of
# the expert counters after each one, so every category gets its own expert profile. Nothing here changes GLM.
# A snapshot: write trigger-<tag> into the counts folder, send one tiny request so every pipeline stage runs a step, and
# wait until every stage's <tag>-rank<R>.pt exists. Categories run two requests at a time.
# Args: COUNTS_DIR (host path of $CACHE/expert-count) DATA_DIR (bank-verified.jsonl, ecount-verses-01.jsonl, review/*)
import glob, json, os, random, sys, threading, time, urllib.request
CD, DD = sys.argv[1], sys.argv[2]; BASE = "http://127.0.0.1:8000"; RANKS = [0, 2, 4, 6, 8]; random.seed(11)
LOG = open(os.path.join(CD, "calib-log.jsonl"), "a")
def call(messages, max_tokens=700, tools=None):
    body = {"model": "glm", "messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
    if tools: body["tools"] = tools
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    return r, time.time() - t0
def metrics():
    # vLLM's own totals, logged at every snapshot: if they grew by more than this driver sent, someone else (Itko) used GLM
    # during that category, and the counts for it are mixed
    t = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for k in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:request_success_total"):
        out[k.split(":")[1]] = sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines() if l.startswith(k + "{"))
    return out
def snap(tag):
    LOG.write(json.dumps({"snapshot": tag, "metrics": metrics(), "t": time.time()}) + "\n"); LOG.flush()
    open(os.path.join(CD, "trigger-" + tag), "w").close()
    call([{"role": "user", "content": "Say ok."}], 4)
    for _ in range(120):
        if all(os.path.exists(os.path.join(CD, f"{tag}-rank{r}.pt")) for r in RANKS): return True
        time.sleep(1); call([{"role": "user", "content": "Say ok."}], 4)
    return False
TOOLS = [{"type": "function", "function": {"name": "run_shell", "description": "Run a shell command on node02 and return its output",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
AGENT = ["Check how hot each GPU on node02 is right now and tell me which is hottest.", "Find out whether the GLM container is running and how long it has been up.",
         "Read /workspace/notes/todo.md and summarise the open items.", "Check free disk space on the model disk and warn me if it is under 500 GB.",
         "List the Itko task folders in the workspace and tell me which ones have no commit yet.", "Find the last 20 lines of the GLM server log and say whether anything failed.",
         "Work out how much VRAM each card is using and whether any card is near full.", "Check whether port 8020 answers and what it serves.",
         "Read FINDINGS.md and list the three most recent findings.", "Check the fan speed of every card and flag any under 800 rpm while hot.",
         "Plan the steps to move two GPUs from one server to another without losing any running work.", "Write a short checklist for restarting the GLM server safely."]
GEN = ["Why did the Roman Empire split into east and west?", "How do fish see colours underwater, and does lure colour matter?", "What makes a good game level for beginners?",
       "How do I draw hands that look natural?", "Why is the sky blue but sunsets red?", "How does a refrigerator move heat out of the box?", "What caused the First World War?",
       "How do volcanoes form island chains like Hawaii?", "What is the difference between weather and climate?", "How do I make a simple fishing lure from a spoon?",
       "Why do cats purr?", "How does a game engine draw 3D graphics on a flat screen?", "What was the Silk Road and why did it matter?", "How do airplanes stay in the air?",
       "What is a black hole?", "How did the printing press change Europe?", "Why do we have leap years?", "How do noise-cancelling headphones work?",
       "What are good habits for learning to draw characters?", "How do tides work?"]
MED = ["How does the immune system recognise a virus?", "What does insulin do in the body?", "Why do antibiotics not work on colds?", "How does the heart pump blood around the body?",
       "What happens in the body during a fever?", "How do vaccines train the immune system?", "What is the difference between a bacterium and a virus?", "How do kidneys filter blood?",
       "Why is sleep important for memory?", "What does cholesterol do, and why do doctors measure it?", "How does a broken bone heal?", "What is blood pressure and what do the two numbers mean?",
       "How do painkillers like ibuprofen work?", "What causes allergies?", "How does the eye focus light?", "What are red and white blood cells for?"]
CODE = ["Write a bash function that prints the hottest GPU from /sys/bus/pci/devices/*/hwmon/hwmon*/temp2_input.", "Write a Python script that reads a JSONL file and prints the mean of a numeric field.",
        "Explain what `set -o pipefail` changes in a bash script, with an example.", "Write a Python function that retries an HTTP request three times with backoff.",
        "Fix this bash line so it fails if cp fails: cp a b && echo done; sed -i s/x/y/ b", "Write a small HIP kernel that adds two float arrays.",
        "Explain the difference between podman and docker for rootless containers.", "Write a Python script that watches a log file and prints lines containing 'error'."]
def tasks_for(cat):
    if cat == "agent-tools":
        out = []
        for q in AGENT * 3:
            out.append(("tool", q))
        return out
    if cat == "code":
        out = [("chat", q, 900) for q in CODE * 2]
        for f in sorted(glob.glob(os.path.join(DD, "review", "*")))[:6]:
            out.append(("chat", "Explain briefly what this script does and point out one risk:\n```\n" + open(f).read()[:9000] + "\n```", 900))
        return out
    if cat == "tutor-math":
        bank = [json.loads(l) for l in open(os.path.join(DD, "bank-verified.jsonl"))]; random.shuffle(bank)
        return [("chat", "Solve this and explain your steps simply for a 14-year-old:\n" + b["question"], 600) for b in bank[:60]]
    if cat == "long-context":
        out = []
        for n in (1500, 2500, 3500, 4500, 5500, 6500):
            t = random.randint(1, n); out.append(("needle", n, t))
        return out
    if cat == "arabic-urdu":
        vs = [json.loads(l) for l in open(os.path.join(DD, "ecount-verses-01.jsonl"))]; out = []
        for v in vs[:20]: out.append(("chat", "اس آیت کا اردو میں ترجمہ کریں اور مختصر وضاحت لکھیں:\n" + v["ar"], 500))
        for v in vs[20:40]: out.append(("chat", "Translate this Urdu sentence into English: " + v["ur"], 300))
        for v in vs[40:60]: out.append(("chat", "Translate this Arabic into English and say which words carry the main meaning: " + v["ar"], 400))
        return out
    if cat == "general":
        return [("chat", q, 600) for q in GEN * 2]
    if cat == "medicine":
        return [("chat", q, 600) for q in MED * 2]
def run_task(cat, t):
    try:
        if t[0] == "tool":
            msgs = [{"role": "system", "content": "You are Itko, an agent that works on the node02 GPU server through tools."}, {"role": "user", "content": t[1]}]
            r, dt = call(msgs, 700, TOOLS); m = r["choices"][0]["message"]; u = r["usage"]
            if m.get("tool_calls"):
                tc = m["tool_calls"][0]; msgs += [{"role": "assistant", "content": None, "tool_calls": [tc]},
                    {"role": "tool", "tool_call_id": tc["id"], "content": "G1 61C 24% fan\nG7 78C 41% fan\nG9 66C 30% fan\nok"}]
                r2, dt2 = call(msgs, 500, TOOLS); u = {k: u.get(k, 0) + r2["usage"].get(k, 0) for k in ("prompt_tokens", "completion_tokens")}; dt += dt2
        elif t[0] == "needle":
            n, tg = t[1], t[2]
            msg = "\n".join(f"Record {i}: gate {i} uses code K{(i * 7919) % 9973}." for i in range(1, n + 1)) + f"\n\nWhat code does gate {tg} use? Answer with the code only."
            r, dt = call([{"role": "user", "content": msg}], 64); u = r["usage"]
        else:
            r, dt = call([{"role": "user", "content": t[1]}], t[2]); u = r["usage"]
        LOG.write(json.dumps({"cat": cat, "kind": t[0], "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "s": round(dt, 1)}) + "\n"); LOG.flush()
    except Exception as e:
        LOG.write(json.dumps({"cat": cat, "kind": t[0], "error": repr(e)[:200]}) + "\n"); LOG.flush()
def run_cat(cat):
    q = tasks_for(cat); lk = threading.Lock()
    def lane():
        while True:
            with lk:
                if not q: return
                t = q.pop(0)
            run_task(cat, t)
    th = [threading.Thread(target=lane) for _ in range(2)]; [x.start() for x in th]; [x.join() for x in th]
CATS = ["agent-tools", "code", "tutor-math", "long-context", "arabic-urdu", "general", "medicine"]
print("snapshot 00-start:", snap("00-start"), flush=True)
for i, c in enumerate(CATS, 1):
    t0 = time.time(); run_cat(c); ok = snap(f"{i:02d}-{c}")
    print(f"{i:02d} {c}: {time.time() - t0:.0f} s, snapshot {'saved' if ok else 'NOT SAVED'}", flush=True)
print("DONE", flush=True)
