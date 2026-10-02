#!/bin/bash
# model-test.sh 2.0: serve any downloaded model on rc3c's stack, measure it like the brain, and record everything.
#
#   model-test.sh REPO CARDS [K]
#     REPO   a model in ~/models, e.g. cyankiwi/GLM-4.5-Air-AWQ-4bit
#     CARDS  itko     G7 and G9 (2 cards); the brain keeps running
#            brain:N  the brain's first N cards (2, 4 or 8); the brain stops and is always restored, also on Ctrl-C
#     K      optional MTP draft tokens, 1 to 3; default 0
#   Environment: ATTN=<backend> (default TRITON_ATTN; ATTN=auto lets vLLM choose, needed for MLA models)
#                MAXLEN=<tokens> (default: the smaller of 81920 and the model's own maximum)
#                IMG=<image> (default localhost/n02-brain:rc3c; e.g. the stock AMD image, to compare stock with ours)
#                ENVS="NAME=value ..." extra container environment; ARGS="--flag value ..." extra vllm serve arguments
#
# Every run gets ~/handtest/model-runs/<model>-<UTC time>/ with run.txt, server.log (complete, captured live),
# timeline.txt, errors.txt, kernels.txt, probes.txt, measure.txt, sanity.txt; and one line in
# ~/handtest/model-runs/summary.tsv. Serves on port 8010 as "brain"; own compile cache per model.
set -u
export PATH=$HOME/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
REPO=${1:-}; CARDS=${2:-}; K=${3:-0}; ATTN=${ATTN:-TRITON_ATTN}; ENVS=${ENVS:-}; ARGS=${ARGS:-}
IMG=${IMG:-localhost/n02-brain:rc3c}; PORT=8010; G7=d580007fa41ec467; G9=fec7cca764f40561
[ -n "$REPO" ] && [ -n "$CARDS" ] || { echo "usage: model-test.sh REPO itko|brain:N [K]   (env ATTN=auto for MLA models)"; exit 1; }
case $K in 0|1|2|3) ;; *) echo "ABORT: K must be 0 to 3"; exit 1;; esac
EE=(); for e in $ENVS; do [[ "$e" =~ ^[A-Za-z_][A-Za-z0-9_]*=.+ ]] || { echo "ABORT: ENVS entry '$e' is not NAME=value"; exit 1; }; EE+=(-e "$e"); done
read -r -a XA <<< "$ARGS"
MD=$HOME/models/hub/models--${REPO/\//--}
SNAP=$(ls -d $MD/snapshots/*/ 2>/dev/null | head -1)
[ -n "$SNAP" ] || { echo "ABORT: $REPO is not in ~/models"; exit 1; }
[ -n "$(find $MD -name '*.incomplete' 2>/dev/null | head -1)" ] && { echo "ABORT: $REPO has unfinished download files"; exit 1; }
podman image exists $IMG || { echo "ABORT: $IMG is missing"; exit 1; }
curl -s -m 2 localhost:$PORT/v1/models >/dev/null && { echo "ABORT: something already answers on port $PORT"; exit 1; }
M=/models/hub/${SNAP#$HOME/models/hub/}
SHORT=$(echo ${REPO#*/} | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '-' | cut -c1-40); NAME=n02-test-$SHORT
MPE=$(python3 -c "import json; c=json.load(open('${SNAP}config.json')); t=c.get('text_config') or {}; print(t.get('max_position_embeddings') or c.get('max_position_embeddings') or 0)" 2>/dev/null)
if [ -z "${MAXLEN:-}" ]; then MAXLEN=81920; [ -n "$MPE" ] && [ "$MPE" -gt 0 ] && [ "$MPE" -lt $MAXLEN ] && MAXLEN=$MPE; fi
ITAG=""; [ "$IMG" != localhost/n02-brain:rc3c ] && ITAG=-$(echo ${IMG##*:} | tr -c 'a-z0-9.\n' '-' | cut -c1-24)
CACHE=$HOME/brain/cache/test-$SHORT$ITAG; TCACHE=$CACHE-triton; mkdir -p $CACHE $TCACHE
STAMP=$(date -u +%Y%m%d-%H%M%S); RUNS=$HOME/handtest/model-runs; R=$RUNS/$SHORT-$STAMP; mkdir -p $R
FLEET=$(n02-fleet resolve 2>/dev/null)
idx_of() { echo "$FLEET" | awk -v u=$1 '$2 == u {print $1}'; }
used_of() { echo "$FLEET" | awk -v u=$1 '$2 == u {print $3 + 0}'; }
if [ "$CARDS" = itko ]; then
  TP=2; IDX="$(idx_of $G7),$(idx_of $G9)"
  [ "$(used_of $G7)" = 0 ] && [ "$(used_of $G9)" = 0 ] || { echo "ABORT: G7 or G9 is in use (is Itko working?)"; rmdir $R 2>/dev/null; exit 1; }
elif [[ "$CARDS" =~ ^brain:(2|4|8)$ ]]; then
  TP=${CARDS#brain:}; RESERVE="$G7 $G9"; [ -f $HOME/brain/pkg/cards.conf ] && . $HOME/brain/pkg/cards.conf
  IDX=$(echo "$FLEET" | awk -v res=" $RESERVE " '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9a-f]+$/ && index(res, " " $2 " ") == 0 {print $1}' | head -$TP | paste -sd,)
else
  echo "ABORT: CARDS must be itko or brain:2, brain:4 or brain:8"; rmdir $R 2>/dev/null; exit 1
fi
[ "$(echo "$IDX" | tr ',' '\n' | grep -c '^[0-9]\+$')" = "$TP" ] || { echo "ABORT: could not resolve $TP cards (got: $IDX)"; rmdir $R 2>/dev/null; exit 1; }

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $R/run.txt; }
STATUS=started; LOAD_S=""; STOPPED_BRAIN=0; LOGPID=""
summary() {
  [ -f $RUNS/summary.tsv ] && ! head -1 $RUNS/summary.tsv | grep -q "extra$" && sed -i '1s/$/\textra/' $RUNS/summary.tsv
  [ -f $RUNS/summary.tsv ] || printf "time_utc\timage\tmodel\tcards\ttp\tk\tattn\twindow\tstatus\tload_s\tallreduce\tfp32_op\tinstalled\ttriton\tdecode16k\tdecode60k\ttokens_per_step16k\trealistic\tfirst_error\trun_dir\textra\n" > $RUNS/summary.tsv
  local e=""
  case $STATUS in load_failed|load_timeout|podman_run_failed) e=$(head -1 $R/errors.txt 2>/dev/null);; interrupted) e="interrupted by the operator";; esac
  [ -z "$e" ] && [ ! -s $R/.d16 ] && [ "$STATUS" = measured ] && e=$(cat $R/probes.txt $R/measure.txt 2>/dev/null | grep -m1 -E "error|REFUSED|FAILED|no measurable")
  e=$(echo "$e" | head -1 | tr '\t\r' '  ' | cut -c1-150)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$STAMP" "$IMG" "$REPO" "$CARDS" "$TP" "$K" "$ATTN" "$MAXLEN" "$STATUS" "$LOAD_S" \
    "$(cat $R/.ar 2>/dev/null)" "$(cat $R/.fp32 2>/dev/null)" "$(cat $R/.inst 2>/dev/null)" "$(cat $R/.trit 2>/dev/null)" "$(cat $R/.d16 2>/dev/null)" "$(cat $R/.d60 2>/dev/null)" \
    "$(cat $R/.c16 2>/dev/null)" "$(cat $R/.real 2>/dev/null)" "$e" "$R" "${ENVS}${ENVS:+ }${ARGS}" >> $RUNS/summary.tsv
}
capture_logs() { podman logs $NAME > $R/server.log 2>&1; }
extract_errors() {
  # exception messages first ("SomethingError: text", NCCL warnings, out of memory); generic engine-death lines are not a cause
  grep -E "[A-Za-z_.]+(Error|Exception): |NCCL WARN|out of memory|OutOfMemory" $R/server.log 2>/dev/null \
    | grep -v -E "QuarkOCP|resource_tracker|warnings\.warn" | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //' | awk '!seen[$0]++' | head -40 > $R/errors.txt
  local rc; rc=$(grep -v -E "Engine core initialization failed|EngineDeadError|EngineCore encountered an issue|Error in completion stream" $R/errors.txt | head -1)
  case $STATUS in
    load_failed|load_timeout|podman_run_failed) [ -n "$rc" ] && { echo "ROOT CAUSE (likely): $rc"; cat $R/errors.txt; } > $R/errors.tmp && mv $R/errors.tmp $R/errors.txt;;
    *) [ -s $R/errors.txt ] && { echo "(error lines from the server log; the server did not fail to load)"; cat $R/errors.txt; } > $R/errors.tmp && mv $R/errors.tmp $R/errors.txt;;
  esac
}
timeline() {
  grep -E "Loading weights took|Model loading took|Directly load|torch.compile takes|Compiling a graph|profiling/warmup run took|Available KV cache memory|GPU KV cache size|Graph capturing finished|init engine|startup complete|Enabled graph-only RDNA|[Aa]utotun" $R/server.log 2>/dev/null \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //' | awk '!seen[$0]++' > $R/timeline.txt
}
restore() {
  [ -n "$LOGPID" ] && kill $LOGPID 2>/dev/null
  capture_logs; timeline; extract_errors
  say "stopping and removing the test container"; podman stop -t 60 $NAME >/dev/null 2>&1; podman rm -f $NAME >/dev/null 2>&1
  if [ $STOPPED_BRAIN = 1 ]; then say "restarting the brain service"; systemctl --user start --no-block n02-brain.service; fi
  summary
  say "run folder: $R"
}
trap 'say "interrupted"; STATUS=interrupted; restore; exit 130' INT TERM

{ echo "model-test.sh 2.0"; echo "command: model-test.sh $*"; echo "ATTN=$ATTN MAXLEN=$MAXLEN (model maximum ${MPE:-unknown}) K=$K"; echo "ENVS=${ENVS:-none}"; echo "ARGS=${ARGS:-none}"
  echo "repo $REPO, snapshot $M"; echo "image $IMG $(podman image inspect --format '{{.Id}}' $IMG 2>/dev/null | cut -c1-12)"
  echo "cards $CARDS: TP$TP, torch indices $IDX"; echo "compile cache $CACHE"; echo; } > $R/run.txt
if [ "$CARDS" != itko ]; then say "stopping the brain service for this test"; systemctl --user stop n02-brain.service; STOPPED_BRAIN=1; sleep 15; fi
AB=(--attention-backend $ATTN); SA=",\"attention_backend\":\"$ATTN\""; [ "$ATTN" = auto ] && AB=() && SA=""
SPEC=(); [ "$K" != 0 ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K$SA}")
say "$REPO on $CARDS (TP$TP, indices $IDX), drafts $K, attention $ATTN, window $MAXLEN, image $IMG"; [ -n "$ENVS$ARGS" ] && say "extra: ${ENVS:+env $ENVS }${ARGS:+args $ARGS}"
podman run -d --replace --name $NAME --device /dev/kfd --device /dev/dri --group-add keep-groups --ipc=host --network=host \
  --security-opt seccomp=unconfined --pids-limit=-1 --ulimit memlock=-1:-1 \
  -v "$HOME/models:/models" -e HF_HOME=/models -v "$CACHE:/root/.cache/vllm-test" -e VLLM_CACHE_ROOT=/root/.cache/vllm-test \
  -v "$TCACHE:/root/.triton" -e FENSTONE_FP32_PREFILL=installed -e NCCL_ALGO=Ring -e OMP_NUM_THREADS=8 -e CUDA_VISIBLE_DEVICES=$IDX \
  -e FENSTONE_RDNA_AR=1 "${EE[@]}" \
  --entrypoint vllm $IMG serve $M --served-model-name brain --tensor-parallel-size $TP --max-model-len $MAXLEN \
  --gpu-memory-utilization 0.92 "${AB[@]}" --enable-prefix-caching --trust-remote-code "${SPEC[@]}" "${XA[@]}" --port $PORT > $R/.cid 2>&1 \
  || { say "podman run failed: $(tail -2 $R/.cid)"; STATUS=podman_run_failed; restore; exit 1; }
podman logs -f $NAME > $R/server.live.log 2>&1 & LOGPID=$!
T0=$(date +%s); up=0
for i in $(seq 240); do
  curl -s -m 2 localhost:$PORT/v1/models >/dev/null && { up=1; LOAD_S=$(( $(date +%s) - T0 )); say "up after $LOAD_S s"; break; }
  if [ "$(podman inspect --format '{{.State.Running}}' $NAME 2>/dev/null)" != "true" ]; then
    STATUS=load_failed; say "RESULT: the server exited while loading"; capture_logs; extract_errors
    echo "--- errors (full list in $R/errors.txt):"; head -12 $R/errors.txt | cut -c1-220; restore; exit 1
  fi
  [ $((i % 6)) = 0 ] && say "loading... $((i * 10)) s"
  sleep 10
done
[ $up = 1 ] || { STATUS=load_timeout; say "RESULT: not up after 40 minutes"; restore; exit 1; }
STATUS=loaded; capture_logs

# ---- kernels and the all-reduce
AR=$(grep -c "Enabled graph-only RDNA HIP all-reduce" $R/server.log); echo "$AR" > $R/.ar
GD=$(ls -td $CACHE/torch_compile_cache/*/ 2>/dev/null | head -1)
f=$([ -n "$GD" ] && ls -S $GD/rank_*_0/*/computation_graph.py 2>/dev/null | head -1)
{ echo "RDNA all-reduce enabled on $AR rank(s) (want $TP)"; grep -m1 -E "RDNA all-reduce.*(disabled|not)" $R/server.log | sed -E 's/^\([^)]*\) //'
  if [ -n "$f" ]; then
    echo "compiled graph (the largest in the newest cache folder, i.e. the main model, not a draft head): $f"
    echo "fp32 op (ours) $(grep -o _fenstone_C.gptq_gemm_rdna3_fp32 "$f" | wc -l), installed $(grep -o '_rocm_C.gptq_gemm_rdna3\b' "$f" | wc -l), triton $(grep -c triton_w4a16_gemm_kernel "$f")"
    echo "top operations:"; grep -o -E "torch\.ops\.[a-z_0-9]+\.[a-z_0-9]+" "$f" | sort | uniq -c | sort -rn | head -10
    grep -o _fenstone_C.gptq_gemm_rdna3_fp32 "$f" | wc -l > $R/.fp32; grep -o '_rocm_C.gptq_gemm_rdna3\b' "$f" | wc -l > $R/.inst; grep -c triton_w4a16_gemm_kernel "$f" > $R/.trit
  else echo "no compiled graph file found in $CACHE"; fi; } > $R/kernels.txt
echo "=== kernels"; head -3 $R/kernels.txt | cut -c1-200

# ---- probes, measurement and sanity: one Python program, written into the run folder
cat > $R/measure.py <<'PY'
import json, sys, time, urllib.request
port, maxlen, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
base = f"http://localhost:{port}"
log = open(out + "/measure.txt", "w"); probes = open(out + "/probes.txt", "w"); results = {}
def post(path, body, stream, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)
def note(f, msg): print(msg, file=f, flush=True)
# probe 1: short, non-streamed, the measurement's settings
try:
    d = json.load(post("/v1/completions", {"model": "brain", "prompt": "The quick brown fox jumps over the lazy dog. The quick brown fox",
        "max_tokens": 8, "min_tokens": 8, "temperature": 0, "ignore_eos": True, "logprobs": 1}, False))
    note(probes, "probe 1 (short, not streamed): " + ("ok " + repr(d["choices"][0]["text"][:50]) if "choices" in d else "REFUSED " + json.dumps(d)[:300]))
except Exception as e:
    note(probes, f"probe 1 FAILED: {type(e).__name__}: {str(e)[:300]}")
# exact decode, true_rate's method: repeated text, 64 forced tokens, streamed, tokens counted from logprobs
def exact(ctx, run):
    p = "The quick brown fox jumps over the lazy dog. " * (ctx // 10)
    body = {"model": "brain", "prompt": p, "max_tokens": 64, "min_tokens": 64, "temperature": 0, "ignore_eos": True, "stream": True, "logprobs": 1}
    t0 = time.time(); t_first = t_last = None; first = total = chunks = 0; other = []
    try:
        for line in post("/v1/completions", body, True):
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if not d.get("choices"): other.append(json.dumps(d)[:300]); continue
            n = len((d["choices"][0].get("logprobs") or {}).get("tokens") or [])
            if n == 0: continue
            now = time.time()
            if t_first is None: t_first, first = now, n
            t_last = now; total += n; chunks += 1
    except Exception as e:
        other.append(f"REQUEST FAILED: {type(e).__name__}: {str(e)[:250]}")
    if other: note(probes, f"ctx {ctx} run {run}: {len(other)} non-token message(s); first: {other[0]}")
    if chunks == 0 or total - first <= 0:
        note(log, f"ctx~{ctx:>6} run {run}: no measurable tokens" + (f" ({other[0][:160]})" if other else "")); return None
    ms = (t_last - t_first) * 1000 / (total - first)
    note(log, f"ctx~{ctx:>6} run {run}: {total} tokens in {chunks} chunks ({total / chunks:.2f} per chunk), first token {t_first - t0:5.2f}s, decode {ms:5.1f} ms/token = {1000 / ms:5.1f} tok/s")
    return round(1000 / ms, 1), round(total / chunks, 2)
for ctx in ([16000, 60000] if maxlen >= 61000 else [16000]):
    r = None
    for run in (1, 2): r = exact(ctx, run)
    results[ctx] = r
if maxlen < 61000: note(log, f"(60K skipped: the window is {maxlen} tokens)")
# realistic generation: ordinary text, nothing forced, timed first token to last
def realistic():
    body = {"model": "brain", "messages": [{"role": "user", "content": "Explain step by step how a CPU fetches, decodes and executes one instruction, with a short worked example."}],
            "max_tokens": 400, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t_first = t_last = None; usage = None; other = []
    try:
        for line in post("/v1/chat/completions", body, True):
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if not ch:
                if "error" in d: other.append(json.dumps(d)[:300])
                continue
            delta = ch[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                now = time.time(); t_first = t_first or now; t_last = now
    except Exception as e:
        other.append(f"REQUEST FAILED: {type(e).__name__}: {str(e)[:250]}")
    if other: note(probes, "realistic: " + other[0])
    if not usage or not t_first or t_last == t_first:
        note(log, "realistic generation: not measurable" + (f" ({other[0][:160]})" if other else "")); return None
    n = usage.get("completion_tokens", 0)
    rate = (n - 1) / (t_last - t_first)
    note(log, f"realistic generation: {n} tokens, decode {rate:5.1f} tok/s (ordinary text, no forced length)")
    return round(rate, 1)
results["realistic"] = realistic()
# sanity: one real answer, reasoning kept apart
try:
    d = json.load(post("/v1/chat/completions", {"model": "brain", "messages": [{"role": "user", "content": "In one short sentence: what is the capital of France, and what river runs through it?"}],
        "max_tokens": 600, "temperature": 0}, False, 180))
    m = d["choices"][0]["message"]; s = open(out + "/sanity.txt", "w")
    print("content:", (m.get("content") or "").strip()[:600], file=s)
    print("reasoning (separate field):", (m.get("reasoning_content") or m.get("reasoning") or "")[:400], file=s)
except Exception as e:
    open(out + "/sanity.txt", "w").write(f"sanity FAILED: {type(e).__name__}: {str(e)[:300]}\n")
json.dump({str(k): v for k, v in results.items()}, open(out + "/measure.json", "w"))
PY
echo "=== probes, measurement, sanity (a few minutes)"
python3 $R/measure.py $PORT $MAXLEN $R > $R/measure.stdout 2>&1 || echo "(measure.py stopped early: $(tail -1 $R/measure.stdout | cut -c1-160))"
cat $R/probes.txt 2>/dev/null | cut -c1-220; cat $R/measure.txt 2>/dev/null; head -2 $R/sanity.txt 2>/dev/null | cut -c1-220
python3 - $R <<'PY' 2>/dev/null
import json, sys, os
r = sys.argv[1]; d = json.load(open(r + "/measure.json"))
w = lambda n, v: open(os.path.join(r, n), "w").write("" if v is None else str(v))
a = d.get("16000"); b = d.get("60000")
w(".d16", a[0] if a else None); w(".c16", a[1] if a else None); w(".d60", b[0] if b else None); w(".real", d.get("realistic"))
PY
STATUS=measured
restore
