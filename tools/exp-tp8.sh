#!/usr/bin/env bash
# Experiment: the brain at TP8 with the RDNA3 W4A16 patch, in a SEPARATE container (n02-exp)
# made from a snapshot of n02-vllm. Production's container is never modified.
# Same model, same flags as ~/launch/brain-tp8.sh, except: container n02-exp, port 8010,
# log /tmp/exp-tp8.log, and the patched rdna3_w4a16.py mounted read-only over the original.
set -euo pipefail
SNAP=localhost/n02-vllm:snap-20260925
PATCH=$HOME/handtest/src/rdna3_w4a16_patched.py
DEST=/opt/python/lib/python3.14/site-packages/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py
WANT=444c2f24fae5d99de091f4caa5b39f52
M=/models/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/6e134bae811fb5adac50ee042ae5f029ac6779aa
UUIDS='90072484815efb41|cbfe8a265e29739c|78cd25d463191e98|1937d5cffa12fb08|4a763ce278eafb6a|0741fd7a35d7ea9f|1b9fb8181ce9991e|a25cdb16a49e2dcf'

# ---- pre-checks: abort before touching anything
[ "$(md5sum < "$PATCH" | cut -d' ' -f1)" = "$WANT" ] || { echo "ABORT: $PATCH hash is not $WANT"; exit 1; }
podman image exists "$SNAP" || { echo "ABORT: snapshot image $SNAP does not exist"; exit 1; }
if podman container exists n02-exp; then echo "ABORT: a container named n02-exp already exists; inspect or remove it by hand"; exit 1; fi
if curl -s -m 2 localhost:8000/v1/models >/dev/null; then echo "ABORT: the production brain is still serving on 8000; stop it first"; exit 1; fi
RES=$(n02-fleet resolve)
IDX=$(echo "$RES" | awk -v u="$UUIDS" '$2 ~ u {print $1}' | paste -sd,)
n=$(echo "$IDX" | tr ',' '\n' | grep -c . || true)
[ "$n" = 8 ] || { echo "ABORT: resolved $n brain cards, expected 8"; echo "$RES"; exit 1; }
if ! echo "$RES" | awk -v u="$UUIDS" '$2 ~ u && $3+0 > 1.0 {bad=1} END {exit bad}'; then
  echo "ABORT: a brain card still holds memory; the production server has not released it"; echo "$RES"; exit 1
fi
echo "pre-checks passed: patch $WANT, snapshot present, 8000 down, 8 empty cards ($IDX)"

# ---- the experiment container
podman run -d --name n02-exp \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ipc=host --network=host --security-opt seccomp=unconfined \
  --pids-limit=-1 --ulimit memlock=-1:-1 \
  -v "$HOME/models:/models" -e HF_HOME=/models \
  -v "$PATCH:$DEST:ro" \
  "$SNAP" tail -f /dev/null
got=$(podman exec n02-exp md5sum "$DEST" | cut -d' ' -f1)
[ "$got" = "$WANT" ] || { echo "ABORT: inside n02-exp the kernel file hash is $got, not the patch"; exit 1; }
echo "n02-exp running, patched kernel file mounted ($got)"

podman exec -d n02-exp bash -c "VLLM_CACHE_ROOT=/root/.cache/vllm-exp CUDA_VISIBLE_DEVICES=$IDX NCCL_ALGO=Tree OMP_NUM_THREADS=8 vllm serve $M --served-model-name brain --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.95 --attention-backend TRITON_ATTN --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --enable-prefix-caching --port 8010 > /tmp/exp-tp8.log 2>&1"

# ---- wait up to 40 minutes, but stop at once if the server process dies (e.g. the patch's guard)
for i in $(seq 240); do
  if curl -s -m 2 localhost:8010/v1/models >/dev/null; then echo "up after $((i * 10)) s"; break; fi
  # pure /proc scan: the image may not ship pgrep, and a missing tool must not look like a dead server.
  # [v]llm stops the scan from matching its own command line, which also contains the pattern.
  if ! podman exec n02-exp bash -c 'for p in /proc/[0-9]*; do tr "\0" " " < $p/cmdline 2>/dev/null; echo; done | grep -q "[v]llm serve"'; then
    echo "SERVER PROCESS EXITED during load. Last lines of the log:"; podman exec n02-exp tail -30 /tmp/exp-tp8.log; exit 1
  fi
  [ $((i % 6)) = 0 ] && echo "[$(date -u +%H:%M:%S)] loading... $((i * 10)) s"
  sleep 10
done
curl -s -m 2 localhost:8010/v1/models >/dev/null || { echo "NOT UP after 40 minutes. Last lines:"; podman exec n02-exp tail -30 /tmp/exp-tp8.log; exit 1; }
podman exec n02-exp grep -i -E "fenstone|Traceback|Error|selected via|GPU KV cache size" /tmp/exp-tp8.log | cut -c1-160 | head -12 || true
