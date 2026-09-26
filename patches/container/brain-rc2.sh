#!/bin/bash
# brain-rc2.sh: start the rc2 brain from the captured image. Refuses rather than guesses.
#   ./brain-rc2.sh            port 8000
#   PORT=8010 ./brain-rc2.sh   another port (for a side-by-side test)
# The server is the container's main process: if it dies, the container stops (see: podman logs n02-brain).
# Both compile caches live on the host (~/brain/cache/rc2 and ~/brain/cache/rc2-triton), so restarts are warm.
set +H
IMG=localhost/n02-brain:rc2
NAME=${NAME:-n02-brain}
PORT=${PORT:-8000}
CACHE=$HOME/brain/cache/rc2
TCACHE=$HOME/brain/cache/rc2-triton
M=/models/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/6e134bae811fb5adac50ee042ae5f029ac6779aa
U='90072484815efb41|cbfe8a265e29739c|78cd25d463191e98|1937d5cffa12fb08|4a763ce278eafb6a|0741fd7a35d7ea9f|1b9fb8181ce9991e|a25cdb16a49e2dcf'
SP=/opt/python/lib/python3.14/site-packages/vllm
EXPECT="6eea73a19eafef97f5aab74da9ab2621 42840ce685a5cbea7046abae6021f344 d543b90d6ad7266db6744778a6edcba8"
die() { echo "ABORT: $*"; exit 1; }
podman image exists $IMG || die "image $IMG is missing; build it with Containerfile.rc2"
H=$(podman run --rm --network=none --entrypoint bash $IMG -c "md5sum < $SP/v1/attention/backends/triton_attn.py | cut -c1-32; md5sum < $SP/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py | cut -c1-32; md5sum < /opt/fenstone/fen_fp32_op.so | cut -c1-32" | paste -sd' ')
[ "$H" = "$EXPECT" ] || die "the image's files are not rc2's (found: $H)"
curl -s -m 2 localhost:$PORT/v1/models >/dev/null && die "something already answers on port $PORT"
used=$(n02-fleet resolve | awk -v u="$U" '$2 ~ u {s+=$3} END {print s+0}')
awk -v x="$used" 'BEGIN{exit !(x < 1)}' || die "the brain's cards hold $used GB; stop whatever is on them first"
IDX=$(n02-fleet resolve | awk -v u="$U" '$2 ~ u {print $1}' | paste -sd,)
[ "$(echo "$IDX" | tr ',' '\n' | grep -c .)" = 8 ] || die "did not find all 8 brain cards (found: $IDX)"
mkdir -p "$CACHE" "$TCACHE"
podman rm -f $NAME >/dev/null 2>&1
echo "[$(date -u +%H:%M:%S)] starting rc2 as $NAME on port $PORT, cards $IDX, caches $CACHE and $TCACHE"
podman run -d --name $NAME --device /dev/kfd --device /dev/dri --group-add keep-groups --ipc=host --network=host \
  --security-opt seccomp=unconfined --pids-limit=-1 --ulimit memlock=-1:-1 \
  -v "$HOME/models:/models" -e HF_HOME=/models -v "$CACHE:/root/.cache/vllm-rc2" -e VLLM_CACHE_ROOT=/root/.cache/vllm-rc2 \
  -v "$TCACHE:/root/.triton" -e FENSTONE_FP32_PREFILL=installed -e NCCL_ALGO=Ring -e OMP_NUM_THREADS=8 -e CUDA_VISIBLE_DEVICES=$IDX \
  --entrypoint vllm $IMG serve $M --served-model-name brain --tensor-parallel-size 8 --max-model-len 131072 \
  --gpu-memory-utilization 0.95 --attention-backend TRITON_ATTN --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"attention_backend":"TRITON_ATTN"}' --port $PORT >/dev/null \
  || die "podman run failed"
up=0
for i in $(seq 240); do
  curl -s -m 2 localhost:$PORT/v1/models >/dev/null && { up=1; echo "[$(date -u +%H:%M:%S)] up after $((i * 10)) s"; break; }
  [ "$(podman inspect --format '{{.State.Running}}' $NAME 2>/dev/null)" = "true" ] || { echo "SERVER EXITED during load:"; podman logs --tail 25 $NAME 2>&1; exit 1; }
  [ $((i % 6)) = 0 ] && echo "[$(date -u +%H:%M:%S)] loading... $((i * 10)) s"
  sleep 10
done
[ $up = 1 ] || { echo "NOT UP after 40 minutes"; podman logs --tail 25 $NAME 2>&1; exit 1; }
f=$(ls -t "$CACHE"/torch_compile_cache/*/rank_7_0/backbone/computation_graph.py 2>/dev/null | head -1)
echo "graph: fp32 op $(grep -o _fenstone_C.gptq_gemm_rdna3_fp32 "$f" | wc -l), installed $(grep -o '_rocm_C.gptq_gemm_rdna3\b' "$f" | wc -l), triton $(grep -c triton_w4a16_gemm_kernel "$f")  (want 510, 0, 0)"
echo "settings in the server: $(podman exec $NAME bash -c 'tr "\0" "\n" < /proc/1/environ | grep -E "^(FENSTONE_FP32_PREFILL|NCCL_ALGO|VLLM_CACHE_ROOT)=" | paste -sd" "')"
w=$(curl -s -m 120 localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d '{"model":"brain","prompt":"Warm-up.","max_tokens":8,"temperature":0}' | grep -o '"completion_tokens":[0-9]*')
echo "warm-up request: ${w:-FAILED}"
echo "rc2 brain ready on port $PORT"
