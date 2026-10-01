#!/bin/bash
# rc3c-spec.sh K: one test window for rc3c (rc3b plus the 4-token 3D attention path) with K draft tokens (1 to 4).
# Stops the brain service, runs rc3c with the service's exact settings except num_speculative_tokens=K, measures the
# exact decode rate and tokens per step at 16K and 60K, compares text with rc3b's reference, then restores the
# brain service, also on Ctrl-C. K = 1 is the regression check against rc3b.
set -u
export PATH=$HOME/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
K=${1:-3}; case $K in 1|2|3|4) ;; *) echo "ABORT: K must be 1, 2, 3 or 4"; exit 1;; esac
IMG=localhost/n02-brain:rc3c; NAME=n02-brain-rc3c-k$K; PORT=8000
CACHE=$HOME/brain/cache/rc2; TCACHE=$HOME/brain/cache/rc2-triton
M=/models/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/6e134bae811fb5adac50ee042ae5f029ac6779aa
BRAIN_TP=8; RESERVE=""; [ -f $HOME/brain/pkg/cards.conf ] && . $HOME/brain/pkg/cards.conf
podman image exists $IMG || { echo "ABORT: $IMG is missing"; exit 1; }
IDX=$(n02-fleet resolve 2>/dev/null | awk -v res=" $RESERVE " '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9a-f]+$/ && index(res, " " $2 " ") == 0 {print $1}' | head -$BRAIN_TP | paste -sd,)
[ "$(echo "$IDX" | tr ',' '\n' | grep -c .)" = "$BRAIN_TP" ] || { echo "ABORT: could not resolve $BRAIN_TP brain cards (got: $IDX)"; exit 1; }
restore() { echo "[$(date -u +%H:%M:%S)] stopping the test and restarting the brain service"; podman stop -t 60 $NAME >/dev/null 2>&1; systemctl --user start --no-block n02-brain.service; }
trap 'echo "[$(date -u +%H:%M:%S)] interrupted"; restore; exit 130' INT TERM
echo "[$(date -u +%H:%M:%S)] stopping the brain service; testing rc3c with $K draft tokens on cards $IDX"
systemctl --user stop n02-brain.service; sleep 15
podman run -d --rm --replace --name $NAME --device /dev/kfd --device /dev/dri --group-add keep-groups --ipc=host --network=host \
  --security-opt seccomp=unconfined --pids-limit=-1 --ulimit memlock=-1:-1 \
  -v "$HOME/models:/models" -e HF_HOME=/models -v "$CACHE:/root/.cache/vllm-rc2" -e VLLM_CACHE_ROOT=/root/.cache/vllm-rc2 \
  -v "$TCACHE:/root/.triton" -e FENSTONE_FP32_PREFILL=installed -e NCCL_ALGO=Ring -e OMP_NUM_THREADS=8 -e CUDA_VISIBLE_DEVICES=$IDX \
  -e FENSTONE_RDNA_AR=1 \
  --entrypoint vllm $IMG serve $M --served-model-name brain --tensor-parallel-size $BRAIN_TP --max-model-len 131072 \
  --gpu-memory-utilization 0.95 --attention-backend TRITON_ATTN --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":'$K',"attention_backend":"TRITON_ATTN"}' --port $PORT >/dev/null \
  || { echo "podman run failed"; restore; exit 1; }
up=0
for i in $(seq 240); do
  curl -s -m 2 localhost:$PORT/v1/models >/dev/null && { up=1; echo "[$(date -u +%H:%M:%S)] rc3c k=$K up after $((i * 10)) s"; break; }
  [ "$(podman inspect --format '{{.State.Running}}' $NAME 2>/dev/null)" = "true" ] || { echo "k=$K EXITED while loading:"; podman logs --tail 25 $NAME 2>&1 | cut -c1-200; restore; exit 1; }
  [ $((i % 6)) = 0 ] && echo "[$(date -u +%H:%M:%S)] loading... $((i * 10)) s"
  sleep 10
done
[ $up = 1 ] || { echo "k=$K not up after 40 minutes"; podman logs --tail 25 $NAME 2>&1 | cut -c1-200; restore; exit 1; }
echo "=== gate 1: the RDNA all-reduce took over (want an 'Enabled graph-only RDNA HIP all-reduce: gfx1100 TP8' line, no warnings)"
podman logs $NAME 2>&1 | grep -E "RDNA HIP all-reduce|RDNA all-reduce" | sort | uniq -c | head -5 | cut -c1-200
f=$(ls -t "$CACHE"/torch_compile_cache/*/rank_*_0/backbone/computation_graph.py 2>/dev/null | head -1)
echo "=== gate 2: same compiled graph (want 510, 0, 0; the op name is unchanged, only its kernel): fp32 op $(grep -o _fenstone_C.gptq_gemm_rdna3_fp32 "$f" | wc -l), installed $(grep -o '_rocm_C.gptq_gemm_rdna3\b' "$f" | wc -l), triton $(grep -c triton_w4a16_gemm_kernel "$f")"
echo "=== exact decode rate, rc3c with $K draft tokens (rc3b k=1: 16K 113, 60K 110 at 1.78 to 1.83 per step; rc3b k=3 before the fix: 46.4 and 15.9 at 2.67)"
( cd $HOME/handtest && python3 true_rate.py $PORT 2>&1 | grep -E "run 2" )
echo "=== text against rc3b's reference (speculation should not change answers beyond numerics; the brain against itself: 5 of 12, largest 0.20)"
( cd $HOME/handtest && python3 ref_outputs.py $PORT rc3c-k$K-A 2>&1 | tail -1 && python3 compare_refs.py rc3b-A rc3c-k$K-A 2>&1 | tail -1 )
restore
