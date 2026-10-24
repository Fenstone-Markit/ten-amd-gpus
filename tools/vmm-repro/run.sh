#!/bin/bash
# run.sh: run vmm-repro.py on one card in a podman container.
# Usage: CARD=<card unique_id> [IMG=<image>] bash run.sh TAG H_MiB ROUNDS MODE [extra podman -e flags]
# MODE: ec (empty_cache after odd rounds), none (never release), oom (no empty_cache, allocate until PyTorch runs out).
set -u
CARD=${CARD:?set CARD to the unique_id of one card (cat /sys/bus/pci/devices/*/unique_id)}
IMG=${IMG:-docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0}
TAG=$1; H=$2; R=$3; M=$4; shift 4
D=$(cd "$(dirname "$0")" && pwd)
podman run --rm --name vmm-repro-$TAG --entrypoint python3 --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --network=none -e ROCR_VISIBLE_DEVICES=GPU-$CARD \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" -v "$D":/w:ro "$IMG" /w/vmm-repro.py "$H" "$R" "$M"
