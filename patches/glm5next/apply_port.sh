#!/bin/bash
# Build the patched vLLM tree for GLM-5.3-Flash on RDNA3 (gfx1100) that Chapter 5 measured.
# Order: the installed vLLM package, then djdeniro's GLM-5.3-Flash-rocm-r9700 overlay (MIT) over it,
# then this folder's vllm/ files over that. Run inside AMD's image
# docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0, then serve with
# PYTHONPATH=/tmp/ov so Python imports the patched copy instead of the installed one.
#   OVERLAY=/path/to/overlay/app/vllm/vllm PATCHES=/path/to/patches/glm5next/vllm ./apply_port.sh
set -euo pipefail
INSTALLED="${INSTALLED:-/opt/python/lib/python3.14/site-packages/vllm}"
OVERLAY="${OVERLAY:?set OVERLAY to the overlay's app/vllm/vllm folder}"
PATCHES="${PATCHES:?set PATCHES to this folder's vllm/ subfolder}"
BASE=/tmp/ov/vllm
mkdir -p /tmp/ov
rm -rf "$BASE"
cp -a "$INSTALLED/." "$BASE/"
cp -a "$OVERLAY/." "$BASE/"
(cd "$PATCHES" && find . -name '*.py' -exec cp --parents {} "$BASE/" \;)
# Purge bytecode only where files were replaced; a full purge costs minutes of recompiling.
{ (cd "$OVERLAY" && find . -name '*.py' -exec dirname {} \;); (cd "$PATCHES" && find . -name '*.py' -exec dirname {} \;); } \
  | sort -u | while IFS= read -r d; do rm -rf "$BASE/$d/__pycache__"; done
echo "built $BASE: $(find "$OVERLAY" -name '*.py' | wc -l) overlay files, $(find "$PATCHES" -name '*.py' | wc -l) patched files"
