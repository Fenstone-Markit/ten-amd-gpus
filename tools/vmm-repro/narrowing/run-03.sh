#!/bin/bash
# run-03: vmm-repro-03 (fixed 1892 MiB per round, 40 rounds) on card 90072484815efb41, control stack then ROCm 7.2.4.
cd ~/glm-serve/vmm-repro-tool/probe
for P in "ctl localhost/n02-sandbox:hermes-10" "724 docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"; do
  set -- $P
  timeout 150 podman run --rm --entrypoint python3 --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined \
    --network=none -e ROCR_VISIBLE_DEVICES=GPU-90072484815efb41 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v ~/glm-serve/vmm-repro-tool/probe:/w:ro $2 /w/vmm-repro-03.py 1892 40 > r03-$1.txt 2>&1
  echo "$1 exit $?"
done
for f in r03-ctl.txt r03-724.txt; do echo "== $f hits $(grep -c ^HIT $f)"; grep -E "^round (0|1|2|3|8|9|39):|RESULT|weights" $f | cut -c1-120; grep -m2 -iE "warn|error" $f | cut -c1-160; done
