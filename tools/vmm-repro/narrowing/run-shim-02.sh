#!/bin/bash
# run-shim-02: build hipmem-shim-01 in the serving image, then vmm-repro-03 (1892 MiB x 40) on card 90072484815efb41 under each mode,
# plus vmm-repro.py in oom mode under recycle. One card, nothing else touched.
cd ~/glm-serve/vmm-repro-tool/probe; I=localhost/n02-sandbox:hermes-10
podman run --rm --network=none --entrypoint bash -v $PWD:/w $I -c "cd /w && gcc -O2 -shared -fPIC -Wl,--version-script=hipmem-shim-01.map -o hipmem-shim-01.so hipmem-shim-01.c -ldl -lpthread" || { echo BUILD FAILED; exit 1; }
nm -D --defined-only hipmem-shim-01.so | grep hipMem
run() { T=$1; M=$2; shift 2; timeout 150 podman run --rm --entrypoint python3 --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined \
  --network=none -e ROCR_VISIBLE_DEVICES=GPU-90072484815efb41 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e LD_PRELOAD=/w/hipmem-shim-01.so -e SHIM_MODE=$M \
  -v ~/glm-serve/vmm-repro-tool/probe:/w:ro -v ~/glm-serve/vmm-repro-tool:/t:ro $I "$@" > shim-$T.txt 2>&1; echo "$T exit $?"; }
run log log /w/vmm-repro-03.py 1892 40
run recycle-a recycle /w/vmm-repro-03.py 1892 40
run recycle-b recycle /w/vmm-repro-03.py 1892 40
run recycle-oom recycle /t/vmm-repro.py 150 40 oom
run norelease norelease /w/vmm-repro-03.py 1892 40
for T in log recycle-a recycle-b recycle-oom norelease; do f=shim-$T.txt; echo "== $T hits $(grep -c ^HIT $f)"; grep -E "^SHIM|RESULT" $f | cut -c1-200; grep -E "^round (1|2|3|39):" $f | cut -c1-100; done
