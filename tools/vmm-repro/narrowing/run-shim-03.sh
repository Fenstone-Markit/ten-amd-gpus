#!/bin/bash
# run-shim-03: build hipmem-shim-02 in the serving image; one card: log and recycle (vmm-repro-03); two cards in one process: log and recycle (vmm-repro-04-2card).
cd ~/glm-serve/vmm-repro-tool/probe; I=localhost/n02-sandbox:hermes-10
podman run --rm --network=none --entrypoint bash -v $PWD:/w $I -c "cd /w && gcc -O2 -Wall -shared -fPIC -Wl,--version-script=hipmem-shim-02.map -o hipmem-shim-02.so hipmem-shim-02.c -ldl -lpthread" || { echo BUILD FAILED; exit 1; }
nm -D --defined-only hipmem-shim-02.so | grep hipMem; md5sum hipmem-shim-02.so | cut -c1-8
run() { T=$1; M=$2; V=$3; shift 3; timeout 170 podman run --rm --entrypoint python3 --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined \
  --network=none -e ROCR_VISIBLE_DEVICES=$V -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e LD_PRELOAD=/w/hipmem-shim-02.so -e SHIM_MODE=$M \
  -v ~/glm-serve/vmm-repro-tool/probe:/w:ro $I "$@" > s2-$T.txt 2>&1; echo "$T exit $?"; }
C1=GPU-90072484815efb41; C2=GPU-90072484815efb41,GPU-78cd25d463191e98
run 1card-log log $C1 /w/vmm-repro-03.py 1892 40
run 1card-recycle recycle $C1 /w/vmm-repro-03.py 1892 40
run 2card-log log $C2 /w/vmm-repro-04-2card.py 1892 40
run 2card-recycle recycle $C2 /w/vmm-repro-04-2card.py 1892 40
for T in 1card-log 1card-recycle 2card-log 2card-recycle; do f=s2-$T.txt; echo "== $T hits $(grep -c ^HIT $f)"; grep -E "RESULT|pid .* (start|exit)" $f | cut -c1-330; grep -E "^round (2|3|39)( card [01])?:" $f | cut -c1-80; grep -m2 -iE "error|Traceback" $f | cut -c1-160; done
