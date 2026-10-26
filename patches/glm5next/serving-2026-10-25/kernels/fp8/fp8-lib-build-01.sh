#!/bin/bash
# fp8-lib-build-01: on node02, build libfp8rdna3.so from fp8-rdna3-lib-01.hip with the GLM image's hipcc (ROCm 10), CPU only.
cd ~/fp8-rdna3-01 || exit 1
podman run --rm --network=none --entrypoint bash -v $PWD:/w localhost/n02-sandbox:hermes-10 -c \
  'cd /w && H=$(command -v hipcc || echo /opt/rocm/bin/hipcc); $H -O3 --offload-arch=gfx1100 -std=c++17 -shared -fPIC -Rpass-analysis=kernel-resource-usage -o libfp8rdna3.so fp8-rdna3-lib-01.hip 2> build-lib-01.txt; echo "hipcc exit $?"; grep -E "error" build-lib-01.txt | head; grep -E "Function Name|VGPRs:|ScratchSize" build-lib-01.txt | sed "s/.*remark: //; s/ \[-Rpass.*//" | paste - - - | head -12; nm -D --defined-only libfp8rdna3.so | grep fp8r3'
