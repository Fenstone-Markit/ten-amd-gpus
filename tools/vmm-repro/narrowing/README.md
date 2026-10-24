# Narrowing it down

The tests behind [Chapter 6's "Since then"](../../../paper/06-the-memory-underneath.md#since-then-2026-10-24) and the shim in [Chapter 8](../../../paper/08-serving-it.md). One RX 7900 XTX (gfx1100), ROCm 10.0.0, HIP 7.15.26333, PyTorch 2.12.0, 2026-10-23 and 24. Every result file is the run's own output, unedited.

The runner scripts here are exactly as run, so they carry this machine's card id and its local image name (`localhost/n02-sandbox:hermes-10`, built on AMD's `rocm/vllm` 0.27.0 image with the same ROCm, HIP and PyTorch). To run one elsewhere, change the card id and set the image to `docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`. The libraries build with `gcc -O2 -shared -fPIC -Wl,--version-script=<name>.map -o <name>.so <name>.c -ldl -lpthread` inside that image.

## The tests

All use `vmm-repro-03.py` unless named: four marked 512 KiB blocks, the card filled to about 2 GiB free, then 40 rounds of 1,892 MiB of bf16 buffers, freed each round, with `empty_cache()` after odd rounds. A hit is a whole block changed.

| Question | Script | Result file | Result |
| --- | --- | --- | --- |
| The control, fixed pressure per round | `run-03.sh` | `r03-ctl.txt` | 21 hits |
| ROCm 7.2.4, same test | `run-03.sh` | `r03-724.txt` | 0 hits, but out of memory in 38 of 40 rounds: released memory never comes back |
| Does PyTorch ever unmap or release the overwritten page? (`hipmem-watch-01`) | like the shim tests, with the watcher preloaded | `w01.txt`, `w01b.txt` | 0 unmaps over it, 0 releases of its handle, 0 other handles mapped onto it; 125 and 87 hits (each of the three blocks on that page 16 times in `w01b`) |
| Is it the memory or only PyTorch's view of it? (`hipmem-watch-02` maps the page again at B; `vmm-repro-03c.py` reads both) | like the shim tests | `w02-1.txt`, `w02-2.txt` | Every hit on blocks 0 to 2 shows at B too (15 and 30), identical bytes |
| Can a plain `hipMalloc` land on a live tensor? (`vmm-repro-05.py`, 1,300 or 600 MiB after each release, filled with `0xAB`) | like the shim tests, no library | `r05-1300.txt`, `r05-600.txt` | 25 and 19 hits right after the `hipMalloc`, reading `abababab`; with 798 MiB of the card free in the 600 MiB case |
| Keep released handles in the process and reuse them (`hipmem-shim-01`, `recycle`) | `run-shim-02.sh` | `shim-recycle-a.txt`, `shim-recycle-b.txt`, `shim-recycle-oom.txt` | 0 hits in all three; in the first two, no out-of-memory, 2,247 of 3,494 creates served from the pool and 0 releases reach the runtime (the third allocates until out of memory on purpose) |
| The same library passing everything through (`log`) | `run-shim-02.sh` | `shim-log.txt` | 21 hits, the same as without it |
| Drop releases instead (`norelease`, a leak) | `run-shim-02.sh` | `shim-norelease.txt` | 0 hits, out of memory in 38 rounds, like ROCm 7.2.4 |
| The deployable version (`hipmem-shim-02`: pool keyed by size and allocation properties, only PyTorch's own calls) on one card and on two cards in one process | `run-shim-03.sh` | `s2-*.txt` | One card: 20 hits passing through, 0 recycling. Two cards (`vmm-repro-04-2card.py`): 0 either way, and no errors with the pool keyed by card |
| The same calls in plain HIP, without PyTorch (`hipvmm-repro-01` to `-03`: one range or a small-pool range, 2 or 8 small pages, creates driven to failure) | run directly in the image | `h01-*.txt`, `h02-*.txt`, `h03-*.txt` | 0 hits |
| An exact replay of PyTorch's own 11,314 calls (`hipmem-trace-01` recorded `trace-03b.txt`; `hipvmm-replay-01.py` replays it at the same addresses) | run directly in the image | `rp01-30.txt` | 0 hits in 4,791 checks |

## What it shows

The overwrite needs a release to reach the ROCm runtime. What gets overwritten is memory behind a handle that is still live and still mapped, and PyTorch never touches that page. Plain HIP making the same calls does not trigger it, so something else PyTorch does is part of the trigger. Whether the runtime or the kernel driver hands out the live memory is not shown.
