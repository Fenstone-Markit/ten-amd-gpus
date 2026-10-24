# vmm-repro: live memory overwritten on a nearly full card

The one-card test behind [Chapter 6](../../paper/06-the-memory-underneath.md). Plain PyTorch, no model, about 30 seconds per run.

## What it does

1. Allocates four 512 KiB int32 blocks with a known pattern (PyTorch's small pool), and keeps a host copy.
2. Fills the card with "weights" (256 MiB float32 tensors) until about H + 2,048 MiB is free.
3. Each round: allocates bf16 buffers of 578, 64, 20, 6, 1 and 0.25 MiB in turn, filled with random values, until less than H MiB is free; reads the four blocks back and compares; frees the round's buffers.
4. Never writes to the blocks after creating them. A hit means something else's data is in their memory.

MODE decides how memory goes back:

| MODE | Release |
| --- | --- |
| `ec` | `torch.cuda.empty_cache()` after every odd round |
| `none` | never released; later rounds reuse PyTorch's cached memory |
| `oom` | no `empty_cache()`, but each round allocates until PyTorch raises out-of-memory, which first makes PyTorch release cached pages itself |

## Run

```bash
CARD=<unique_id of one idle card> bash run.sh check 150 40 ec
```

`CARD` is read from `/sys/bus/pci/devices/*/unique_id`; the container sees only that card (`ROCR_VISIBLE_DEVICES=GPU-<id>`). `IMG` defaults to AMD's public `rocm/vllm` 0.27.0 image (`docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`), with ROCm 10.0.0, HIP 7.15.26333 and PyTorch 2.12.0+rocm10.0.0. The runs in the table before 2026-10-24 used a local image built on it, with the same ROCm, HIP and PyTorch. Extra `-e NAME=value` flags after the four arguments go to the container, for example `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` or a ROCm runtime switch. A healthy stack prints `0 block hits`.

## Results on an RX 7900 XTX (gfx1100), kernel 7.0.0-38, 2026-10-23

| Run | Allocator / mode | Free kept | Rounds | Block hits | First hit |
| --- | --- | --- | --- | --- | --- |
| out-loose | expandable, `ec` | 4,096 MiB | 20 | 0 | |
| out-tight | expandable, `ec` | 150 MiB | 20 | 4 | round 8 |
| out-base40 | expandable, `ec` | 150 MiB | 40 | 12 | round 8 |
| out2-ec | expandable, `ec` | 150 MiB | 40 | 48 | round 8 |
| out2-none | expandable, `none` | 150 MiB | 40 | 0 | |
| out2-oom | expandable, `oom` | 150 MiB | 40 | 4 | round 0 |
| out-esoff | classic allocator, `ec` | 150 MiB | 40 | 0 (20 rounds hit out-of-memory first) | |
| out-sdma0 | expandable, `ec`, `HSA_ENABLE_SDMA=0` | 150 MiB | 40 | 12 | round 8 |
| out-nofrag | expandable, `ec`, `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` | 150 MiB | 40 | 12 | round 8 |
| out3-public-ec (2026-10-24, this `run.sh` with its default public image, after a reboot) | expandable, `ec` | 150 MiB | 40 | 3 | round 38 |

The `out-` runs used the first version of the script, whose only mode was `ec`; the `out2-` runs used this one. This exact `run.sh` and `vmm-repro.py` gave 12 hits in 20 rounds, first at round 8, on the same card before publishing. A hit is all 131,072 words of a block changed, to zeros or to bf16 values from the round's buffers.

## Narrowing it down

[narrowing/](narrowing/) holds the tests that followed: which call has to reach the runtime for the overwrite to happen, a second mapping of the overwritten page, a plain `hipMalloc` landing on a live tensor, a library that keeps released memory in the process (loaded as a backstop when GLM serves, [Chapter 8](../../paper/08-serving-it.md)), and plain-HIP versions that do not trigger it.

## Related

- [ROCm/legacy-rocm-build#6692](https://github.com/ROCm/legacy-rocm-build/issues/6692): the same 2 MiB overwrites with ROCm's virtual-memory calls on gfx1201.
- [ROCm/TheRock#9274](https://github.com/ROCm/TheRock/issues/9274): live buffers reading back as zeros under expandable segments, ROCm 7.14 but not 7.2.
