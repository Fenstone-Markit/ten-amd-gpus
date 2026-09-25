# Tools

The scripts behind Chapters 1.2 and 2.1. Every figure in those chapters came from one of these, and
the raw output is in `bench/node02-2026-09-25/`.

They were written for one machine (Node02) and assume a working folder at `~/handtest`. Card UUIDs,
the model path and the container names are set at the top of each script.

## Measuring the server

| Script | What it does | Chapter |
| --- | --- | --- |
| `brain_bench.py PORT LABEL` | Decode speed at 2K, 16K and 60K context: 64 tokens forced with `min_tokens` and `ignore_eos`, timed from the first streamed token to the last, two runs each. Refuses to run if the server is busy | 1.2 |
| `ref_outputs.py PORT LABEL` | Records greedy output for a fixed set of 12 prompts, with every token's log-probability and the top five alternatives | 2.1 |
| `compare_refs.py A B` | Compares two recorded runs: where the text first diverges, and how far the probabilities moved before that | 2.1 |
| `score_texts.py PORT LABEL` and `--compare A B` | Scores the same fixed text on a server (echo with log-probabilities) and compares two servers. A broken model shows up as a perplexity orders of magnitude higher | 2.1 |

## The native kernel patch

| File | What it does | Chapter |
| --- | --- | --- |
| `rdna3_w4a16_uint4.patch` | The change to vLLM's `rdna3_w4a16.py`: accept uint4 checkpoints, transpose their zero points into the kernel's layout, and call the kernel without GPTQ's +1 offset | 1.2 |
| `make_patch.py` | Builds the patched file from the installed original, refusing unless the original's hash matches, and prints the diff | 1.2 |
| `exp-tp8.sh` | Starts the patched server in a separate container made from a snapshot of production, with its own compile cache, and refuses to start unless every pre-check passes | 1.2 |

After any change to which kernel vLLM selects, the server needs an empty compile cache
(`VLLM_CACHE_ROOT`). Otherwise it reuses the graph compiled for the old kernel and serves garbage
while reporting healthy (Chapter 2.1).

## Kernel measurements, `kernel/`

Each test runs on one card in a throwaway container, times kernels inside CUDA graphs with enough
distinct weight copies to defeat the 96 MB Infinity Cache, and gates every candidate against a
reference, with a deliberately broken control that has to fail.

| Script | What it measures |
| --- | --- |
| `test1.py` | Instrument checks (a known-answer copy, a read-only comparator, cache on and off) and the native kernel's size sweep |
| `test2.py` | Tile size 256 against 128, built from source with its own namespace |
| `test3.py` | Tiles 256, 128 and 64 at batch sizes 1, 2, 4 and 8 |
| `test4.py` | The cost of the output-zeroing fill kernel |
| `test5.py` | Production's Triton kernel against the native one, at production's merged shapes |
| `gate_fp32.py` | The frozen gate for the fp32-accumulation kernel: every kernel scored against an exact float64 reference |

## Attention gate, `gate/`

The batched correctness check for the Triton attention change in Chapter 2.1. It builds batches
and includes a deliberately broken control, because the earlier single-request check passed a kernel
that wrote out of bounds.
