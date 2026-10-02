# Models that run here

Measured on 2026-10-02, on this machine's current stack: the public AMD vLLM image plus the patches in
[`patches/`](patches/) and [`patches/rc3/`](patches/rc3/) (image `localhost/n02-brain:rc3c`). The stack was tuned on
one model, the agent's Qwen3.8-27B. This page is the test of whether that work carries to other models.

Every run below was made with [`tools/model-test.sh`](tools/model-test.sh), and its raw record is in
[`bench/models/`](bench/models/): the exact command and settings, the startup timeline, the kernels the compiled graph
uses, every probe, every speed line, and one real answer. [`bench/models/summary.tsv`](bench/models/summary.tsv) lists
every run, including the ones that failed.

## How to read the numbers

- **One request at a time.** These are single-user decode speeds, exact, timed from the first streamed token to the last.
  Speed with several users at once is not measured yet.
- **16K and 60K** are 64 forced tokens after a prompt of that length, made of one repeated sentence.
- **Realistic** is 400 tokens of ordinary text, nothing forced. With draft tokens off, it agrees with the benchmark.
  With draft tokens on, the repeated-sentence benchmark flatters speculation, so the realistic figure is the honest one.
- **Cards** are RX 7900 XTX, connected by PCIe Gen4 x8 through one switch. **Our all-reduce** is the eight-card RDNA3
  all-reduce: vLLM pull request #57767's RDNA3 kernel for 2 and 4 cards, extended to 8 cards in `patches/rc3/`, with the hook that
  makes vLLM use it. **Our INT4 kernel** is the fixed-order kernel; it only applies to dense 4-bit layers.

## Results with a raw record

| Model | Type | Format | Cards | 16K tok/s | 60K tok/s | Realistic tok/s | Our all-reduce | Our INT4 kernel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B (the agent's model, 3 draft tokens) | dense, linear-attention hybrid | INT4 (compressed-tensors) | 8 | 130 to 165 | 144 to 162 | **134 on real agent work at 104K** | yes | yes |
| Qwen3.6-35B-A3B | MoE hybrid, 3B active | AWQ | 2 | 99.7 | 95.3 | 104.4 | yes | not used (its dense layers are bf16) |
| Qwen3.6-35B-A3B | MoE hybrid, 3B active | AWQ | 4 | 116.2 | 111.1 | 123.9 | yes | not used |
| Qwen3.6-35B-A3B | MoE hybrid, 3B active | bf16 | 4 | 117.7 | 113.5 | 124.9 | yes | not used |
| Qwen3-Coder-Next | MoE hybrid, 3B active | GPTQ | 4 | 100.3 | 95.0 | 104.5 | yes | not used |
| Qwen2.5-72B-Instruct | dense | AWQ | 8 | 70.8 | (32K model) | 76.4 | yes | **yes** |
| Qwen2.5-72B-Instruct | dense | bf16 | 8 | 38.5 | (32K model) | 40.7 | yes | not used |
| MiniMax-M2.7 (229B) | MoE | AWQ | 8 | 53.9 | 49.2 | 56.4 | yes | **yes** |

The agent's model is measured by the production service, not by `model-test.sh`; its records are in
[`bench/rc3/`](bench/rc3/).

## Results without a raw record yet

These ran earlier the same day, before `model-test.sh` kept a record of each run. The numbers are from the terminal.
They will be re-run and recorded.

| Model | Type | Format | Cards | 16K tok/s | 60K tok/s | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B, 3 draft tokens | MoE hybrid | AWQ | 2 | 197.8 | 198.4 | 3.76 tokens per step on the repeated-sentence benchmark, a ceiling rather than a working speed |
| GLM-4.5-Air | MoE, full attention | INT4 (compressed-tensors) | 4 | 56.0 | 46.3 | our INT4 kernel on its dense layers |
| Qwen2.5-72B-Instruct | dense | AWQ | 4 | 48.9 | (32K model) | our INT4 kernel on every dense layer |
| GLM-4.7-REAP-218B-A32B | MoE, 32B active | INT4 (compressed-tensors) | 8 | 27.2 | 24.8 | our INT4 kernel on its dense layers |

## What the runs show

- **The stack carries.** Every model in the first table loaded, from 2 cards to 8, and the RDNA3 all-reduce engaged in every
  run, at 2, 4 and 8 cards. The stack was tuned on one model.
- **4-bit is a speed choice on dense models and only a capacity choice on small-active MoE.** On the same 8 cards the dense
  72B ran 1.84 times faster at 4-bit than at bf16. On the same 4 cards the 3B-active MoE ran within 2 percent either way;
  4-bit only let it fit on half the cards.
- **More cards help dense models more.** The dense 72B gained 45 percent from 4 cards to 8; the 3B-active MoE gained
  17 percent from 2 cards to 4.
- **Long context costs depend on the model's design.** The linear-attention hybrids barely slow down from 16K to 60K.
  GLM-4.5-Air, with full attention in every layer, slowed by 17 percent.
- **Draft tokens need a good drafter.** N-gram speculation made the dense 72B slower (76.4 to 48.2 realistic): it guessed
  almost nothing right, and checking four tokens at once costs that model about 11 ms extra per step. The agent's model,
  with its own draft head, gains 46 percent from three draft tokens.

## What does not run yet

| Model | What happens | Status |
| --- | --- | --- |
| GLM-4.7-Flash (MLA attention) | loads with vLLM's own attention choice and answers short prompts; the first long prompt kills the engine with `AttributeError: 'ColumnParallelLinear' object has no attribute 'weight'` | open: the stock image will show whether the bug is vLLM's or in our patched layer |
| NVIDIA Nemotron-3-Super | not attempted on this stack | a known vLLM bug in the RDNA3 MoE path for models with a ReLU-squared activation |

## Caveats

- **Speed is not intelligence.** Only the agent's model has been scored for quality on this stack (8 of 10 on Task 01,
  [`tasks/`](tasks/)). Nothing here ranks the other models' answers.
- **One request at a time.** A model's speed with several users at once can rank differently.
- **One machine.** Ten RX 7900 XTX on one EPYC board and one PCIe switch. Other machines will differ.

## Reproduce

```
model-test.sh REPO CARDS [K]
```

`REPO` is a model already downloaded to `~/models`, `CARDS` is `itko` (two cards set aside for the agent) or `brain:N` (2,
4 or 8 of the serving cards), and `K` is the number of draft tokens for models with a draft head. Settings: `ATTN=auto`
for MLA models, `MAXLEN=`, `IMG=` to compare another image, and `ENVS=` and `ARGS=` for extra environment and vLLM
arguments. The script refuses before changing anything if a setting is wrong, always restores the serving model when it
used its cards, and keeps a folder for every run. Its paths and card IDs are this machine's; set them to yours.
