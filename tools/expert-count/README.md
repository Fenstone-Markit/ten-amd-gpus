# expert-count: which of GLM-5.3-Flash's experts each kind of work picks

The count behind [Chapter 8](../../paper/08-serving-it.md#which-experts-do-the-work). Read-only: the model's routing and output are untouched, and nothing is pruned.

## Files

| File | What it is |
| --- | --- |
| `glm5_ecount.py` | The counter: GLM's routing rule (sigmoid scores, bias-corrected top-8, n_group 1), counters made when each layer is built so compiled graphs see fixed tensors, and in-place adds inside the forward |
| `model-ecount-02.diff` | The three hooks in the GLM model file of my agent's port: import, counter set-up in the MoE layer, one call after the router. Off unless `GLM5_EXPERT_COUNT=1` |
| `gpu_worker-ecount-02.diff` | The worker side: at each step start, tensor-parallel rank 0 of each pipeline stage looks for `trigger-<tag>` files in `GLM5_ECOUNT_DIR` and saves its counters as `<tag>-rank<R>.pt` |
| `route-check-01.py` | Checks the counter's rule against vLLM's `topk_sigmoid` kernel: same eight experts for 20,000 of 20,000 random inputs, on vLLM 0.23 (not re-run on the 0.27 build the count used) |
| `ecount-calib-02.py` | The driver: seven kinds of work, two requests at a time, a snapshot after each, and vLLM's own token totals logged at each snapshot so outside traffic shows |
| `ecount-table-01.py` | Turns the snapshots into the table, CPU only |
| `results/` | The table (`ecount-table-01.md`), per-layer figures, the experts that lean towards one kind, the raw per-kind counts (`ecount-deltas-01.pt`) and the driver's log |

## Not included

The driver reads three inputs that are not here: the tutor question bank (made by GLM during the soak in `bench/glm-soak-2026-10-24`), 60 Quran verses with Urdu and English translations, and six of my scripts for the code reviews. Use your own of each kind.

## Limits

A few dozen prompts per kind, English except Arabic and Urdu, the long-context kind synthetic. Routing weights were meant to be summed too, but their totals went NaN during start-up on four of the five stages (fixed in the next version by skipping values that are not finite), so the table uses pick counts only. The shared expert is not routed and not counted.
