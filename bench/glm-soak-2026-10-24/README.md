# GLM-5.3-Flash soak, 2026-10-24

The run behind [Chapter 8](../../paper/08-serving-it.md#the-soak): 2 hours 18 minutes, two lanes, all ten cards, started 08:42 UTC.

| File | What it is |
| --- | --- |
| `soak-03.py` | The soak driver: tutor maths generated and solved blind, tool calls streamed and not, needle prompts from 2,000 to 116,000 tokens, script reviews; a heat and power guard every 2 s (stop at 100 C or 3,500 W) |
| `summary.md` | Its own summary at the end |
| `results.jsonl` | One line per request |
| `thermal.jsonl` | Hottest card and summed power, every 10 s |
| `warmup-26.txt`, `long-01.txt` | The serving start's six warm-up prompts, and a 116,399-token prompt right after (41.2 s, correct) |
| `memsnap-23-rank4-log.txt` | PyTorch's memory on one middle-stage card through a start-up with the vision encoder still profiled: 19.15 GB reserved after weights, then 0 bytes free and two out-of-memory retries at the peak |

The reviews GLM wrote are not included: they quote scripts that carry this machine's paths. Their verdicts are summarised in the chapter.
