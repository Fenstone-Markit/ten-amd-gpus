# GLM soak 2026-10-24 08:42 UTC, 138 minutes

Ended: time up. Hottest card 99 C (G7); highest ten-card power 2891 W.

| Kind | Requests | Correct or complete | Not | Errors | Median latency (s) | Largest prompt (tokens) |
| --- | --- | --- | --- | --- | --- | --- |
| needle-long | 24 | 16 | 8 | 0 | 79.8 | 116400 |
| needle-mid | 23 | 18 | 5 | 0 | 29.4 | 39283 |
| needle-short | 92 | 90 | 2 | 0 | 3.8 | 7382 |
| review | 23 | 23 | 0 | 0 | 179.4 | 8203 |
| tool-call | 56 | 56 | 0 | 0 | 1.5 | 185 |
| tool-call-stream | 46 | 46 | 0 | 0 | 1.5 | 185 |
| tool-result | 56 | 56 | 0 | 0 | 1.4 | 238 |
| tool-result-stream | 46 | 46 | 0 | 0 | 1.4 | 238 |
| tutor-generate | 162 | 161 | 1 | 0 | 18.3 | 151 |
| tutor-solve | 537 | 530 | 7 | 0 | 2.3 | 121 |

Long prompts read: 74104 tokens in 107 s, 100524 tokens in 36 s, 74103 tokens in 79 s, 100526 tokens in 90 s, 116399 tokens in 141 s, 100525 tokens in 44 s, 47971 tokens in 64 s, 100525 tokens in 152 s, 74104 tokens in 105 s, 47971 tokens in 64 s, 116400 tokens in 49 s, 47971 tokens in 64 s

Tutor bank: 566 problems passed the arithmetic check, 530 confirmed by a blind re-solve, 7 disagreements, 240 rejected.
Reviews written: 23 (unverified; read before acting on any).
