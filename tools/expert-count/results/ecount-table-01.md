# GLM-5.3-Flash expert counts by category (count only, nothing pruned)

Snapshots: 00-start, 01-agent-tools, 02-code, 03-tutor-math, 04-long-context, 05-arabic-urdu, 06-general, 07-medicine. Stages (TP rank 0 of each): [0, 2, 4, 6, 8]. MoE layers: 3 to 44 (42). 288 routed experts per layer, 8 picked per token.

## Tokens: what the counter saw against what was sent

Counter tokens are tokens the model actually computed; they fall short of vLLM's prompt totals when a prompt prefix was cached and is not recomputed, and run over when padding is counted. Outside traffic means vLLM served more than this driver sent in that window, so that category is mixed with someone else's work.

| Category | Requests ok / errors | Driver prompt + completion tokens | vLLM prompt + generation tokens | Counter tokens | Picks per token | Outside traffic |
| --- | --- | --- | --- | --- | --- | --- |
| 01-agent-tools | 36 / 0 | 18,977 + 4,675 | 18,992 + 4,679 | 23,877 | 8.000 | no |
| 02-code | 22 / 0 | 8,723 + 10,984 | 8,738 + 10,988 | 19,794 | 8.000 | no |
| 03-tutor-math | 60 / 0 | 3,524 + 14,305 | 3,539 + 14,309 | 18,030 | 8.000 | no |
| 04-long-context | 6 / 0 | 394,007 + 32 | 394,022 + 36 | 394,078 | 8.000 | no |
| 05-arabic-urdu | 60 / 0 | 7,956 + 18,646 | 7,971 + 18,650 | 26,784 | 8.000 | no |
| 06-general | 40 / 0 | 854 + 15,745 | 869 + 15,749 | 16,740 | 8.000 | no |
| 07-medicine | 32 / 0 | 664 + 11,441 | 679 + 11,445 | 12,222 | 8.000 | no |

## Spread of experts per category (averaged over the 42 layers)

Effective experts: exp(entropy) of the pick distribution, 288 if every expert were used equally. k50 / k90: fewest experts that cover 50 % / 90 % of picks. Unused: experts never picked in that category (in a layer, averaged). Top expert: the busiest expert's share, as a multiple of an even share (1/288). Picks per expert if even: how often each expert would be picked if routing were even; unused counts mean little when this is small.

| Category | Counter tokens | Picks per expert if even | Effective experts | k50 | k90 | Unused experts | Top expert x even share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01-agent-tools | 23,877 | 663 | 181.0 | 51.8 | 174.6 | 2.4 | 10.3 |
| 02-code | 19,794 | 550 | 224.6 | 76.0 | 205.5 | 0.2 | 7.3 |
| 03-tutor-math | 18,030 | 501 | 171.8 | 48.7 | 168.2 | 1.1 | 12.3 |
| 04-long-context | 394,078 | 10,947 | 99.4 | 23.7 | 95.6 | 1.9 | 14.7 |
| 05-arabic-urdu | 26,784 | 744 | 116.4 | 27.7 | 143.9 | 0.6 | 24.9 |
| 06-general | 16,740 | 465 | 187.2 | 57.8 | 169.0 | 0.9 | 7.8 |
| 07-medicine | 12,222 | 340 | 150.1 | 42.1 | 153.9 | 2.2 | 16.4 |
| all categories | 511,525 | 14,209 | 148.8 | 34.0 | 167.5 | 0.0 | 11.5 |

## Per layer, all categories together

| Layer | Effective experts | k50 | k90 | Unused in every category | Busiest expert (id, x even share) |
| --- | --- | --- | --- | --- | --- |
| 3 | 208.9 | 59 | 197 | 0 | 92, 4.7 |
| 4 | 210.9 | 60 | 197 | 0 | 277, 5.4 |
| 5 | 217.3 | 61 | 205 | 0 | 73, 4.8 |
| 6 | 217.4 | 63 | 199 | 0 | 84, 4.4 |
| 7 | 193.3 | 53 | 186 | 0 | 29, 7.1 |
| 8 | 188.5 | 51 | 181 | 0 | 226, 6.2 |
| 9 | 182.1 | 46 | 181 | 0 | 104, 6.3 |
| 10 | 199.1 | 56 | 189 | 0 | 269, 7.4 |
| 11 | 192.1 | 52 | 182 | 0 | 32, 6.5 |
| 12 | 174.4 | 43 | 172 | 0 | 235, 7.0 |
| 13 | 157.8 | 39 | 174 | 0 | 190, 13.7 |
| 14 | 177.7 | 45 | 181 | 0 | 77, 8.4 |
| 15 | 172.4 | 42 | 176 | 0 | 167, 8.6 |
| 16 | 179.1 | 46 | 175 | 0 | 238, 7.4 |
| 17 | 178.8 | 44 | 178 | 0 | 121, 7.4 |
| 18 | 134.4 | 28 | 154 | 0 | 237, 10.9 |
| 19 | 132.4 | 28 | 151 | 0 | 45, 12.1 |
| 20 | 124.0 | 23 | 156 | 0 | 182, 14.2 |
| 21 | 136.4 | 27 | 163 | 0 | 102, 9.8 |
| 22 | 131.0 | 26 | 160 | 0 | 110, 14.9 |
| 23 | 140.9 | 30 | 164 | 0 | 280, 11.9 |
| 24 | 147.4 | 31 | 168 | 0 | 130, 11.5 |
| 25 | 114.5 | 21 | 149 | 0 | 156, 13.9 |
| 26 | 115.6 | 20 | 157 | 0 | 85, 13.9 |
| 27 | 106.6 | 18 | 152 | 0 | 233, 15.1 |
| 28 | 113.9 | 20 | 157 | 0 | 134, 15.4 |
| 29 | 123.5 | 23 | 160 | 0 | 97, 11.3 |
| 30 | 107.8 | 18 | 154 | 0 | 278, 14.6 |
| 31 | 115.5 | 21 | 157 | 0 | 84, 15.6 |
| 32 | 117.4 | 21 | 157 | 0 | 222, 18.7 |
| 33 | 92.6 | 14 | 143 | 0 | 5, 19.7 |
| 34 | 99.5 | 16 | 156 | 0 | 176, 18.2 |
| 35 | 96.8 | 15 | 155 | 0 | 35, 18.8 |
| 36 | 118.3 | 22 | 156 | 0 | 173, 14.3 |
| 37 | 114.7 | 21 | 152 | 0 | 129, 17.0 |
| 38 | 110.2 | 20 | 151 | 0 | 208, 18.1 |
| 39 | 117.3 | 21 | 153 | 0 | 109, 14.2 |
| 40 | 138.3 | 27 | 166 | 0 | 128, 13.7 |
| 41 | 147.5 | 33 | 165 | 0 | 46, 11.6 |
| 42 | 166.0 | 40 | 168 | 0 | 121, 9.2 |
| 43 | 171.8 | 43 | 172 | 0 | 53, 9.7 |
| 44 | 164.6 | 39 | 165 | 0 | 190, 8.3 |

## How alike the categories are (cosine similarity of pick distributions, averaged over layers; 1.00 = identical)

| Category | agent-tools | code | tutor-math | long-context | arabic-urdu | general | medicine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agent-tools | 1.00 | 0.70 | 0.52 | 0.26 | 0.32 | 0.52 | 0.43 |
| code | 0.70 | 1.00 | 0.54 | 0.32 | 0.36 | 0.59 | 0.49 |
| tutor-math | 0.52 | 0.54 | 1.00 | 0.24 | 0.34 | 0.58 | 0.46 |
| long-context | 0.26 | 0.32 | 0.24 | 1.00 | 0.16 | 0.28 | 0.25 |
| arabic-urdu | 0.32 | 0.36 | 0.34 | 0.16 | 1.00 | 0.34 | 0.27 |
| general | 0.52 | 0.59 | 0.58 | 0.28 | 0.34 | 1.00 | 0.68 |
| medicine | 0.43 | 0.49 | 0.46 | 0.25 | 0.27 | 0.68 | 1.00 |

## Experts that lean towards one category

An expert leans towards a category when its share of that category's picks is at least 4x its average share in the other categories, and it was picked at least 200 times there. Counted per layer, summed over layers. This is a description of routing, not a claim that the expert holds that subject.

| Category | Leaning experts (all layers) | Their share of that category's picks |
| --- | --- | --- |
| 01-agent-tools | 719 | 20.1 % per layer on average |
| 02-code | 645 | 13.6 % per layer on average |
| 03-tutor-math | 537 | 18.4 % per layer on average |
| 04-long-context | 1352 | 57.0 % per layer on average |
| 05-arabic-urdu | 552 | 38.1 % per layer on average |
| 06-general | 439 | 12.3 % per layer on average |
| 07-medicine | 344 | 22.0 % per layer on average |

## Not measured here

- Routing weights: the weight totals went NaN during GLM's start-up on 4 of 5 stages in this run (fix: claude-ecount-03 drops non-finite values), so this table uses pick counts only.
- Whether an unused or rarely used expert matters: counts say how often an expert is picked, not what removing it would cost. That needs a quality test with the expert masked, which is pruning work and is not part of this run.
- The shared expert (always on) is not routed, so it is not counted.
