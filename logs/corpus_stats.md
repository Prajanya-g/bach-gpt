# Corpus statistics

| corpus | files | round-trip pass | tokens | mean len | median len | min | max | vocab used / total |
|---|---|---|---|---|---|---|---|---|
| JSB | 30 | 30/30 (100.0%) | 31,119 | 1037 | 1006 | 519 | 2222 | 75/167 (45%) |

## Token budget (projected from sample means)

Per-corpus projection = (mean tokens / sample file) × (full-corpus file count). GigaMIDI training split = the all-instruments-with-drums training-V1.1-80% split used as pretraining; JSB Chorales is held out for zero-shot probing.

| corpus | full-corpus files | sample size | mean tokens / file | projected total tokens | role |
|---|---|---|---|---|---|
| JSB | 371 | 30 | 1,037 | 384,838 | probing (held out) |

## Top-20 tokens by corpus

### JSB

| rank | token | count | share |
|---|---|---|---|
| 1 | `V5` | 8,965 | 28.81% |
| 2 | `D18` | 4,944 | 15.89% |
| 3 | `D13` | 2,778 | 8.93% |
| 4 | `TS13` | 2,029 | 6.52% |
| 5 | `D22` | 755 | 2.43% |
| 6 | `TS18` | 728 | 2.34% |
| 7 | `P62` | 593 | 1.91% |
| 8 | `P69` | 584 | 1.88% |
| 9 | `BAR_START` | 581 | 1.87% |
| 10 | `BAR_END` | 581 | 1.87% |
| 11 | `P64` | 566 | 1.82% |
| 12 | `P57` | 558 | 1.79% |
| 13 | `P67` | 489 | 1.57% |
| 14 | `P60` | 399 | 1.28% |
| 15 | `P55` | 351 | 1.13% |
| 16 | `P59` | 338 | 1.09% |
| 17 | `P65` | 324 | 1.04% |
| 18 | `P66` | 321 | 1.03% |
| 19 | `P50` | 318 | 1.02% |
| 20 | `P52` | 304 | 0.98% |
