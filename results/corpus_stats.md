# Corpus statistics

| corpus | files | round-trip pass | tokens | mean len | median len | min | max | vocab used / total |
|---|---|---|---|---|---|---|---|---|
| JSB | 30 | 30/30 (100.0%) | 31,119 | 1037 | 1006 | 519 | 2222 | 77/167 (46%) |
| GigaMIDI | 100 | 100/100 (100.0%) | 676,198 | 6762 | 5922 | 3 | 26606 | 160/167 (96%) |

## Token budget (projected from sample means)

Per-corpus projection = (mean tokens / sample file) × (full-corpus file count). GigaMIDI training split = the all-instruments-with-drums training-V1.1-80% split used as pretraining; JSB Chorales is held out for zero-shot probing.

| corpus | full-corpus files | sample size | mean tokens / file | projected total tokens | role |
|---|---|---|---|---|---|
| JSB | 371 | 30 | 1,037 | 384,838 | probing (held out) |
| GigaMIDI | 137,241 | 100 | 6,762 | 928,020,897 | pretraining |

## GigaMIDI genre distribution (sample)

10 of 100 sampled files carry a genre label across the five metadata columns (curated, scraped, Discogs, Last.fm, Tagtraum). Top entries:

| genre | count |
|---|---|
| game | 2 |
| pop | 2 |
| rock | 2 |
| dance | 1 |
| jazz | 1 |
| alternative-indie | 1 |
| latin | 1 |

## Top-20 tokens by corpus

### JSB

| rank | token | count | share |
|---|---|---|---|
| 1 | `V5` | 8,965 | 28.81% |
| 2 | `D18` | 4,733 | 15.21% |
| 3 | `D13` | 2,700 | 8.68% |
| 4 | `TS13` | 1,963 | 6.31% |
| 5 | `D22` | 752 | 2.42% |
| 6 | `TS18` | 698 | 2.24% |
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
| 17 | `D20` | 330 | 1.06% |
| 18 | `P65` | 324 | 1.04% |
| 19 | `P66` | 321 | 1.03% |
| 20 | `P50` | 318 | 1.02% |

### GigaMIDI

| rank | token | count | share |
|---|---|---|---|
| 1 | `V6` | 58,387 | 8.63% |
| 2 | `V5` | 48,142 | 7.12% |
| 3 | `V7` | 42,721 | 6.32% |
| 4 | `V4` | 29,657 | 4.39% |
| 5 | `D9` | 27,133 | 4.01% |
| 6 | `D13` | 21,978 | 3.25% |
| 7 | `D11` | 18,934 | 2.80% |
| 8 | `TS13` | 18,130 | 2.68% |
| 9 | `D4` | 14,760 | 2.18% |
| 10 | `V3` | 12,718 | 1.88% |
| 11 | `D12` | 11,449 | 1.69% |
| 12 | `TS9` | 11,382 | 1.68% |
| 13 | `D16` | 8,796 | 1.30% |
| 14 | `P57` | 8,260 | 1.22% |
| 15 | `D14` | 7,926 | 1.17% |
| 16 | `P62` | 7,833 | 1.16% |
| 17 | `D18` | 7,829 | 1.16% |
| 18 | `BAR_START` | 7,806 | 1.15% |
| 19 | `BAR_END` | 7,806 | 1.15% |
| 20 | `P64` | 7,638 | 1.13% |
