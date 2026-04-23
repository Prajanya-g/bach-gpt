# bach-gpt

Mechanistic interpretability of a small GPT-style transformer trained on symbolic music (MIDI).

## Layout

```
bach-gpt/
├── src/
│   ├── tokenizer.py       # 167-token vocab + encode/decode + round-trip test
│   └── corpus_stats.py    # JSB + GigaMIDI stats, figures, markdown report
├── data/                  # NOT COMMITTED — see "Datasets" below for download links
│   ├── Final_GigaMIDI_V1.1_Final/   # GigaMIDI v1.1 (pretraining)
│   └── gigamidi/sample/             # per-run 100-file sample extracted from the training zip
├── figures/               # PNGs written by corpus_stats.py
├── results/
│   └── corpus_stats.md    # markdown summary written by corpus_stats.py
├── checkin.md             # project check-in (markdown)
├── checkin.docx           # project check-in (Word)
├── requirements.txt
└── README.md
```

## Datasets (not committed)

The `data/` directory is in `.gitignore`. Download the corpora locally before running the pipeline.

- **GigaMIDI v1.1** (pretraining corpus, ~12 GB total, training split ~4.4 GB zipped): https://huggingface.co/datasets/Metacreation/GigaMIDI — the same corpus used to train MIDI-GPT. We sample from `training-V1.1-80%/all-instruments-with-drums.zip`. Drop the extracted `Final_GigaMIDI_V1.1_Final/` folder under `./data/`.
- **JSB Chorales** (probing corpus): no download required — pulled automatically from `music21.corpus.chorales.Iterator()` the first time `corpus_stats.py` runs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 src/tokenizer.py       # smoke-test the tokenizer
python3 src/corpus_stats.py    # JSB + GigaMIDI stats → figures/ + results/
```

On first run, `corpus_stats.py` extracts 100 random MIDI files from the GigaMIDI training zip into `data/gigamidi/sample/` and caches them. Delete that folder to force a fresh sample.

## Status

Check-in (2026-04-22): tokenizer + round-trip test + three-corpus statistics pipeline complete, pretraining corpus switched from MAESTRO to GigaMIDI. No model trained, no probing yet.
