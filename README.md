# bach-gpt

Mechanistic interpretability of a small GPT-style transformer trained on symbolic music (MIDI).

## Layout

```
bach-gpt/
├── src/
│   ├── tokenizer.py       # 167-token vocab + encode/decode + round-trip test
│   ├── corpus_stats.py    # JSB + GigaMIDI stats, figures, markdown report
│   └── dataset.py         # MIDI → chunked token tensors, train/val DataLoaders
├── data/                  # NOT COMMITTED — see "Datasets" below
│   ├── Final_GigaMIDI_V1.1_Final/   # GigaMIDI v1.1 splits (zips + optional unzip)
│   └── gigamidi/sample/             # 100-file sample extracted from the training zip
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

- **GigaMIDI v1.1** (pretraining): [Metacreation/GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI) on Hugging Face (gated — log in and accept terms). The training split is a large zip; place artifacts under `./data/Final_GigaMIDI_V1.1_Final/`.
  - After you unzip `training-V1.1-80%.zip`, you should have:
    `data/Final_GigaMIDI_V1.1_Final/training-V1.1-80%/all-instruments-with-drums.zip`
  - `corpus_stats.py` samples from that inner zip into `data/gigamidi/sample/`.
- **JSB Chorales** (probing): no download — loaded from `music21.corpus.chorales.Iterator()` the first time `corpus_stats.py` runs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For `src/dataset.py` (PyTorch `DataLoader`), install PyTorch for your platform (not listed in `requirements.txt`):

```bash
pip install torch
```

([PyTorch install selector](https://pytorch.org/get-started/locally/) if you need a specific CUDA build.)

## Run

```bash
python3 src/tokenizer.py       # smoke-test the tokenizer
python3 src/corpus_stats.py    # JSB + GigaMIDI stats → figures/ + results/
python3 src/dataset.py         # build train/val loaders from data/gigamidi/sample/ (needs torch)
```

On first run, `corpus_stats.py` extracts 100 random MIDI files from the GigaMIDI training zip into `data/gigamidi/sample/` and caches them. Delete that folder to force a fresh sample.

`dataset.py` encodes those MIDIs, concatenates with `EOS` between pieces, chunks into fixed windows, does a 90/10 train/val split on chunks, and prints basic sanity checks (vocab bounds, token counts, random decode preview).

## Status

Check-in (2026-04-22): tokenizer + round-trip test + GigaMIDI/JSB statistics pipeline; pretraining corpus is GigaMIDI v1.1.

Update (2026-04-28): `dataset.py` adds batched next-token examples from the GigaMIDI sample. Model training and probing are still to come.
