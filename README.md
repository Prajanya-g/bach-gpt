# bach-gpt

Mechanistic interpretability of a small GPT-style transformer trained on symbolic music (MIDI).

## Layout

```
bach-gpt/
├── src/
│   ├── tokenizer.py       # 167-token vocab + encode/decode + round-trip test
│   ├── corpus_stats.py    # JSB + GigaMIDI stats, figures, markdown report
│   ├── dataset.py         # MIDI → chunked token tensors, train/val DataLoaders
│   ├── model.py           # minimal GPT (manual causal attention, tied embeddings)
│   ├── train.py           # training loop + cosine LR + checkpoints + CSV logs
│   ├── generate.py        # checkpoint sampling (random/jsb/custom MIDI prompts)
│   ├── extract_gigamidi_sample.py  # parquet -> sampled .mid files
│   └── download.py        # gated Hugging Face snapshot download helper
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

- **GigaMIDI v1.1/v2.0** (pretraining): [Metacreation/GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI) on Hugging Face (gated — log in and accept terms).
  - **Zip layout path** (older workflow): after unzipping `training-V1.1-80%.zip`, you should have `data/Final_GigaMIDI_V1.1_Final/training-V1.1-80%/all-instruments-with-drums.zip`, and `corpus_stats.py` can sample from that zip.
  - **Parquet layout path** (HF snapshot workflow): data lives in `data/Final_GigaMIDI_V1.1_Final/all-instruments-with-drums/train.parquet` (no raw `.mid` files on disk). Use `src/extract_gigamidi_sample.py` to write sampled `.mid` files into `data/gigamidi/sample/`.
- **JSB Chorales** (probing): no download — loaded from `music21.corpus.chorales.Iterator()` the first time `corpus_stats.py` runs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For `src/dataset.py`, `src/model.py`, `src/train.py`, and `src/generate.py`, install PyTorch for your platform (not listed in `requirements.txt`):

```bash
pip install torch
```

([PyTorch install selector](https://pytorch.org/get-started/locally/) if you need a specific CUDA build.)

### Python version note

- `music21>=9.0` requires Python 3.10+.
- On Python 3.9 environments (common on HPC), JSB-related features may need an older `music21`, but the GigaMIDI parquet -> sample -> train path works without JSB.

## Run

```bash
python3 src/tokenizer.py       # smoke-test the tokenizer
python3 src/corpus_stats.py    # JSB + GigaMIDI stats → figures/ + results/
python3 src/dataset.py         # build train/val loaders from data/gigamidi/sample/ (needs torch)
python3 src/train.py           # train model, save checkpoints, write training_log.csv
python3 src/generate.py --checkpoint results/checkpoints/best_model.pt --prompt random --out results/generated.mid
```

On first run, `corpus_stats.py` extracts 100 random MIDI files from the GigaMIDI training zip into `data/gigamidi/sample/` and caches them. Delete that folder to force a fresh sample.

`dataset.py` encodes those MIDIs, concatenates with `EOS` between pieces, chunks into fixed windows, does a 90/10 train/val split on chunks, and prints basic sanity checks (vocab bounds, token counts, random decode preview).

## HPC parquet workflow (recommended)

If your Hugging Face download produced parquet files (e.g. `all-instruments-with-drums/train.parquet`), run:

```bash
python3 src/extract_gigamidi_sample.py \
  --parquet data/Final_GigaMIDI_V1.1_Final/all-instruments-with-drums/train.parquet \
  --out-dir data/gigamidi/sample \
  --n-samples 100 \
  --seed 17
```

Then train as usual:

```bash
python3 src/dataset.py
python3 src/train.py --max-epochs 10 --batch-size 32 --block-size 512
python3 src/generate.py --checkpoint results/checkpoints/best_model.pt --prompt random --out results/generated.mid
```

## Status

Check-in (2026-04-22): tokenizer + round-trip test + GigaMIDI/JSB statistics pipeline; pretraining corpus is GigaMIDI v1.1.

Update (2026-04-28): training stack now includes `dataset.py`, `model.py`, `train.py`, and `generate.py`, plus parquet sample extraction via `extract_gigamidi_sample.py`.
