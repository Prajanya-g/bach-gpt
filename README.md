# bach-gpt

Mechanistic interpretability experiments on a small GPT-style transformer trained on symbolic music (MIDI tokens).

## What this repo contains

- MIDI tokenizer with a fixed symbolic vocabulary and round-trip checks.
- Data pipeline for building token chunks from sampled GigaMIDI files.
- GPT training + checkpointing + CSV logging + Weights and Biases logging.
- MIDI generation from checkpoints (random, JSB, or custom MIDI prompts).
- Probing scripts for linear probes, attention analysis, and phrase completion.

## Project layout

```text
bach-gpt/
├── src/
│   ├── tokenizer.py
│   ├── corpus_stats.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── download.py
│   ├── extract_gigamidi_sample.py
│   ├── probe_linear.py
│   ├── probe_attention.py
│   └── probe_completion.py
├── data/            # not committed
├── figures/         # plots produced by stats/probing scripts
├── results/         # markdown summaries, logs, checkpoints
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+ (because `music21>=9.0` requires it).
- Dependencies from `requirements.txt`.
- PyTorch installed separately for your platform.

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch
```

For CUDA-specific wheels, use the [PyTorch install selector](https://pytorch.org/get-started/locally/).

## Datasets (not committed)

`data/` is gitignored. You need local dataset files before training.

- **GigaMIDI** (pretraining corpus, gated): [Metacreation/GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI)
- **JSB Chorales** (probing/statistics): loaded through `music21` corpus APIs (no manual download in this repo).

### Download GigaMIDI from Hugging Face

After accepting dataset terms and creating a read token:

```bash
export HF_TOKEN=hf_...
python3 src/download.py
```

This downloads into `data/Final_GigaMIDI_V1.1_Final/` by default.

## Build a local training sample

### Recommended (parquet workflow)

If your snapshot contains parquet files (for example `all-instruments-with-drums/train.parquet`):

```bash
python3 src/extract_gigamidi_sample.py \
  --parquet data/Final_GigaMIDI_V1.1_Final/all-instruments-with-drums/train.parquet \
  --out-dir data/gigamidi/sample \
  --n-samples 100 \
  --seed 17
```

### Zip workflow

If you have `training-V1.1-80%/all-instruments-with-drums.zip` under `data/Final_GigaMIDI_V1.1_Final/`, `src/corpus_stats.py` can auto-extract a 100-file sample into `data/gigamidi/sample/` on first run.

## Typical run sequence

```bash
python3 src/tokenizer.py
python3 src/corpus_stats.py
python3 src/dataset.py
python3 src/train.py --max-epochs 10 --batch-size 32 --block-size 512
python3 src/generate.py \
  --checkpoint results/checkpoints/best_model.pt \
  --prompt random \
  --out results/generated.mid
```

Outputs:

- `results/training_log.csv`
- `results/checkpoints/*.pt`
- `results/generated.mid` (+ continuation-only MIDI)
- figures and markdown summaries in `figures/` and `results/`

## Training notes

- `src/train.py` logs to Weights and Biases (`wandb`) by default.
- If you do not want cloud logging, run with offline mode:

```bash
WANDB_MODE=offline python3 src/train.py
```

- Device is auto-selected (`cuda` -> `mps` -> `cpu`).

## Probing and analysis scripts

After training (or with an existing checkpoint):

```bash
python3 src/probe_linear.py --checkpoint results/checkpoints/best_model.pt
python3 src/probe_attention.py --checkpoint results/checkpoints/best_model.pt
python3 src/probe_completion.py --checkpoint results/checkpoints/best_model.pt
```

These write figures to `figures/` and markdown summaries to `results/`.
