# bach-gpt

Mechanistic interpretability experiments on a small GPT-style transformer trained on symbolic music (MIDI tokens).

## What this repo contains

- MIDI tokenizer with a fixed symbolic vocabulary and round-trip checks.
- Data pipeline for building token chunks from sampled GigaMIDI files.
- GPT training + checkpointing + CSV logging + Weights and Biases logging.
- MIDI generation from checkpoints (random, JSB, or custom MIDI prompts).
- Captioning pipeline (`caption_midi.py` -> `llm_caption.py`) for text-conditioned training data.
- Caption-conditioned dataloaders with fixed 512-token windows (`src/caption_dataloader.py`).
- Contrastive MIDI-text architecture (`src/contrastive_model.py`) with frozen encoders + trainable projection heads.
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
│   ├── caption_dataloader.py
│   ├── contrastive_model.py
│   ├── train_contrastive.py
│   ├── prefix_projector.py
│   ├── train_prefix.py
│   ├── inference_pipeline.py
│   ├── generate_conditional.py
│   ├── eval_retrieval.py
│   ├── download.py
│   ├── extract_gigamidi_sample.py
│   ├── probe_linear.py
│   ├── probe_attention.py
│   └── probe_completion.py
├── caption_midi.py
├── llm_caption.py
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

## Caption dataset pipeline

Generate JSONL captions from local MIDI files:

```bash
python3 caption_midi.py \
  --midi_dir data/gigamidi/sample \
  --output data/captions.jsonl \
  --limit 10000
```

Rewrite captions with an LLM:

```bash
python3 llm_caption.py \
  --input data/captions.jsonl \
  --output data/captions_llm.jsonl \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --batch_size 64
```

Use `src/caption_dataloader.py` to read `captions_llm.jsonl` for training:

- Uses `caption` by default; falls back to `caption_template` when empty.
- Tokenizes each MIDI from `path` with the existing tokenizer.
- Handles parse failures by sampling a random valid record (no crash).
- Uses `max_seq_len=512` windows:
  - train: random crop
  - val: deterministic first 512 tokens
- Returns `input_ids` and all-ones `attention_mask` with shape `(B, 512)`.
- Default split/dataloader settings:
  - split: 95/5 with fixed seed
  - `batch_size=64`
  - train: `shuffle=True`, `drop_last=True`
  - `num_workers=4`
  - `pin_memory=True` when CUDA is available

## Contrastive model architecture (Stage A)

`src/contrastive_model.py` defines `MidiTextContrastiveModel` with four components:

1. **MIDI encoder (frozen)**  
   Wraps the trained GPT from `src/model.py`, extracts last-layer hidden states, then applies masked mean pooling with the dataloader attention mask.

2. **Text encoder (frozen)**  
   Uses `sentence-transformers/all-MiniLM-L6-v2` (384-d embeddings) via `sentence_transformers`.

3. **Two trainable projection heads**  
   MIDI and text each use:
   - `Linear(input_dim, 512) -> GELU -> LayerNorm(512) -> Linear(512, 256)`
   producing a shared `embed_dim=256`.

4. **Learnable temperature**  
   Stores `log_temperature`, initialized at `log(0.07)`, and uses `exp(log_temperature)` clamped to `[0.01, 1.0]` during forward.

### Hidden-state extraction behavior

- If GPT supports `output_hidden_states=True`, the model uses that output.
- Otherwise it falls back to a forward hook on the last transformer block (works with the current custom GPT).

### Stage A trainable parameters

Only these are trained initially:
- MIDI projection head
- Text projection head
- `log_temperature` (typically with 10x smaller LR than projection heads)

Helper available in code:
- `make_optimizer_param_groups(proj_lr, temperature_lr_scale=0.1)`

### Symmetric InfoNCE loss (Phase 2c)

`src/contrastive_model.py` includes `symmetric_infonce_loss(...)` and uses it in model `forward`.

For a batch of `N` aligned pairs:

1. L2-normalize projected embeddings:
   - MIDI matrix `M` with shape `(N, 256)`
   - text matrix `T` with shape `(N, 256)`
2. Compute similarity logits:
   - `logits = (M @ T.T) / temperature` with shape `(N, N)`
3. Use diagonal labels:
   - `labels = [0, 1, ..., N-1]`
4. Compute bidirectional cross-entropy:
   - `loss_midi_to_text = CE(logits, labels)`
   - `loss_text_to_midi = CE(logits.T, labels)`
   - `loss = 0.5 * (loss_midi_to_text + loss_text_to_midi)`

The forward pass returns:
- `logits_midi_to_text`, `logits_text_to_midi`
- `loss_midi_to_text`, `loss_text_to_midi`, `loss`
- retrieval metrics `acc_midi_to_text` and `acc_text_to_midi`

Batch-size note:
- Contrastive learning strength depends heavily on in-batch negatives.
- `batch_size=64` is the target; `32` is a practical minimum when memory-constrained.

## Contrastive training loop (Phase 2d)

`src/train_contrastive.py` trains `MidiTextContrastiveModel` using symmetric InfoNCE.

### Optimizer and parameter groups

- Optimizer: `AdamW`
- Projection heads:
  - `lr=1e-4`
  - `weight_decay=0.01`
- Temperature (`log_temperature`):
  - `lr=1e-5`
  - `weight_decay=0.0`
- Text encoder (added at epoch 11):
  - `lr=1e-5`
  - added via `optimizer.add_param_group(...)` (optimizer is not recreated)

### Schedule and stability settings

- Total epochs: `30`
- Stage A: epochs `1-10` (projection heads + temperature only)
- Stage B: epochs `11-30` (text encoder unfrozen)
- LR schedule: cosine annealing over full run to `lr * 0.01`
- Gradient clipping: global norm clipped to `1.0` every step

### Run training

```bash
python3 src/train_contrastive.py \
  --captions-jsonl data/captions_llm.jsonl \
  --midi-checkpoint results/checkpoints/best_model.pt \
  --epochs 30 \
  --unfreeze-text-epoch 11 \
  --batch-size 64 \
  --proj-lr 1e-4 \
  --temp-lr 1e-5 \
  --text-lr 1e-5
```

Outputs:

- `results/contrastive_training_log.csv`
- `results/checkpoints_contrastive/clap_latest.pt`
- `results/checkpoints_contrastive/clap_best.pt`
- periodic `results/checkpoints_contrastive/clap_epoch_XXX.pt`

### Metrics tracked each epoch

- Primary model-selection metric: validation loss (`val_loss`)
- Retrieval R@1:
  - `val_r1_m2t` (MIDI -> text)
  - `val_r1_t2m` (text -> MIDI)
- Temperature value (`temperature`)
- Training diagnostics:
  - `train_loss`
  - `train_acc_m2t`
  - `train_acc_t2m`

Qualitative retrieval sanity check:
- Every 5 epochs (configurable via `--qualitative-every`)
- Uses 5 fixed prompts and prints top-3 MIDI matches over the full dataset by cosine similarity
- Prompt list can be overridden with `--qual-prompts`

R@1 interpretation guide (batch size 64):
- around `0.015`: random chance (`1/64`)
- around `0.10`: learning signal appears
- `0.30+`: strong alignment
- `0.50+`: excellent alignment

### Checkpointing

`src/train_contrastive.py` writes three checkpoint types:

- `clap_best.pt`
  - best validation loss so far (recommended for Phase 3 loading)
- `clap_latest.pt`
  - most recent epoch (resume target)
- `clap_epoch_XXX.pt`
  - periodic snapshots (`--checkpoint-every`, default `10`)

Each checkpoint stores:
- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `val_loss`
- `args` (training launch args for reconstruction)

## Potential failure modes to watch for

- **Loss does not decrease in Stage A**
  - Likely issue: incorrect MIDI representation extraction.
  - Checks:
    - confirm you are using hidden states from the last transformer block (not logits)
    - confirm attention masks are correct
    - confirm masked mean pooling is not averaging over padding

- **Loss decreases, then spikes at epoch 11**
  - Likely issue: unfreezing text encoder destabilizes optimization.
  - Mitigations:
    - lower text encoder LR (for example to `5e-6`)
    - optionally add a short warmup for the new text param group (for example 2 epochs ramping LR from 0)

- **R@1 stays near zero while loss decreases**
  - Likely issue: captions are too semantically similar across samples.
  - Checks and fixes:
    - run a caption cosine-similarity audit across the dataset
    - if mean max similarity is above ~`0.88`, increase caption diversity and/or filter near-duplicate pairs

- **Temperature collapses to the minimum (`0.01`)**
  - Likely issue: batch negatives are too weak for caption diversity.
  - Mitigations:
    - increase batch size
    - or reduce temperature LR (for example to `1e-6`) so it evolves more slowly

## Phase 3: Prefix conditioning

Phase 3 trains a small prefix projector that maps CLAP text embeddings into soft prefix tokens for the frozen MIDI GPT.

### Frozen vs trainable

- Frozen:
  - CLAP model (`MidiTextContrastiveModel` encoders + projection heads)
  - MIDI GPT (all parameters)
- Trainable:
  - prefix projector only (`PrefixProjector`)

`src/prefix_projector.py` includes hard freeze verification:
- non-projector parameters are set `requires_grad=False`
- runtime guard asserts total trainable params == projector params

### Prefix projector architecture

Input:
- 256-d L2-normalized CLAP text embedding

Output:
- `K` soft tokens in GPT hidden space, shape `(B, K, d_model)` (default `K=8`)

MLP:
- `Linear(256, d_model*2) -> GELU -> Dropout(0.1) -> LayerNorm(d_model*2) -> Linear(d_model*2, d_model*K)`
- reshape to `(B, K, d_model)`
- output `LayerNorm(d_model)` for scale control

Initialization:
- final linear layer weights/bias are scaled by `0.01` for near-zero initial prefixes

### Prefix injection forward path

Implemented in `forward_prefix_conditioned_logits(...)` and `phase3_prefix_lm_loss(...)`:

1. Encode caption with frozen CLAP text tower (no grad)
2. Project to prefix embeddings (grad flows through projector)
3. Lookup GPT token embeddings for `input_ids` (frozen)
4. Concatenate prefix + token embeddings
5. Call GPT with `inputs_embeds` and explicit `position_ids`
6. Slice/use only MIDI token positions for supervision

`src/model.py` now supports:
- `idx` or `inputs_embeds` input mode
- optional `position_ids`

### Phase 3 loss (LM only)

Standard next-token cross-entropy on MIDI tokens only:
- full label tensor shape `(B, K+T)` initialized to `-100`
- shifted targets filled for MIDI positions
- prefix positions ignored via `ignore_index=-100`

No Phase 2 contrastive loss is used in Phase 3.

### Phase 3 training setup

`src/train_prefix.py`:
- Optimizer: `AdamW` on projector params only
  - `lr=1e-4`, `weight_decay=0.01`
- LR schedule: 100-step linear warmup then cosine decay to `lr * 0.01`
- Gradient clipping: `1.0`
- Epochs: default `20` (works for 15–20 target)
- Batch size: default `64` (32 is supported)
- Dataset: same as Phase 2 (`captions_llm.jsonl`, same tokenizer/crops)

### Phase 3 metrics and checks

Per epoch:
- `val_loss` (primary metric)
- no-prefix baseline val loss (`baseline_val`) for comparison
- perplexity gap: `ppl_with_prefix - ppl_without_prefix`
- conditional perplexity gap by genre over held-out 200 examples:
  - `rock`, `jazz`, `classical`, `electronic` (plus `other`)

Every 5 epochs:
- qualitative short generation check:
  - unconditional sample
  - 3 fixed text prompts
  - compare token beginnings for divergence

### Pre-Phase-4 verification hooks

Implemented in `train_prefix.py`:
- **Verify 1:** same MIDI with correct vs wrong caption prefix; compare losses
- **Verify 2:** check projector final layer grad is non-`None` and non-zero after backward

### Optional Phase 3 regularization

`phase3_prefix_lm_loss(...)` supports optional prefix-attention regularization:
- `--prefix-attn-reg-weight`
- `--prefix-attn-min-mean`

When enabled, it penalizes low mean attention from MIDI query positions to prefix key positions.

Scale diagnostics are printed each epoch:
- mean norm of prefix embeddings vs GPT token embeddings
- warning when ratio is severely mismatched

## Phase 4: Inference and evaluation

### Pipeline assembly (4a)

`src/inference_pipeline.py` loads checkpoints in strict order:
1. MIDI GPT checkpoint
2. CLAP checkpoint (reconstructed from saved `args`)
3. Prefix projector checkpoint

All modules are set to `.eval()`, including the sentence-transformer text encoder inside CLAP.

A smoke-test forward pass is run immediately to catch:
- device mismatches
- missing/incompatible weights
- shape errors in prefix injection

### Conditional generation (4b/4c)

`src/generate_conditional.py` implements the full text -> MIDI pipeline as separate debug-friendly functions:

1. text encoding
2. prefix projection
3. BOS seed embedding + prefix concatenation
4. autoregressive decoding with KV cache
5. structural-boundary truncation for malformed tails
6. save + reload verification with `pretty_midi`

Sampling controls:
- temperature (default `0.9`)
- top-p nucleus sampling (default `0.92`, primary sampler)
- top-k safety cap (default `50`)
- optional repetition penalty + recent-token window

Example:

```bash
python3 src/generate_conditional.py \
  --prompt "A bright fast piano étude with rising melodic contour." \
  --midi-checkpoint results/checkpoints/best_model.pt \
  --clap-checkpoint results/checkpoints_contrastive/clap_best.pt \
  --prefix-checkpoint results/checkpoints_prefix/prefix_projector_best.pt \
  --out results/conditional_generated.mid
```

### Retrieval evaluation (4d)

`src/eval_retrieval.py` runs held-out retrieval metrics on the 5% validation split:

- MIDI->text: `R@1`, `R@5`, `R@10`, median rank
- text->MIDI: `R@1`, `R@5`, `R@10`, median rank
- random baseline: `1/N_val`
- genre `R@1` breakdown for:
  - `rock`, `jazz`, `classical`, `electronic`

Outputs:
- console summary
- `results/retrieval_eval.json`

Example:

```bash
python3 src/eval_retrieval.py \
  --captions-jsonl data/captions_llm.jsonl \
  --midi-checkpoint results/checkpoints/best_model.pt \
  --clap-checkpoint results/checkpoints_contrastive/clap_best.pt
```

## Probing and analysis scripts

After training (or with an existing checkpoint):

```bash
python3 src/probe_linear.py --checkpoint results/checkpoints/best_model.pt
python3 src/probe_attention.py --checkpoint results/checkpoints/best_model.pt
python3 src/probe_completion.py --checkpoint results/checkpoints/best_model.pt
```

These write figures to `figures/` and markdown summaries to `results/`.
