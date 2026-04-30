#!/bin/bash
#SBATCH --job-name=bach-compound-test
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=75G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pg2963/bach-gpt/logs/pipeline_test_compound_%j.out
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/pg2963/bach-gpt
cd $ROOT || exit 1
source "$ROOT/.venv/bin/activate"
mkdir -p logs results/test_compound/checkpoints_compound

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "========================================"

fail() {
    echo ""
    echo "FAILED at: $1"
    exit 1
}

pass() {
    echo "PASSED: $1  ($(date))"
    echo "----------------------------------------"
}

# -- Prepare 100-file sample dir ------------------------------------------------
echo ""
echo "========================================"
echo "Preparing 100-file LMD sample dir"
echo "========================================"

python3 << 'PYEOF'
import random
import shutil
from pathlib import Path

src = Path("data/midicaps/lmd_full")
dst = Path("data/lmd_sample_100")
dst.mkdir(exist_ok=True)

all_files = list(src.rglob("*.mid"))
if not all_files:
    raise RuntimeError(f"No MIDI files found under {src}")

random.seed(42)
sample = random.sample(all_files, min(100, len(all_files)))

copied = 0
for f in sample:
    dest = dst / f.name
    if not dest.exists():
        shutil.copy(f, dest)
    copied += 1

print(f"Sample dir ready: {copied} files -> {dst}")
PYEOF

[ $? -eq 0 ] || fail "sample dir preparation"
pass "sample dir preparation"

# -- Phase 1a: compound tokenizer + dataset smoke -------------------------------
echo ""
echo "========================================"
echo "PHASE 1a — Compound tokenizer + dataloader smoke"
echo "========================================"

python3 << 'PYEOF'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from compound_dataset import build_compound_dataloaders

train_loader, val_loader, stats = build_compound_dataloaders(
    sample_dir=Path("data/lmd_sample_100"),
    block_size=512,
    batch_size=8,
    split_ratio=0.9,
    seed=17,
)

if stats.n_files_encoded == 0:
    raise RuntimeError("No MIDI files were successfully encoded.")
if stats.n_chunks_total == 0:
    raise RuntimeError("Compound stream produced zero chunks.")

xb, yb = next(iter(train_loader))
print(
    f"[compound] files={stats.n_files_encoded}/{stats.n_files_seen} "
    f"chunks={stats.n_chunks_total} train={stats.n_train_chunks} val={stats.n_val_chunks}"
)
print(f"[compound] batch x={tuple(xb.shape)} y={tuple(yb.shape)}")
PYEOF

[ $? -eq 0 ] || fail "Phase 1a"
pass "Phase 1a"

# -- Phase 1b: short compound GPT optimization run ------------------------------
echo ""
echo "========================================"
echo "PHASE 1b — Compound GPT short train (100 files)"
echo "========================================"

python3 << 'PYEOF'
import sys
from pathlib import Path
import torch

sys.path.insert(0, "src")
from compound_dataset import build_compound_dataloaders
from compound_model import CompoundGPT, default_compound_config, compound_loss

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)
print(f"[compound-train] device={device}")

train_loader, val_loader, _ = build_compound_dataloaders(
    sample_dir=Path("data/lmd_sample_100"),
    block_size=512,
    batch_size=8,
    split_ratio=0.9,
    seed=17,
)

cfg = default_compound_config()
cfg.block_size = 512
cfg.dropout = 0.1
model = CompoundGPT(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

max_steps = 200
step = 0
model.train()
for xb, yb in train_loader:
    xb = xb.to(device)
    yb = yb.to(device)
    opt.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = compound_loss(logits, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    if step % 25 == 0:
        print(f"[compound-train] step={step} loss={loss.item():.4f}")
    if step >= max_steps:
        break

model.eval()
val_losses = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        val_losses.append(float(compound_loss(logits, yb).item()))
        if len(val_losses) >= 20:
            break

val_loss = sum(val_losses) / max(1, len(val_losses))
print(f"[compound-train] val_loss={val_loss:.4f}")

ckpt_path = Path("results/test_compound/checkpoints_compound/compound_best.pt")
ckpt_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "val_loss": val_loss,
    },
    ckpt_path,
)
print(f"[compound-train] checkpoint -> {ckpt_path}")
PYEOF

[ $? -eq 0 ] || fail "Phase 1b"
pass "Phase 1b"

COMPOUND_CKPT="results/test_compound/checkpoints_compound/compound_best.pt"
[ -f "$COMPOUND_CKPT" ] || fail "compound checkpoint not found at $COMPOUND_CKPT"

# -- Phase 2: compound contrastive smoke ----------------------------------------
echo ""
echo "========================================"
echo "PHASE 2 — Compound contrastive smoke"
echo "========================================"

python3 << 'PYEOF'
import json
import random
import sys
from pathlib import Path

import pretty_midi
import torch

sys.path.insert(0, "src")
from compound import SENTINELS, STEP_PAD, encode_compound
from compound_model import CompoundGPT, default_compound_config
from contrastive_model import CompoundMidiTextContrastiveModel

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)
print(f"[compound-contrastive] device={device}")

cfg = default_compound_config()
cfg.block_size = 512
midi_encoder = CompoundGPT(cfg).to(device)

ckpt = torch.load(
    "results/test_compound/checkpoints_compound/compound_best.pt",
    map_location=device,
)
midi_encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
midi_encoder.eval()

model = CompoundMidiTextContrastiveModel(
    midi_compound_gpt=midi_encoder,
    device=device,
)
model.train()

records = []
with open("data/midicaps_100.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
if not records:
    raise RuntimeError("No records found in data/midicaps_100.jsonl")

random.seed(17)
random.shuffle(records)

batch_size = 8
max_seq_len = 256
compound_batches = []
captions_batches = []
cur_steps = []
cur_caps = []
for r in records:
    path = r.get("path") or r.get("midi_path") or r.get("midi")
    cap = (r.get("caption") or r.get("caption_template") or "").strip()
    if not path or not cap:
        continue
    try:
        pm = pretty_midi.PrettyMIDI(path)
        steps = encode_compound(pm)
    except Exception:
        continue
    if not steps:
        continue

    trimmed = steps[:max_seq_len]
    pad_step = list(SENTINELS)
    pad_step[0] = STEP_PAD
    if len(trimmed) < max_seq_len:
        trimmed = trimmed + [pad_step] * (max_seq_len - len(trimmed))

    cur_steps.append(trimmed)
    cur_caps.append(cap)
    if len(cur_steps) == batch_size:
        compound_batches.append(torch.tensor(cur_steps, dtype=torch.long))
        captions_batches.append(list(cur_caps))
        cur_steps, cur_caps = [], []
    if len(compound_batches) >= 4:
        break

if not compound_batches:
    raise RuntimeError("Could not build any valid compound-caption batches.")

opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.01)

for step_idx, (compound_input, captions) in enumerate(
    zip(compound_batches, captions_batches), start=1
):
    compound_input = compound_input.to(device)
    opt.zero_grad(set_to_none=True)
    out = model(compound_input=compound_input, captions=captions)
    loss = out["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(
        f"[compound-contrastive] step={step_idx} "
        f"loss={loss.item():.4f} "
        f"acc=({out['acc_midi_to_text'].item():.3f},{out['acc_text_to_midi'].item():.3f})"
    )

out_dir = Path("results/test_compound/checkpoints_contrastive_compound")
out_dir.mkdir(parents=True, exist_ok=True)
out_ckpt = out_dir / "clap_compound_smoke.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "steps_ran": len(compound_batches),
        "max_seq_len": max_seq_len,
    },
    out_ckpt,
)
print(f"[compound-contrastive] checkpoint -> {out_ckpt}")
PYEOF

[ $? -eq 0 ] || fail "Phase 2"
pass "Phase 2"

COMPOUND_CLAP_CKPT="results/test_compound/checkpoints_contrastive_compound/clap_compound_smoke.pt"
[ -f "$COMPOUND_CLAP_CKPT" ] || fail "compound contrastive checkpoint not found at $COMPOUND_CLAP_CKPT"

echo ""
echo "========================================"
echo "COMPOUND PIPELINE TEST PASSED"
echo "Finished: $(date)"
echo "========================================"
echo "Checkpoint: $COMPOUND_CKPT"
echo "Compound contrastive checkpoint: $COMPOUND_CLAP_CKPT"
