#!/bin/bash
#SBATCH --job-name=bach-compound-10k
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=75G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/pg2963/bach-gpt/logs/overnight_compound_10k_%j.out
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/pg2963/bach-gpt
cd $ROOT || exit 1
if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/venv/bin/activate"
else
  echo "WARNING: No .venv or venv under $ROOT; using default python" >&2
fi
mkdir -p logs results/overnight_compound/checkpoints_compound

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

# -- Prepare 10k-file sample dir ------------------------------------------------
echo ""
echo "========================================"
echo "Preparing 10k-file LMD sample dir"
echo "========================================"

python3 << 'PYEOF'
import random
import shutil
from pathlib import Path

src = Path("data/midicaps/lmd_full")
dst = Path("data/lmd_sample_10000")
dst.mkdir(exist_ok=True)

all_files = list(src.rglob("*.mid"))
if not all_files:
    raise RuntimeError(f"No MIDI files found under {src}")

random.seed(42)
sample = random.sample(all_files, min(10000, len(all_files)))

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

# -- Phase 1a: compound tokenizer + dataloader smoke ----------------------------
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
    sample_dir=Path("data/lmd_sample_10000"),
    block_size=512,
    batch_size=16,
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

# -- Phase 1b: overnight compound GPT training ----------------------------------
echo ""
echo "========================================"
echo "PHASE 1b — Compound GPT overnight train (10k files)"
echo "========================================"

python3 << 'PYEOF'
import sys
import time
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
    sample_dir=Path("data/lmd_sample_10000"),
    block_size=512,
    batch_size=16,
    split_ratio=0.9,
    seed=17,
)

cfg = default_compound_config()
cfg.block_size = 512
cfg.dropout = 0.1
model = CompoundGPT(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

epochs = 8
best_val = float("inf")
ckpt_dir = Path("results/overnight_compound/checkpoints_compound")
ckpt_dir.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
for epoch in range(1, epochs + 1):
    model.train()
    train_losses = []
    for step, (xb, yb) in enumerate(train_loader, start=1):
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = compound_loss(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_losses.append(float(loss.item()))
        if step % 100 == 0:
            print(
                f"[compound-train] epoch={epoch} step={step} "
                f"loss={loss.item():.4f}"
            )

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            val_losses.append(float(compound_loss(logits, yb).item()))

    train_loss = sum(train_losses) / max(1, len(train_losses))
    val_loss = sum(val_losses) / max(1, len(val_losses))
    print(
        f"[compound-train] epoch={epoch}/{epochs} "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
    )

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(ckpt, ckpt_dir / "compound_latest.pt")
    if epoch % 2 == 0:
        torch.save(ckpt, ckpt_dir / f"compound_epoch_{epoch:03d}.pt")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(ckpt, ckpt_dir / "compound_best.pt")

elapsed = time.perf_counter() - t0
print(f"[compound-train] done in {elapsed/60:.1f} min, best_val={best_val:.4f}")
PYEOF

[ $? -eq 0 ] || fail "Phase 1b"
pass "Phase 1b"

COMPOUND_CKPT="results/overnight_compound/checkpoints_compound/compound_best.pt"
[ -f "$COMPOUND_CKPT" ] || fail "compound checkpoint not found at $COMPOUND_CKPT"

echo ""
echo "========================================"
echo "OVERNIGHT COMPOUND RUN PASSED"
echo "Finished: $(date)"
echo "========================================"
echo "Best checkpoint: $COMPOUND_CKPT"
