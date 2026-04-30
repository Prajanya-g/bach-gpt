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
source .venv/bin/activate
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
from pathlib import Path
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
from pathlib import Path
import torch
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

echo ""
echo "========================================"
echo "COMPOUND PIPELINE TEST PASSED"
echo "Finished: $(date)"
echo "========================================"
echo "Checkpoint: $COMPOUND_CKPT"
