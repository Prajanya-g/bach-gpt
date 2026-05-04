#!/bin/bash
#SBATCH --job-name=bach-compound-10k-v2-p1
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=75G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/pg2963/bach-gpt/logs/compound_10k_v2_p1_%j.out
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/pg2963/bach-gpt
cd $ROOT || exit 1
mkdir -p logs results/compound_10k_v2/checkpoints_compound

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "========================================"

fail() { echo "FAILED at: $1"; exit 1; }
pass() { echo "PASSED: $1  ($(date))"; echo "----------------------------------------"; }

# -- Phase 1a: smoke test ------------------------------------------------------
echo ""
echo "========================================"
echo "PHASE 1a — Dataloader smoke"
echo "========================================"

python3 << 'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from compound_dataset import build_compound_dataloaders

train_loader, val_loader, stats = build_compound_dataloaders(
    sample_dir=Path("data/lmd_sample_10000"),
    block_size=512,
    batch_size=32,
    split_ratio=0.9,
    seed=17,
)
if stats.n_files_encoded == 0:
    raise RuntimeError("No files encoded")
xb, yb = next(iter(train_loader))
print(f"[compound] files={stats.n_files_encoded} chunks={stats.n_chunks_total} "
      f"train={stats.n_train_chunks} val={stats.n_val_chunks}")
print(f"[compound] batch x={tuple(xb.shape)} y={tuple(yb.shape)}")
PYEOF

[ $? -eq 0 ] || fail "Phase 1a"
pass "Phase 1a"

# -- Phase 1b: train bigger CompoundGPT ---------------------------------------
echo ""
echo "========================================"
echo "PHASE 1b — Compound GPT v2 (10k, 20 epochs, 45M params)"
echo "========================================"

python3 << 'PYEOF'
import sys, time, math
from pathlib import Path
import torch

sys.path.insert(0, "src")
from compound_dataset import build_compound_dataloaders
from compound_model import CompoundGPT, default_compound_config, compound_loss

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"[compound-train] device={device}")

train_loader, val_loader, stats = build_compound_dataloaders(
    sample_dir=Path("data/lmd_sample_10000"),
    block_size=512,
    batch_size=32,
    split_ratio=0.9,
    seed=17,
)
print(f"[compound-train] train={stats.n_train_chunks} val={stats.n_val_chunks}")

# bigger model
cfg = default_compound_config()
cfg.block_size = 512
cfg.d_model    = 768
cfg.n_layers   = 8
cfg.n_heads    = 12
cfg.d_ff       = 3072
cfg.dropout    = 0.1

model = CompoundGPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[compound-train] parameters={n_params:,} (~{n_params/1e6:.1f}M)")

epochs      = 20
accum_steps = 2   # effective batch = 64
total_steps = epochs * (len(train_loader) // accum_steps)
warmup      = 500

opt = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

def get_lr(step):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)

best_val = float("inf")
ckpt_dir = Path("results/compound_10k_v2/checkpoints_compound")
ckpt_dir.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
global_step = 0

for epoch in range(1, epochs + 1):
    model.train()
    train_losses = []
    opt.zero_grad(set_to_none=True)

    for micro_step, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        loss = compound_loss(model(xb), yb) / accum_steps
        loss.backward()
        train_losses.append(float(loss.item()) * accum_steps)

        if (micro_step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 200 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"[compound-train] epoch={epoch} step={global_step} "
                      f"loss={train_losses[-1]:.4f} lr={lr:.2e}")

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_losses.append(float(compound_loss(model(xb), yb).item()))

    train_loss = sum(train_losses) / max(1, len(train_losses))
    val_loss   = sum(val_losses)   / max(1, len(val_losses))
    elapsed    = time.perf_counter() - t0
    print(
        f"[compound-train] epoch={epoch}/{epochs} "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"val_ppl={math.exp(val_loss):.2f} elapsed={elapsed/60:.1f}min"
    )

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(ckpt, ckpt_dir / "compound_latest.pt")
    torch.save(ckpt, ckpt_dir / f"compound_epoch_{epoch:03d}.pt")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(ckpt, ckpt_dir / "compound_best.pt")
        print(f"[compound-train] new best val_loss={val_loss:.4f}")

elapsed = time.perf_counter() - t0
print(f"[compound-train] done in {elapsed/60:.1f} min, best_val={best_val:.4f}")
PYEOF

[ $? -eq 0 ] || fail "Phase 1b"
pass "Phase 1b"

COMPOUND_CKPT="results/compound_10k_v2/checkpoints_compound/compound_best.pt"
[ -f "$COMPOUND_CKPT" ] || fail "checkpoint not found at $COMPOUND_CKPT"

echo ""
echo "========================================"
echo "PHASE 1 COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo "Checkpoint: $COMPOUND_CKPT"
