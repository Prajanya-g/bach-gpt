#!/bin/bash
#SBATCH --job-name=bach-compound-50k
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=75G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/pg2963/bach-gpt/logs/overnight_compound_50k_%j.out
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
mkdir -p logs results/overnight_compound_50k/checkpoints_compound

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

# -- Phase 1a: compound dataloader smoke on full 50k ---------------------------
echo ""
echo "========================================"
echo "PHASE 1a — Compound dataloader smoke (50k)"
echo "========================================"

python3 << 'PYEOF'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from compound_dataset import build_compound_dataloaders

train_loader, val_loader, stats = build_compound_dataloaders(
    sample_dir=Path("data/midicaps/lmd_full"),
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
    f"failed={stats.n_files_failed} "
    f"chunks={stats.n_chunks_total} train={stats.n_train_chunks} val={stats.n_val_chunks}"
)
print(f"[compound] batch x={tuple(xb.shape)} y={tuple(yb.shape)}")
PYEOF

[ $? -eq 0 ] || fail "Phase 1a"
pass "Phase 1a"

# -- Phase 1b: compound GPT training on 50k ------------------------------------
echo ""
echo "========================================"
echo "PHASE 1b — Compound GPT training (50k files, 10 epochs)"
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
    else torch.device("cpu")
)
print(f"[compound-train] device={device}")

train_loader, val_loader, stats = build_compound_dataloaders(
    sample_dir=Path("data/midicaps/lmd_full"),
    block_size=512,
    batch_size=32,
    split_ratio=0.9,
    seed=17,
)
print(
    f"[compound-train] train_chunks={stats.n_train_chunks} "
    f"val_chunks={stats.n_val_chunks}"
)

cfg = default_compound_config()
cfg.block_size = 512
cfg.dropout = 0.1
cfg.d_model = 512
cfg.n_layers = 6
cfg.n_heads = 8
model = CompoundGPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[compound-train] parameters={n_params:,} (~{n_params/1e6:.1f}M)")

opt = torch.optim.AdamW(
    model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
)

epochs = 10
best_val = float("inf")
ckpt_dir = Path("results/overnight_compound_50k/checkpoints_compound")
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
        if step % 200 == 0:
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
    elapsed = time.perf_counter() - t0
    print(
        f"[compound-train] epoch={epoch}/{epochs} "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"elapsed={elapsed/60:.1f}min"
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
        print(f"[compound-train] new best val_loss={val_loss:.4f}")

elapsed = time.perf_counter() - t0
print(f"[compound-train] done in {elapsed/60:.1f} min, best_val={best_val:.4f}")
PYEOF

[ $? -eq 0 ] || fail "Phase 1b"
pass "Phase 1b"

COMPOUND_CKPT="results/overnight_compound_50k/checkpoints_compound/compound_best.pt"
[ -f "$COMPOUND_CKPT" ] || fail "compound checkpoint not found at $COMPOUND_CKPT"
echo "Compound GPT checkpoint: $COMPOUND_CKPT"

# -- Phase 2: compound contrastive on 50k pairs --------------------------------
echo ""
echo "========================================"
echo "PHASE 2 — Compound CLAP (50k pairs, 20 epochs)"
echo "========================================"

python3 src/train_contrastive_compound.py \
    --captions-jsonl      data/midicaps_50k.jsonl \
    --compound-checkpoint $COMPOUND_CKPT \
    --results-dir         results/overnight_compound_50k \
    --epochs              20 \
    --batch-size          32 \
    --max-seq-len         512 \
    --num-workers         4 \
    --unfreeze-text-epoch 11 \
    --checkpoint-every    5 \
    --seed                17 \
    || fail "Phase 2"

pass "Phase 2"

CLAP_CKPT="results/overnight_compound_50k/checkpoints_contrastive_compound/clap_compound_best.pt"
[ -f "$CLAP_CKPT" ] || fail "clap checkpoint not found at $CLAP_CKPT"
echo "CLAP checkpoint: $CLAP_CKPT"

# -- Phase 3: prefix projector -------------------------------------------------
echo ""
echo "========================================"
echo "PHASE 3 — Prefix projector (50k pairs, 15 epochs)"
echo "========================================"

python3 src/train_prefix_compound.py \
    --captions-jsonl       data/midicaps_50k.jsonl \
    --compound-checkpoint  $COMPOUND_CKPT \
    --clap-checkpoint      $CLAP_CKPT \
    --results-dir          results/overnight_compound_50k \
    --epochs               15 \
    --batch-size           32 \
    --max-seq-len          504 \
    --num-workers          4 \
    --n-prefix-tokens      8 \
    --seed                 17 \
    || fail "Phase 3"

pass "Phase 3"

PREFIX_CKPT="results/overnight_compound_50k/checkpoints_prefix/prefix_projector_best.pt"
[ -f "$PREFIX_CKPT" ] || fail "prefix checkpoint not found at $PREFIX_CKPT"
echo "Prefix checkpoint: $PREFIX_CKPT"

# -- Phase 4: generation -------------------------------------------------------
echo ""
echo "========================================"
echo "PHASE 4 — Conditional generation (8 prompts)"
echo "========================================"

mkdir -p results/overnight_compound_50k/generated

PROMPTS=(
    "a slow melancholic piano piece in a minor key with sparse flowing notes"
    "an upbeat rock band with electric guitar bass and drums in a major key"
    "a fast jazz trio with saxophone piano and bass syncopated and energetic"
    "a soft ambient electronic piece with synthesizer pads slow and atmospheric"
    "a gentle classical piece for piano and strings moderate tempo expressive"
    "a funky horn driven arrangement with brass guitar and bass with strong rhythm"
    "a loud driving hard rock piece with heavy guitar and dense busy texture"
    "a sparse solo acoustic guitar piece fingerpicked quiet and introspective"
)

for i in "${!PROMPTS[@]}"; do
    echo "Generating prompt $((i+1))/8: ${PROMPTS[$i]}"
    python3 src/generate_conditional_compound.py \
        --compound-midi-checkpoint $COMPOUND_CKPT \
        --compound-clap-checkpoint $CLAP_CKPT \
        --prefix-checkpoint        $PREFIX_CKPT \
        --prompt                   "${PROMPTS[$i]}" \
        --max-new-steps            512 \
        --n-prefix-tokens          8 \
        --out                      "results/overnight_compound_50k/generated/prompt_${i}.mid" \
        || echo "WARNING: generation failed for prompt $i"
done

pass "Phase 4"

echo ""
echo "========================================"
echo "OVERNIGHT COMPOUND 50k COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo "GPT:    $COMPOUND_CKPT"
echo "CLAP:   $CLAP_CKPT"
echo "Prefix: $PREFIX_CKPT"
echo "MIDIs:  results/overnight_compound_50k/generated/"
