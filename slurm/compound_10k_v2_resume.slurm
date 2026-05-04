#!/bin/bash
#SBATCH --job-name=bach-10k-v2-resume
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=75G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/pg2963/bach-gpt/logs/compound_10k_v2_resume_%j.out
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/pg2963/bach-gpt
cd $ROOT || exit 1

RESUME_CKPT="results/compound_10k_v2/checkpoints_compound/compound_epoch_014.pt"
RESULTS_DIR="results/compound_10k_v2"
CAPTIONS_JSONL="data/midicaps_10k.jsonl"

mkdir -p logs "$RESULTS_DIR/generated"

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "Resuming:  $RESUME_CKPT"
echo "========================================"

fail() { echo "FAILED at: $1"; exit 1; }
pass() { echo "PASSED: $1  ($(date))"; echo "----------------------------------------"; }

[ -f "$RESUME_CKPT" ] || fail "resume checkpoint not found"

# -- Phase 1 resume: epochs 15-20 ---------------------------------------------
echo ""
echo "========================================"
echo "PHASE 1 resume — epochs 15-20"
echo "========================================"

python3 << 'PYEOF'
import sys, time, math
from pathlib import Path
import torch

sys.path.insert(0, "src")
from compound_dataset import build_compound_dataloaders
from compound_model import CompoundGPT, CompoundGPTConfig, default_compound_config, compound_loss

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

# load checkpoint
ckpt_path = Path("results/compound_10k_v2/checkpoints_compound/compound_epoch_014.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

# rebuild config from checkpoint
cfg = default_compound_config()
raw_cfg = ckpt.get("config", {})
for k in CompoundGPTConfig.__dataclass_fields__.keys():
    if k in raw_cfg:
        setattr(cfg, k, raw_cfg[k])

model = CompoundGPT(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
n_params = sum(p.numel() for p in model.parameters())
print(f"[compound-train] parameters={n_params:,} (~{n_params/1e6:.1f}M)")
print(f"[compound-train] resuming from epoch={ckpt['epoch']} val_loss={ckpt['val_loss']:.4f}")

start_epoch = ckpt["epoch"] + 1
total_epochs = 20
accum_steps  = 2
total_steps  = total_epochs * (len(train_loader) // accum_steps)
warmup       = 500

opt = torch.optim.AdamW(
    model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
)

def get_lr(step):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)

# fast-forward scheduler to match where we left off
steps_done = (start_epoch - 1) * (len(train_loader) // accum_steps)
for _ in range(steps_done):
    scheduler.step()

best_val = ckpt["val_loss"]
ckpt_dir = Path("results/compound_10k_v2/checkpoints_compound")

t0 = time.perf_counter()
global_step = steps_done

for epoch in range(start_epoch, total_epochs + 1):
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
        f"[compound-train] epoch={epoch}/{total_epochs} "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"val_ppl={math.exp(val_loss):.2f} elapsed={elapsed/60:.1f}min"
    )

    ckpt_out = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(ckpt_out, ckpt_dir / "compound_latest.pt")
    torch.save(ckpt_out, ckpt_dir / f"compound_epoch_{epoch:03d}.pt")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(ckpt_out, ckpt_dir / "compound_best.pt")
        print(f"[compound-train] new best val_loss={val_loss:.4f}")

elapsed = time.perf_counter() - t0
print(f"[compound-train] done in {elapsed/60:.1f} min, best_val={best_val:.4f}")
PYEOF

[ $? -eq 0 ] || fail "Phase 1 resume"
pass "Phase 1 resume"

COMPOUND_CKPT="results/compound_10k_v2/checkpoints_compound/compound_best.pt"
[ -f "$COMPOUND_CKPT" ] || fail "compound best checkpoint not found"
echo "GPT checkpoint: $COMPOUND_CKPT"

# -- Phase 2: CLAP -------------------------------------------------------------
echo ""
echo "========================================"
echo "PHASE 2 — Compound CLAP (10k, 20 epochs)"
echo "========================================"

python3 src/train_contrastive_compound.py \
    --captions-jsonl       $CAPTIONS_JSONL \
    --compound-checkpoint  $COMPOUND_CKPT \
    --results-dir          $RESULTS_DIR \
    --epochs               20 \
    --batch-size           32 \
    --max-seq-len          512 \
    --num-workers          4 \
    --unfreeze-text-epoch  11 \
    --checkpoint-every     5 \
    --seed                 17 \
    || fail "Phase 2"

pass "Phase 2"

CLAP_CKPT="$RESULTS_DIR/checkpoints_contrastive_compound/clap_compound_best.pt"
[ -f "$CLAP_CKPT" ] || fail "clap checkpoint not found"
echo "CLAP: $CLAP_CKPT"

# -- Phase 3: prefix -----------------------------------------------------------
echo ""
echo "========================================"
echo "PHASE 3 — Prefix projector (10k, 15 epochs)"
echo "========================================"

python3 src/train_prefix_compound.py \
    --compound-checkpoint  $COMPOUND_CKPT \
    --clap-checkpoint      $CLAP_CKPT \
    --captions-jsonl       $CAPTIONS_JSONL \
    --results-dir          $RESULTS_DIR \
    --epochs               15 \
    --batch-size           32 \
    --max-seq-len          504 \
    --num-workers          4 \
    --n-prefix-tokens      8 \
    --seed                 17 \
    || fail "Phase 3"

pass "Phase 3"

PREFIX_CKPT="$RESULTS_DIR/checkpoints_prefix/prefix_projector_best.pt"
[ -f "$PREFIX_CKPT" ] || fail "prefix checkpoint not found"
echo "Prefix: $PREFIX_CKPT"

# -- Phase 4: generation -------------------------------------------------------
echo ""
echo "========================================"
echo "PHASE 4 — Generation (8 prompts x 2 temps)"
echo "========================================"

PROMPTS=(
    "a slow melancholic piano piece in a minor key with sparse flowing notes"
    "an upbeat jazz trio with piano bass and drums syncopated and energetic"
    "ambient electronic music with synthesizer pads slow and atmospheric"
    "fast energetic rock band with electric guitar and drums"
    "a gentle classical piece for piano and strings moderate tempo"
    "a funky groove with bass guitar and brass instruments"
    "a soft acoustic guitar piece fingerpicked quiet and introspective"
    "an orchestral piece with strings and brass building to a climax"
)

for i in "${!PROMPTS[@]}"; do
    for temp in 0.8 1.1; do
        tname=$(echo $temp | tr '.' '_')
        OUT="$RESULTS_DIR/generated/prompt_${i}_temp${tname}.mid"
        echo "Generating prompt $((i+1))/8 temp=$temp"
        python3 src/generate_conditional_compound.py \
            --compound-midi-checkpoint $COMPOUND_CKPT \
            --compound-clap-checkpoint $CLAP_CKPT \
            --prefix-checkpoint        $PREFIX_CKPT \
            --prompt                   "${PROMPTS[$i]}" \
            --max-new-steps            500 \
            --temperature              $temp \
            --top-k                    40 \
            --top-p                    0.92 \
            --n-prefix-tokens          8 \
            --out                      "$OUT" \
            || echo "WARNING: failed prompt $i temp $temp"
    done
done

pass "Phase 4"

echo ""
echo "========================================"
echo "COMPOUND 10k v2 RESUME COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo "GPT:    $COMPOUND_CKPT"
echo "CLAP:   $CLAP_CKPT"
echo "Prefix: $PREFIX_CKPT"
echo "MIDIs:  $RESULTS_DIR/generated/"
