#!/bin/bash
#SBATCH --job-name=bach-compound-new-100
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=75G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/pg2963/bach-gpt/logs/pipeline_test_compound_new_100_%j.out
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/pg2963/bach-gpt
cd $ROOT || exit 1
source "$ROOT/.venv/bin/activate"

# Existing checkpoint + 100-sample data
CAPTIONS_JSONL=${CAPTIONS_JSONL:-data/midicaps_100.jsonl}
COMPOUND_MIDI_CKPT=${COMPOUND_MIDI_CKPT:-results/test_compound/checkpoints_compound/compound_best.pt}
RESULTS_DIR=${RESULTS_DIR:-results/test_compound_new_features_100}
N_PREFIX_TOKENS=${N_PREFIX_TOKENS:-8}

mkdir -p logs "$RESULTS_DIR" "$RESULTS_DIR/generated" "$RESULTS_DIR/checkpoints_prefix"

echo "========================================"
echo "Job ID:              $SLURM_JOB_ID"
echo "Node:                $SLURMD_NODENAME"
echo "Started:             $(date)"
echo "CAPTIONS_JSONL:      $CAPTIONS_JSONL"
echo "COMPOUND_MIDI_CKPT:  $COMPOUND_MIDI_CKPT"
echo "RESULTS_DIR:         $RESULTS_DIR"
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

[ -f "$CAPTIONS_JSONL" ] || fail "captions jsonl not found at $CAPTIONS_JSONL"
[ -f "$COMPOUND_MIDI_CKPT" ] || fail "compound checkpoint not found at $COMPOUND_MIDI_CKPT"

# -- Phase A: train compound contrastive from existing CompoundGPT --------------
echo ""
echo "========================================"
echo "PHASE A — train_contrastive_compound.py (100-sample smoke)"
echo "========================================"

python3 src/train_contrastive_compound.py \
    --captions-jsonl       "$CAPTIONS_JSONL" \
    --compound-checkpoint  "$COMPOUND_MIDI_CKPT" \
    --results-dir          "$RESULTS_DIR" \
    --epochs               2 \
    --batch-size           8 \
    --max-seq-len          256 \
    --num-workers          4 \
    --unfreeze-text-epoch  2 \
    --checkpoint-every     1 \
    || fail "Phase A"

pass "Phase A"

COMPOUND_CLAP_CKPT="$RESULTS_DIR/checkpoints_contrastive_compound/clap_compound_best.pt"
[ -f "$COMPOUND_CLAP_CKPT" ] || fail "compound clap checkpoint not found at $COMPOUND_CLAP_CKPT"
echo "Compound CLAP checkpoint: $COMPOUND_CLAP_CKPT"

# -- Phase B: smoke-test new Phase 3 compound helpers ---------------------------
echo ""
echo "========================================"
echo "PHASE B — prefix_projector compound helper smoke"
echo "========================================"

python3 - <<'PYEOF'
import torch
from pathlib import Path
import sys

sys.path.insert(0, "src")
from caption_dataloader import build_compound_caption_dataloaders
from prefix_projector import (
    load_phase3_compound_components,
    phase3_compound_prefix_lm_loss,
)

results_dir = Path("results/test_compound_new_features_100")
compound_midi_ckpt = Path("results/test_compound/checkpoints_compound/compound_best.pt")
compound_clap_ckpt = results_dir / "checkpoints_contrastive_compound" / "clap_compound_best.pt"
n_prefix_tokens = 8

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)
print(f"[phaseB] device={device}")

train_loader, _, _ = build_compound_caption_dataloaders(
    jsonl_path="data/midicaps_100.jsonl",
    max_seq_len=128,
    batch_size=4,
    split_ratio=0.95,
    seed=17,
    num_workers=0,
)
batch = next(iter(train_loader))
compound_input = batch["compound_input"].to(device)
captions = batch["captions"]
print(f"[phaseB] batch compound_input shape={tuple(compound_input.shape)}")

clap_model, compound_gpt, projector, counts = load_phase3_compound_components(
    compound_midi_checkpoint=str(compound_midi_ckpt),
    compound_clap_checkpoint=str(compound_clap_ckpt),
    n_prefix_tokens=n_prefix_tokens,
    device=device,
)
print(
    f"[phaseB] freeze check trainable={counts.n_total_trainable} "
    f"projector={counts.n_projector_params}"
)

loss, logits = phase3_compound_prefix_lm_loss(
    clap_model=clap_model,
    midi_compound_gpt=compound_gpt,
    prefix_projector=projector,
    compound_input=compound_input,
    captions=captions,
)
loss.backward()
grad_norm = float(projector.fc2.weight.grad.norm().item())
print(f"[phaseB] loss={loss.item():.4f} grad_norm(fc2)={grad_norm:.6f}")

prefix_ckpt = results_dir / "checkpoints_prefix" / "prefix_projector_compound_smoke.pt"
prefix_ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "projector_state_dict": projector.state_dict(),
        "args": {"n_prefix_tokens": n_prefix_tokens},
    },
    prefix_ckpt,
)
print(f"[phaseB] wrote {prefix_ckpt}")
PYEOF

[ $? -eq 0 ] || fail "Phase B"
pass "Phase B"

PREFIX_CKPT="$RESULTS_DIR/checkpoints_prefix/prefix_projector_compound_smoke.pt"
[ -f "$PREFIX_CKPT" ] || fail "prefix checkpoint not found at $PREFIX_CKPT"
echo "Prefix checkpoint: $PREFIX_CKPT"

# -- Phase C: compound conditional generation -----------------------------------
echo ""
echo "========================================"
echo "PHASE C — generate_conditional_compound.py"
echo "========================================"

python3 src/generate_conditional_compound.py \
    --compound-midi-checkpoint "$COMPOUND_MIDI_CKPT" \
    --compound-clap-checkpoint "$COMPOUND_CLAP_CKPT" \
    --prefix-checkpoint        "$PREFIX_CKPT" \
    --prompt                   "A bright fast piano etude with rising contour." \
    --max-new-steps            256 \
    --top-k                    30 \
    --temperature              0.9 \
    --n-prefix-tokens          "$N_PREFIX_TOKENS" \
    --out                      "$RESULTS_DIR/generated/compound_conditional_smoke.mid" \
    || fail "Phase C"

pass "Phase C"

echo ""
echo "========================================"
echo "COMPOUND NEW FEATURES (100-SAMPLE) PASSED"
echo "Finished: $(date)"
echo "========================================"
echo "Compound MIDI:  $COMPOUND_MIDI_CKPT"
echo "Compound CLAP:  $COMPOUND_CLAP_CKPT"
echo "Prefix:         $PREFIX_CKPT"
echo "Generated MIDI: $RESULTS_DIR/generated/compound_conditional_smoke.mid"
