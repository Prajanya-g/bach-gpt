#!/bin/bash
#SBATCH --job-name=bach-phase2-to-gen
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=75G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pg2963/bach-gpt/logs/pipeline_phase2_to_gen_%j.out
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/pg2963/bach-gpt
cd $ROOT || exit 1
source "$ROOT/.venv/bin/activate"

# ------------------------------------------------------------------------------
# Reusable config (override via environment variables at submit time)
# Example:
#   sbatch --export=ALL,CAPTIONS_JSONL=data/midicaps_100.jsonl,RESULTS_DIR=results/test_resume slurm/pipeline_phase2_to_gen.slurm
# ------------------------------------------------------------------------------
CAPTIONS_JSONL=${CAPTIONS_JSONL:-data/midicaps_100.jsonl}
RESULTS_DIR=${RESULTS_DIR:-results/test_resume}
MIDI_CKPT=${MIDI_CKPT:-results/test/checkpoints/best_model.pt}
PROMPT=${PROMPT:-"A slow melancholic piano piece in a minor key"}
GEN_OUT=${GEN_OUT:-$RESULTS_DIR/generated/conditional_output.mid}

# Contrastive settings
PHASE2_EPOCHS=${PHASE2_EPOCHS:-2}
PHASE2_BATCH_SIZE=${PHASE2_BATCH_SIZE:-8}
PHASE2_NUM_WORKERS=${PHASE2_NUM_WORKERS:-4}
PHASE2_UNFREEZE_TEXT_EPOCH=${PHASE2_UNFREEZE_TEXT_EPOCH:-2}
PHASE2_MAX_SEQ_LEN=${PHASE2_MAX_SEQ_LEN:-512}

# Prefix settings
PHASE3_EPOCHS=${PHASE3_EPOCHS:-2}
PHASE3_BATCH_SIZE=${PHASE3_BATCH_SIZE:-8}
PHASE3_NUM_WORKERS=${PHASE3_NUM_WORKERS:-4}
PHASE3_N_PREFIX_TOKENS=${PHASE3_N_PREFIX_TOKENS:-8}
PHASE3_MAX_SEQ_LEN=${PHASE3_MAX_SEQ_LEN:-504}

# Generation settings
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-128}

mkdir -p logs "$RESULTS_DIR" "$RESULTS_DIR/generated"

echo "========================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $SLURMD_NODENAME"
echo "Started:        $(date)"
echo "CAPTIONS_JSONL: $CAPTIONS_JSONL"
echo "RESULTS_DIR:    $RESULTS_DIR"
echo "MIDI_CKPT:      $MIDI_CKPT"
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
[ -f "$MIDI_CKPT" ] || fail "midi checkpoint not found at $MIDI_CKPT"

# -- Phase 2: contrastive -------------------------------------------------------
echo ""
echo "========================================"
echo "PHASE 2 — MidiTextContrastiveModel training"
echo "========================================"

python3 src/train_contrastive.py \
    --captions-jsonl      "$CAPTIONS_JSONL" \
    --midi-checkpoint     "$MIDI_CKPT" \
    --epochs              "$PHASE2_EPOCHS" \
    --batch-size          "$PHASE2_BATCH_SIZE" \
    --num-workers         "$PHASE2_NUM_WORKERS" \
    --unfreeze-text-epoch "$PHASE2_UNFREEZE_TEXT_EPOCH" \
    --max-seq-len         "$PHASE2_MAX_SEQ_LEN" \
    --qualitative-every   "$PHASE2_EPOCHS" \
    --results-dir         "$RESULTS_DIR" \
    || fail "Phase 2"

pass "Phase 2"

CLAP_CKPT="$RESULTS_DIR/checkpoints_contrastive/clap_best.pt"
[ -f "$CLAP_CKPT" ] || fail "clap checkpoint not found at $CLAP_CKPT"
echo "CLAP checkpoint: $CLAP_CKPT"

# -- Phase 3: prefix projector --------------------------------------------------
echo ""
echo "========================================"
echo "PHASE 3 — Prefix projector"
echo "========================================"

python3 src/train_prefix.py \
    --captions-jsonl  "$CAPTIONS_JSONL" \
    --midi-checkpoint "$MIDI_CKPT" \
    --clap-checkpoint "$CLAP_CKPT" \
    --epochs          "$PHASE3_EPOCHS" \
    --batch-size      "$PHASE3_BATCH_SIZE" \
    --num-workers     "$PHASE3_NUM_WORKERS" \
    --max-seq-len     "$PHASE3_MAX_SEQ_LEN" \
    --n-prefix-tokens "$PHASE3_N_PREFIX_TOKENS" \
    --results-dir     "$RESULTS_DIR" \
    || fail "Phase 3"

pass "Phase 3"

PREFIX_CKPT="$RESULTS_DIR/checkpoints_prefix/prefix_projector_best.pt"
[ -f "$PREFIX_CKPT" ] || fail "prefix checkpoint not found at $PREFIX_CKPT"
echo "Prefix checkpoint: $PREFIX_CKPT"

# -- Phase 4b: conditional generation ------------------------------------------
echo ""
echo "========================================"
echo "PHASE 4b — Conditional generation"
echo "========================================"

python3 src/generate_conditional.py \
    --midi-checkpoint   "$MIDI_CKPT" \
    --clap-checkpoint   "$CLAP_CKPT" \
    --prefix-checkpoint "$PREFIX_CKPT" \
    --prompt            "$PROMPT" \
    --out               "$GEN_OUT" \
    --max-new-tokens    "$GEN_MAX_NEW_TOKENS" \
    || fail "Phase 4b"

pass "Phase 4b"

echo ""
echo "========================================"
echo "PHASE 2 -> PREFIX -> GENERATION PASSED"
echo "Finished: $(date)"
echo "========================================"
echo "MIDI:   $MIDI_CKPT"
echo "CLAP:   $CLAP_CKPT"
echo "Prefix: $PREFIX_CKPT"
echo "MIDI out: $GEN_OUT"
