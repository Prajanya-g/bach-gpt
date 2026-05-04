#!/bin/bash
#SBATCH --job-name=bach-compound-10k-v2-p234
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=75G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pg2963/bach-gpt/logs/compound_10k_v2_p234_%j.out
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/pg2963/bach-gpt
cd $ROOT || exit 1

COMPOUND_CKPT="results/compound_10k_v2/checkpoints_compound/compound_best.pt"
RESULTS_DIR="results/compound_10k_v2"
CAPTIONS_JSONL="data/midicaps_10k.jsonl"

mkdir -p logs "$RESULTS_DIR/generated"

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "GPT:       $COMPOUND_CKPT"
echo "========================================"

fail() { echo "FAILED at: $1"; exit 1; }
pass() { echo "PASSED: $1  ($(date))"; echo "----------------------------------------"; }

[ -f "$COMPOUND_CKPT" ]  || fail "compound checkpoint not found"
[ -f "$CAPTIONS_JSONL" ] || fail "captions jsonl not found"

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
echo "COMPOUND 10k v2 COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo "GPT:    $COMPOUND_CKPT"
echo "CLAP:   $CLAP_CKPT"
echo "Prefix: $PREFIX_CKPT"
echo "MIDIs:  $RESULTS_DIR/generated/"
