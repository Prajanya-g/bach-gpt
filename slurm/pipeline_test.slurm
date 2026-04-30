#!/bin/bash
#SBATCH --job-name=bach-gpt-test
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=75G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pg2963/bach-gpt/logs/pipeline_test_%j.out
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/pg2963/bach-gpt
source .venv/bin/activate
cd $ROOT || exit 1
mkdir -p logs results/test/checkpoints results/test/generated

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "========================================"

fail() {
    echo ""
    echo "FAILED at: $1"
    echo "See: /scratch/pg2963/bach-gpt/logs/pipeline_test_${SLURM_JOB_ID}.err"
    exit 1
}

pass() {
    echo "PASSED: $1  ($(date))"
    echo "----------------------------------------"
}

# ── Prepare 100-file flat sample dir for Phase 1 ─────────────────────────────

echo ""
echo "========================================"
echo "Preparing 100-file LMD sample dir"
echo "========================================"

python3 << 'PYEOF'
import random, shutil
from pathlib import Path

src = Path("data/midicaps/lmd_full")
dst = Path("data/lmd_sample_100")
dst.mkdir(exist_ok=True)

all_files = list(src.rglob("*.mid"))
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

# ── Phase 1 — train MIDI GPT on 100 LMD files ────────────────────────────────

echo ""
echo "========================================"
echo "PHASE 1 — MIDI GPT pretraining (100 files, 5 epochs)"
echo "========================================"

python3 src/train.py \
    --sample-dir  data/lmd_sample_100 \
    --max-epochs  5 \
    --batch-size  8 \
    --block-size  512 \
    --results-dir results/test \
    || fail "Phase 1"

pass "Phase 1"

MIDI_CKPT="results/test/checkpoints/best_model.pt"
[ -f "$MIDI_CKPT" ] || fail "Phase 1 checkpoint not found at $MIDI_CKPT"
echo "GPT checkpoint: $MIDI_CKPT"

# ── Phase 2 — CLAP contrastive training ──────────────────────────────────────

echo ""
echo "========================================"
echo "PHASE 2 — CLAP contrastive training (100 pairs, 2 epochs)"
echo "========================================"

python3 src/train_contrastive.py \
    --captions-jsonl      data/midicaps_100.jsonl \
    --midi-checkpoint     $MIDI_CKPT \
    --epochs              2 \
    --batch-size          8 \
    --num-workers         4 \
    --unfreeze-text-epoch 2 \
    --qualitative-every   2 \
    --results-dir         results/test \
    || fail "Phase 2"

pass "Phase 2"

CLAP_CKPT="results/test/checkpoints_contrastive/clap_best.pt"
[ -f "$CLAP_CKPT" ] || fail "Phase 2 checkpoint not found at $CLAP_CKPT"
echo "CLAP checkpoint: $CLAP_CKPT"

# ── Phase 3 — Prefix projector ────────────────────────────────────────────────

echo ""
echo "========================================"
echo "PHASE 3 — Prefix projector (100 pairs, 2 epochs)"
echo "========================================"

python3 src/train_prefix.py \
    --captions-jsonl  data/midicaps_100.jsonl \
    --midi-checkpoint $MIDI_CKPT \
    --clap-checkpoint $CLAP_CKPT \
    --epochs          2 \
    --batch-size      8 \
    --num-workers     4 \
    --max-seq-len     504 \
    --n-prefix-tokens 8 \
    --results-dir     results/test \
    || fail "Phase 3"

pass "Phase 3"

PREFIX_CKPT="results/test/checkpoints_prefix/prefix_projector_best.pt"
[ -f "$PREFIX_CKPT" ] || fail "Phase 3 checkpoint not found at $PREFIX_CKPT"
echo "Prefix checkpoint: $PREFIX_CKPT"

# ── Phase 4a — inference pipeline smoke test ─────────────────────────────────

echo ""
echo "========================================"
echo "PHASE 4a — inference pipeline smoke test"
echo "========================================"

python3 src/inference_pipeline.py \
    --midi-checkpoint   $MIDI_CKPT \
    --clap-checkpoint   $CLAP_CKPT \
    --prefix-checkpoint $PREFIX_CKPT \
    --test-prompt       "A slow melancholic piano piece in a minor key" \
    || fail "Phase 4a"

pass "Phase 4a"

# ── Phase 4b — conditional generation ────────────────────────────────────────

echo ""
echo "========================================"
echo "PHASE 4b — conditional generation"
echo "========================================"

python3 src/generate_conditional.py \
    --midi-checkpoint   $MIDI_CKPT \
    --clap-checkpoint   $CLAP_CKPT \
    --prefix-checkpoint $PREFIX_CKPT \
    --prompt            "A slow melancholic piano piece in a minor key" \
    --out               results/test/generated/test_output.mid \
    --max-new-tokens    128 \
    || fail "Phase 4b"

pass "Phase 4b"

# ── Phase 4c — retrieval evaluation ──────────────────────────────────────────

echo ""
echo "========================================"
echo "PHASE 4c — retrieval evaluation"
echo "========================================"

python3 src/eval_retrieval.py \
    --captions-jsonl  data/midicaps_100.jsonl \
    --midi-checkpoint $MIDI_CKPT \
    --clap-checkpoint $CLAP_CKPT \
    --batch-size      8 \
    --out-json        results/test/retrieval_eval.json \
    || fail "Phase 4c"

pass "Phase 4c"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "ALL PHASES PASSED — ready for overnight run"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Checkpoints:"
echo "  GPT:    $MIDI_CKPT"
echo "  CLAP:   $CLAP_CKPT"
echo "  Prefix: $PREFIX_CKPT"
echo ""
echo "Outputs:"
echo "  Generated MIDI: results/test/generated/test_output.mid"
echo "  Retrieval eval: results/test/retrieval_eval.json"
echo ""

python3 -c "
import json
from pathlib import Path
p = Path('results/test/retrieval_eval.json')
if p.exists():
    r = json.load(open(p))
    m = r['overall']
    print('Retrieval results:')
    print(f'  random R@1:   {m[\"random_r1\"]:.4f}')
    print(f'  m2t R@1/5/10: {m[\"m2t_r1\"]:.4f} / {m[\"m2t_r5\"]:.4f} / {m[\"m2t_r10\"]:.4f}')
    print(f'  t2m R@1/5/10: {m[\"t2m_r1\"]:.4f} / {m[\"t2m_r5\"]:.4f} / {m[\"t2m_r10\"]:.4f}')
    if 'genre_r1' in r:
        print('Genre R@1:')
        for g, v in r['genre_r1'].items():
            print(f'  {g}: {v:.4f}')
"
