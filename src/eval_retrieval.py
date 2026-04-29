"""Phase 4d retrieval evaluation on held-out validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from caption_dataloader import build_caption_dataloaders  # noqa: E402
from inference_pipeline import (  # noqa: E402
    _pick_device,
    load_clap,
    load_midi_gpt,
)


def _infer_genre_label(caption: str) -> str:
    text = caption.lower()
    if "jazz" in text or "swing" in text or "bebop" in text:
        return "jazz"
    if "electronic" in text or "synth" in text or "edm" in text:
        return "electronic"
    if "classical" in text or "orchestral" in text or "baroque" in text:
        return "classical"
    if "rock" in text or "guitar" in text or "band" in text:
        return "rock"
    return "other"


def _ranks_from_similarity(sim: torch.Tensor) -> torch.Tensor:
    """Return 1-indexed rank of correct pair for each row."""
    n = sim.size(0)
    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(n, device=sim.device).unsqueeze(1)
    matches = sorted_idx.eq(labels)
    rank0 = torch.argmax(matches.to(torch.int64), dim=1)
    return rank0 + 1


def _recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= k).float().mean().item())


def _median_rank(ranks: torch.Tensor) -> float:
    return float(torch.median(ranks.to(torch.float32)).item())


@torch.no_grad()
def collect_val_embeddings(
    clap,
    val_loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    midi_chunks: List[torch.Tensor] = []
    text_chunks: List[torch.Tensor] = []
    captions_all: List[str] = []

    clap.eval()
    clap.text_encoder.eval()

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        captions = batch["captions"]

        midi_feat = clap.encode_midi(input_ids, attention_mask)
        text_feat = clap.encode_text(captions, device=device)

        midi_emb = F.normalize(clap.midi_projection(midi_feat), p=2, dim=-1)
        text_emb = F.normalize(clap.text_projection(text_feat), p=2, dim=-1)

        midi_chunks.append(midi_emb.cpu())
        text_chunks.append(text_emb.cpu())
        captions_all.extend(captions)

    return (
        torch.cat(midi_chunks, dim=0),
        torch.cat(text_chunks, dim=0),
        captions_all,
    )


def evaluate_retrieval(
    midi_embs: torch.Tensor,
    text_embs: torch.Tensor,
) -> Dict[str, float]:
    sim = midi_embs @ text_embs.t()

    ranks_m2t = _ranks_from_similarity(sim)
    ranks_t2m = _ranks_from_similarity(sim.t())

    out: Dict[str, float] = {
        "n_val": float(sim.size(0)),
        "random_r1": 1.0 / float(sim.size(0)),
        "m2t_r1": _recall_at_k(ranks_m2t, 1),
        "m2t_r5": _recall_at_k(ranks_m2t, 5),
        "m2t_r10": _recall_at_k(ranks_m2t, 10),
        "m2t_median_rank": _median_rank(ranks_m2t),
        "t2m_r1": _recall_at_k(ranks_t2m, 1),
        "t2m_r5": _recall_at_k(ranks_t2m, 5),
        "t2m_r10": _recall_at_k(ranks_t2m, 10),
        "t2m_median_rank": _median_rank(ranks_t2m),
    }
    return out


def genre_r1_breakdown(
    midi_embs: torch.Tensor,
    text_embs: torch.Tensor,
    captions: List[str],
    top_genres: List[str],
) -> Dict[str, float]:
    sim = midi_embs @ text_embs.t()
    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(sim.size(0)).unsqueeze(1)
    top1 = sorted_idx[:, :1]
    correct_top1 = top1.eq(labels).squeeze(1)

    genres = [_infer_genre_label(c) for c in captions]
    out: Dict[str, float] = {}
    for g in top_genres:
        idxs = [i for i, gg in enumerate(genres) if gg == g]
        if not idxs:
            out[g] = float("nan")
            continue
        mask = torch.tensor(idxs, dtype=torch.long)
        out[g] = float(correct_top1[mask].float().mean().item())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CLAP retrieval metrics.")
    p.add_argument(
        "--results-dir",
        type=str,
        default=str(_ROOT / "results"),
    )
    p.add_argument(
        "--captions-jsonl",
        type=str,
        default=str(_ROOT / "data" / "captions_llm.jsonl"),
    )
    p.add_argument(
        "--midi-checkpoint",
        type=str,
        default="",
    )
    p.add_argument(
        "--clap-checkpoint",
        type=str,
        default="",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--split-ratio", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--out-json",
        type=str,
        default="",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not args.midi_checkpoint:
        args.midi_checkpoint = str(
            results_dir / "checkpoints" / "best_model.pt"
        )
    if not args.clap_checkpoint:
        args.clap_checkpoint = str(
            results_dir / "checkpoints_contrastive" / "clap_best.pt"
        )
    if not args.out_json:
        args.out_json = str(results_dir / "retrieval_eval.json")
    device = _pick_device()
    print(f"[retrieval] device={device}")

    _, val_loader, stats = build_caption_dataloaders(
        jsonl_path=args.captions_jsonl,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(
        "[retrieval] val split size="
        f"{stats.n_val_records} (total={stats.n_total_records})"
    )

    midi_gpt, _ = load_midi_gpt(Path(args.midi_checkpoint), device=device)
    clap, _ = load_clap(
        Path(args.clap_checkpoint), midi_gpt=midi_gpt, device=device
    )

    midi_embs, text_embs, captions = collect_val_embeddings(
        clap=clap,
        val_loader=val_loader,
        device=device,
    )
    metrics = evaluate_retrieval(midi_embs=midi_embs, text_embs=text_embs)
    genre_r1 = genre_r1_breakdown(
        midi_embs=midi_embs,
        text_embs=text_embs,
        captions=captions,
        top_genres=["rock", "jazz", "classical", "electronic"],
    )
    result = {"overall": metrics, "genre_r1": genre_r1}

    print(
        "[retrieval] random_r1="
        f"{metrics['random_r1']:.6f} | "
        f"m2t R@1/5/10={metrics['m2t_r1']:.4f}/"
        f"{metrics['m2t_r5']:.4f}/{metrics['m2t_r10']:.4f} "
        f"medR={metrics['m2t_median_rank']:.1f}"
    )
    print(
        "[retrieval] t2m R@1/5/10="
        f"{metrics['t2m_r1']:.4f}/{metrics['t2m_r5']:.4f}/"
        f"{metrics['t2m_r10']:.4f} "
        f"medR={metrics['t2m_median_rank']:.1f}"
    )
    print(
        "[retrieval] genre R@1 "
        + " ".join(f"{k}:{v:.4f}" for k, v in genre_r1.items())
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[retrieval] wrote {out_path}")


if __name__ == "__main__":
    main()
