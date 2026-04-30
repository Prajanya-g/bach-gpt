"""Pre-tokenize MIDI files with the compound tokenizer and cache chunks.

This is useful for preparing data while longer training jobs are running.
It writes train/val chunk tensors and a small JSON stats file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compound_dataset import (
    concat_sequences,
    chunk_compound_stream,
    load_encoded_compound_sequences,
    split_chunks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize MIDI set with compound tokenizer and cache chunks."
    )
    p.add_argument(
        "--sample-dir",
        type=str,
        default="data/lmd_sample_10000",
        help="Directory containing .mid/.midi files.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/compound_cache",
        help="Directory to write cached tensors + stats JSON.",
    )
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--split-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample dir not found: {sample_dir}")

    midi_paths = sorted(sample_dir.rglob("*.mid")) + sorted(
        sample_dir.rglob("*.midi")
    )
    sequences, n_failed = load_encoded_compound_sequences(sample_dir)
    stream = concat_sequences(sequences)
    chunks = chunk_compound_stream(stream, block_size=args.block_size)
    train_chunks, val_chunks = split_chunks(
        chunks, split_ratio=args.split_ratio, seed=args.seed
    )

    train_tensor = (
        torch.stack(train_chunks, dim=0)
        if train_chunks
        else torch.empty(0, args.block_size, 7, dtype=torch.long)
    )
    val_tensor = (
        torch.stack(val_chunks, dim=0)
        if val_chunks
        else torch.empty(0, args.block_size, 7, dtype=torch.long)
    )

    torch.save(train_tensor, out_dir / "compound_train_chunks.pt")
    torch.save(val_tensor, out_dir / "compound_val_chunks.pt")

    stats = {
        "sample_dir": str(sample_dir),
        "block_size": args.block_size,
        "split_ratio": args.split_ratio,
        "seed": args.seed,
        "n_files_seen": len(midi_paths),
        "n_files_encoded": len(sequences),
        "n_files_failed": n_failed,
        "n_steps_total": len(stream),
        "n_chunks_total": len(chunks),
        "n_train_chunks": len(train_chunks),
        "n_val_chunks": len(val_chunks),
        "train_tensor_shape": list(train_tensor.shape),
        "val_tensor_shape": list(val_tensor.shape),
    }
    (out_dir / "compound_cache_stats.json").write_text(json.dumps(stats, indent=2))

    print(
        "[compound-cache] files seen/encoded/failed: "
        f"{stats['n_files_seen']}/{stats['n_files_encoded']}/{stats['n_files_failed']}"
    )
    print(
        "[compound-cache] steps/chunks/train/val: "
        f"{stats['n_steps_total']}/{stats['n_chunks_total']}/"
        f"{stats['n_train_chunks']}/{stats['n_val_chunks']}"
    )
    print(
        "[compound-cache] wrote: "
        f"{out_dir / 'compound_train_chunks.pt'}, "
        f"{out_dir / 'compound_val_chunks.pt'}, "
        f"{out_dir / 'compound_cache_stats.json'}"
    )


if __name__ == "__main__":
    main()
