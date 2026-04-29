"""Extract a random MIDI sample from GigaMIDI parquet shards.

This script converts rows from:
  data/Final_GigaMIDI_V1.1_Final/all-instruments-with-drums/train.parquet
into:
  data/gigamidi/sample/<md5>.mid

Each parquet row stores raw MIDI bytes in the ``music`` column and an ``md5``
identifier. We sample rows by index, write bytes to .mid files, and skip any
invalid records.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq


def _default_input_parquet() -> Path:
    root = Path(__file__).resolve().parent.parent
    return (
        root
        / "data"
        / "Final_GigaMIDI_V1.1_Final"
        / "all-instruments-with-drums"
        / "train.parquet"
    )


def _default_output_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "gigamidi" / "sample"


def extract_sample(
    parquet_path: Path,
    out_dir: Path,
    n_samples: int,
    seed: int,
    overwrite: bool,
) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    if total_rows == 0:
        raise ValueError("Parquet file has zero rows.")

    n_pick = min(n_samples, total_rows)
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_rows), n_pick))
    selected_set = set(selected)

    print(f"[extract] parquet: {parquet_path}")
    print(
        f"[extract] total_rows={total_rows:,} "
        f"n_samples={n_pick:,} seed={seed}"
    )
    print(f"[extract] output_dir: {out_dir}")

    written = 0
    skipped_existing = 0
    skipped_invalid = 0

    offset = 0
    for rg_idx in range(pf.num_row_groups):
        rg_meta = pf.metadata.row_group(rg_idx)
        rg_rows = rg_meta.num_rows
        rg_start = offset
        rg_end = offset + rg_rows
        offset = rg_end

        in_group = [i for i in selected if rg_start <= i < rg_end]
        if not in_group:
            continue

        table = pf.read_row_group(rg_idx, columns=["md5", "music"])
        md5_col = table.column("md5")
        music_col = table.column("music")

        for global_idx in in_group:
            local_idx = global_idx - rg_start
            md5 = md5_col[local_idx].as_py()
            music = music_col[local_idx].as_py()

            if not md5 or not music:
                skipped_invalid += 1
                continue

            out_path = out_dir / f"{md5}.mid"
            if out_path.exists() and not overwrite:
                skipped_existing += 1
                continue

            if isinstance(music, memoryview):
                payload = music.tobytes()
            elif isinstance(music, (bytes, bytearray)):
                payload = bytes(music)
            else:
                skipped_invalid += 1
                continue

            out_path.write_bytes(payload)
            written += 1

    print(
        "[extract] complete: "
        f"written={written} "
        f"skipped_existing={skipped_existing} "
        f"skipped_invalid={skipped_invalid}"
    )
    print(
        f"[extract] expected_total={len(selected_set)} actual_handled="
        f"{written + skipped_existing + skipped_invalid}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract random .mid files from GigaMIDI parquet rows."
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=_default_input_parquet(),
        help="Path to train.parquet",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_default_output_dir(),
        help="Output directory for .mid files",
    )
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .mid files if present",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_sample(
        parquet_path=args.parquet,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        overwrite=args.overwrite,
    )
