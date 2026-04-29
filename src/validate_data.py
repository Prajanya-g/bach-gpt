"""Quick validator for local GigaMIDI data layout.

Checks the folder structure expected by this repo's scripts:
- data/Final_GigaMIDI_V1.1_Final/
- data/Final_GigaMIDI_V1.1_Final/training-V1.1-80%/all-instruments-with-drums.zip

Usage:
  python3 src/validate_data.py
  python3 src/validate_data.py --data-root /scratch/$USER/gigamidi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def _count_midis(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob("*.mid"))) + len(list(path.rglob("*.midi")))


def validate_layout(data_root: Path) -> int:
    giga_root = data_root / "Final_GigaMIDI_V1.1_Final"
    train_outer_zip = giga_root / "training-V1.1-80%.zip"
    val_outer_zip = giga_root / "validation-V1.1-10%.zip"
    test_outer_zip = giga_root / "test-V1.1-10%.zip"

    train_dir = giga_root / "training-V1.1-80%"
    all_inst_zip = train_dir / "all-instruments-with-drums.zip"
    drums_zip = train_dir / "drums-only.zip"
    nodrums_zip = train_dir / "no-drums.zip"

    sample_dir = data_root / "gigamidi" / "sample"

    checks = [
        ("data root", data_root.exists()),
        ("Final_GigaMIDI_V1.1_Final directory", giga_root.exists()),
        ("training-V1.1-80%.zip", train_outer_zip.exists()),
        ("validation-V1.1-10%.zip", val_outer_zip.exists()),
        ("test-V1.1-10%.zip", test_outer_zip.exists()),
        ("training-V1.1-80% directory", train_dir.exists()),
        ("all-instruments-with-drums.zip", all_inst_zip.exists()),
        ("drums-only.zip", drums_zip.exists()),
        ("no-drums.zip", nodrums_zip.exists()),
    ]

    print("[validate_data] Expected layout checks:")
    for label, ok in checks:
        print(f"  - {_status(ok):>7} : {label}")

    sample_count = _count_midis(sample_dir)
    print("\n[validate_data] Sample cache:")
    print(f"  - {_status(sample_dir.exists()):>7} : {sample_dir}")
    print(f"  - {'INFO':>7} : sample MIDI files found = {sample_count}")

    hard_requirements_ok = all(
        [
            data_root.exists(),
            giga_root.exists(),
            train_dir.exists(),
            all_inst_zip.exists(),
        ]
    )

    print("\n[validate_data] Result:")
    if hard_requirements_ok:
        print("  PASS: required training layout exists for corpus_stats.py.")
        return 0

    print("  FAIL: required training layout missing.")
    print(
        "  Needed: data/Final_GigaMIDI_V1.1_Final/training-V1.1-80%/"
        "all-instruments-with-drums.zip"
    )
    return 1


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent / "data"

    p = argparse.ArgumentParser(description="Validate GigaMIDI data folder layout")
    p.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        help=f"Data root to validate (default: {default_root})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(validate_layout(args.data_root))
