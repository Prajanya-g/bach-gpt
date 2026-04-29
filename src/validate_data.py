"""Quick validator for local GigaMIDI data layout.

Supports common layouts:
- v1.1 local unzip: training-V1.1-80%/all-instruments-with-drums.zip or that folder
- Hugging Face snapshot: Final_GigaMIDI_V1.1_Final/all-instruments-with-drums/

Usage:
  python3 src/validate_data.py
  python3 src/validate_data.py --data-root /scratch/$USER/bach-gpt/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def _count_midis(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob("*.mid"))) + len(list(path.rglob("*.midi")))


def _find_all_instruments_dir(giga_root: Path) -> Optional[Path]:
    candidates = [
        giga_root / "training-V1.1-80%" / "all-instruments-with-drums",
        giga_root / "all-instruments-with-drums",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    for child in giga_root.iterdir():
        if child.is_dir() and child.name.strip() == "all-instruments-with-drums":
            return child
    return None


def validate_layout(data_root: Path) -> int:
    giga_root = data_root / "Final_GigaMIDI_V1.1_Final"
    train_outer_zip = giga_root / "training-V1.1-80%.zip"
    val_outer_zip = giga_root / "validation-V1.1-10%.zip"
    test_outer_zip = giga_root / "test-V1.1-10%.zip"

    train_dir = giga_root / "training-V1.1-80%"
    nested_zip = train_dir / "all-instruments-with-drums.zip"
    flat_zip = giga_root / "all-instruments-with-drums.zip"
    all_inst_dir = _find_all_instruments_dir(giga_root)
    train_midi_count = _count_midis(all_inst_dir) if all_inst_dir else 0

    drums_zip = train_dir / "drums-only.zip"
    nodrums_zip = train_dir / "no-drums.zip"

    sample_dir = data_root / "gigamidi" / "sample"

    has_zip = nested_zip.exists() or flat_zip.exists()
    has_dir_midis = train_midi_count > 0
    has_train_source = has_zip or has_dir_midis

    checks = [
        ("data root", data_root.exists()),
        ("Final_GigaMIDI_V1.1_Final directory", giga_root.exists()),
        ("training-V1.1-80%.zip (outer split)", train_outer_zip.exists()),
        ("validation-V1.1-10%.zip (outer split)", val_outer_zip.exists()),
        ("test-V1.1-10%.zip (outer split)", test_outer_zip.exists()),
        ("training-V1.1-80% directory (v1.1 nested)", train_dir.exists()),
        ("training-V1.1-80%/all-instruments-with-drums.zip", nested_zip.exists()),
        ("Final_.../all-instruments-with-drums.zip (flat)", flat_zip.exists()),
        ("all-instruments-with-drums/ (extracted)", all_inst_dir is not None),
        ("MIDI files under all-instruments-with-drums", has_dir_midis),
        ("drums-only.zip (optional)", drums_zip.exists()),
        ("no-drums.zip (optional)", nodrums_zip.exists()),
    ]

    print("[validate_data] Layout checks:")
    for label, ok in checks:
        print(f"  - {_status(ok):>7} : {label}")

    if all_inst_dir and not has_dir_midis:
        print(
            f"\n[validate_data] NOTE: found directory {all_inst_dir} "
            "but no .mid/.midi files (still unzipping or empty?)."
        )

    sample_count = _count_midis(sample_dir)
    print("\n[validate_data] Sample cache:")
    print(f"  - {_status(sample_dir.exists()):>7} : {sample_dir}")
    print(f"  - {'INFO':>7} : sample MIDI files found = {sample_count}")

    hard_requirements_ok = data_root.exists() and giga_root.exists() and (
        has_train_source
    )

    print("\n[validate_data] Result:")
    if hard_requirements_ok:
        if nested_zip.exists():
            print("  PASS: nested zip (v1.1 layout).")
        elif flat_zip.exists():
            print("  PASS: flat all-instruments-with-drums.zip (Hub / flat layout).")
        else:
            print("  PASS: extracted all-instruments-with-drums/ with MIDI files.")
        return 0

    print("  FAIL: no usable GigaMIDI training source.")
    print("  Need one of:")
    print(
        "    - .../training-V1.1-80%/all-instruments-with-drums.zip\n"
        "    - .../all-instruments-with-drums.zip\n"
        "    - .../all-instruments-with-drums/ (recursive .mid/.midi)\n"
        "    (folder name may have stray spaces; we match strip().)"
    )
    return 1


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent / "data"

    p = argparse.ArgumentParser(
        description="Validate GigaMIDI data folder layout",
    )
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
