"""Download Metacreation/GigaMIDI from the Hugging Face Hub (gated dataset).

Requires a token: accept the dataset terms on the hub, then either
  - export HF_TOKEN=...  or  export HUGGING_FACE_HUB_TOKEN=...
  - or run `huggingface-cli login` once on the cluster
  - or pass --token (avoid in shared history; prefer env)

Usage (from repo root):
  python3 src/download.py
  python3 src/download.py --local-dir /scratch/$USER/gigamidi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import get_token, snapshot_download, whoami

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_LOCAL = _REPO_ROOT / "data" / "Final_GigaMIDI_V1.1_Final"


def _resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return get_token()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download GigaMIDI (gated) from Hugging Face",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=_DEFAULT_LOCAL,
        help=f"Download directory (default: {_DEFAULT_LOCAL})",
    )
    p.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (optional; prefer env or huggingface-cli login)",
    )
    args = p.parse_args()

    token = _resolve_token(args.token)
    if not token:
        print(
            "No Hugging Face token found.\n\n"
            "GigaMIDI is gated. Do one of the following on this machine:\n"
            "  1) export HF_TOKEN=hf_...   (or HUGGING_FACE_HUB_TOKEN)\n"
            "  2) huggingface-cli login\n"
            "  3) python3 src/download.py --token hf_...\n\n"
            "Create a read token at: https://huggingface.co/settings/tokens\n"
            "Accept the dataset at: https://huggingface.co/datasets/Metacreation/GigaMIDI",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        user = whoami(token=token)
        print(f"[download] authenticated as: {user}")
    except Exception as e:
        print(
            f"[download] whoami failed (check token / network): {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    args.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] repo=Metacreation/GigaMIDI -> {args.local_dir}")
    snapshot_download(
        repo_id="Metacreation/GigaMIDI",
        repo_type="dataset",
        local_dir=str(args.local_dir),
        token=token,
    )
    print("[download] done.")


if __name__ == "__main__":
    main()
