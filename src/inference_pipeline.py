"""Phase 4 inference pipeline assembly and smoke test.

Loads checkpoints in strict order:
1) MIDI GPT
2) CLAP model
3) Prefix projector
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from contrastive_model import MidiTextContrastiveModel  # noqa: E402
from model import GPT, GPTConfig, default_gpt_config  # noqa: E402
from prefix_projector import PrefixProjector, clap_text_for_prefix_projector  # noqa: E402
from tokenizer import PHRASE_START  # noqa: E402


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_gpt_config_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(GPTConfig.__dataclass_fields__.keys())
    return {k: raw[k] for k in keys if k in raw}


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict):
        return ckpt
    return {"model_state_dict": ckpt}


def _resolve_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    for key in (
        "model_state_dict",
        "projector_state_dict",
        "state_dict",
        "model",
    ):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def load_midi_gpt(
    midi_checkpoint: Path, device: torch.device
) -> Tuple[GPT, Dict[str, Any]]:
    ckpt = _load_checkpoint(midi_checkpoint, device)
    cfg = default_gpt_config()
    raw_cfg = ckpt.get("config")
    if isinstance(raw_cfg, dict):
        for k, v in _extract_gpt_config_dict(raw_cfg).items():
            setattr(cfg, k, v)

    model = GPT(cfg).to(device)
    model.load_state_dict(_resolve_state_dict(ckpt), strict=False)
    model.eval()
    return model, ckpt


def load_clap(
    clap_checkpoint: Path,
    midi_gpt: GPT,
    device: torch.device,
) -> Tuple[MidiTextContrastiveModel, Dict[str, Any]]:
    ckpt = _load_checkpoint(clap_checkpoint, device)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    clap = MidiTextContrastiveModel(
        midi_gpt=midi_gpt,
        text_model_name=args.get(
            "text_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embed_dim=args.get("embed_dim", 256),
        init_temperature=args.get("init_temperature", 0.07),
        min_temperature=args.get("min_temperature", 0.01),
        max_temperature=args.get("max_temperature", 1.0),
        device=device,
    )
    clap.load_state_dict(_resolve_state_dict(ckpt), strict=False)
    clap.eval()
    # Explicitly force sentence-transformer eval mode (dropout off).
    clap.text_encoder.eval()
    return clap, ckpt


def load_prefix_projector(
    prefix_checkpoint: Path,
    gpt_d_model: int,
    device: torch.device,
    n_prefix_tokens_override: Optional[int] = None,
) -> Tuple[PrefixProjector, Dict[str, Any]]:
    ckpt = _load_checkpoint(prefix_checkpoint, device)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    n_prefix_tokens = n_prefix_tokens_override
    if n_prefix_tokens is None:
        n_prefix_tokens = int(args.get("n_prefix_tokens", 8))

    projector = PrefixProjector(
        clap_embed_dim=256,
        gpt_d_model=gpt_d_model,
        n_prefix_tokens=n_prefix_tokens,
    ).to(device)
    projector.load_state_dict(_resolve_state_dict(ckpt), strict=False)
    projector.eval()
    return projector, ckpt


@torch.no_grad()
def smoke_test_forward(
    clap: MidiTextContrastiveModel,
    midi_gpt: GPT,
    projector: PrefixProjector,
    test_prompt: str,
    device: torch.device,
) -> None:
    text_emb = clap_text_for_prefix_projector(clap, [test_prompt], device)
    prefix_embeds = projector(text_emb)

    input_ids = torch.tensor([[PHRASE_START]], dtype=torch.long, device=device)
    token_embeds = midi_gpt.wte(input_ids)
    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
    seq_len = inputs_embeds.size(1)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0)

    logits = midi_gpt(inputs_embeds=inputs_embeds, position_ids=position_ids)
    print(
        "[phase4] smoke OK: "
        f"prefix={tuple(prefix_embeds.shape)} "
        f"inputs_embeds={tuple(inputs_embeds.shape)} "
        f"logits={tuple(logits.shape)}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load Phase 4 inference pipeline.")
    p.add_argument(
        "--midi-checkpoint",
        type=str,
        default=str(_ROOT / "results" / "checkpoints" / "best_model.pt"),
    )
    p.add_argument(
        "--clap-checkpoint",
        type=str,
        default=str(
            _ROOT / "results" / "checkpoints_contrastive" / "clap_best.pt"
        ),
    )
    p.add_argument(
        "--prefix-checkpoint",
        type=str,
        default=str(
            _ROOT
            / "results"
            / "checkpoints_prefix"
            / "prefix_projector_best.pt"
        ),
    )
    p.add_argument("--n-prefix-tokens", type=int, default=0)
    p.add_argument(
        "--test-prompt",
        type=str,
        default="A bright fast piano étude with rising melodic contour.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _pick_device()
    print(f"[phase4] device={device}")

    midi_gpt, _ = load_midi_gpt(Path(args.midi_checkpoint), device=device)
    clap, _ = load_clap(
        Path(args.clap_checkpoint),
        midi_gpt=midi_gpt,
        device=device,
    )
    override = None if args.n_prefix_tokens <= 0 else args.n_prefix_tokens
    projector, _ = load_prefix_projector(
        Path(args.prefix_checkpoint),
        gpt_d_model=midi_gpt.config.d_model,
        device=device,
        n_prefix_tokens_override=override,
    )

    # Entire inference path is eval + no_grad.
    midi_gpt.eval()
    clap.eval()
    clap.text_encoder.eval()
    projector.eval()

    smoke_test_forward(
        clap=clap,
        midi_gpt=midi_gpt,
        projector=projector,
        test_prompt=args.test_prompt,
        device=device,
    )


if __name__ == "__main__":
    main()
