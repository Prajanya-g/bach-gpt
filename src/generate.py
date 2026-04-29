"""Autoregressive MIDI token generation from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pretty_midi
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bpe import (  # noqa: E402
    Merge,
    apply_bpe,
    load as load_bpe_merges,
    unapply_bpe,
)
from model import GPT, GPTConfig, default_gpt_config  # noqa: E402
from tokenizer import ID2TOKEN, decode, encode  # noqa: E402

DEFAULT_BPE_MERGES_PATH = _ROOT / "data" / "bpe" / "merges.json"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only top-k logits per row and mask the rest."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _extract_gpt_config_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(GPTConfig.__dataclass_fields__.keys())
    return {k: raw[k] for k in keys if k in raw}


def _load_config_from_sources(
    checkpoint_data: Dict[str, Any], config_path: str
) -> GPTConfig:
    cfg = default_gpt_config()

    ckpt_cfg = checkpoint_data.get("config")
    if isinstance(ckpt_cfg, dict):
        for k, v in _extract_gpt_config_dict(ckpt_cfg).items():
            setattr(cfg, k, v)

    if config_path:
        loaded = json.loads(Path(config_path).read_text())
        if not isinstance(loaded, dict):
            raise ValueError("--config must point to a JSON object.")
        for k, v in _extract_gpt_config_dict(loaded).items():
            setattr(cfg, k, v)

    return cfg


def _load_jsb_prompt(seed: int) -> Tuple[List[int], str]:
    try:
        from music21 import corpus
    except Exception as e:
        raise RuntimeError(
            "JSB prompt mode requires music21 to be installed."
        ) from e

    rng = random.Random(seed)
    all_scores = list(
        corpus.chorales.Iterator(
            numberingSystem="bwv",
            returnType="stream",
        )
    )
    if not all_scores:
        raise RuntimeError("No JSB chorales found via music21 corpus.")

    idx = rng.randrange(len(all_scores))
    score = all_scores[idx]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        score.write("midi", fp=tmp.name)
        pm = pretty_midi.PrettyMIDI(tmp.name)

    return encode(pm), f"jsb chorale #{idx}"


def _load_prompt_tokens(
    prompt: str,
    prompt_tokens: int,
    seed: int,
    merges: Sequence[Merge],
    vocab_size: int,
) -> Tuple[List[int], str]:
    if prompt == "random":
        rng = random.Random(seed)
        ids = [rng.randrange(vocab_size) for _ in range(prompt_tokens)]
        return ids, "random"

    if prompt == "jsb":
        ids, label = _load_jsb_prompt(seed=seed)
        if merges:
            ids = apply_bpe(ids, merges)
        return ids[:prompt_tokens], label

    midi_path = Path(prompt)
    if not midi_path.exists():
        raise FileNotFoundError(f"Prompt MIDI not found: {midi_path}")
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    ids = encode(pm)
    if merges:
        ids = apply_bpe(ids, merges)
    return ids[:prompt_tokens], str(midi_path)


@torch.no_grad()
def generate_tokens(
    model: GPT,
    prompt_ids: torch.Tensor,
    gen_tokens: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    generated = prompt_ids
    for _ in range(gen_tokens):
        context = generated[:, -model.config.block_size:]
        logits = model(context)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


def _token_text(ids: List[int], max_len: int = 120) -> str:
    toks = [ID2TOKEN.get(i, f"UNK({i})") for i in ids[:max_len]]
    suffix = " ..." if len(ids) > max_len else ""
    return " ".join(toks) + suffix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate MIDI from a trained GPT checkpoint"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_ROOT / "results" / "checkpoints" / "best_model.pt"),
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON config path; overrides checkpoint config.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="random",
        help='Prompt source: "jsb", "random", or path to .mid/.midi file.',
    )
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "results" / "generated.mid"),
    )
    p.add_argument(
        "--bpe-merges",
        type=str,
        default=str(DEFAULT_BPE_MERGES_PATH),
        help="BPE merges JSON path. Skipped silently if file missing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _pick_device()
    print(f"[generate] device={device}")

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    cfg = _load_config_from_sources(ckpt, args.config)
    model = GPT(cfg).to(device)

    state = (
        ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    )
    model.load_state_dict(state)
    model.eval()

    merges_path = Path(args.bpe_merges)
    merges: List[Merge] = (
        load_bpe_merges(merges_path) if merges_path.exists() else []
    )
    if merges:
        print(f"[generate] BPE merges loaded: {len(merges)} from {merges_path}")

    prompt_ids_list, prompt_label = _load_prompt_tokens(
        prompt=args.prompt,
        prompt_tokens=args.prompt_tokens,
        seed=args.seed,
        merges=merges,
        vocab_size=cfg.vocab_size,
    )
    if not prompt_ids_list:
        raise ValueError("Prompt produced zero tokens.")

    x = torch.tensor([prompt_ids_list], dtype=torch.long, device=device)
    out_ids = generate_tokens(
        model=model,
        prompt_ids=x,
        gen_tokens=args.gen_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()

    prompt_len = len(prompt_ids_list)
    cont_ids = out_ids[prompt_len:]

    base_out_ids = unapply_bpe(out_ids, merges) if merges else out_ids
    base_cont_ids = unapply_bpe(cont_ids, merges) if merges else cont_ids

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decode(base_out_ids).write(str(out_path))

    cont_path = out_path.with_name(
        f"{out_path.stem}_continuation{out_path.suffix}"
    )
    if base_cont_ids:
        decode(base_cont_ids).write(str(cont_path))

    print(f"[generate] prompt: {prompt_len} tokens ({prompt_label})")
    print(f"[generate] generated: {len(cont_ids)} tokens")
    print(f"[generate] temperature={args.temperature}, top_k={args.top_k}")
    print(f"[generate] output -> {out_path}")
    if cont_ids:
        print(f"[generate] continuation_only -> {cont_path}")
    print("[generate] token preview:")
    print(_token_text(out_ids))


if __name__ == "__main__":
    main()
