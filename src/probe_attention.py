"""Attention probing for bach-gpt on JSB chorales.

Outputs:
- Per-head attention heatmaps:
  figures/attention/attention_layer{L}_head{H}.png
- Summary charts:
  figures/attention/attention_entropy_by_head.png
  figures/attention/attention_distance_by_head.png
- Text report:
  results/attention_probe_summary.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import pretty_midi
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from model import GPT, GPTConfig, default_gpt_config  # noqa: E402
from tokenizer import BAR_START, ID2TOKEN, PHRASE_START, encode  # noqa: E402


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


def _load_config_from_sources(
    ckpt: Dict[str, Any], config_path: str
) -> GPTConfig:
    cfg = default_gpt_config()
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        for k, v in _extract_gpt_config_dict(ckpt_cfg).items():
            setattr(cfg, k, v)
    if config_path:
        loaded = json.loads(Path(config_path).read_text())
        if not isinstance(loaded, dict):
            raise ValueError("--config must be a JSON object.")
        for k, v in _extract_gpt_config_dict(loaded).items():
            setattr(cfg, k, v)
    return cfg


def _token_kind(token_id: int) -> str:
    tok = ID2TOKEN.get(token_id, "")
    if tok.startswith("P"):
        return "pitch"
    if tok.startswith("D"):
        return "duration"
    if tok.startswith("TS"):
        return "timeshift"
    if tok.startswith("V"):
        return "velocity"
    if tok == "BAR_START":
        return "bar_start"
    if tok == "PHRASE_START":
        return "phrase_start"
    return "structural"


def _load_jsb_sequences(
    n_files: int,
    seq_len: int,
    seed: int,
) -> List[List[int]]:
    from music21 import corpus

    rng = random.Random(seed)
    chorales = list(
        corpus.chorales.Iterator(
            numberingSystem="bwv",
            returnType="stream",
        )
    )
    rng.shuffle(chorales)

    seqs: List[List[int]] = []
    for score in chorales:
        if len(seqs) >= n_files:
            break
        try:
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
                score.write("midi", fp=tmp.name)
                pm = pretty_midi.PrettyMIDI(tmp.name)
            ids = encode(pm)
            if len(ids) < seq_len:
                continue
            seqs.append(ids[:seq_len])
        except Exception:
            continue
    return seqs


def _plot_heatmaps(mean_attn: torch.Tensor, out_dir: Path) -> None:
    n_layers, n_heads, _, _ = mean_attn.shape
    for layer in range(n_layers):
        for head in range(n_heads):
            arr = mean_attn[layer, head].cpu().numpy()
            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.imshow(arr, origin="lower", aspect="auto", cmap="magma")
            ax.set_title(f"Layer {layer} Head {head} mean attention")
            ax.set_xlabel("key position")
            ax.set_ylabel("query position")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            out = out_dir / f"attention_layer{layer}_head{head}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)


def _plot_head_metric(
    metric: torch.Tensor,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    n_layers, n_heads = metric.shape
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(n_heads))
    width = 0.8 / n_layers
    for layer in range(n_layers):
        offs = [(i - 0.4) + width * (layer + 0.5) for i in x]
        vals = metric[layer].cpu().tolist()
        ax.bar(offs, vals, width=width, label=f"L{layer}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("head index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncols=min(4, n_layers), fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_summary(
    out_path: Path,
    entropy: torch.Tensor,
    distance: torch.Tensor,
    token_mass: Dict[str, torch.Tensor],
    bar_focus: torch.Tensor,
    phrase_focus: torch.Tensor,
) -> None:
    n_layers, n_heads = entropy.shape
    rows: List[str] = [
        "# Attention probe summary",
        "",
        "Lower entropy = more focused head; "
        "higher distance = longer-range attention.",
        "",
        "| layer | head | entropy | avg_distance | "
        "bar_focus | phrase_focus | top_token_kind |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    kind_names = sorted(token_mass.keys())
    for layer in range(n_layers):
        for head in range(n_heads):
            best_kind = max(kind_names, key=lambda k: float(token_mass[k][layer, head].item()))
            rows.append(
                f"| {layer} | {head} | {entropy[layer, head]:.4f} | "
                f"{distance[layer, head]:.2f} | {bar_focus[layer, head]:.4f} | "
                f"{phrase_focus[layer, head]:.4f} | {best_kind} |"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows))


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description="Probe GPT attention patterns on JSB."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_ROOT / "results" / "checkpoints" / "best_model.pt"),
    )
    p.add_argument("--config", type=str, default="")
    p.add_argument("--n-chorales", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fig-dir",
        type=str,
        default=str(_ROOT / "figures" / "attention"),
    )
    p.add_argument(
        "--summary-out",
        type=str,
        default=str(_ROOT / "results" / "attention_probe_summary.md"),
    )
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_out = Path(args.summary_out)

    device = _pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = _load_config_from_sources(ckpt, args.config)
    model = GPT(cfg).to(device)
    state = (
        ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    )
    model.load_state_dict(state)
    model.eval()

    seq_len = min(args.seq_len, cfg.block_size)
    seqs = _load_jsb_sequences(
        args.n_chorales,
        seq_len=seq_len,
        seed=args.seed,
    )
    if not seqs:
        raise RuntimeError(
            "No JSB chorales loaded; check music21 install/dataset."
        )

    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    T = seq_len

    attn_sum = torch.zeros(n_layers, n_heads, T, T, device=device)
    entropy_sum = torch.zeros(n_layers, n_heads, device=device)
    distance_sum = torch.zeros(n_layers, n_heads, device=device)
    bar_focus_sum = torch.zeros(n_layers, n_heads, device=device)
    phrase_focus_sum = torch.zeros(n_layers, n_heads, device=device)
    token_mass_sum: Dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros(n_layers, n_heads, device=device)
    )

    pos = torch.arange(T, device=device)
    dist_matrix = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()

    n_used = 0
    for ids in seqs:
        x = torch.tensor([ids[:T]], dtype=torch.long, device=device)
        _, attn_list = model(x, return_attn=True)
        per_layer = torch.stack([a[0] for a in attn_list], dim=0)  # L,H,T,T
        attn_sum += per_layer

        eps = 1e-9
        entropy = -(per_layer * (per_layer + eps).log()).sum(
            dim=-1
        ).mean(dim=-1)
        entropy_sum += entropy

        avg_dist = (per_layer * dist_matrix).sum(dim=-1).mean(dim=-1)
        distance_sum += avg_dist

        key_kind = [_token_kind(tid) for tid in ids[:T]]
        key_mass = per_layer.mean(dim=2)  # L,H,T (avg over query positions)
        for k_idx, kind in enumerate(key_kind):
            token_mass_sum[kind] += key_mass[:, :, k_idx]

        bar_positions = [i for i, tid in enumerate(ids[:T]) if tid == BAR_START]
        phrase_positions = [
            i for i, tid in enumerate(ids[:T]) if tid == PHRASE_START
        ]
        if bar_positions:
            bar_idx = torch.tensor(bar_positions, dtype=torch.long, device=device)
            bar_focus_sum += per_layer[:, :, :, bar_idx].sum(
                dim=-1
            ).mean(dim=-1)
        if phrase_positions:
            phrase_idx = torch.tensor(
                phrase_positions,
                dtype=torch.long,
                device=device,
            )
            phrase_focus_sum += per_layer[:, :, :, phrase_idx].sum(
                dim=-1
            ).mean(dim=-1)

        n_used += 1

    mean_attn = attn_sum / n_used
    entropy_mean = entropy_sum / n_used
    distance_mean = distance_sum / n_used
    bar_focus_mean = bar_focus_sum / n_used
    phrase_focus_mean = phrase_focus_sum / n_used
    token_mass_mean = {k: v / n_used for k, v in token_mass_sum.items()}

    _plot_heatmaps(mean_attn=mean_attn, out_dir=fig_dir)
    _plot_head_metric(
        metric=entropy_mean,
        ylabel="entropy",
        title="Average attention entropy by head and layer",
        out_path=fig_dir / "attention_entropy_by_head.png",
    )
    _plot_head_metric(
        metric=distance_mean,
        ylabel="avg |query-key| distance",
        title="Average attention distance by head and layer",
        out_path=fig_dir / "attention_distance_by_head.png",
    )
    _write_summary(
        out_path=summary_out,
        entropy=entropy_mean.cpu(),
        distance=distance_mean.cpu(),
        token_mass={k: v.cpu() for k, v in token_mass_mean.items()},
        bar_focus=bar_focus_mean.cpu(),
        phrase_focus=phrase_focus_mean.cpu(),
    )

    flat_entropy = [
        (float(entropy_mean[layer_idx, head_idx].item()), layer_idx, head_idx)
        for layer_idx in range(n_layers)
        for head_idx in range(n_heads)
    ]
    flat_entropy.sort(key=lambda x: x[0])
    print(f"[probe_attention] chorales_used={n_used} seq_len={T}")
    print(f"[probe_attention] wrote heatmaps to {fig_dir}")
    print(f"[probe_attention] wrote summary to {summary_out}")
    print("[probe_attention] most focused heads (lowest entropy):")
    for ent, layer, head in flat_entropy[:5]:
        print(
            f"  layer={layer} head={head} entropy={ent:.4f} "
            f"distance={distance_mean[layer, head].item():.2f} "
            f"bar_focus={bar_focus_mean[layer, head].item():.4f} "
            f"phrase_focus={phrase_focus_mean[layer, head].item():.4f}"
        )


if __name__ == "__main__":
    main()
