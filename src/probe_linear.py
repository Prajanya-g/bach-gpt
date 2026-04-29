"""Linear probing on frozen GPT activations for musical structure signals."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import pretty_midi
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from model import GPT, GPTConfig, default_gpt_config  # noqa: E402
from tokenizer import ID2TOKEN, PHRASE_END, encode  # noqa: E402

N_VOICE_BINS = 8


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


def _load_jsb_sequences(
    n_files: int, seq_len: int, seed: int
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
            with tempfile.NamedTemporaryFile(
                suffix=".mid",
                delete=True,
            ) as tmp:
                score.write("midi", fp=tmp.name)
                pm = pretty_midi.PrettyMIDI(tmp.name)
            ids = encode(pm)
            if len(ids) < seq_len:
                continue
            seqs.append(ids[:seq_len])
        except Exception:
            continue
    return seqs


def _build_labels(ids: List[int]) -> Dict[str, np.ndarray]:
    """Build per-position labels from token sequence heuristics."""
    n = len(ids)
    beat = np.zeros(n, dtype=np.int64)
    pitch_class = np.full(n, -1, dtype=np.int64)
    cadence = np.zeros(n, dtype=np.int64)
    voice = np.zeros(n, dtype=np.int64)

    ts_since_bar = 0
    last_voice = 0
    for i, tid in enumerate(ids):
        tok = ID2TOKEN.get(tid, "")
        if tok == "BAR_START":
            ts_since_bar = 0
        elif tok.startswith("TS"):
            ts_since_bar += 1
        elif tok.startswith("V"):
            try:
                last_voice = int(tok[1:])
            except ValueError:
                last_voice = 0

        beat[i] = ts_since_bar % 4
        voice[i] = max(0, min(N_VOICE_BINS - 1, last_voice))

        if tok.startswith("P"):
            try:
                pitch = int(tok[1:])
                pitch_class[i] = pitch % 12
            except ValueError:
                pitch_class[i] = -1

        end = min(n, i + 9)
        if PHRASE_END in ids[i + 1:end]:
            cadence[i] = 1

    return {
        "beat_position": beat,
        "pitch_class": pitch_class,
        "cadence_soon": cadence,
        "voice_bin": voice,
    }


def _collect_layer_activations(
    model: GPT,
    ids: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Return tensor (L, T, D) of block output activations for one sequence."""
    activations: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            # TransformerBlock forward returns (x, attn_weights).
            x = output[0] if isinstance(output, tuple) else output
            activations[layer_idx] = x.detach()

        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    try:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _ = model(x)
    finally:
        for h in hooks:
            h.remove()

    layer_tensors = [
        activations[i][0].cpu() for i in range(model.config.n_layers)
    ]
    return torch.stack(layer_tensors, dim=0)


def _probe_one_target(
    X_by_layer: List[np.ndarray],
    y: np.ndarray,
    seed: int,
) -> List[float]:
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    y_train = y[train_idx]
    y_val = y[val_idx]

    scores: List[float] = []
    for X in X_by_layer:
        X_train = X[train_idx]
        X_val = X[val_idx]
        clf = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
            multi_class="auto",
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        scores.append(float(accuracy_score(y_val, pred)))
    return scores


def _plot_probe_lines(
    results: Dict[str, List[float]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    n_layers = len(next(iter(results.values())))
    xs = list(range(n_layers))
    for name, vals in results.items():
        ax.plot(xs, vals, marker="o", linewidth=2, label=name)
    ax.set_xlabel("layer")
    ax.set_ylabel("validation accuracy")
    ax.set_xticks(xs)
    ax.set_title("Linear probe accuracy by layer")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary_md(
    out_path: Path,
    results: Dict[str, List[float]],
    baselines: Dict[str, float],
) -> None:
    rows = [
        "# Linear probe summary",
        "",
        "| target | random_baseline | layer_0 | layer_1 | "
        "layer_2 | layer_3 | best_layer | best_acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, vals in results.items():
        best_layer = int(np.argmax(vals))
        best_acc = float(np.max(vals))
        padded = vals + [np.nan] * max(0, 4 - len(vals))
        rows.append(
            f"| {target} | {baselines[target]:.3f} | "
            f"{padded[0]:.3f} | {padded[1]:.3f} | {padded[2]:.3f} | "
            f"{padded[3]:.3f} | {best_layer} | {best_acc:.3f} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows))


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description="Linear probing on frozen GPT activations."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_ROOT / "results" / "checkpoints" / "best_model.pt"),
    )
    p.add_argument("--config", type=str, default="")
    p.add_argument("--n-chorales", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--plot-out",
        type=str,
        default=str(_ROOT / "figures" / "linear_probe_accuracy_by_layer.png"),
    )
    p.add_argument(
        "--summary-out",
        type=str,
        default=str(_ROOT / "results" / "linear_probe_summary.md"),
    )
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    X_layers: List[List[np.ndarray]] = [[] for _ in range(n_layers)]
    labels_all: Dict[str, List[np.ndarray]] = {
        "beat_position": [],
        "pitch_class": [],
        "cadence_soon": [],
        "voice_bin": [],
    }

    for ids in seqs:
        acts = _collect_layer_activations(
            model,
            ids=ids[:seq_len],
            device=device,
        )
        # acts shape: (L, T, D)
        labels = _build_labels(ids[:seq_len])
        for layer_idx in range(n_layers):
            X_layers[layer_idx].append(acts[layer_idx].numpy())
        for k in labels_all:
            labels_all[k].append(labels[k])

    X_by_layer = [np.concatenate(xlist, axis=0) for xlist in X_layers]
    y_targets = {k: np.concatenate(v, axis=0) for k, v in labels_all.items()}

    results: Dict[str, List[float]] = {}
    for target, y in y_targets.items():
        if target == "pitch_class":
            valid = y >= 0
            y_use = y[valid]
            X_use = [X[valid] for X in X_by_layer]
        else:
            y_use = y
            X_use = X_by_layer
        results[target] = _probe_one_target(X_use, y_use, seed=args.seed)

    baselines = {
        "beat_position": 0.25,
        "pitch_class": 1.0 / 12.0,
        "cadence_soon": 0.15,
        "voice_bin": 1.0 / N_VOICE_BINS,
    }
    _plot_probe_lines(results, out_path=Path(args.plot_out))
    _write_summary_md(
        Path(args.summary_out),
        results=results,
        baselines=baselines,
    )

    print(f"[probe_linear] chorales_used={len(seqs)} seq_len={seq_len}")
    print(f"[probe_linear] plot -> {args.plot_out}")
    print(f"[probe_linear] summary -> {args.summary_out}")
    for target, vals in results.items():
        best = max(vals)
        print(
            f"[probe_linear] {target}: "
            f"layers={['%.3f' % v for v in vals]} best={best:.3f} "
            f"(baseline≈{baselines[target]:.3f})"
        )


if __name__ == "__main__":
    main()
