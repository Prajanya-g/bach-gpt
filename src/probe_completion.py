"""Phrase completion evaluation on JSB chorales.

For each JSB piece, we build (prompt, ground_truth_continuation) pairs by
splitting near the midpoint BAR_END. We then compare:
  - model completions
  - random-token completions
  - ground-truth Bach continuations
using tonal stability, rhythmic regularity, and phrase closure metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from model import GPT, GPTConfig, default_gpt_config  # noqa: E402
from tokenizer import (  # noqa: E402
    BAR_END,
    ID2TOKEN,
    PHRASE_END,
    VOCAB_SIZE,
    encode,
)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


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


def _pitch_class_from_id(tid: int) -> int | None:
    tok = ID2TOKEN.get(tid, "")
    if not tok.startswith("P"):
        return None
    try:
        return int(tok[1:]) % 12
    except ValueError:
        return None


def _bar_lengths(ids: List[int]) -> List[int]:
    lengths: List[int] = []
    count = 0
    in_bar = False
    for tid in ids:
        if tid == BAR_END:
            if in_bar:
                lengths.append(max(1, count))
            in_bar = False
            count = 0
            continue
        count += 1
        tok = ID2TOKEN.get(tid, "")
        if tok == "BAR_START":
            in_bar = True
    return lengths


def _infer_tonic(prompt_ids: List[int]) -> int:
    pcs = [
        pc for tid in prompt_ids if (pc := _pitch_class_from_id(tid)) is not None
    ]
    if not pcs:
        return 0
    counts = np.bincount(np.array(pcs, dtype=np.int64), minlength=12)
    return int(np.argmax(counts))


def _scale_pitch_classes(tonic: int) -> Tuple[set[int], set[int]]:
    major = {0, 2, 4, 5, 7, 9, 11}
    minor = {0, 2, 3, 5, 7, 8, 10}
    maj = {(tonic + x) % 12 for x in major}
    minr = {(tonic + x) % 12 for x in minor}
    return maj, minr


def tonal_stability(prompt_ids: List[int], gen_ids: List[int]) -> float:
    tonic = _infer_tonic(prompt_ids)
    maj, minr = _scale_pitch_classes(tonic)
    pcs = [pc for tid in gen_ids if (pc := _pitch_class_from_id(tid)) is not None]
    if not pcs:
        return 0.0
    in_maj = sum(1 for pc in pcs if pc in maj) / len(pcs)
    in_min = sum(1 for pc in pcs if pc in minr) / len(pcs)
    return float(max(in_maj, in_min))


def rhythmic_regularity(prompt_ids: List[int], gen_ids: List[int]) -> float:
    p_bars = _bar_lengths(prompt_ids)
    g_bars = _bar_lengths(gen_ids)
    if not p_bars or not g_bars:
        return 0.0
    p_len = float(np.mean(p_bars))
    g_len = float(np.mean(g_bars))
    if p_len <= 0:
        return 0.0
    score = 1.0 - abs(g_len - p_len) / p_len
    return float(max(0.0, min(1.0, score)))


def phrase_closure(prompt_ids: List[int], gen_ids: List[int]) -> float:
    if not gen_ids:
        return 0.0
    tail = gen_ids[-8:]
    tonic = _infer_tonic(prompt_ids)
    tonic_present = any(_pitch_class_from_id(tid) == tonic for tid in tail)
    if PHRASE_END in tail or BAR_END in tail or tonic_present:
        return 1.0
    return 0.0


def _metrics(
    prompt_ids: List[int], continuation_ids: List[int]
) -> Dict[str, float]:
    return {
        "tonal_stability": tonal_stability(prompt_ids, continuation_ids),
        "rhythmic_regularity": rhythmic_regularity(prompt_ids, continuation_ids),
        "phrase_closure": phrase_closure(prompt_ids, continuation_ids),
    }


def _build_phrase_pairs(
    n_pairs: int,
    min_prompt_tokens: int,
    min_cont_tokens: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    from music21 import corpus

    rng = random.Random(seed)
    chorales = list(
        corpus.chorales.Iterator(
            numberingSystem="bwv",
            returnType="stream",
        )
    )
    rng.shuffle(chorales)

    pairs: List[Tuple[List[int], List[int]]] = []
    for score in chorales:
        if len(pairs) >= n_pairs:
            break
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mid",
                delete=True,
            ) as tmp:
                score.write("midi", fp=tmp.name)
                pm = pretty_midi.PrettyMIDI(tmp.name)
            ids = encode(pm)
        except Exception:
            continue

        bar_end_pos = [i for i, tid in enumerate(ids) if tid == BAR_END]
        if len(bar_end_pos) < 2:
            continue
        split = bar_end_pos[len(bar_end_pos) // 2]
        prompt = ids[:split + 1]
        continuation = ids[split + 1:]

        if (
            len(prompt) < min_prompt_tokens
            or len(continuation) < min_cont_tokens
        ):
            continue
        pairs.append((prompt, continuation))
    return pairs


@torch.no_grad()
def _generate_continuation(
    model: GPT,
    prompt_ids: List[int],
    n_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> List[int]:
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = x
    for _ in range(n_tokens):
        ctx = out[:, -model.config.block_size:]
        logits = model(ctx)[:, -1, :] / temperature
        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, nxt], dim=1)
    full = out[0].tolist()
    return full[len(prompt_ids):]


def _plot_metrics(
    system_scores: Dict[str, Dict[str, float]], out_path: Path
) -> None:
    metrics = ["tonal_stability", "rhythmic_regularity", "phrase_closure"]
    systems = list(system_scores.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(systems)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for idx, name in enumerate(systems):
        offs = x - 0.4 + width * (idx + 0.5)
        vals = [system_scores[name][m] for m in metrics]
        ax.bar(offs, vals, width=width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(["tonal", "rhythmic", "closure"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Phrase completion metrics")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary(
    out_path: Path,
    n_pairs: int,
    n_samples: int,
    system_scores: Dict[str, Dict[str, float]],
) -> None:
    rows = [
        "# Phrase completion evaluation",
        "",
        f"- Pairs evaluated: {n_pairs}",
        f"- Model samples per prompt: {n_samples}",
        "",
        "| system | tonal_stability | rhythmic_regularity | phrase_closure |",
        "|---|---:|---:|---:|",
    ]
    for name, scores in system_scores.items():
        rows.append(
            f"| {name} | {scores['tonal_stability']:.3f} | "
            f"{scores['rhythmic_regularity']:.3f} | {scores['phrase_closure']:.3f} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="JSB phrase completion evaluation."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_ROOT / "results" / "checkpoints" / "best_model.pt"),
    )
    p.add_argument("--config", type=str, default="")
    p.add_argument("--n-pairs", type=int, default=60)
    p.add_argument("--samples-per-prompt", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--plot-out",
        type=str,
        default=str(_ROOT / "figures" / "phrase_completion_metrics.png"),
    )
    p.add_argument(
        "--summary-out",
        type=str,
        default=str(_ROOT / "results" / "phrase_completion_summary.md"),
    )
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
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

    pairs = _build_phrase_pairs(
        n_pairs=args.n_pairs,
        min_prompt_tokens=32,
        min_cont_tokens=32,
        seed=args.seed,
    )
    if not pairs:
        raise RuntimeError("No phrase pairs extracted from JSB.")

    metric_names = ["tonal_stability", "rhythmic_regularity", "phrase_closure"]
    systems: Dict[str, Dict[str, List[float]]] = {
        "random": {k: [] for k in metric_names},
        "model_t1.0_topk40": {k: [] for k in metric_names},
        "ground_truth": {k: [] for k in metric_names},
    }

    rng = random.Random(args.seed + 123)
    for prompt_ids, gt_cont in pairs:
        target_len = len(gt_cont)

        gt_scores = _metrics(prompt_ids, gt_cont)
        for k, v in gt_scores.items():
            systems["ground_truth"][k].append(v)

        rand_ids = [rng.randrange(VOCAB_SIZE) for _ in range(target_len)]
        rand_scores = _metrics(prompt_ids, rand_ids)
        for k, v in rand_scores.items():
            systems["random"][k].append(v)

        for _ in range(args.samples_per_prompt):
            gen_ids = _generate_continuation(
                model=model,
                prompt_ids=prompt_ids,
                n_tokens=target_len,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
            gen_scores = _metrics(prompt_ids, gen_ids)
            for k, v in gen_scores.items():
                systems["model_t1.0_topk40"][k].append(v)

    system_means: Dict[str, Dict[str, float]] = {}
    for name, metrics in systems.items():
        system_means[name] = {
            k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()
        }

    _plot_metrics(system_means, out_path=Path(args.plot_out))
    _write_summary(
        out_path=Path(args.summary_out),
        n_pairs=len(pairs),
        n_samples=args.samples_per_prompt,
        system_scores=system_means,
    )

    print(
        f"[probe_completion] pairs={len(pairs)} "
        f"samples_per_prompt={args.samples_per_prompt}"
    )
    print(f"[probe_completion] plot -> {args.plot_out}")
    print(f"[probe_completion] summary -> {args.summary_out}")
    for system, scores in system_means.items():
        print(
            f"[probe_completion] {system}: "
            f"tonal={scores['tonal_stability']:.3f} "
            f"rhythmic={scores['rhythmic_regularity']:.3f} "
            f"closure={scores['phrase_closure']:.3f}"
        )


if __name__ == "__main__":
    main()
