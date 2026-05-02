"""Phase 3 prefix projector training (compound path)."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from caption_dataloader import build_compound_caption_dataloaders  # noqa: E402
from prefix_projector import (  # noqa: E402
    load_phase3_compound_components,
    phase3_compound_prefix_lm_loss,
)
from compound import SENTINELS, STEP_PAD  # noqa: E402
from compound_model import compound_loss  # noqa: E402
from tokenizer import PHRASE_START  # noqa: E402


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 prefix projector (compound path)")
    p.add_argument("--compound-checkpoint", type=str, required=True)
    p.add_argument("--clap-checkpoint", type=str, required=True)
    p.add_argument("--captions-jsonl", type=str, required=True)
    p.add_argument("--n-prefix-tokens", type=int, default=8)
    p.add_argument("--results-dir", type=str, default=str(_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=504)
    p.add_argument("--split-ratio", type=float, default=0.95)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-scale", type=float, default=0.01)
    return p.parse_args()


def _set_lr(optimizer, step, total_steps, warmup_steps, base_lr, min_lr_scale):
    if step < warmup_steps:
        mult = float(step + 1) / float(max(1, warmup_steps))
    else:
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        mult = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    optimizer.param_groups[0]["lr"] = base_lr * mult


@torch.no_grad()
def _eval_loss(clap_model, compound_gpt, projector, loader, device):
    projector.eval()
    total = 0.0
    n = 0
    for batch in loader:
        loss, _ = phase3_compound_prefix_lm_loss(
            clap_model=clap_model,
            midi_compound_gpt=compound_gpt,
            prefix_projector=projector,
            compound_input=batch["compound_input"].to(device),
            captions=batch["captions"],
        )
        total += float(loss.item())
        n += 1
    projector.train()
    return total / max(1, n)


@torch.no_grad()
def _baseline_loss(compound_gpt, loader, device):
    """Loss without prefix — for comparison."""
    compound_gpt.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["compound_input"].to(device)
        logits = compound_gpt(idx=x)
        # shift: predict next step
        loss = compound_loss(
            logits_per_axis=[l[:, :-1, :] for l in logits],
            targets=x[:, 1:, :],
            pad_step_value=STEP_PAD,
            ignore_pad_steps=True,
        )
        total += float(loss.item())
        n += 1
    compound_gpt.eval()
    return total / max(1, n)


def main() -> None:
    args = parse_args()
    device = _pick_device()
    print(f"[phase3-compound] device={device}")

    clap_model, compound_gpt, projector, counts = load_phase3_compound_components(
        compound_midi_checkpoint=args.compound_checkpoint,
        compound_clap_checkpoint=args.clap_checkpoint,
        n_prefix_tokens=args.n_prefix_tokens,
        device=device,
    )

    train_loader, val_loader, stats = build_compound_caption_dataloaders(
        jsonl_path=args.captions_jsonl,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    print(f"[phase3-compound] dataset total/train/val="
          f"{stats.n_total_records}/{stats.n_train_records}/{stats.n_val_records}")
    print(f"[phase3-compound] CLAP params (frozen): {counts.n_clap_params:,}")
    print(f"[phase3-compound] GPT params (frozen):  {counts.n_gpt_params:,}")
    print(f"[phase3-compound] projector params:     {counts.n_projector_params:,}")

    optimizer = AdamW(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    print(f"[phase3-compound] epochs={args.epochs} "
          f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

    ckpt_dir = Path(args.results_dir) / "checkpoints_prefix"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val = float("inf")
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        projector.train()
        train_loss_sum = 0.0
        n_train = 0

        for batch in train_loader:
            _set_lr(optimizer, global_step, total_steps,
                    args.warmup_steps, args.lr, args.min_lr_scale)
            optimizer.zero_grad(set_to_none=True)

            loss, _ = phase3_compound_prefix_lm_loss(
                clap_model=clap_model,
                midi_compound_gpt=compound_gpt,
                prefix_projector=projector,
                compound_input=batch["compound_input"].to(device),
                captions=batch["captions"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip_norm)
            optimizer.step()

            train_loss_sum += float(loss.item())
            n_train += 1
            global_step += 1

        train_loss = train_loss_sum / max(1, n_train)
        val_loss = _eval_loss(clap_model, compound_gpt, projector, val_loader, device)
        baseline = _baseline_loss(compound_gpt, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[phase3-compound] epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"baseline={baseline:.4f} gap={val_loss - baseline:+.4f} "
            f"lr={lr:.2e}"
        )

        ckpt = {
            "projector_state_dict": projector.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_dir / "prefix_projector_latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "prefix_projector_best.pt")

    elapsed = time.perf_counter() - t0
    print(f"[phase3-compound] done in {elapsed/60:.1f} min, best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
