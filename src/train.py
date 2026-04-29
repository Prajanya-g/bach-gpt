"""Train GPT on MIDI token chunks: checkpoints, CSV log, val tracking."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import build_dataloaders  # noqa: E402
from model import GPT, default_gpt_config  # noqa: E402
from tokenizer import VOCAB_SIZE  # noqa: E402


def _lr_lambda_factory(warmup_steps: int, total_steps: int):
    """Warmup then cosine: LR multiplier 1.0 → 0.1 over non-warmup steps."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        t = (current_step - warmup_steps) / float(total_steps - warmup_steps)
        t = min(1.0, max(0.0, t))
        min_f = 0.1
        return min_f + (1.0 - min_f) * 0.5 * (1.0 + math.cos(math.pi * t))

    return lr_lambda


@torch.no_grad()
def evaluate(model: GPT, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n_tokens = 0
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        total += loss.item() * y.numel()
        n_tokens += y.numel()
    model.train()
    return total / max(1, n_tokens)


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: AdamW,
    scheduler: LambdaLR,
    global_step: int,
    epoch: int,
    config_dict: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "config": config_dict,
        },
        path,
    )


def save_best(
    path: Path,
    model: GPT,
    val_loss: float,
    global_step: int,
    config_dict: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "val_loss": val_loss,
            "global_step": global_step,
            "config": config_dict,
        },
        path,
    )


def append_csv_row(
    csv_path: Path,
    fieldnames: list[str],
    row: Dict[str, Any],
    write_header: bool,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[train] device={device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, stats = build_dataloaders(
        sample_dir=Path(args.sample_dir) if args.sample_dir else None,
        block_size=args.block_size,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    print(
        f"[train] data: train_chunks={stats.n_train_chunks} "
        f"val_chunks={stats.n_val_chunks} tokens={stats.n_tokens_total}"
    )

    cfg = default_gpt_config()
    cfg.block_size = args.block_size
    cfg.dropout = args.dropout
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] parameters={n_params:,} (~{n_params / 1e6:.2f}M)")

    base_lr = 3e-4
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.max_epochs * steps_per_epoch)
    if total_steps < args.warmup_steps:
        print(
            f"[train] warning: total_steps={total_steps} < "
            f"warmup={args.warmup_steps}; LR schedule may be odd."
        )

    scheduler = LambdaLR(
        optimizer,
        _lr_lambda_factory(args.warmup_steps, total_steps),
        last_epoch=-1,
    )

    config_dict: Dict[str, Any] = asdict(cfg)
    config_dict.update(
        {
            "vocab_size": VOCAB_SIZE,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }
    )

    results_dir = Path(args.results_dir)
    log_csv = results_dir / "training_log.csv"
    ckpt_dir = results_dir / "checkpoints"
    best_path = ckpt_dir / "best_model.pt"

    fieldnames = [
        "step",
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "train_ppl",
        "val_ppl",
    ]
    if not log_csv.exists():
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    random_ce = math.log(VOCAB_SIZE)
    print(
        f"[train] random baseline CE≈{random_ce:.3f} (nats), "
        f"ppl≈{math.exp(random_ce):.1f} (≈vocab {VOCAB_SIZE})"
    )

    best_val = float("inf")
    global_step = 0
    train_loss_accum = 0.0
    train_loss_count = 0
    last_val_loss: Optional[float] = None

    model.train()
    t0 = time.perf_counter()

    for epoch in range(args.max_epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            train_loss_accum += loss.item()
            train_loss_count += 1

            lr = optimizer.param_groups[0]["lr"]

            if global_step % args.train_log_every == 0:
                avg_train = train_loss_accum / max(1, train_loss_count)
                try:
                    train_ppl = math.exp(avg_train)
                except OverflowError:
                    train_ppl = float("inf")
                print(
                    f"[train] step={global_step} epoch={epoch} "
                    f"train_loss={avg_train:.4f} train_ppl={train_ppl:.2f} "
                    f"lr={lr:.2e}"
                )
                append_csv_row(
                    log_csv,
                    fieldnames,
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "lr": lr,
                        "train_loss": f"{avg_train:.6f}",
                        "val_loss": "" if last_val_loss is None else f"{last_val_loss:.6f}",
                        "train_ppl": f"{train_ppl:.4f}",
                        "val_ppl": (
                            ""
                            if last_val_loss is None
                            else f"{math.exp(last_val_loss):.4f}"
                        ),
                    },
                    write_header=False,
                )
                train_loss_accum = 0.0
                train_loss_count = 0

            if global_step % args.val_every == 0:
                val_loss = evaluate(model, val_loader, device)
                last_val_loss = val_loss
                val_ppl = math.exp(val_loss)
                print(
                    f"[val] step={global_step} val_loss={val_loss:.4f} "
                    f"val_ppl={val_ppl:.2f}"
                )
                append_csv_row(
                    log_csv,
                    fieldnames,
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "lr": lr,
                        "train_loss": "",
                        "val_loss": f"{val_loss:.6f}",
                        "train_ppl": "",
                        "val_ppl": f"{val_ppl:.4f}",
                    },
                    write_header=False,
                )

                if val_loss < best_val:
                    best_val = val_loss
                    save_best(best_path, model, val_loss, global_step, config_dict)
                    print(
                        f"[train] new best val_loss={val_loss:.4f} "
                        f"→ {best_path}"
                    )

            if global_step % args.checkpoint_every == 0:
                ckpt_path = ckpt_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    epoch,
                    config_dict,
                )
                print(f"[train] saved {ckpt_path}")

    elapsed = time.perf_counter() - t0
    print(f"[train] finished in {elapsed / 60:.1f} min, best_val={best_val:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bach-gpt on MIDI tokens")
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--split-ratio", type=float, default=0.9)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--train-log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument(
        "--sample-dir",
        type=str,
        default="",
        help="Override GigaMIDI sample directory (default: data/gigamidi/sample)",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=str(_ROOT / "results"),
        help="Directory for training_log.csv and checkpoints/",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
