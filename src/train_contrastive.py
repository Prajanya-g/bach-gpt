"""Train the MIDI-text contrastive model (Phase 2)."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from caption_dataloader import (  # noqa: E402
    MidiCaptionDataset,
    _collate_caption_batch,
    _load_jsonl_records,
    build_caption_dataloaders,
)
from contrastive_model import MidiTextContrastiveModel  # noqa: E402
from model import GPT, GPTConfig, default_gpt_config  # noqa: E402


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


def _load_gpt_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = default_gpt_config()
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(ckpt_cfg, dict):
        for k, v in _extract_gpt_config_dict(ckpt_cfg).items():
            setattr(cfg, k, v)
    model = GPT(cfg).to(device)
    state = (
        ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _set_cosine_lrs(
    optimizer: AdamW,
    current_step: int,
    total_steps: int,
    min_lr_scale: float = 0.01,
) -> None:
    if total_steps <= 1:
        mult = 1.0
    else:
        progress = min(1.0, max(0.0, current_step / float(total_steps - 1)))
        mult = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        base_lr = group.get("initial_lr", group["lr"])
        group["lr"] = base_lr * mult


@torch.no_grad()
def evaluate(
    model: MidiTextContrastiveModel,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    midi_chunks: List[torch.Tensor] = []
    text_chunks: List[torch.Tensor] = []

    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            captions=batch["captions"],
        )
        loss_sum += float(out["loss"].item())
        midi_chunks.append(out["midi_embeds"].detach().cpu())
        text_chunks.append(out["text_embeds"].detach().cpu())
        n_batches += 1
    model.train()
    if n_batches == 0:
        return {"loss": 0.0, "r1_m2t": 0.0, "r1_t2m": 0.0}

    midi_all = torch.cat(midi_chunks, dim=0)
    text_all = torch.cat(text_chunks, dim=0)
    logits = midi_all @ text_all.t()
    labels = torch.arange(logits.size(0))
    r1_m2t = float((torch.argmax(logits, dim=1) == labels).float().mean().item())
    r1_t2m = float(
        (torch.argmax(logits.t(), dim=1) == labels).float().mean().item()
    )

    return {
        "loss": loss_sum / n_batches,
        "r1_m2t": r1_m2t,
        "r1_t2m": r1_t2m,
    }


@torch.no_grad()
def qualitative_retrieval_check(
    model: MidiTextContrastiveModel,
    loader: DataLoader,
    prompts: List[str],
    device: torch.device,
    top_k: int = 3,
) -> None:
    model.eval()
    midi_chunks: List[torch.Tensor] = []
    captions: List[str] = []
    paths: List[str] = []

    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            captions=batch["captions"],
        )
        midi_chunks.append(out["midi_embeds"].detach().cpu())
        captions.extend(batch["captions"])
        paths.extend(batch["paths"])

    if not midi_chunks:
        print("[qualitative] no valid samples found.")
        model.train()
        return

    midi_all = torch.cat(midi_chunks, dim=0).to(device)
    text_feats = model.encode_text(prompts, device=device)
    text_proj = model.text_projection(text_feats)
    text_embeds = torch.nn.functional.normalize(text_proj, p=2, dim=-1)
    sims = text_embeds @ midi_all.t()

    print("\n[qualitative] fixed-prompt retrieval (top-3)")
    for i, prompt in enumerate(prompts):
        top_idx = torch.topk(sims[i], k=min(top_k, sims.size(1))).indices.tolist()
        print(f"\nPrompt: {prompt}")
        for rank, idx in enumerate(top_idx, start=1):
            cap = captions[idx].replace("\n", " ")
            print(f"  {rank}. {Path(paths[idx]).name} :: {cap[:160]}")
    model.train()


def train(args: argparse.Namespace) -> None:
    device = _pick_device()
    print(f"[contrastive] device={device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, data_stats = build_caption_dataloaders(
        jsonl_path=args.captions_jsonl,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(
        "[contrastive] records total/train/val="
        f"{data_stats.n_total_records}/{data_stats.n_train_records}/"
        f"{data_stats.n_val_records}"
    )
    full_records = _load_jsonl_records(Path(args.captions_jsonl))
    full_eval_loader = DataLoader(
        MidiCaptionDataset(
            records=full_records,
            is_train=False,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_caption_batch,
    )

    midi_gpt = _load_gpt_from_checkpoint(
        Path(args.midi_checkpoint), device=device
    )
    model = MidiTextContrastiveModel(
        midi_gpt=midi_gpt,
        text_model_name=args.text_model,
        embed_dim=args.embed_dim,
        init_temperature=args.init_temperature,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
        device=device,
    )
    model.train()

    optimizer = AdamW(
        [
            {
                "params": model.midi_projection.parameters(),
                "lr": args.proj_lr,
                "initial_lr": args.proj_lr,
                "weight_decay": args.proj_weight_decay,
            },
            {
                "params": model.text_projection.parameters(),
                "lr": args.proj_lr,
                "initial_lr": args.proj_lr,
                "weight_decay": args.proj_weight_decay,
            },
            {
                "params": [model.log_temperature],
                "lr": args.temp_lr,
                "initial_lr": args.temp_lr,
                "weight_decay": 0.0,
            },
        ]
    )

    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    print(
        f"[contrastive] epochs={args.epochs} "
        f"steps_per_epoch={steps_per_epoch} "
        f"total_steps={total_steps}"
    )

    results_dir = Path(args.results_dir)
    ckpt_dir = results_dir / "checkpoints_contrastive"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = results_dir / "contrastive_training_log.csv"
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "step",
                    "stage",
                    "train_loss",
                    "train_acc_m2t",
                    "train_acc_t2m",
                    "val_loss",
                    "val_r1_m2t",
                    "val_r1_t2m",
                    "lr_proj",
                    "lr_temp",
                    "lr_text",
                    "temperature",
                ],
            ).writeheader()

    global_step = 0
    best_val = float("inf")
    text_group_added = False
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_text_epoch and not text_group_added:
            model.unfreeze_text_encoder()
            optimizer.add_param_group(
                {
                    "params": model.text_encoder.parameters(),
                    "lr": args.text_lr,
                    "initial_lr": args.text_lr,
                    "weight_decay": args.text_weight_decay,
                }
            )
            text_group_added = True
            print(f"[contrastive] epoch={epoch}: unfroze text encoder.")

        stage = "A" if epoch < args.unfreeze_text_epoch else "B"
        train_loss_sum = 0.0
        train_acc_m2t_sum = 0.0
        train_acc_t2m_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            _set_cosine_lrs(
                optimizer=optimizer,
                current_step=global_step,
                total_steps=total_steps,
                min_lr_scale=args.min_lr_scale,
            )

            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                captions=batch["captions"],
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip_norm
            )
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_acc_m2t_sum += float(out["acc_midi_to_text"].item())
            train_acc_t2m_sum += float(out["acc_text_to_midi"].item())
            train_batches += 1
            global_step += 1

        train_loss = train_loss_sum / max(1, train_batches)
        train_acc_m2t = train_acc_m2t_sum / max(1, train_batches)
        train_acc_t2m = train_acc_t2m_sum / max(1, train_batches)
        val = evaluate(model, val_loader, device)

        lrs = [g["lr"] for g in optimizer.param_groups]
        lr_proj = lrs[0]
        lr_temp = lrs[2]
        lr_text = lrs[3] if len(lrs) > 3 else 0.0
        temp_val = float(model.get_temperature().item())

        print(
            f"[contrastive] epoch={epoch}/{args.epochs} stage={stage} "
            f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"train_acc=({train_acc_m2t:.3f},{train_acc_t2m:.3f}) "
            f"val_r1=({val['r1_m2t']:.3f},{val['r1_t2m']:.3f}) "
            f"temp={temp_val:.4f}"
        )

        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "step",
                    "stage",
                    "train_loss",
                    "train_acc_m2t",
                    "train_acc_t2m",
                    "val_loss",
                    "val_r1_m2t",
                    "val_r1_t2m",
                    "lr_proj",
                    "lr_temp",
                    "lr_text",
                    "temperature",
                ],
            ).writerow(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "stage": stage,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc_m2t": f"{train_acc_m2t:.6f}",
                    "train_acc_t2m": f"{train_acc_t2m:.6f}",
                    "val_loss": f"{val['loss']:.6f}",
                    "val_r1_m2t": f"{val['r1_m2t']:.6f}",
                    "val_r1_t2m": f"{val['r1_t2m']:.6f}",
                    "lr_proj": f"{lr_proj:.8e}",
                    "lr_temp": f"{lr_temp:.8e}",
                    "lr_text": f"{lr_text:.8e}",
                    "temperature": f"{temp_val:.6f}",
                }
            )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": float(val["loss"]),
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_dir / "clap_latest.pt")

        if epoch % args.checkpoint_every == 0:
            torch.save(ckpt, ckpt_dir / f"clap_epoch_{epoch:03d}.pt")

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(ckpt, ckpt_dir / "clap_best.pt")

        if epoch % args.qualitative_every == 0:
            qualitative_retrieval_check(
                model=model,
                loader=full_eval_loader,
                prompts=args.qual_prompts,
                device=device,
                top_k=3,
            )

    elapsed = time.perf_counter() - t0
    print(
        f"[contrastive] finished in {elapsed/60:.1f} min, "
        f"best_val={best_val:.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MIDI-text contrastive model"
    )
    p.add_argument(
        "--captions-jsonl",
        type=str,
        default=str(_ROOT / "data" / "captions_llm.jsonl"),
    )
    p.add_argument(
        "--midi-checkpoint",
        type=str,
        default=str(_ROOT / "results" / "checkpoints" / "best_model.pt"),
    )
    p.add_argument(
        "--text-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p.add_argument("--results-dir", type=str, default=str(_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--unfreeze-text-epoch", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--split-ratio", type=float, default=0.95)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--init-temperature", type=float, default=0.07)
    p.add_argument("--min-temperature", type=float, default=0.01)
    p.add_argument("--max-temperature", type=float, default=1.0)

    p.add_argument("--proj-lr", type=float, default=1e-4)
    p.add_argument("--temp-lr", type=float, default=1e-5)
    p.add_argument("--text-lr", type=float, default=1e-5)
    p.add_argument("--proj-weight-decay", type=float, default=0.01)
    p.add_argument("--text-weight-decay", type=float, default=0.01)
    p.add_argument("--min-lr-scale", type=float, default=0.01)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--qualitative-every", type=int, default=5)
    p.add_argument(
        "--qual-prompts",
        nargs="+",
        default=[
            "A bright fast piano étude with rising melodic contour.",
            "A slow melancholic minor-key piece with gentle dynamics.",
            "A syncopated groove with drums, bass, and electric guitar.",
            "A dense orchestral texture with strings and brass swells.",
            "An ambient electronic track with sustained synth pads.",
        ],
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
