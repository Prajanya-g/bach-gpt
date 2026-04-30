"""Train compound MIDI-text contrastive model from a CompoundGPT checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pretty_midi
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compound import SENTINELS, STEP_PAD, encode_compound  # noqa: E402
from compound_model import (  # noqa: E402
    CompoundGPT,
    CompoundGPTConfig,
    default_compound_config,
)
from contrastive_model import CompoundMidiTextContrastiveModel  # noqa: E402


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _caption_from_record(record: Dict[str, Any]) -> str:
    llm_caption = str(record.get("caption", "")).strip()
    if llm_caption:
        return llm_caption
    return str(record.get("caption_template", "")).strip()


def _path_from_record(record: Dict[str, Any]) -> str:
    for k in ("path", "midi_path", "midi"):
        value = str(record.get(k, "")).strip()
        if value:
            return value
    return ""


class CompoundCaptionDataset(Dataset):
    """Dataset yielding padded compound steps + captions."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        is_train: bool,
        max_seq_len: int,
        seed: int,
    ) -> None:
        self.records = list(records)
        self.is_train = is_train
        self.max_seq_len = max_seq_len
        self._rng = random.Random(seed)
        self._cache: Dict[int, Optional[List[List[int]]]] = {}
        self._valid_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.records)

    def _encode_idx(self, idx: int) -> Optional[List[List[int]]]:
        if idx in self._cache:
            return self._cache[idx]
        rec = self.records[idx]
        path = _path_from_record(rec)
        if not path:
            self._cache[idx] = None
            return None
        try:
            pm = pretty_midi.PrettyMIDI(path)
            steps = encode_compound(pm)
        except Exception:
            self._cache[idx] = None
            return None
        if not steps:
            self._cache[idx] = None
            return None
        self._cache[idx] = steps
        self._valid_indices.add(idx)
        return steps

    def _crop(self, steps: List[List[int]]) -> List[List[int]]:
        if self.is_train:
            max_start = len(steps) - self.max_seq_len
            start = 0 if max_start <= 0 else self._rng.randint(0, max_start)
            return steps[start : start + self.max_seq_len]
        return steps[: self.max_seq_len]

    def _pad(self, steps: List[List[int]]) -> List[List[int]]:
        if len(steps) >= self.max_seq_len:
            return steps
        pad_step = list(SENTINELS)
        pad_step[0] = STEP_PAD
        return steps + [pad_step] * (self.max_seq_len - len(steps))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        steps = self._encode_idx(idx)
        rec = self.records[idx]
        if steps is None:
            if not self._valid_indices:
                for j in range(len(self.records)):
                    self._encode_idx(j)
            if not self._valid_indices:
                raise RuntimeError("No valid MIDI records for compound contrastive.")
            repl_idx = self._rng.choice(list(self._valid_indices))
            steps = self._encode_idx(repl_idx)
            rec = self.records[repl_idx]
        assert steps is not None
        steps = self._pad(self._crop(steps))
        return {
            "compound_input": torch.tensor(steps, dtype=torch.long),
            "caption": _caption_from_record(rec),
            "path": _path_from_record(rec),
        }


def _collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "compound_input": torch.stack([x["compound_input"] for x in items], dim=0),
        "captions": [x["caption"] for x in items],
        "paths": [x["path"] for x in items],
    }


def _split_records(
    records: List[Dict[str, Any]], split_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_train = int(len(idx) * split_ratio)
    train = [records[i] for i in idx[:n_train]]
    val = [records[i] for i in idx[n_train:]]
    return train, val


def _set_cosine_lrs(
    optimizer: AdamW, current_step: int, total_steps: int, min_lr_scale: float
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
    model: CompoundMidiTextContrastiveModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    midi_chunks: List[torch.Tensor] = []
    text_chunks: List[torch.Tensor] = []
    for batch in loader:
        out = model(
            compound_input=batch["compound_input"].to(device),
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
    r1_t2m = float((torch.argmax(logits.t(), dim=1) == labels).float().mean().item())
    return {"loss": loss_sum / n_batches, "r1_m2t": r1_m2t, "r1_t2m": r1_t2m}


def _load_compound_gpt(
    ckpt_path: Path, device: torch.device, block_size: int
) -> CompoundGPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = default_compound_config()
    cfg.block_size = block_size
    raw_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(raw_cfg, dict):
        for k in CompoundGPTConfig.__dataclass_fields__.keys():
            if k in raw_cfg:
                setattr(cfg, k, raw_cfg[k])
    model = CompoundGPT(cfg).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def train(args: argparse.Namespace) -> None:
    device = _pick_device()
    print(f"[compound-contrastive] device={device}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_records = _load_jsonl_records(Path(args.captions_jsonl))
    train_records, val_records = _split_records(
        all_records, split_ratio=args.split_ratio, seed=args.seed
    )
    print(
        "[compound-contrastive] records total/train/val="
        f"{len(all_records)}/{len(train_records)}/{len(val_records)}"
    )

    train_ds = CompoundCaptionDataset(
        records=train_records,
        is_train=True,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    val_ds = CompoundCaptionDataset(
        records=val_records,
        is_train=False,
        max_seq_len=args.max_seq_len,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )

    midi_encoder = _load_compound_gpt(
        ckpt_path=Path(args.compound_checkpoint),
        device=device,
        block_size=args.max_seq_len,
    )
    model = CompoundMidiTextContrastiveModel(
        midi_compound_gpt=midi_encoder,
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
    global_step = 0
    best_val = float("inf")
    text_group_added = False

    results_dir = Path(args.results_dir)
    ckpt_dir = results_dir / "checkpoints_contrastive_compound"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_csv = results_dir / "contrastive_compound_training_log.csv"
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
                    "temperature",
                ],
            ).writeheader()

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
            print(f"[compound-contrastive] epoch={epoch}: unfroze text encoder.")

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
                compound_input=batch["compound_input"].to(device),
                captions=batch["captions"],
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
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
        temp_val = float(model.get_temperature().item())
        print(
            f"[compound-contrastive] epoch={epoch}/{args.epochs} stage={stage} "
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
        torch.save(ckpt, ckpt_dir / "clap_compound_latest.pt")
        if epoch % args.checkpoint_every == 0:
            torch.save(ckpt, ckpt_dir / f"clap_compound_epoch_{epoch:03d}.pt")
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(ckpt, ckpt_dir / "clap_compound_best.pt")

    elapsed = time.perf_counter() - t0
    print(
        f"[compound-contrastive] finished in {elapsed/60:.1f} min, "
        f"best_val={best_val:.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train compound MIDI-text contrastive model")
    p.add_argument(
        "--captions-jsonl",
        type=str,
        default=str(_ROOT / "data" / "midicaps_100.jsonl"),
    )
    p.add_argument(
        "--compound-checkpoint",
        type=str,
        default=str(
            _ROOT / "results" / "test_compound" / "checkpoints_compound" / "compound_best.pt"
        ),
    )
    p.add_argument(
        "--text-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p.add_argument("--results-dir", type=str, default=str(_ROOT / "results" / "test_compound"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--unfreeze-text-epoch", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
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
    p.add_argument("--checkpoint-every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
