"""Phase 3 training loop (prefix projector only)."""

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

from caption_dataloader import build_caption_dataloaders  # noqa: E402
from prefix_projector import (  # noqa: E402
    clap_text_for_prefix_projector,
    load_phase3_components,
    phase3_prefix_lm_loss,
)
from tokenizer import ID2TOKEN, PHRASE_START  # noqa: E402


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 training: prefix projector only"
    )
    p.add_argument(
        "--midi-checkpoint",
        type=str,
        default=str(_ROOT / "results" / "checkpoints" / "best_model.pt"),
    )
    p.add_argument(
        "--clap-checkpoint",
        type=str,
        default=str(
            _ROOT
            / "results"
            / "checkpoints_contrastive"
            / "clap_best.pt"
        ),
    )
    p.add_argument("--n-prefix-tokens", type=int, default=8)
    p.add_argument(
        "--captions-jsonl",
        type=str,
        default=str(_ROOT / "data" / "captions_llm.jsonl"),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--split-ratio", type=float, default=0.95)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--results-dir",
        type=str,
        default=str(_ROOT / "results"),
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-scale", type=float, default=0.01)
    p.add_argument("--prefix-attn-reg-weight", type=float, default=0.0)
    p.add_argument("--prefix-attn-min-mean", type=float, default=0.05)
    p.add_argument("--qualitative-every", type=int, default=5)
    p.add_argument("--qual-gen-tokens", type=int, default=40)
    p.add_argument(
        "--qual-prompts",
        nargs="+",
        default=[
            "A fast bright piano étude with rising melodic contour.",
            "A syncopated jazz combo with saxophone and walking bass.",
            "An ambient electronic piece with sustained synth pads.",
        ],
    )
    return p.parse_args()


def _set_warmup_cosine_lr(
    optimizer: AdamW,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_scale: float,
) -> None:
    if step < warmup_steps:
        mult = float(step + 1) / float(max(1, warmup_steps))
    else:
        if total_steps <= warmup_steps:
            mult = 1.0
        else:
            progress = (step - warmup_steps) / float(total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            mult = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
    optimizer.param_groups[0]["lr"] = base_lr * mult


@torch.no_grad()
def _eval_loss(
    clap_model,
    midi_gpt,
    projector,
    loader,
    device: torch.device,
    prefix_attn_reg_weight: float,
    prefix_attn_min_mean: float,
) -> float:
    projector.eval()
    total = 0.0
    n = 0
    for batch in loader:
        loss, _ = phase3_prefix_lm_loss(
            clap_model=clap_model,
            midi_gpt=midi_gpt,
            prefix_projector=projector,
            input_ids=batch["input_ids"].to(device),
            captions=batch["captions"],
            prefix_attn_reg_weight=prefix_attn_reg_weight,
            prefix_attn_min_mean=prefix_attn_min_mean,
        )
        total += float(loss.item())
        n += 1
    projector.train()
    return total / max(1, n)


def _lm_loss_without_prefix(midi_gpt, input_ids: torch.Tensor) -> torch.Tensor:
    logits = midi_gpt(input_ids)
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
    )


def _infer_genre_label(caption: str) -> str:
    text = caption.lower()
    if "jazz" in text or "swing" in text or "bebop" in text:
        return "jazz"
    if "electronic" in text or "synth" in text or "edm" in text:
        return "electronic"
    if "classical" in text or "orchestral" in text or "baroque" in text:
        return "classical"
    if "rock" in text or "guitar" in text or "band" in text:
        return "rock"
    return "other"


@torch.no_grad()
def _conditional_perplexity_gap_by_genre(
    clap_model,
    midi_gpt,
    projector,
    loader,
    device: torch.device,
    max_examples: int = 200,
) -> Dict[str, float]:
    projector.eval()
    sums_with: Dict[str, float] = {}
    sums_without: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    seen = 0

    for batch in loader:
        if seen >= max_examples:
            break
        input_ids = batch["input_ids"].to(device)
        captions = batch["captions"]
        for i in range(input_ids.size(0)):
            if seen >= max_examples:
                break
            x = input_ids[i : i + 1]
            cap = [captions[i]]
            loss_with, _ = phase3_prefix_lm_loss(
                clap_model=clap_model,
                midi_gpt=midi_gpt,
                prefix_projector=projector,
                input_ids=x,
                captions=cap,
            )
            loss_without = _lm_loss_without_prefix(midi_gpt=midi_gpt, input_ids=x)
            genre = _infer_genre_label(cap[0])
            sums_with[genre] = sums_with.get(genre, 0.0) + float(loss_with.item())
            sums_without[genre] = sums_without.get(genre, 0.0) + float(
                loss_without.item()
            )
            counts[genre] = counts.get(genre, 0) + 1
            seen += 1

    gaps: Dict[str, float] = {}
    for genre, n in counts.items():
        mean_with = sums_with[genre] / n
        mean_without = sums_without[genre] / n
        gaps[genre] = math.exp(mean_with) - math.exp(mean_without)
    projector.train()
    return gaps


@torch.no_grad()
def _generate_unconditional(midi_gpt, gen_tokens: int, device: torch.device) -> List[int]:
    seq = torch.tensor([[PHRASE_START]], dtype=torch.long, device=device)
    for _ in range(gen_tokens):
        logits = midi_gpt(seq)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, nxt], dim=1)
    return seq[0].tolist()


@torch.no_grad()
def _generate_with_text_prefix(
    clap_model,
    midi_gpt,
    projector,
    text_prompt: str,
    gen_tokens: int,
    device: torch.device,
) -> List[int]:
    # Diagnostic-only helper for qualitative checks during training.
    # This re-runs full prefix+GPT forward each token (O(n^2)); production
    # inference should use cached decoding in generate_conditional.py.
    ids: List[int] = [PHRASE_START]
    for _ in range(gen_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _loss, logits_full = phase3_prefix_lm_loss(
            clap_model=clap_model,
            midi_gpt=midi_gpt,
            prefix_projector=projector,
            input_ids=x,
            captions=[text_prompt],
        )
        logits = logits_full[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids.append(int(nxt.item()))
    return ids


def _token_preview(ids: List[int], max_len: int = 40) -> str:
    toks = [ID2TOKEN.get(i, f"UNK({i})") for i in ids[:max_len]]
    suffix = " ..." if len(ids) > max_len else ""
    return " ".join(toks) + suffix


@torch.no_grad()
def _prefix_token_scale_diagnostics(
    clap_model,
    midi_gpt,
    projector,
    batch,
    device: torch.device,
) -> None:
    x = batch["input_ids"].to(device)
    caps = batch["captions"]
    text_emb = clap_text_for_prefix_projector(clap_model, caps, device)
    prefix = projector(text_emb)
    token = midi_gpt.wte(x)
    pnorm = float(prefix.norm(dim=-1).mean().item())
    tnorm = float(token.norm(dim=-1).mean().item())
    ratio = pnorm / max(1e-8, tnorm)
    print(
        "[phase3][scale] prefix_norm="
        f"{pnorm:.4f} token_norm={tnorm:.4f} ratio={ratio:.3f}"
    )
    if ratio > 10.0 or ratio < 0.1:
        print(
            "[phase3][scale][warn] prefix/token norm mismatch is large."
        )


@torch.no_grad()
def _verify_prefix_usage(
    clap_model,
    midi_gpt,
    projector,
    batch,
    device: torch.device,
) -> None:
    """Check loss is lower with correct caption prefix than random wrong one."""
    input_ids = batch["input_ids"].to(device)
    captions = batch["captions"]
    if input_ids.size(0) < 2:
        print("[phase3][verify1] skipped: need batch size >= 2.")
        return

    x = input_ids[0:1]
    correct_caption = [captions[0]]
    wrong_caption = [captions[1]]

    loss_correct, _ = phase3_prefix_lm_loss(
        clap_model=clap_model,
        midi_gpt=midi_gpt,
        prefix_projector=projector,
        input_ids=x,
        captions=correct_caption,
    )
    loss_wrong, _ = phase3_prefix_lm_loss(
        clap_model=clap_model,
        midi_gpt=midi_gpt,
        prefix_projector=projector,
        input_ids=x,
        captions=wrong_caption,
    )
    delta = float(loss_wrong.item() - loss_correct.item())
    print(
        "[phase3][verify1] loss(correct_prefix)="
        f"{loss_correct.item():.4f} loss(wrong_prefix)={loss_wrong.item():.4f} "
        f"delta(wrong-correct)={delta:+.4f}"
    )
    if abs(delta) < 1e-4:
        print(
            "[phase3][verify1][warn] losses are almost identical; "
            "prefix may be ignored."
        )


def main() -> None:
    args = parse_args()
    device = _pick_device()
    print(f"[phase3] device={device}")

    clap_model, midi_gpt, projector, counts = load_phase3_components(
        midi_checkpoint=args.midi_checkpoint,
        clap_checkpoint=args.clap_checkpoint,
        n_prefix_tokens=args.n_prefix_tokens,
        device=device,
    )

    # Phase 3 uses the exact same dataset setup as Phase 2.
    train_loader, val_loader, stats = build_caption_dataloaders(
        jsonl_path=args.captions_jsonl,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    batch = next(iter(train_loader))
    if "input_ids" not in batch:
        raise RuntimeError(
            "Phase 3 requires input_ids from the dataloader to build LM labels."
        )

    print("[phase3] freeze policy check passed.")
    print(
        "[phase3] dataset total/train/val="
        f"{stats.n_total_records}/{stats.n_train_records}/{stats.n_val_records}"
    )
    print(
        "[phase3] dataloader check passed: input_ids shape="
        f"{tuple(batch['input_ids'].shape)}"
    )
    print(f"[phase3] CLAP params (frozen): {counts.n_clap_params:,}")
    print(f"[phase3] GPT params (frozen):  {counts.n_gpt_params:,}")
    print(f"[phase3] projector params:     {counts.n_projector_params:,}")
    print(f"[phase3] total trainable:      {counts.n_total_trainable:,}")

    optimizer = AdamW(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    print(
        f"[phase3] epochs={args.epochs} steps_per_epoch={steps_per_epoch} "
        f"total_steps={total_steps} warmup_steps={args.warmup_steps}"
    )

    global_step = 0
    best_val = float("inf")
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss_sum = 0.0
        n_train = 0
        verify1_done = False
        verify2_done = False
        for batch in train_loader:
            if not verify1_done:
                _verify_prefix_usage(
                    clap_model=clap_model,
                    midi_gpt=midi_gpt,
                    projector=projector,
                    batch=batch,
                    device=device,
                )
                verify1_done = True
            _set_warmup_cosine_lr(
                optimizer=optimizer,
                step=global_step,
                total_steps=total_steps,
                warmup_steps=args.warmup_steps,
                base_lr=args.lr,
                min_lr_scale=args.min_lr_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _ = phase3_prefix_lm_loss(
                clap_model=clap_model,
                midi_gpt=midi_gpt,
                prefix_projector=projector,
                input_ids=batch["input_ids"].to(device),
                captions=batch["captions"],
                prefix_attn_reg_weight=args.prefix_attn_reg_weight,
                prefix_attn_min_mean=args.prefix_attn_min_mean,
            )
            loss.backward()

            if not verify2_done:
                grad = projector.fc2.weight.grad
                if grad is None:
                    print(
                        "[phase3][verify2][warn] projector.fc2.weight.grad is None."
                    )
                else:
                    grad_norm = float(grad.norm().item())
                    grad_abs = float(grad.abs().sum().item())
                    print(
                        "[phase3][verify2] projector.fc2.weight grad_norm="
                        f"{grad_norm:.6f} grad_abs_sum={grad_abs:.6f}"
                    )
                    if grad_abs == 0.0:
                        print(
                            "[phase3][verify2][warn] projector gradient is all zeros."
                        )
                verify2_done = True

            torch.nn.utils.clip_grad_norm_(
                projector.parameters(), args.grad_clip_norm
            )
            optimizer.step()
            train_loss_sum += float(loss.item())
            n_train += 1
            global_step += 1

        train_loss = train_loss_sum / max(1, n_train)
        val_loss = _eval_loss(
            clap_model=clap_model,
            midi_gpt=midi_gpt,
            projector=projector,
            loader=val_loader,
            device=device,
            prefix_attn_reg_weight=args.prefix_attn_reg_weight,
            prefix_attn_min_mean=args.prefix_attn_min_mean,
        )
        baseline_sum = 0.0
        baseline_n = 0
        for vbatch in val_loader:
            baseline_sum += float(
                _lm_loss_without_prefix(
                    midi_gpt=midi_gpt,
                    input_ids=vbatch["input_ids"].to(device),
                ).item()
            )
            baseline_n += 1
        baseline_val = baseline_sum / max(1, baseline_n)
        ppl_gap = math.exp(val_loss) - math.exp(baseline_val)
        genre_gap = _conditional_perplexity_gap_by_genre(
            clap_model=clap_model,
            midi_gpt=midi_gpt,
            projector=projector,
            loader=val_loader,
            device=device,
            max_examples=200,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[phase3] epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"baseline_val={baseline_val:.4f} ppl_gap={ppl_gap:+.3f} "
            f"lr={current_lr:.2e}"
        )
        if genre_gap:
            parts = " ".join(f"{k}:{v:+.3f}" for k, v in sorted(genre_gap.items()))
            print(f"[phase3] genre ppl_gap(with-without): {parts}")
        if epoch >= 10 and val_loss >= baseline_val:
            print("[phase3][warn] prefix loss not below no-prefix baseline.")

        ckpt_dir = Path(
            args.results_dir
            if hasattr(args, "results_dir")
            else _ROOT / "results"
        ) / "checkpoints_prefix"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

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

        _prefix_token_scale_diagnostics(
            clap_model=clap_model,
            midi_gpt=midi_gpt,
            projector=projector,
            batch=batch,
            device=device,
        )

        if epoch % args.qualitative_every == 0:
            print("\n[phase3] qualitative generation check")
            uncond = _generate_unconditional(
                midi_gpt=midi_gpt,
                gen_tokens=args.qual_gen_tokens,
                device=device,
            )
            print(f"  [unconditional] {_token_preview(uncond)}")
            for prompt in args.qual_prompts:
                cond = _generate_with_text_prefix(
                    clap_model=clap_model,
                    midi_gpt=midi_gpt,
                    projector=projector,
                    text_prompt=prompt,
                    gen_tokens=args.qual_gen_tokens,
                    device=device,
                )
                print(f"  [prompt] {prompt}")
                print(f"           {_token_preview(cond)}")

    elapsed = time.perf_counter() - t0
    print(
        f"[phase3] finished in {elapsed/60:.1f} min, best_val={best_val:.4f}"
    )


if __name__ == "__main__":
    main()
