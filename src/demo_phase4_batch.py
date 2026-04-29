"""Phase 4f demo artifact builder: 8 prompts x 2 seeds + retrieval report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from generate_conditional import (
    EOS,
    autoregressive_decode,
    encode_text_prompt,
    make_initial_context,
    project_prefix,
    save_and_verify_midi,
    truncate_to_last_boundary,
)
from inference_pipeline import (
    _pick_device,
    load_clap,
    load_midi_gpt,
    load_prefix_projector,
)

DEMO_PROMPTS: List[str] = [
    (
        "a slow, melancholic piano piece in a minor key "
        "with sparse, flowing notes"
    ),
    (
        "an upbeat rock band with electric guitar, bass, and drums "
        "in a major key"
    ),
    (
        "a fast jazz trio with saxophone, piano, and bass - "
        "syncopated and energetic"
    ),
    (
        "a soft, ambient electronic piece with synthesizer pads, "
        "slow and atmospheric"
    ),
    (
        "a loud, driving hard rock piece with heavy guitar "
        "and a dense, busy texture"
    ),
    "a gentle classical piece for piano and strings, moderate tempo, expressive",
    (
        "a funky horn-driven arrangement with brass, guitar, and bass "
        "with strong rhythm"
    ),
    (
        "a sparse solo acoustic guitar piece, fingerpicked, "
        "quiet and introspective"
    ),
]


def _fixed_window(
    ids: List[int], max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    window = ids[:max_seq_len]
    valid_len = len(window)
    if valid_len < max_seq_len:
        window = window + [0] * (max_seq_len - valid_len)
    attention_mask = [1] * valid_len + [0] * (max_seq_len - valid_len)
    input_ids = torch.tensor([window], dtype=torch.long)
    mask = torch.tensor([attention_mask], dtype=torch.long)
    return input_ids, mask


def _write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        prompt_idx = int(row["prompt_idx"])
        by_prompt.setdefault(prompt_idx, []).append(row)

    lines: List[str] = ["# Phase 4 Demo Artifact", ""]
    for prompt_idx in sorted(by_prompt):
        entries = by_prompt[prompt_idx]
        entries = sorted(entries, key=lambda x: int(x["take"]))
        best = max(
            entries,
            key=lambda x: (int(x["is_top1"]), float(x["correct_similarity"])),
        )
        lines.append(f"## Prompt {prompt_idx + 1}")
        lines.append(f"- Text: {entries[0]['prompt']}")
        for row in entries:
            lines.append(
                "- Take {take} (seed={seed}): file=`{midi_path}` "
                "rank={correct_rank}/8 sim={correct_similarity:.4f} top1={is_top1}".format(
                    take=row["take"],
                    seed=row["seed"],
                    midi_path=row["midi_path"],
                    correct_rank=row["correct_rank"],
                    correct_similarity=float(row["correct_similarity"]),
                    is_top1=bool(row["is_top1"]),
                )
            )
        lines.append(
            "- Selected for demo: take {take} (`{midi_path}`)".format(
                take=best["take"], midi_path=best["midi_path"]
            )
        )
        lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Phase 4 batch demo artifact."
    )
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--midi-checkpoint", type=str, default="")
    p.add_argument("--clap-checkpoint", type=str, default="")
    p.add_argument("--prefix-checkpoint", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--repetition-window", type=int, default=64)
    p.add_argument("--n-prefix-tokens", type=int, default=0)
    p.add_argument("--seed-a", type=int, default=17)
    p.add_argument("--seed-b", type=int, default=29)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    midi_ckpt = Path(args.midi_checkpoint) if args.midi_checkpoint else (
        results_dir / "checkpoints" / "best_model.pt"
    )
    clap_ckpt = Path(args.clap_checkpoint) if args.clap_checkpoint else (
        results_dir / "checkpoints_contrastive" / "clap_best.pt"
    )
    prefix_ckpt = Path(args.prefix_checkpoint) if args.prefix_checkpoint else (
        results_dir / "checkpoints_prefix" / "prefix_projector_best.pt"
    )
    out_dir = (
        Path(args.out_dir) if args.out_dir else (results_dir / "demo_phase4")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_out_dir = out_dir / "midi"
    midi_out_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device()
    print(f"[demo4f] device={device}")
    midi_gpt, _ = load_midi_gpt(midi_ckpt, device=device)
    clap, _ = load_clap(clap_ckpt, midi_gpt=midi_gpt, device=device)
    override = None if args.n_prefix_tokens <= 0 else args.n_prefix_tokens
    projector, _ = load_prefix_projector(
        prefix_ckpt,
        gpt_d_model=midi_gpt.config.d_model,
        device=device,
        n_prefix_tokens_override=override,
    )
    midi_gpt.eval()
    clap.eval()
    clap.text_encoder.eval()
    projector.eval()

    with torch.no_grad():
        prompt_text = clap.encode_text(DEMO_PROMPTS, device=device)
        prompt_embs = F.normalize(clap.text_projection(prompt_text), p=2, dim=-1)

    seeds = [args.seed_a, args.seed_b]
    rows: List[Dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(DEMO_PROMPTS):
        for take_idx, seed in enumerate(seeds, start=1):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            with torch.no_grad():
                text_emb = encode_text_prompt(clap, prompt, device=device)
                prefix_embeds = project_prefix(projector, text_emb)
                inputs_embeds = make_initial_context(midi_gpt, prefix_embeds)
                max_required = inputs_embeds.size(1) + args.max_new_tokens
                if max_required > midi_gpt.config.block_size:
                    raise ValueError(
                        "Requested generation exceeds GPT block size: "
                        f"{max_required} > {midi_gpt.config.block_size}"
                    )
                generated_ids = autoregressive_decode(
                    midi_gpt=midi_gpt,
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    repetition_window=args.repetition_window,
                    eos_token_id=EOS,
                )
                generated_ids = truncate_to_last_boundary(generated_ids)

            midi_path = (
                midi_out_dir
                / f"prompt_{prompt_idx + 1:02d}_take_{take_idx}.mid"
            )
            n_notes, duration = save_and_verify_midi(generated_ids, midi_path)

            input_ids, attn_mask = _fixed_window(generated_ids, args.max_seq_len)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            with torch.no_grad():
                midi_feat = clap.encode_midi(input_ids, attn_mask)
                midi_emb = F.normalize(clap.midi_projection(midi_feat), p=2, dim=-1)
                sims = (midi_emb @ prompt_embs.t()).squeeze(0)
                sorted_idx = torch.argsort(sims, descending=True)
                rank = (
                    int(
                        (sorted_idx == prompt_idx)
                        .nonzero(as_tuple=False)[0]
                        .item()
                    )
                    + 1
                )
                best_idx = int(sorted_idx[0].item())

            row: Dict[str, Any] = {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "take": take_idx,
                "seed": seed,
                "midi_path": str(midi_path),
                "n_notes": n_notes,
                "duration_sec": float(duration),
                "correct_rank": rank,
                "is_top1": rank == 1,
                "correct_similarity": float(sims[prompt_idx].item()),
                "top1_prompt_idx": best_idx,
                "top1_similarity": float(sims[best_idx].item()),
            }
            rows.append(row)
            print(
                f"[demo4f] prompt={prompt_idx + 1} take={take_idx} seed={seed} "
                f"rank={rank}/8 file={midi_path.name}"
            )

    json_path = out_dir / "demo_phase4_results.json"
    md_path = out_dir / "demo_phase4_report.md"
    json_path.write_text(
        json.dumps({"prompts": DEMO_PROMPTS, "rows": rows}, indent=2)
    )
    _write_markdown(md_path, rows)
    print(f"[demo4f] wrote {json_path}")
    print(f"[demo4f] wrote {md_path}")


if __name__ == "__main__":
    main()
