"""Text-conditioned compound generation (Phase 4, compound path)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import pretty_midi
import torch
import torch.nn.functional as F

from compound import (
    AXIS_NAMES,
    SENTINELS,
    STEP_BAR_END,
    STEP_BOS,
    STEP_CHORD_END,
    STEP_EOS,
    STEP_PB,
    decode_compound,
)
from inference_pipeline import _pick_device
from prefix_projector import (
    PrefixProjector,
    clap_text_for_prefix_projector,
    load_phase3_compound_components,
)


def _compound_step_embeds(midi_compound_gpt, step_ids: torch.Tensor) -> torch.Tensor:
    """step_ids: (B, T, N_AXES) -> embeds: (B, T, d_model)."""
    out = midi_compound_gpt.input_embeds[0](step_ids[..., 0])
    for a in range(1, midi_compound_gpt.n_axes):
        out = out + midi_compound_gpt.input_embeds[a](step_ids[..., a])
    return out


def _sample_axis(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")

    l = logits.clone() / temperature
    if top_k > 0 and top_k < l.numel():
        values, _ = torch.topk(l, top_k)
        cutoff = values[-1]
        l = torch.where(l < cutoff, torch.tensor(float("-inf"), device=l.device), l)

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(l, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumprobs > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        l_filtered = torch.full_like(l, float("-inf"))
        l_filtered.scatter_(0, sorted_idx, sorted_logits)
        l = l_filtered

    probs = F.softmax(l, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def autoregressive_decode_compound(
    clap_model,
    midi_compound_gpt,
    prefix_projector: PrefixProjector,
    prompt: str,
    max_new_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> List[List[int]]:
    device = next(prefix_projector.parameters()).device

    text_emb = clap_text_for_prefix_projector(clap_model, [prompt], device=device)
    prefix_embeds = prefix_projector(text_emb)  # (1, K, d_model)

    generated_steps: List[List[int]] = []
    bos = list(SENTINELS)
    bos[0] = STEP_BOS
    generated_steps.append(bos)

    max_steps = (
        midi_compound_gpt.config.block_size - prefix_embeds.size(1) - 1
    )  # -1 for BOS

    for _ in range(min(max_new_steps, max_steps)):
        step_ids = torch.tensor([generated_steps], dtype=torch.long, device=device)
        token_embeds = _compound_step_embeds(midi_compound_gpt, step_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        seq_len = inputs_embeds.size(1)
        if seq_len > midi_compound_gpt.config.block_size:
            raise ValueError(
                "Requested generation exceeds CompoundGPT block size: "
                f"{seq_len} > {midi_compound_gpt.config.block_size}"
            )
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(
            0
        )
        logits_per_axis = midi_compound_gpt(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )

        next_step: List[int] = []
        for axis_idx, axis_logits in enumerate(logits_per_axis):
            axis_next = _sample_axis(
                logits=axis_logits[0, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_step.append(axis_next)

        # Keep EOS structurally valid: only step-axis carries EOS tag.
        if next_step[0] == STEP_EOS:
            next_step = [STEP_EOS] + SENTINELS[1:]
            generated_steps.append(next_step)
            break

        generated_steps.append(next_step)

    return generated_steps


def _truncate_to_last_boundary(steps: Sequence[Sequence[int]]) -> List[List[int]]:
    boundaries = {STEP_EOS, STEP_BAR_END, STEP_CHORD_END}
    last = -1
    for i, s in enumerate(steps):
        if int(s[0]) in boundaries:
            last = i
    if last == -1:
        return [list(s) for s in steps]
    return [list(s) for s in steps[: last + 1]]


def save_and_verify_compound_midi(
    steps: Sequence[Sequence[int]], out_path: Path
) -> Tuple[int, float]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_steps = [list(s) for s in steps if int(s[0]) != STEP_PB]
    decode_compound(safe_steps).write(str(out_path))
    pm = pretty_midi.PrettyMIDI(str(out_path))
    n_notes = sum(len(inst.notes) for inst in pm.instruments)
    duration = pm.get_end_time()
    if n_notes == 0 or duration < 1.0:
        raise RuntimeError(
            "Generated compound MIDI is empty or too short; "
            "check sampling/decoding behavior."
        )
    return n_notes, duration


def _step_preview(steps: Sequence[Sequence[int]], max_len: int = 20) -> str:
    rows = []
    for s in steps[:max_len]:
        rows.append(
            ", ".join(f"{name}={int(v)}" for name, v in zip(AXIS_NAMES, s))
        )
    suffix = " ..." if len(steps) > max_len else ""
    return " | ".join(rows) + suffix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate compound MIDI from text prompt."
    )
    p.add_argument(
        "--compound-midi-checkpoint",
        type=str,
        default="results/test_compound/checkpoints_compound/compound_best.pt",
    )
    p.add_argument(
        "--compound-clap-checkpoint",
        type=str,
        default="results/test_compound/checkpoints_contrastive_compound/clap_compound_best.pt",
    )
    p.add_argument(
        "--prefix-checkpoint",
        type=str,
        default="results/test_compound/checkpoints_prefix/prefix_projector_best.pt",
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument(
        "--out",
        type=str,
        default="results/test_compound/generated_compound_conditional.mid",
    )
    p.add_argument("--max-new-steps", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--n-prefix-tokens", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _pick_device()
    print(f"[gen_cond_compound] device={device}")

    clap_model, midi_compound_gpt, projector, _ = load_phase3_compound_components(
        compound_midi_checkpoint=args.compound_midi_checkpoint,
        compound_clap_checkpoint=args.compound_clap_checkpoint,
        n_prefix_tokens=args.n_prefix_tokens,
        device=device,
    )
    prefix_ckpt = torch.load(
        Path(args.prefix_checkpoint), map_location=device, weights_only=True
    )
    proj_state = None
    if isinstance(prefix_ckpt, dict):
        proj_state = (
            prefix_ckpt.get("projector_state_dict")
            or prefix_ckpt.get("model_state_dict")
            or prefix_ckpt.get("model")
        )
    if proj_state is None:
        proj_state = prefix_ckpt
    projector.load_state_dict(proj_state, strict=False)
    projector.eval()
    midi_compound_gpt.eval()
    clap_model.eval()

    steps = autoregressive_decode_compound(
        clap_model=clap_model,
        midi_compound_gpt=midi_compound_gpt,
        prefix_projector=projector,
        prompt=args.prompt,
        max_new_steps=args.max_new_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    steps = _truncate_to_last_boundary(steps)

    out_path = Path(args.out)
    n_notes, duration = save_and_verify_compound_midi(steps, out_path)
    print(f"[gen_cond_compound] output -> {out_path}")
    print(f"[gen_cond_compound] notes={n_notes} duration={duration:.2f}s")
    print(f"[gen_cond_compound] steps={len(steps)}")
    print(f"[gen_cond_compound] step preview: {_step_preview(steps)}")


if __name__ == "__main__":
    main()
