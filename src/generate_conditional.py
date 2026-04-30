"""Phase 4b text-conditioned MIDI generation with soft prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pretty_midi
import torch
import torch.nn.functional as F

from inference_pipeline import (  # noqa: E402
    _pick_device,
    load_clap,
    load_midi_gpt,
    load_prefix_projector,
)
from prefix_projector import clap_text_for_prefix_projector  # noqa: E402
from tokenizer import BAR_END, EOS, ID2TOKEN, PHRASE_END, PHRASE_START, decode


def encode_text_prompt(
    clap, prompt: str, device: torch.device
) -> torch.Tensor:
    """Step 1: CLAP 256-d projected text emb (Phase 3 / contrastive space)."""
    return clap_text_for_prefix_projector(clap, [prompt], device)


def project_prefix(projector, text_emb: torch.Tensor) -> torch.Tensor:
    """Step 2: deterministic prefix projection to (1, K, d_model)."""
    return projector(text_emb)


def make_initial_context(
    midi_gpt, prefix_embeds: torch.Tensor
) -> torch.Tensor:
    """Step 3: BOS token embedding + prefix concat -> (1, K+1, d_model)."""
    bos = torch.tensor(
        [[PHRASE_START]], dtype=torch.long, device=prefix_embeds.device
    )
    token_embeds = midi_gpt.wte(bos)
    return torch.cat([prefix_embeds, token_embeds], dim=1)


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    recent_tokens: List[int],
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    logits = logits.clone() / temperature

    if repetition_penalty > 1.0 and recent_tokens:
        unique_recent = set(recent_tokens)
        idx = torch.tensor(
            list(unique_recent), dtype=torch.long, device=logits.device
        )
        logits[:, idx] = logits[:, idx] / repetition_penalty

    if top_k > 0 and top_k < logits.size(-1):
        values, _ = torch.topk(logits, top_k)
        cutoff = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)

        # Keep the smallest prefix with cumulative prob >= top_p.
        remove = cumprobs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered.scatter_(1, sorted_idx, sorted_logits)
        logits = logits_filtered

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def autoregressive_decode(
    midi_gpt,
    inputs_embeds: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
    eos_token_id: int,
) -> List[int]:
    """Step 4: cached autoregressive decoding."""
    generated = [PHRASE_START]
    seq_len = inputs_embeds.size(1)
    position_ids = torch.arange(
        seq_len, device=inputs_embeds.device, dtype=torch.long
    ).unsqueeze(0)

    out = midi_gpt(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        use_cache=True,
    )
    logits, past_key_values = out

    next_token = _sample_token(
        logits=logits[:, -1, :],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        recent_tokens=generated[-repetition_window:],
    )
    token_id = int(next_token.item())
    generated.append(token_id)
    if token_id == eos_token_id:
        return generated

    cur_pos = seq_len
    for _ in range(max_new_tokens - 1):
        step_pos = torch.tensor(
            [[cur_pos]], device=inputs_embeds.device, dtype=torch.long
        )
        out = midi_gpt(
            idx=next_token,
            position_ids=step_pos,
            use_cache=True,
            past_key_values=past_key_values,
        )
        logits, past_key_values = out
        next_token = _sample_token(
            logits=logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            recent_tokens=generated[-repetition_window:],
        )
        token_id = int(next_token.item())
        generated.append(token_id)
        cur_pos += 1
        if token_id == eos_token_id:
            break
    return generated


def truncate_to_last_boundary(ids: List[int]) -> List[int]:
    """Step 5 helper: truncate malformed tails at safe structural boundary."""
    boundaries = {EOS, PHRASE_END, BAR_END}
    last = -1
    for i, tid in enumerate(ids):
        if tid in boundaries:
            last = i
    if last == -1:
        return ids
    return ids[: last + 1]


def save_and_verify_midi(ids: List[int], out_path: Path) -> Tuple[int, float]:
    """Step 6: write MIDI and verify it's readable/non-empty."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decode(ids).write(str(out_path))
    pm = pretty_midi.PrettyMIDI(str(out_path))
    n_notes = sum(len(inst.notes) for inst in pm.instruments)
    duration = pm.get_end_time()
    if n_notes == 0 or duration < 1.0:
        raise RuntimeError(
            "Generated MIDI is empty or too short; "
            "check EOS behavior / decode logic."
        )
    return n_notes, duration


def _token_preview(ids: List[int], max_len: int = 60) -> str:
    toks = [ID2TOKEN.get(i, f"UNK({i})") for i in ids[:max_len]]
    suffix = " ..." if len(ids) > max_len else ""
    return " ".join(toks) + suffix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MIDI from text prompt.")
    p.add_argument(
        "--midi-checkpoint",
        type=str,
        default="results/checkpoints/best_model.pt",
    )
    p.add_argument(
        "--clap-checkpoint",
        type=str,
        default="results/checkpoints_contrastive/clap_best.pt",
    )
    p.add_argument(
        "--prefix-checkpoint",
        type=str,
        default="results/checkpoints_prefix/prefix_projector_best.pt",
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument(
        "--out", type=str, default="results/conditional_generated.mid"
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--repetition-window", type=int, default=64)
    p.add_argument("--n-prefix-tokens", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _pick_device()
    print(f"[gen_cond] device={device}")

    midi_gpt, _ = load_midi_gpt(Path(args.midi_checkpoint), device=device)
    clap, _ = load_clap(
        Path(args.clap_checkpoint), midi_gpt=midi_gpt, device=device
    )
    override = None if args.n_prefix_tokens <= 0 else args.n_prefix_tokens
    projector, _ = load_prefix_projector(
        Path(args.prefix_checkpoint),
        gpt_d_model=midi_gpt.config.d_model,
        device=device,
        n_prefix_tokens_override=override,
    )

    with torch.no_grad():
        # Step 1
        text_emb = encode_text_prompt(clap, args.prompt, device=device)
        # Step 2
        prefix_embeds = project_prefix(projector, text_emb)
        # Step 3
        inputs_embeds = make_initial_context(midi_gpt, prefix_embeds)

        max_required = inputs_embeds.size(1) + args.max_new_tokens
        if max_required > midi_gpt.config.block_size:
            raise ValueError(
                "Requested generation exceeds GPT block size: "
                f"{max_required} > {midi_gpt.config.block_size}"
            )

        # Step 4
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

        # Step 5
        generated_ids = truncate_to_last_boundary(generated_ids)

    # Step 6
    out_path = Path(args.out)
    n_notes, duration = save_and_verify_midi(generated_ids, out_path)
    print(f"[gen_cond] output -> {out_path}")
    print(f"[gen_cond] notes={n_notes} duration={duration:.2f}s")
    print(f"[gen_cond] token preview: {_token_preview(generated_ids)}")


if __name__ == "__main__":
    main()
