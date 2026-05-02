"""Sample MIDI/WAV from a CompoundGPT checkpoint only (no CLAP / prefix weights)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pretty_midi
import scipy.io.wavfile
import torch
import torch.nn.functional as F

from compound import (
    SENTINELS,
    STEP_BAR_END,
    STEP_BOS,
    STEP_CHORD_END,
    STEP_EOS,
    STEP_PB,
    decode_compound,
)
from compound_model import CompoundGPT, CompoundGPTConfig, default_compound_config
from inference_pipeline import _pick_device


def _load_compound_gpt(ckpt_path: Path, device: torch.device) -> CompoundGPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = default_compound_config()
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


def _truncate_to_last_boundary(steps: Sequence[Sequence[int]]) -> List[List[int]]:
    boundaries = {STEP_EOS, STEP_BAR_END, STEP_CHORD_END}
    last = -1
    for i, s in enumerate(steps):
        if int(s[0]) in boundaries:
            last = i
    if last == -1:
        return [list(s) for s in steps]
    return [list(s) for s in steps[: last + 1]]


@torch.no_grad()
def _generate_one_sequence(
    model: CompoundGPT,
    device: torch.device,
    max_new_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> List[List[int]]:
    generated_steps: List[List[int]] = []
    bos = list(SENTINELS)
    bos[0] = STEP_BOS
    generated_steps.append(bos)

    for _ in range(max_new_steps):
        step_ids = torch.tensor([generated_steps], dtype=torch.long, device=device)
        if step_ids.size(1) > model.config.block_size:
            raise ValueError(
                f"sequence length {step_ids.size(1)} > block_size {model.config.block_size}"
            )
        position_ids = torch.arange(
            step_ids.size(1), device=device, dtype=torch.long
        ).unsqueeze(0)
        logits_per_axis = model(idx=step_ids, position_ids=position_ids)

        next_step: List[int] = []
        for axis_logits in logits_per_axis:
            axis_next = _sample_axis(
                logits=axis_logits[0, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_step.append(axis_next)

        if next_step[0] == STEP_EOS:
            next_step = [STEP_EOS] + SENTINELS[1:]
            generated_steps.append(next_step)
            break

        generated_steps.append(next_step)

    return _truncate_to_last_boundary(generated_steps)


def _steps_to_safe_midi_steps(steps: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(s) for s in steps if int(s[0]) != STEP_PB]


def _append_pm(dst: pretty_midi.PrettyMIDI, src: pretty_midi.PrettyMIDI, t0: float) -> None:
    for inst in src.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name
        )
        for n in inst.notes:
            new_inst.notes.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start + t0,
                    end=n.end + t0,
                )
            )
        for cc in inst.control_changes:
            new_inst.control_changes.append(
                pretty_midi.ControlChange(
                    number=cc.number,
                    value=cc.value,
                    time=float(cc.time) + t0,
                )
            )
        for pb in inst.pitch_bends:
            new_inst.pitch_bends.append(
                pretty_midi.PitchBend(pitch=pb.pitch, time=pb.time + t0)
            )
        if (
            new_inst.notes
            or new_inst.control_changes
            or new_inst.pitch_bends
        ):
            dst.instruments.append(new_inst)


def _synthesize_wav_numpy(pm: pretty_midi.PrettyMIDI, sample_rate: int) -> np.ndarray:
    """Fallback PCM when FluidSynth/pyfluidsynth is unavailable (simple additive tones)."""
    duration = float(pm.get_end_time())
    n_samples = int(np.ceil(duration * sample_rate)) + 1
    y = np.zeros(n_samples, dtype=np.float64)
    twopi = 2.0 * np.pi

    for inst in pm.instruments:
        for note in inst.notes:
            f = 440.0 * (2.0 ** ((float(note.pitch) - 69.0) / 12.0))
            i0 = max(0, int(note.start * sample_rate))
            i1 = min(n_samples, int(np.ceil(note.end * sample_rate)))
            if i1 <= i0:
                continue
            seg_len = i1 - i0
            t = (np.arange(seg_len, dtype=np.float64) + i0) / sample_rate
            ph = twopi * f * t
            vel = float(note.velocity) / 127.0
            sig = vel * (
                0.55 * np.sin(ph)
                + 0.28 * np.sin(2.0 * ph)
                + 0.12 * np.sin(3.0 * ph)
                + 0.05 * np.sin(4.0 * ph)
            )
            atk = max(1, int(0.008 * sample_rate))
            rel = max(1, int(0.04 * sample_rate))
            env = np.ones(seg_len, dtype=np.float64)
            env[:atk] *= np.linspace(0.0, 1.0, atk, endpoint=False)
            tail = min(rel, seg_len)
            env[-tail:] *= np.linspace(1.0, 0.0, tail, endpoint=False)
            y[i0:i1] += sig * env

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak * 0.85
    return y.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unconditional compound GPT sampling (checkpoint weights only)."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="compound_best.pt",
        help="CompoundGPT checkpoint (e.g. compound_best.pt).",
    )
    p.add_argument("--out-midi", type=str, default="results/compound_unconditional.mid")
    p.add_argument("--out-wav", type=str, default="results/compound_unconditional.wav")
    p.add_argument(
        "--target-seconds",
        type=float,
        default=60.0,
        help="Accumulate decoded MIDI segments until at least this duration.",
    )
    p.add_argument(
        "--max-segments",
        type=int,
        default=64,
        help="Safety cap on number of BOS..EOS sequences to stitch.",
    )
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=44100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _pick_device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpt_path = Path(args.checkpoint)
    print(f"[gen_compound_uncond] device={device} ckpt={ckpt_path}")

    model = _load_compound_gpt(ckpt_path, device=device)
    bs = model.config.block_size
    max_new = bs - 1

    pm_out = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    t_off = 0.0
    n_segments = 0

    while t_off < args.target_seconds and n_segments < args.max_segments:
        torch.manual_seed(args.seed + n_segments)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + n_segments)

        steps = _generate_one_sequence(
            model=model,
            device=device,
            max_new_steps=max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        safe = _steps_to_safe_midi_steps(steps)
        if len(safe) < 2:
            print("[gen_compound_uncond] warning: empty segment, retrying sweep")
            break
        seg = decode_compound(safe)
        dur = seg.get_end_time()
        if dur < 0.05:
            print("[gen_compound_uncond] warning: near-empty decode, stopping")
            break
        _append_pm(pm_out, seg, t_off)
        t_off = pm_out.get_end_time()
        n_segments += 1
        print(
            f"[gen_compound_uncond] segment={n_segments} steps={len(steps)} "
            f"seg_dur={dur:.2f}s total={t_off:.2f}s"
        )

    if t_off < 1.0:
        raise RuntimeError(
            "Generated MIDI is too short; try different --seed or sampling params."
        )

    midi_path = Path(args.out_midi)
    wav_path = Path(args.out_wav)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    pm_out.write(str(midi_path))
    print(f"[gen_compound_uncond] midi -> {midi_path} duration={t_off:.2f}s")

    try:
        audio = pm_out.fluidsynth(fs=args.sample_rate)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    except (ImportError, OSError, ValueError) as e:
        print(
            f"[gen_compound_uncond] fluidsynth unavailable ({e!s}); "
            "using numpy additive synthesizer fallback."
        )
        audio = _synthesize_wav_numpy(pm_out, args.sample_rate)
    max_samples = int(args.target_seconds * args.sample_rate)
    if audio.size > max_samples:
        audio = audio[:max_samples]
    audio = np.clip(audio, -1.0, 1.0)
    scipy.io.wavfile.write(
        str(wav_path), args.sample_rate, (audio * 32767.0).astype(np.int16)
    )
    print(f"[gen_compound_uncond] wav -> {wav_path} samples={audio.size}")


if __name__ == "__main__":
    main()
