"""MIDI tokenizer for bach-gpt.

Vocabulary
----------
Structural:  PAD, EOS, BAR_START, BAR_END, PHRASE_START, PHRASE_END, REST (7)
Pitch:       MIDI 21..108                                                  (88)
Duration:    32 log-quantized bins over [0.03125, 4.0] seconds             (32)
Time-shift:  32 log-quantized bins over [0.03125, 4.0] seconds             (32)
Velocity:    8 uniform bins over [0, 127]                                  (8)

Total vocab size: 167 tokens.

API
---
encode(pm: pretty_midi.PrettyMIDI) -> List[int]
decode(ids: List[int])             -> pretty_midi.PrettyMIDI
round_trip_test(pm)                -> (passed: bool, details: dict)

Note on fidelity: encode/decode preserve the pitch multiset but timing is
fuzzy due to log quantization of durations and time-shifts. BAR_* and
PHRASE_* markers are emitted by encode based on PrettyMIDI downbeats when
a time signature is present; if not, bar markers are dropped. Phrase
markers are emitted at the start/end of each extracted stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pretty_midi


# --- Vocabulary construction --------------------------------------------------

STRUCTURAL = ["PAD", "EOS", "BAR_START", "BAR_END", "PHRASE_START", "PHRASE_END", "REST"]
PITCH_MIN, PITCH_MAX = 21, 108          # 88 pitches (A0..C8)
N_PITCH = PITCH_MAX - PITCH_MIN + 1     # 88

N_DUR_BINS = 32
N_SHIFT_BINS = 32
N_VEL_BINS = 8

# Log-quantization range: 2**-5 s (~31 ms) to 4 s.
TIME_MIN, TIME_MAX = 2 ** -5, 4.0
LOG_TIME_EDGES = np.linspace(math.log(TIME_MIN), math.log(TIME_MAX), N_DUR_BINS + 1)


def _build_vocab() -> Tuple[List[str], Dict[str, int]]:
    tokens: List[str] = list(STRUCTURAL)
    tokens += [f"P{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
    tokens += [f"D{i}" for i in range(N_DUR_BINS)]
    tokens += [f"TS{i}" for i in range(N_SHIFT_BINS)]
    tokens += [f"V{i}" for i in range(N_VEL_BINS)]
    t2i = {t: i for i, t in enumerate(tokens)}
    return tokens, t2i


TOKENS, TOKEN2ID = _build_vocab()
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE = len(TOKENS)

PAD = TOKEN2ID["PAD"]
EOS = TOKEN2ID["EOS"]
BAR_START = TOKEN2ID["BAR_START"]
BAR_END = TOKEN2ID["BAR_END"]
PHRASE_START = TOKEN2ID["PHRASE_START"]
PHRASE_END = TOKEN2ID["PHRASE_END"]
REST = TOKEN2ID["REST"]


# --- Quantization helpers -----------------------------------------------------

def _log_bin(x: float, edges=LOG_TIME_EDGES) -> int:
    """Map a positive time (s) to a log-bin index in [0, N-1]."""
    x = max(x, TIME_MIN)
    x = min(x, TIME_MAX)
    logx = math.log(x)
    # digitize returns 1..len(edges)-1; clip to valid bin range.
    idx = int(np.digitize(logx, edges)) - 1
    return max(0, min(N_DUR_BINS - 1, idx))


def _bin_center(i: int, edges=LOG_TIME_EDGES) -> float:
    lo, hi = edges[i], edges[i + 1]
    return math.exp(0.5 * (lo + hi))


def _vel_bin(v: int) -> int:
    v = max(0, min(127, int(v)))
    return min(N_VEL_BINS - 1, v * N_VEL_BINS // 128)


def _vel_center(i: int) -> int:
    return int((i + 0.5) * 128 / N_VEL_BINS)


# --- Encode -------------------------------------------------------------------

@dataclass
class _Event:
    onset: float
    pitch: int
    duration: float
    velocity: int


def _extract_events(pm: pretty_midi.PrettyMIDI) -> List[_Event]:
    events: List[_Event] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if PITCH_MIN <= n.pitch <= PITCH_MAX:
                events.append(_Event(n.start, n.pitch, max(n.end - n.start, TIME_MIN), n.velocity))
    events.sort(key=lambda e: (e.onset, e.pitch))
    return events


def _downbeats(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    try:
        db = pm.get_downbeats()
        return np.asarray(db) if db is not None else np.array([])
    except Exception:
        return np.array([])


def encode(pm: pretty_midi.PrettyMIDI) -> List[int]:
    """Encode a PrettyMIDI object to a list of vocabulary ids."""
    events = _extract_events(pm)
    ids: List[int] = [PHRASE_START]
    if not events:
        ids.append(PHRASE_END)
        ids.append(EOS)
        return ids

    downbeats = list(_downbeats(pm))
    db_iter = iter(downbeats)
    next_db = next(db_iter, None)
    in_bar = False

    prev_onset = events[0].onset
    # Open first bar at or before first event if downbeats exist.
    while next_db is not None and next_db <= prev_onset + 1e-6:
        if in_bar:
            ids.append(BAR_END)
        ids.append(BAR_START)
        in_bar = True
        next_db = next(db_iter, None)

    for idx, ev in enumerate(events):
        # Emit bar markers that fall at/before this event.
        while next_db is not None and next_db <= ev.onset + 1e-6:
            if in_bar:
                ids.append(BAR_END)
            ids.append(BAR_START)
            in_bar = True
            next_db = next(db_iter, None)

        if idx == 0:
            shift = 0.0
        else:
            shift = ev.onset - prev_onset
        if shift > TIME_MIN:
            ids.append(TOKEN2ID[f"TS{_log_bin(shift)}"])

        ids.append(TOKEN2ID[f"V{_vel_bin(ev.velocity)}"])
        ids.append(TOKEN2ID[f"P{ev.pitch}"])
        ids.append(TOKEN2ID[f"D{_log_bin(ev.duration)}"])

        prev_onset = ev.onset

    if in_bar:
        ids.append(BAR_END)
    ids.append(PHRASE_END)
    ids.append(EOS)
    return ids


# --- Decode -------------------------------------------------------------------

def decode(ids: List[int], tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """Decode a token id list back to a PrettyMIDI. Timing is reconstructed
    from the bin centers of TS and D tokens and is not exact."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0, name="piano")

    current_time = 0.0
    pending_velocity = 64

    # State for pitch-duration pairing: when we see a P token, we look
    # ahead in the stream for its D token (encoded immediately after).
    def _kind(t: str):
        # Structural tokens have explicit names, so check those first.
        if t in STRUCTURAL:
            return ("struct", t)
        if t.startswith("TS") and t[2:].isdigit():
            return ("ts", int(t[2:]))
        if t.startswith("D") and t[1:].isdigit():
            return ("dur", int(t[1:]))
        if t.startswith("V") and t[1:].isdigit():
            return ("vel", int(t[1:]))
        if t.startswith("P") and t[1:].isdigit():
            return ("pitch", int(t[1:]))
        return ("struct", t)

    i = 0
    while i < len(ids):
        kind, val = _kind(ID2TOKEN.get(ids[i], "PAD"))

        if kind == "ts":
            current_time += _bin_center(val)

        elif kind == "vel":
            pending_velocity = _vel_center(val)

        elif kind == "pitch":
            duration = 0.25  # fallback
            j = i + 1
            while j < len(ids):
                nkind, nval = _kind(ID2TOKEN.get(ids[j], "PAD"))
                if nkind == "dur":
                    duration = _bin_center(nval)
                    break
                if nkind in ("pitch", "ts", "vel"):
                    break
                j += 1
            note = pretty_midi.Note(
                velocity=int(pending_velocity),
                pitch=int(val),
                start=current_time,
                end=current_time + max(duration, 0.01),
            )
            inst.notes.append(note)

        elif kind == "struct" and val == "REST":
            current_time += 0.25
        # BAR_*, PHRASE_*, PAD, EOS do not affect decoded timing
        i += 1

    pm.instruments.append(inst)
    return pm


# --- Round-trip test ----------------------------------------------------------

def round_trip_test(pm: pretty_midi.PrettyMIDI) -> Tuple[bool, Dict]:
    """Verify the pitch multiset is preserved through encode+decode.

    Timing is not checked because log quantization is lossy.
    """
    original = sorted(
        n.pitch
        for inst in pm.instruments
        if not inst.is_drum
        for n in inst.notes
        if PITCH_MIN <= n.pitch <= PITCH_MAX
    )
    ids = encode(pm)
    pm2 = decode(ids)
    reconstructed = sorted(
        n.pitch for inst in pm2.instruments for n in inst.notes
    )
    passed = original == reconstructed
    return passed, {
        "n_orig": len(original),
        "n_recon": len(reconstructed),
        "n_tokens": len(ids),
        "vocab_size": VOCAB_SIZE,
    }


# --- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"  structural:   {len(STRUCTURAL)}")
    print(f"  pitch:        {N_PITCH}")
    print(f"  duration:     {N_DUR_BINS}")
    print(f"  time-shift:   {N_SHIFT_BINS}")
    print(f"  velocity:     {N_VEL_BINS}")

    # Smoke test: build a tiny C-major scale and round-trip it.
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    for p in [60, 62, 64, 65, 67, 69, 71, 72]:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=t, end=t + 0.5))
        t += 0.5
    pm.instruments.append(inst)
    ok, info = round_trip_test(pm)
    print(f"\nSmoke test round-trip: {'PASS' if ok else 'FAIL'}  {info}")
