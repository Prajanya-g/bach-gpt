"""MIDI tokenizer for bach-gpt.

Vocabulary
----------
Structural:  PAD, EOS, BAR_START, BAR_END, PHRASE_START, PHRASE_END,
             REST, CHORD_START, CHORD_END                                  (9)
Pitch:       MIDI 21..108                                                  (88)
Duration:    32 log-quantized bins over [0.03125, 4.0] seconds             (32)
Time-shift:  32 log-quantized bins over [0.03125, 4.0] seconds             (32)
Velocity:    16 uniform bins over [0, 127] (V0..V15)                       (16)
Voice/chan:  16 GM families + 1 drums (VC0..VC16)                          (17)
Tempo:       16 log-spaced bins over [40, 240] BPM (T0..T15)               (16)
Position:    16 sub-beat positions per bar (POS0..POS15)                   (16)
Meter:       8 common time signatures + OTHER (METER_*)                     (9)
Voice role:  ROLE_BASS, ROLE_INNER, ROLE_TOP (within chord brackets)        (3)

Total vocab size: 238 tokens.

API
---
encode(pm: pretty_midi.PrettyMIDI) -> List[int]
decode(ids: List[int])             -> pretty_midi.PrettyMIDI
round_trip_test(pm)                -> (passed: bool, details: dict)

Note on fidelity: encode/decode preserve the pitch multiset and instrument
family per note but timing is fuzzy due to log quantization. BAR_* and
PHRASE_* markers are emitted from PrettyMIDI downbeats when a time signature
is present. A tempo token (T*) is emitted at PHRASE_START and on tempo
changes. A VC* token is emitted whenever the active track's instrument
family changes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi


# --- Vocabulary construction --------------------------------------------------

STRUCTURAL = [
    "PAD",
    "EOS",
    "BAR_START",
    "BAR_END",
    "PHRASE_START",
    "PHRASE_END",
    "REST",
    "CHORD_START",
    "CHORD_END",
]
ROLES = ["ROLE_BASS", "ROLE_INNER", "ROLE_TOP"]
# 24 keys: indices 0..11 = C..B major; 12..23 = C..B minor (PrettyMIDI's
# convention via key_number).
KEYS = [f"KEY_{i}" for i in range(24)]
# Bar-header axes: emitted right after each BAR_START as a coarse summary
# of the bar's harmonic, density, and register content.
ROOT_NAMES = [f"ROOT_{i}" for i in range(12)]
N_DENS_BINS = 4   # 0-3, 4-7, 8-15, 16+
N_REG_BINS = 4    # <48, 48-59, 60-71, 72+
DENS_NAMES = [f"DENS_{i}" for i in range(N_DENS_BINS)]
REG_NAMES = [f"REG_{i}" for i in range(N_REG_BINS)]
# Bar-repetition markers: emitted right after bar header when this bar's
# pitch multiset matches a bar K positions earlier (K in {1, 2, 4, 8}).
REF_DISTANCES = [1, 2, 4, 8]
REF_NAMES = [f"REF_BAR_{k}" for k in REF_DISTANCES]
# Caption-segment markers for cross-modal alignment with MidiCaps-style
# multi-sentence captions. Emit CAP_SEG_<i> at PHRASE_START to bind the
# next phrase to the i-th caption segment.
N_CAP_SEGS = 8
CAP_SEG_NAMES = [f"CAP_SEG_{i}" for i in range(N_CAP_SEGS)]

# Pedal tokens: GM CC#64 (sustain), CC#66 (sostenuto), CC#67 (soft).
PEDAL_CC_NUMBERS = {64: "SUS", 66: "SOS", 67: "SFT"}
PEDAL_NAMES = [
    f"PEDAL_{p}_{state}"
    for p in PEDAL_CC_NUMBERS.values()
    for state in ("UP", "DOWN")
]

# Continuous-controller tokens for the most common expressive CCs, each
# quantized to 8 bins over [0, 128).
N_CC_BINS = 8
CC_TYPES = {1: "MOD", 7: "VOL", 10: "PAN", 11: "EXPR"}
CC_NAMES = [
    f"CC_{name}_{i}"
    for name in CC_TYPES.values()
    for i in range(N_CC_BINS)
]

# Pitch-bend tokens. PrettyMIDI gives 14-bit values in [-8192, 8191];
# quantize to 16 uniform bins and emit as PB_<i>.
N_PB_BINS = 16
PB_NAMES = [f"PB_{i}" for i in range(N_PB_BINS)]

# Reverse maps for the decoder: short name -> CC number.
PEDAL_NAME_TO_CC = {v: k for k, v in PEDAL_CC_NUMBERS.items()}
CC_NAME_TO_NUMBER = {v: k for k, v in CC_TYPES.items()}
METERS = [
    "METER_2_4",
    "METER_3_4",
    "METER_4_4",
    "METER_5_4",
    "METER_6_8",
    "METER_7_8",
    "METER_9_8",
    "METER_12_8",
    "METER_OTHER",
]
# Bar length in quarter-notes for each meter (used by decoder).
METER_QUARTERS: Dict[str, float] = {
    "METER_2_4": 2.0,
    "METER_3_4": 3.0,
    "METER_4_4": 4.0,
    "METER_5_4": 5.0,
    "METER_6_8": 3.0,
    "METER_7_8": 3.5,
    "METER_9_8": 4.5,
    "METER_12_8": 6.0,
    "METER_OTHER": 4.0,
}
PITCH_MIN, PITCH_MAX = 21, 108          # 88 pitches (A0..C8)
N_PITCH = PITCH_MAX - PITCH_MIN + 1     # 88

N_DUR_BINS = 32
N_SHIFT_BINS = 32
N_VEL_BINS = 16
# 16 GM instrument families + 1 reserved drum voice (VC16).
N_VOICE_BINS = 17
DRUM_VOICE = 16
N_TEMPO_BINS = 16
# Sub-beat resolution per bar (sixteenth-note grid in 4/4).
N_POS_BINS = 16

# Log-quantization range: 2**-5 s (~31 ms) to 4 s.
TIME_MIN, TIME_MAX = 2 ** -5, 4.0
LOG_TIME_EDGES = np.linspace(math.log(TIME_MIN), math.log(TIME_MAX), N_DUR_BINS + 1)

# Tempo log-quantization range: 40..240 BPM.
TEMPO_MIN, TEMPO_MAX = 40.0, 240.0
LOG_TEMPO_EDGES = np.linspace(
    math.log(TEMPO_MIN), math.log(TEMPO_MAX), N_TEMPO_BINS + 1
)


def _build_vocab() -> Tuple[List[str], Dict[str, int]]:
    tokens: List[str] = list(STRUCTURAL)
    tokens += [f"P{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
    tokens += [f"D{i}" for i in range(N_DUR_BINS)]
    tokens += [f"TS{i}" for i in range(N_SHIFT_BINS)]
    tokens += [f"V{i}" for i in range(N_VEL_BINS)]
    tokens += [f"VC{i}" for i in range(N_VOICE_BINS)]
    tokens += [f"T{i}" for i in range(N_TEMPO_BINS)]
    tokens += [f"POS{i}" for i in range(N_POS_BINS)]
    tokens += list(METERS)
    tokens += list(ROLES)
    tokens += list(KEYS)
    tokens += list(ROOT_NAMES)
    tokens += list(DENS_NAMES)
    tokens += list(REG_NAMES)
    tokens += list(REF_NAMES)
    tokens += list(CAP_SEG_NAMES)
    tokens += list(PEDAL_NAMES)
    tokens += list(CC_NAMES)
    tokens += list(PB_NAMES)
    t2i = {t: i for i, t in enumerate(tokens)}
    return tokens, t2i


TOKENS, TOKEN2ID = _build_vocab()
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE = len(TOKENS)

# Default location for fitted velocity quantiles. See _maybe_load_vel_edges
# at the end of this module for the auto-load.
DEFAULT_VEL_QUANTILES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "tokenizer" / "velocity_quantiles.json"
)

PAD = TOKEN2ID["PAD"]
EOS = TOKEN2ID["EOS"]
BAR_START = TOKEN2ID["BAR_START"]
BAR_END = TOKEN2ID["BAR_END"]
PHRASE_START = TOKEN2ID["PHRASE_START"]
PHRASE_END = TOKEN2ID["PHRASE_END"]
REST = TOKEN2ID["REST"]
CHORD_START = TOKEN2ID["CHORD_START"]
CHORD_END = TOKEN2ID["CHORD_END"]
ROLE_BASS = TOKEN2ID["ROLE_BASS"]
ROLE_INNER = TOKEN2ID["ROLE_INNER"]
ROLE_TOP = TOKEN2ID["ROLE_TOP"]


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


# Optional corpus-fit quantile edges for velocity bins. When set, _vel_bin
# uses these instead of uniform binning. Loaded from a JSON file by
# load_velocity_quantiles(); fit_velocity_quantiles() trains them.
_VEL_EDGES: Optional[np.ndarray] = None


def _vel_bin(v: int) -> int:
    v = max(0, min(127, int(v)))
    if _VEL_EDGES is not None:
        idx = int(np.searchsorted(_VEL_EDGES, v, side="right")) - 1
        return max(0, min(N_VEL_BINS - 1, idx))
    return min(N_VEL_BINS - 1, v * N_VEL_BINS // 128)


def _vel_center(i: int) -> int:
    if _VEL_EDGES is not None:
        lo = float(_VEL_EDGES[i])
        hi = float(_VEL_EDGES[i + 1])
        return int(round(0.5 * (lo + hi)))
    return int((i + 0.5) * 128 / N_VEL_BINS)


def fit_velocity_quantiles(velocities: List[int], n_bins: int = N_VEL_BINS) -> List[float]:
    """Compute quantile bin edges for a corpus of velocity values."""
    if not velocities:
        return [i * 128 / n_bins for i in range(n_bins + 1)]
    vs = np.asarray(velocities, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(vs, qs).tolist()
    edges[0] = 0.0
    edges[-1] = 128.0
    # Force monotonic increasing in case of heavy ties.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-3
    return edges


def load_velocity_quantiles(path: Path) -> bool:
    """Install bin edges from a JSON file. Returns True if loaded."""
    global _VEL_EDGES
    p = Path(path)
    if not p.exists():
        return False
    edges = json.loads(p.read_text())
    if not isinstance(edges, list) or len(edges) != N_VEL_BINS + 1:
        return False
    _VEL_EDGES = np.asarray(edges, dtype=np.float64)
    return True


def save_velocity_quantiles(edges: List[float], path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(list(edges)))


def _tempo_bin(bpm: float) -> int:
    bpm = max(TEMPO_MIN, min(TEMPO_MAX, float(bpm)))
    idx = int(np.digitize(math.log(bpm), LOG_TEMPO_EDGES)) - 1
    return max(0, min(N_TEMPO_BINS - 1, idx))


def _tempo_center(i: int) -> float:
    lo, hi = LOG_TEMPO_EDGES[i], LOG_TEMPO_EDGES[i + 1]
    return math.exp(0.5 * (lo + hi))


def _program_family(program: int) -> int:
    """Map a General MIDI program (0..127) to its 16-family index."""
    return max(0, min(N_VOICE_BINS - 1, int(program) // 8))


# --- Encode -------------------------------------------------------------------

@dataclass
class _Event:
    onset: float
    voice: int               # GM family index 0..15 or DRUM_VOICE
    kind: str = "note"       # 'note' | 'pedal' | 'cc' | 'pb'
    # Note fields
    pitch: int = 0
    duration: float = 0.0
    velocity: int = 0
    # Pedal fields
    pedal_type: str = "SUS"  # 'SUS' | 'SOS' | 'SFT'
    pedal_state: str = "UP"  # 'UP' | 'DOWN'
    # CC fields (continuous controllers)
    cc_type: str = "MOD"     # name from CC_TYPES values
    cc_bin: int = 0
    # Pitch-bend fields
    pb_bin: int = 0


def _cc_bin(value: int) -> int:
    v = max(0, min(127, int(value)))
    return min(N_CC_BINS - 1, v * N_CC_BINS // 128)


def _cc_center(i: int) -> int:
    return int((i + 0.5) * 128 / N_CC_BINS)


def _pb_bin(value: int) -> int:
    v = max(-8192, min(8191, int(value)))
    return min(N_PB_BINS - 1, (v + 8192) * N_PB_BINS // 16384)


def _pb_center(i: int) -> int:
    return int((i + 0.5) * 16384 / N_PB_BINS) - 8192


def _extract_events(pm: pretty_midi.PrettyMIDI) -> List[_Event]:
    events: List[_Event] = []
    for inst in pm.instruments:
        voice = DRUM_VOICE if inst.is_drum else _program_family(inst.program)
        for n in inst.notes:
            if PITCH_MIN <= n.pitch <= PITCH_MAX:
                events.append(
                    _Event(
                        onset=n.start, voice=voice, kind="note",
                        pitch=n.pitch,
                        duration=max(n.end - n.start, TIME_MIN),
                        velocity=n.velocity,
                    )
                )
        for cc in getattr(inst, "control_changes", []) or []:
            num = int(cc.number)
            if num in PEDAL_CC_NUMBERS:
                events.append(
                    _Event(
                        onset=float(cc.time), voice=voice, kind="pedal",
                        pedal_type=PEDAL_CC_NUMBERS[num],
                        pedal_state="DOWN" if int(cc.value) >= 64 else "UP",
                    )
                )
            elif num in CC_TYPES:
                events.append(
                    _Event(
                        onset=float(cc.time), voice=voice, kind="cc",
                        cc_type=CC_TYPES[num],
                        cc_bin=_cc_bin(cc.value),
                    )
                )
        for pb in getattr(inst, "pitch_bends", []) or []:
            events.append(
                _Event(
                    onset=float(pb.time), voice=voice, kind="pb",
                    pb_bin=_pb_bin(pb.pitch),
                )
            )
    # Note events sort by pitch within onset; non-note events keep their input order.
    events.sort(key=lambda e: (e.onset, e.kind != "note", e.voice, e.pitch))
    return events


def _tempo_changes(pm: pretty_midi.PrettyMIDI) -> List[Tuple[float, float]]:
    """Return sorted (time_s, bpm) pairs from a PrettyMIDI's tempo map."""
    try:
        times, tempos = pm.get_tempo_changes()
    except Exception:
        return [(0.0, 120.0)]
    pairs = list(zip(times.tolist(), tempos.tolist())) if len(times) else []
    if not pairs or pairs[0][0] > 1e-6:
        pairs.insert(0, (0.0, pairs[0][1] if pairs else 120.0))
    return pairs


def _meter_token(num: int, den: int) -> str:
    name = f"METER_{num}_{den}"
    return name if name in METER_QUARTERS else "METER_OTHER"


def _key_changes(pm: pretty_midi.PrettyMIDI) -> List[Tuple[float, str]]:
    """Return sorted (time_s, key_name) pairs from key_signature_changes."""
    out: List[Tuple[float, str]] = []
    for ks in getattr(pm, "key_signature_changes", []) or []:
        kn = int(getattr(ks, "key_number", 0)) % 24
        out.append((float(ks.time), f"KEY_{kn}"))
    if not out or out[0][0] > 1e-6:
        out.insert(0, (0.0, out[0][1] if out else "KEY_0"))
    return out


def _meter_changes(pm: pretty_midi.PrettyMIDI) -> List[Tuple[float, str]]:
    """Return sorted (time_s, meter_name) pairs from time-signature changes."""
    out: List[Tuple[float, str]] = []
    for ts in getattr(pm, "time_signature_changes", []) or []:
        out.append((float(ts.time), _meter_token(ts.numerator, ts.denominator)))
    if not out or out[0][0] > 1e-6:
        out.insert(0, (0.0, out[0][1] if out else "METER_4_4"))
    return out


def _bin_density(n: int) -> int:
    if n < 4:
        return 0
    if n < 8:
        return 1
    if n < 16:
        return 2
    return 3


def _bin_register(mean_pitch: float) -> int:
    if mean_pitch < 48:
        return 0
    if mean_pitch < 60:
        return 1
    if mean_pitch < 72:
        return 2
    return 3


def _bar_header_tokens(bar_events: List["_Event"]) -> List[int]:
    """Return [ROOT_<n>, DENS_<n>, REG_<n>] tokens summarizing a bar."""
    if not bar_events:
        return [
            TOKEN2ID[ROOT_NAMES[0]],
            TOKEN2ID[DENS_NAMES[0]],
            TOKEN2ID[REG_NAMES[0]],
        ]
    lowest = min(e.pitch for e in bar_events)
    root_pc = lowest % 12
    n = len(bar_events)
    mean_pitch = sum(e.pitch for e in bar_events) / n
    return [
        TOKEN2ID[ROOT_NAMES[root_pc]],
        TOKEN2ID[DENS_NAMES[_bin_density(n)]],
        TOKEN2ID[REG_NAMES[_bin_register(mean_pitch)]],
    ]


def _group_by_onset(events: List["_Event"], eps: float = TIME_MIN) -> List[List["_Event"]]:
    """Group consecutive *note* events with coincident onsets into chord
    groups. Non-note events (pedal/cc/pb) are emitted as size-1 groups so
    they keep their place in the timeline but never bracket as chords.
    """
    groups: List[List[_Event]] = []
    cur: List[_Event] = []
    cur_onset: Optional[float] = None

    def _flush() -> None:
        nonlocal cur, cur_onset
        if cur:
            groups.append(cur)
            cur = []
            cur_onset = None

    for ev in events:
        if ev.kind != "note":
            _flush()
            groups.append([ev])
            continue
        if cur_onset is None or abs(ev.onset - cur_onset) <= eps:
            cur.append(ev)
            if cur_onset is None:
                cur_onset = ev.onset
        else:
            _flush()
            cur = [ev]
            cur_onset = ev.onset
    _flush()
    return groups


def _downbeats(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    try:
        db = pm.get_downbeats()
        return np.asarray(db) if db is not None else np.array([])
    except Exception:
        return np.array([])


def encode(pm: pretty_midi.PrettyMIDI) -> List[int]:
    """Encode a PrettyMIDI object to a list of vocabulary ids.

    Stream layout: PHRASE_START T<n> METER_X_Y [VC<v>] [BAR_START] ...
    For each onset group (chord = co-located notes):
        [tempo/meter/bar tokens if any cross this onset]
        [POS<p> if in a bar AND position changed; otherwise TS<n> as fallback]
        if size>1: CHORD_START <per-note: VC?, ROLE, V, P, D> CHORD_END
        else:                  <per-note: VC?, V, P, D>
    Notes within a chord are sorted by pitch ascending; lowest gets ROLE_BASS,
    highest ROLE_TOP, middle pitches ROLE_INNER.
    """
    events = _extract_events(pm)
    ids: List[int] = [PHRASE_START]
    if not events:
        ids.append(PHRASE_END)
        ids.append(EOS)
        return ids

    tempo_map = _tempo_changes(pm)
    tempo_iter = iter(tempo_map)
    cur_tempo = next(tempo_iter, (0.0, 120.0))
    next_tempo = next(tempo_iter, None)
    ids.append(TOKEN2ID[f"T{_tempo_bin(cur_tempo[1])}"])

    meter_map = _meter_changes(pm)
    meter_iter = iter(meter_map)
    cur_meter = next(meter_iter, (0.0, "METER_4_4"))
    next_meter = next(meter_iter, None)
    ids.append(TOKEN2ID[cur_meter[1]])

    key_map = _key_changes(pm)
    key_iter = iter(key_map)
    cur_key = next(key_iter, (0.0, "KEY_0"))
    next_key = next(key_iter, None)
    ids.append(TOKEN2ID[cur_key[1]])

    downbeats = list(_downbeats(pm))
    # Precompute *note* events per bar for the header summary. Non-note
    # events (pedals/CC/PB) are excluded so they don't skew ROOT/DENS/REG.
    bar_events_by_idx: Dict[int, List[_Event]] = {}
    if downbeats:
        db_arr = np.asarray(downbeats)
        for ev in events:
            if ev.kind != "note":
                continue
            i = max(
                0,
                int(np.searchsorted(db_arr, ev.onset, side="right")) - 1,
            )
            bar_events_by_idx.setdefault(i, []).append(ev)

    db_idx = 0
    in_bar = False
    bar_start_time: Optional[float] = None
    bar_duration: Optional[float] = None
    last_pos_in_bar: Optional[int] = None
    # History of pitch multisets per emitted bar (for REF_BAR_K matching).
    bar_pitch_history: List[Tuple[int, ...]] = []

    def _bar_pitches(idx: int) -> Tuple[int, ...]:
        return tuple(sorted(e.pitch for e in bar_events_by_idx.get(idx, [])))

    def _emit_bar_start(bar_index: int) -> None:
        ids.append(BAR_START)
        ids.extend(_bar_header_tokens(bar_events_by_idx.get(bar_index, [])))
        fp = _bar_pitches(bar_index)
        for k in REF_DISTANCES:
            if k <= len(bar_pitch_history) and fp and fp == bar_pitch_history[-k]:
                ids.append(TOKEN2ID[f"REF_BAR_{k}"])
                break
        bar_pitch_history.append(fp)

    groups = _group_by_onset(events)
    current_voice = groups[0][0].voice
    ids.append(TOKEN2ID[f"VC{current_voice}"])

    prev_onset = groups[0][0].onset
    while db_idx < len(downbeats) and downbeats[db_idx] <= prev_onset + 1e-6:
        if in_bar:
            ids.append(BAR_END)
        _emit_bar_start(db_idx)
        in_bar = True
        bar_start_time = float(downbeats[db_idx])
        if db_idx + 1 < len(downbeats):
            bar_duration = float(downbeats[db_idx + 1] - downbeats[db_idx])
        last_pos_in_bar = None
        db_idx += 1

    for g_idx, group in enumerate(groups):
        onset = group[0].onset

        while next_tempo is not None and next_tempo[0] <= onset + 1e-6:
            ids.append(TOKEN2ID[f"T{_tempo_bin(next_tempo[1])}"])
            next_tempo = next(tempo_iter, None)

        while next_meter is not None and next_meter[0] <= onset + 1e-6:
            ids.append(TOKEN2ID[next_meter[1]])
            next_meter = next(meter_iter, None)

        while next_key is not None and next_key[0] <= onset + 1e-6:
            ids.append(TOKEN2ID[next_key[1]])
            next_key = next(key_iter, None)

        while db_idx < len(downbeats) and downbeats[db_idx] <= onset + 1e-6:
            if in_bar:
                ids.append(BAR_END)
            _emit_bar_start(db_idx)
            in_bar = True
            bar_start_time = float(downbeats[db_idx])
            if db_idx + 1 < len(downbeats):
                bar_duration = float(downbeats[db_idx + 1] - downbeats[db_idx])
            last_pos_in_bar = None
            db_idx += 1

        if in_bar and bar_duration and bar_duration > 1e-6:
            pos_bin = int(round((onset - bar_start_time) / bar_duration * N_POS_BINS))
            pos_bin = max(0, min(N_POS_BINS - 1, pos_bin))
            if pos_bin != last_pos_in_bar:
                ids.append(TOKEN2ID[f"POS{pos_bin}"])
                last_pos_in_bar = pos_bin
        else:
            shift = 0.0 if g_idx == 0 else onset - prev_onset
            if shift > TIME_MIN:
                ids.append(TOKEN2ID[f"TS{_log_bin(shift)}"])

        if group[0].kind != "note":
            ev = group[0]
            if ev.voice != current_voice:
                ids.append(TOKEN2ID[f"VC{ev.voice}"])
                current_voice = ev.voice
            if ev.kind == "pedal":
                ids.append(TOKEN2ID[f"PEDAL_{ev.pedal_type}_{ev.pedal_state}"])
            elif ev.kind == "cc":
                ids.append(TOKEN2ID[f"CC_{ev.cc_type}_{ev.cc_bin}"])
            elif ev.kind == "pb":
                ids.append(TOKEN2ID[f"PB_{ev.pb_bin}"])
            prev_onset = onset
            continue

        notes = sorted(group, key=lambda e: e.pitch)
        is_chord = len(notes) > 1
        if is_chord:
            ids.append(CHORD_START)
        for n_idx, ev in enumerate(notes):
            if ev.voice != current_voice:
                ids.append(TOKEN2ID[f"VC{ev.voice}"])
                current_voice = ev.voice
            if is_chord:
                if n_idx == 0:
                    ids.append(ROLE_BASS)
                elif n_idx == len(notes) - 1:
                    ids.append(ROLE_TOP)
                else:
                    ids.append(ROLE_INNER)
            ids.append(TOKEN2ID[f"V{_vel_bin(ev.velocity)}"])
            ids.append(TOKEN2ID[f"P{ev.pitch}"])
            ids.append(TOKEN2ID[f"D{_log_bin(ev.duration)}"])
        if is_chord:
            ids.append(CHORD_END)

        prev_onset = onset

    if in_bar:
        ids.append(BAR_END)
    ids.append(PHRASE_END)
    ids.append(EOS)
    return ids


# --- Decode -------------------------------------------------------------------

# GM family -> representative program number (one per family of 8).
FAMILY_PROGRAMS = {
    0: 0,    # Piano
    1: 8,    # Chromatic Percussion
    2: 16,   # Organ
    3: 24,   # Guitar
    4: 32,   # Bass
    5: 40,   # Strings
    6: 48,   # Ensemble
    7: 56,   # Brass
    8: 64,   # Reed
    9: 72,   # Pipe
    10: 80,  # Synth Lead
    11: 88,  # Synth Pad
    12: 96,  # Synth Effects
    13: 104, # Ethnic
    14: 112, # Percussive
    15: 120, # Sound Effects
}


def _kind(t: str):
    """Classify a token name into a (kind, value) pair for the decoder."""
    if t in STRUCTURAL:
        return ("struct", t)
    if t in ROLES:
        return ("role", t)
    if t in METERS:
        return ("meter", t)
    if t.startswith("KEY_") and t[4:].isdigit():
        return ("key", int(t[4:]))
    if t.startswith("ROOT_") and t[5:].isdigit():
        return ("root", int(t[5:]))
    if t.startswith("DENS_") and t[5:].isdigit():
        return ("dens", int(t[5:]))
    if t.startswith("REG_") and t[4:].isdigit():
        return ("reg", int(t[4:]))
    if t.startswith("REF_BAR_") and t[8:].isdigit():
        return ("ref", int(t[8:]))
    if t.startswith("CAP_SEG_") and t[8:].isdigit():
        return ("capseg", int(t[8:]))
    if t.startswith("PEDAL_"):
        return ("pedal", t)
    if t.startswith("CC_"):
        return ("cc", t)
    if t.startswith("PB_") and t[3:].isdigit():
        return ("pb", int(t[3:]))
    if t.startswith("TS") and t[2:].isdigit():
        return ("ts", int(t[2:]))
    if t.startswith("VC") and t[2:].isdigit():
        return ("voice", int(t[2:]))
    if t.startswith("POS") and t[3:].isdigit():
        return ("pos", int(t[3:]))
    if t.startswith("D") and t[1:].isdigit():
        return ("dur", int(t[1:]))
    if t.startswith("V") and t[1:].isdigit():
        return ("vel", int(t[1:]))
    if t.startswith("T") and t[1:].isdigit():
        return ("tempo", int(t[1:]))
    if t.startswith("P") and t[1:].isdigit():
        return ("pitch", int(t[1:]))
    return ("struct", t)


def decode(ids: List[int], default_tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """Decode a token id list back to a PrettyMIDI. Timing is reconstructed
    from POS within bars (using current tempo + meter) or TS deltas as a
    fallback. Pitches and instrument families are preserved exactly.
    """
    initial_tempo = default_tempo
    for tid in ids:
        t = ID2TOKEN.get(tid, "")
        if t.startswith("T") and not t.startswith("TS") and t[1:].isdigit():
            initial_tempo = _tempo_center(int(t[1:]))
            break

    pm = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
    instruments: Dict[int, pretty_midi.Instrument] = {}
    current_voice = 0

    def get_inst(v: int) -> pretty_midi.Instrument:
        if v not in instruments:
            if v == DRUM_VOICE:
                instruments[v] = pretty_midi.Instrument(
                    program=0,
                    is_drum=True,
                    name="drums",
                )
            else:
                prog = FAMILY_PROGRAMS.get(v, 0)
                instruments[v] = pretty_midi.Instrument(
                    program=prog,
                    name=f"family_{v}",
                )
        return instruments[v]

    current_tempo = initial_tempo
    current_meter_quarters = METER_QUARTERS["METER_4_4"]
    bar_duration = current_meter_quarters * 60.0 / current_tempo
    bar_start_time = 0.0
    n_bars_seen = 0
    current_time = 0.0
    pending_velocity = 64

    i = 0
    while i < len(ids):
        kind, val = _kind(ID2TOKEN.get(ids[i], "PAD"))

        if kind == "ts":
            current_time += _bin_center(val)
        elif kind == "pos":
            current_time = bar_start_time + (int(val) / N_POS_BINS) * bar_duration
        elif kind == "voice":
            current_voice = int(val)
        elif kind == "vel":
            pending_velocity = _vel_center(int(val))
        elif kind == "tempo":
            current_tempo = _tempo_center(int(val))
            bar_duration = current_meter_quarters * 60.0 / current_tempo
        elif kind == "meter":
            current_meter_quarters = METER_QUARTERS.get(val, 4.0)
            bar_duration = current_meter_quarters * 60.0 / current_tempo
        elif kind == "key":
            try:
                pm.key_signature_changes.append(
                    pretty_midi.KeySignature(int(val), float(current_time))
                )
            except Exception:
                pass
        elif kind == "pedal":
            # PEDAL_<SUS|SOS|SFT>_<UP|DOWN>
            parts = str(val).split("_")
            if len(parts) == 3:
                ptype, pstate = parts[1], parts[2]
                cc_num = PEDAL_NAME_TO_CC.get(ptype)
                if cc_num is not None:
                    inst = get_inst(current_voice)
                    inst.control_changes.append(
                        pretty_midi.ControlChange(
                            number=cc_num,
                            value=127 if pstate == "DOWN" else 0,
                            time=float(current_time),
                        )
                    )
        elif kind == "cc":
            # CC_<NAME>_<BIN>
            parts = str(val).split("_")
            if len(parts) == 3 and parts[2].isdigit():
                cname, bidx = parts[1], int(parts[2])
                cc_num = CC_NAME_TO_NUMBER.get(cname)
                if cc_num is not None:
                    inst = get_inst(current_voice)
                    inst.control_changes.append(
                        pretty_midi.ControlChange(
                            number=cc_num,
                            value=_cc_center(bidx),
                            time=float(current_time),
                        )
                    )
        elif kind == "pb":
            inst = get_inst(current_voice)
            inst.pitch_bends.append(
                pretty_midi.PitchBend(
                    pitch=_pb_center(int(val)),
                    time=float(current_time),
                )
            )
        elif kind == "struct" and val == "BAR_START":
            if n_bars_seen > 0:
                bar_start_time += bar_duration
            current_time = bar_start_time
            n_bars_seen += 1
        elif kind == "struct" and val == "REST":
            current_time += 0.25
        elif kind == "pitch":
            duration = 0.25
            j = i + 1
            while j < len(ids):
                nt = ID2TOKEN.get(ids[j], "PAD")
                nkind, nval = _kind(nt)
                if nkind == "dur":
                    duration = _bin_center(int(nval))
                    break
                if nkind in (
                    "pitch", "ts", "pos", "voice", "vel",
                    "tempo", "meter", "role", "key",
                    "root", "dens", "reg", "ref", "capseg",
                    "pedal", "cc", "pb",
                ):
                    break
                if nkind == "struct" and nval not in ("CHORD_START", "CHORD_END"):
                    break
                j += 1
            note = pretty_midi.Note(
                velocity=int(pending_velocity),
                pitch=int(val),
                start=current_time,
                end=current_time + max(duration, 0.01),
            )
            get_inst(current_voice).notes.append(note)
        # role / chord brackets / bar_end / phrase / pad / eos: no timing effect
        i += 1

    for voice in sorted(instruments):
        inst = instruments[voice]
        if inst.notes:
            pm.instruments.append(inst)
    return pm


def inject_caption_segments(ids: List[int], n_segs: int = N_CAP_SEGS) -> List[int]:
    """Insert CAP_SEG_<i % n_segs> right after each PHRASE_START.

    Use this when you have a multi-sentence caption split into ``n_segs``
    parts and want to bind the i-th phrase of the encoded MIDI to the
    i-th caption segment. Emission is opt-in because the segment count
    only makes sense in the presence of an external caption.
    """
    if n_segs <= 0 or n_segs > N_CAP_SEGS:
        raise ValueError(f"n_segs must be in [1, {N_CAP_SEGS}]")
    out: List[int] = []
    seen_phrases = 0
    for tid in ids:
        out.append(tid)
        if tid == PHRASE_START:
            out.append(TOKEN2ID[CAP_SEG_NAMES[seen_phrases % n_segs]])
            seen_phrases += 1
    return out


# --- Round-trip test ----------------------------------------------------------

def _voice_label(inst: "pretty_midi.Instrument") -> int:
    return DRUM_VOICE if inst.is_drum else _program_family(inst.program)


def round_trip_test(pm: pretty_midi.PrettyMIDI) -> Tuple[bool, Dict]:
    """Verify the (pitch, voice) multiset is preserved through encode+decode.

    Timing is not checked because log quantization is lossy.
    """
    original = sorted(
        (n.pitch, _voice_label(inst))
        for inst in pm.instruments
        for n in inst.notes
        if PITCH_MIN <= n.pitch <= PITCH_MAX
    )
    ids = encode(pm)
    pm2 = decode(ids)
    reconstructed = sorted(
        (n.pitch, _voice_label(inst))
        for inst in pm2.instruments
        for n in inst.notes
    )
    passed = original == reconstructed
    return passed, {
        "n_orig": len(original),
        "n_recon": len(reconstructed),
        "n_tokens": len(ids),
        "vocab_size": VOCAB_SIZE,
    }


# --- Auto-load fitted velocity quantiles -------------------------------------

if DEFAULT_VEL_QUANTILES_PATH.exists():
    try:
        load_velocity_quantiles(DEFAULT_VEL_QUANTILES_PATH)
    except Exception:
        pass


# --- CLI ----------------------------------------------------------------------

def _cli_fit_velocity_quantiles(sample_dir: Path, out_path: Path) -> None:
    paths = sorted(sample_dir.rglob("*.mid")) + sorted(sample_dir.rglob("*.midi"))
    if not paths:
        raise SystemExit(f"No MIDI files under {sample_dir}")
    velocities: List[int] = []
    n_failed = 0
    for p in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(p))
        except Exception:
            n_failed += 1
            continue
        for inst in pm.instruments:
            for n in inst.notes:
                velocities.append(int(n.velocity))
    edges = fit_velocity_quantiles(velocities, n_bins=N_VEL_BINS)
    save_velocity_quantiles(edges, out_path)
    print(
        f"[velocity] files={len(paths)} failed={n_failed} "
        f"velocities={len(velocities)} -> {out_path}"
    )
    print(f"[velocity] edges = {[round(e, 2) for e in edges]}")


if __name__ == "__main__":
    import argparse as _argparse
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "fit-velocity":
        p = _argparse.ArgumentParser()
        p.add_argument(
            "--sample-dir",
            type=str,
            default=str(Path(__file__).resolve().parent.parent / "data" / "gigamidi" / "sample"),
        )
        p.add_argument("--out", type=str, default=str(DEFAULT_VEL_QUANTILES_PATH))
        args = p.parse_args(_sys.argv[2:])
        _cli_fit_velocity_quantiles(Path(args.sample_dir), Path(args.out))
        _sys.exit(0)

    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"  structural:   {len(STRUCTURAL)}")
    print(f"  pitch:        {N_PITCH}")
    print(f"  duration:     {N_DUR_BINS}")
    print(f"  time-shift:   {N_SHIFT_BINS}")
    print(f"  velocity:     {N_VEL_BINS}")
    print(f"  voice/family: {N_VOICE_BINS}")
    print(f"  tempo:        {N_TEMPO_BINS}")

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

    # Multi-track / multi-velocity test.
    pm2 = pretty_midi.PrettyMIDI(initial_tempo=92.0)
    piano = pretty_midi.Instrument(program=0)      # family 0 (piano)
    bass = pretty_midi.Instrument(program=33)      # family 4 (bass)
    strings = pretty_midi.Instrument(program=48)   # family 6 (ensemble)
    t = 0.0
    for p, v in [(60, 30), (64, 80), (67, 110), (72, 60)]:
        piano.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=t, end=t + 0.5))
        bass.notes.append(pretty_midi.Note(velocity=70, pitch=p - 24, start=t, end=t + 0.5))
        strings.notes.append(pretty_midi.Note(velocity=50, pitch=p + 12, start=t, end=t + 0.5))
        t += 0.5
    pm2.instruments += [piano, bass, strings]
    ok2, info2 = round_trip_test(pm2)
    print(f"Multi-track round-trip: {'PASS' if ok2 else 'FAIL'}  {info2}")

    ids = encode(pm2)
    has_tempo = any(
        ID2TOKEN[i].startswith("T") and not ID2TOKEN[i].startswith("TS")
        and ID2TOKEN[i][1:].isdigit()
        for i in ids
    )
    has_voice = any(ID2TOKEN[i].startswith("VC") for i in ids)
    has_meter = any(ID2TOKEN[i] in METERS for i in ids)
    has_pos = any(ID2TOKEN[i].startswith("POS") for i in ids)
    has_chord = CHORD_START in ids
    has_role = any(t in ids for t in (ROLE_BASS, ROLE_INNER, ROLE_TOP))
    print(
        f"Stream features: tempo={has_tempo} voice={has_voice} "
        f"meter={has_meter} pos={has_pos} chord={has_chord} role={has_role}"
    )
