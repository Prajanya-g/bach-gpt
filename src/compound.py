"""Octuple/CPWord-style compound tokenizer for bach-gpt.

Each event becomes a single "step" with N parallel feature ids instead of
N consecutive tokens. The compound encoder is a thin post-processor over
the 1D ``tokenizer.encode`` stream, so all structural logic (chord
brackets, voice-role, bar headers, REF, key/meter/tempo, etc.) is shared.

Step layout
-----------
A compound step is a tuple of ``N_AXES`` feature ids, one per axis:

    (step_type, pitch, duration, velocity, position, voice, aux)

Step types
----------
  0 PAD              all other axes = sentinel
  1 BOS / PHRASE_START
  2 EOS / PHRASE_END
  3 BAR_START        pos = 0; aux carries packed (root, density, register)
  4 BAR_END
  5 NOTE             pitch/dur/vel filled; voice + position; aux = role
  6 PEDAL_DOWN       voice + position; aux = pedal_type (0/1/2 = SUS/SOS/SFT)
  7 PEDAL_UP         voice + position; aux = pedal_type
  8 CC_CHANGE        voice + position; aux = CC_type * N_CC_BINS + bin
  9 PITCH_BEND       voice + position; aux = pb_bin
 10 CHORD_START      structural; pos optional
 11 CHORD_END        structural

Sentinels
---------
A per-axis "no value" id is the last entry of each axis. Models embed
this just like any other id; the sentinel just means "this axis doesn't
apply at this step type."
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pretty_midi

from tokenizer import (
    BAR_END,
    BAR_START,
    CHORD_END,
    CHORD_START,
    DRUM_VOICE,
    EOS,
    ID2TOKEN,
    METERS,
    METER_QUARTERS,
    N_CC_BINS,
    N_PB_BINS,
    N_POS_BINS,
    N_VEL_BINS,
    N_VOICE_BINS,
    PHRASE_END,
    PHRASE_START,
    PITCH_MAX,
    PITCH_MIN,
    REF_NAMES,
    ROLES,
    ROOT_NAMES,
    DENS_NAMES,
    REG_NAMES,
    PB_NAMES,
    CC_NAMES,
    PEDAL_NAMES,
    CC_TYPES,
    PEDAL_CC_NUMBERS,
    TOKEN2ID,
    VOCAB_SIZE,
    _bin_center,
    _cc_center,
    _pb_center,
    _tempo_center,
    _vel_center,
    decode as decode_1d,
    encode as encode_1d,
)

# --- Step type vocabulary ----------------------------------------------------

STEP_PAD = 0
STEP_BOS = 1
STEP_EOS = 2
STEP_BAR_START = 3
STEP_BAR_END = 4
STEP_NOTE = 5
STEP_PEDAL_DOWN = 6
STEP_PEDAL_UP = 7
STEP_CC = 8
STEP_PB = 9
STEP_CHORD_START = 10
STEP_CHORD_END = 11
N_STEP_TYPES = 12

# Per-axis cardinalities (last entry of each is the sentinel).
N_PITCH_AXIS = (PITCH_MAX - PITCH_MIN + 1) + 1     # 88 + sentinel
N_DUR_AXIS = 32 + 1
N_VEL_AXIS = N_VEL_BINS + 1
N_POS_AXIS = N_POS_BINS + 1
N_VOICE_AXIS = N_VOICE_BINS + 1

# Aux axis is type-multiplexed. We size it for the largest payload.
N_PEDAL_TYPES = 3
N_CC_AUX = len(CC_TYPES) * N_CC_BINS
N_AUX = max(len(ROLES) + 1, N_PEDAL_TYPES, N_CC_AUX, N_PB_BINS)
N_AUX_AXIS = N_AUX + 1

AXIS_SIZES = [
    N_STEP_TYPES,
    N_PITCH_AXIS,
    N_DUR_AXIS,
    N_VEL_AXIS,
    N_POS_AXIS,
    N_VOICE_AXIS,
    N_AUX_AXIS,
]
AXIS_NAMES = ["step", "pitch", "dur", "vel", "pos", "voice", "aux"]
N_AXES = len(AXIS_SIZES)

# Per-axis sentinel ids = the last index in that axis.
SENTINELS = [n - 1 for n in AXIS_SIZES]


def empty_step() -> List[int]:
    """Return a step with all axes set to their sentinel."""
    return list(SENTINELS)


# --- Helpers -----------------------------------------------------------------

PEDAL_TYPE_TO_AUX = {"SUS": 0, "SOS": 1, "SFT": 2}
ROLE_NAME_TO_AUX = {"ROLE_BASS": 0, "ROLE_INNER": 1, "ROLE_TOP": 2}
CC_NAME_TO_AUX_BASE = {
    name: i * N_CC_BINS for i, name in enumerate(CC_TYPES.values())
}


def _classify(name: str) -> Tuple[str, str]:
    """Return (kind, payload) for a 1D token name."""
    if name in ("PAD", "EOS", "PHRASE_START", "PHRASE_END",
                "BAR_START", "BAR_END", "REST",
                "CHORD_START", "CHORD_END"):
        return ("struct", name)
    if name in ROLES:
        return ("role", name)
    if name in METERS:
        return ("meter", name)
    if name in REF_NAMES:
        return ("ref", name)
    if name in PEDAL_NAMES:
        return ("pedal", name)
    if name in CC_NAMES:
        return ("cc", name)
    if name in PB_NAMES:
        return ("pb", name)
    if name in ROOT_NAMES:
        return ("root", name)
    if name in DENS_NAMES:
        return ("dens", name)
    if name in REG_NAMES:
        return ("reg", name)
    if name.startswith("VC") and name[2:].isdigit():
        return ("voice", name[2:])
    if name.startswith("POS") and name[3:].isdigit():
        return ("pos", name[3:])
    if name.startswith("TS") and name[2:].isdigit():
        return ("ts", name[2:])
    if name.startswith("D") and name[1:].isdigit():
        return ("dur", name[1:])
    if name.startswith("V") and name[1:].isdigit():
        return ("vel", name[1:])
    if name.startswith("T") and name[1:].isdigit():
        return ("tempo", name[1:])
    if name.startswith("KEY_") and name[4:].isdigit():
        return ("key", name[4:])
    if name.startswith("CAP_SEG_") and name[8:].isdigit():
        return ("capseg", name[8:])
    if name.startswith("P") and name[1:].isdigit():
        return ("pitch", name[1:])
    return ("struct", name)


# --- Encode ------------------------------------------------------------------

def encode_compound(pm: pretty_midi.PrettyMIDI) -> List[List[int]]:
    """Convert a PrettyMIDI to a list of compound steps via the 1D encoder.

    The 1D stream is walked once; each musical event collapses into a
    single compound step. Bare metadata tokens (KEY/METER/TEMPO/ROLE/
    bar headers/REF/CAP_SEG) update running state but don't emit a step.
    """
    ids_1d = encode_1d(pm)
    steps: List[List[int]] = []
    cur_voice = SENTINELS[5]
    cur_pos = SENTINELS[4]
    pending_role = SENTINELS[6]
    pending_vel = SENTINELS[3]

    i = 0
    while i < len(ids_1d):
        name = ID2TOKEN.get(ids_1d[i], "PAD")
        kind, payload = _classify(name)

        if kind == "struct":
            tag = payload
            if tag in ("PHRASE_START",):
                steps.append([
                    STEP_BOS, SENTINELS[1], SENTINELS[2],
                    SENTINELS[3], SENTINELS[4], SENTINELS[5], SENTINELS[6],
                ])
            elif tag in ("PHRASE_END",):
                steps.append([
                    STEP_EOS, SENTINELS[1], SENTINELS[2],
                    SENTINELS[3], SENTINELS[4], SENTINELS[5], SENTINELS[6],
                ])
            elif tag == "BAR_START":
                steps.append([
                    STEP_BAR_START, SENTINELS[1], SENTINELS[2],
                    SENTINELS[3], 0, cur_voice, SENTINELS[6],
                ])
                cur_pos = 0
            elif tag == "BAR_END":
                steps.append([
                    STEP_BAR_END, SENTINELS[1], SENTINELS[2],
                    SENTINELS[3], SENTINELS[4], cur_voice, SENTINELS[6],
                ])
            elif tag == "CHORD_START":
                steps.append([
                    STEP_CHORD_START, SENTINELS[1], SENTINELS[2],
                    SENTINELS[3], cur_pos, cur_voice, SENTINELS[6],
                ])
            elif tag == "CHORD_END":
                steps.append([
                    STEP_CHORD_END, SENTINELS[1], SENTINELS[2],
                    SENTINELS[3], cur_pos, cur_voice, SENTINELS[6],
                ])
            # PAD, EOS, REST: ignored; metadata-only are skipped quietly
        elif kind == "voice":
            cur_voice = int(payload)
        elif kind == "pos":
            cur_pos = int(payload)
        elif kind == "vel":
            pending_vel = int(payload)
        elif kind == "role":
            pending_role = ROLE_NAME_TO_AUX.get(payload, SENTINELS[6])
        elif kind == "pitch":
            # Look ahead for the duration token bound to this pitch.
            midi = int(payload)
            dur_bin = SENTINELS[2]
            j = i + 1
            while j < len(ids_1d):
                nname = ID2TOKEN.get(ids_1d[j], "PAD")
                nkind, npay = _classify(nname)
                if nkind == "dur":
                    dur_bin = int(npay)
                    break
                if nkind in ("pitch", "vel", "voice", "pos", "ts",
                             "tempo", "meter", "key", "role",
                             "root", "dens", "reg", "ref",
                             "pedal", "cc", "pb", "capseg"):
                    break
                j += 1
            steps.append([
                STEP_NOTE,
                midi - PITCH_MIN,
                dur_bin,
                pending_vel,
                cur_pos,
                cur_voice,
                pending_role,
            ])
            pending_role = SENTINELS[6]
        elif kind == "pedal":
            # PEDAL_<TYPE>_<STATE>
            parts = payload.split("_")
            ptype = PEDAL_TYPE_TO_AUX.get(parts[1], 0)
            stype = STEP_PEDAL_DOWN if parts[2] == "DOWN" else STEP_PEDAL_UP
            steps.append([
                stype, SENTINELS[1], SENTINELS[2], SENTINELS[3],
                cur_pos, cur_voice, ptype,
            ])
        elif kind == "cc":
            # CC_<NAME>_<BIN>
            parts = payload.split("_")
            cname = parts[1]
            cbin = int(parts[2])
            aux = CC_NAME_TO_AUX_BASE.get(cname, 0) + cbin
            steps.append([
                STEP_CC, SENTINELS[1], SENTINELS[2], SENTINELS[3],
                cur_pos, cur_voice, aux,
            ])
        elif kind == "pb":
            steps.append([
                STEP_PB, SENTINELS[1], SENTINELS[2], SENTINELS[3],
                cur_pos, cur_voice, int(payload[3:]) if payload[3:].isdigit() else 0,
            ])
        # Other metadata kinds (tempo/meter/key/ref/root/dens/reg/capseg/ts)
        # update running state implicitly via the 1D decoder; the compound
        # stream omits them since the model can derive them from absolute
        # position + KEY/METER/TEMPO that the 1D decoder also handles.
        i += 1

    return steps


# --- Decode ------------------------------------------------------------------

def decode_compound(steps: Sequence[Sequence[int]]) -> pretty_midi.PrettyMIDI:
    """Reconstruct a PrettyMIDI from a compound step list. Approximates
    timing using the assumption that each BAR_START corresponds to one
    bar at 4/4 + 120 BPM (i.e., 2.0 s per bar). Pitches and instrument
    routing are exact; precise timing requires the 1D stream.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instruments: dict = {}
    bar_duration = 2.0   # 4/4 at 120 BPM
    bar_idx = 0
    bar_start_time = 0.0
    cur_time = 0.0

    def get_inst(v: int) -> pretty_midi.Instrument:
        if v not in instruments:
            from tokenizer import FAMILY_PROGRAMS
            if v == DRUM_VOICE:
                instruments[v] = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
            else:
                instruments[v] = pretty_midi.Instrument(
                    program=FAMILY_PROGRAMS.get(v, 0),
                    name=f"family_{v}",
                )
        return instruments[v]

    for step in steps:
        stype = int(step[0])
        if stype == STEP_BAR_START:
            if bar_idx > 0:
                bar_start_time += bar_duration
            cur_time = bar_start_time
            bar_idx += 1
        elif stype == STEP_NOTE:
            midi_off = int(step[1])
            dur_bin = int(step[2])
            vel_bin = int(step[3])
            pos_bin = int(step[4])
            voice = int(step[5])
            if voice == SENTINELS[5]:
                continue
            cur_time = bar_start_time + (pos_bin / N_POS_BINS) * bar_duration if pos_bin != SENTINELS[4] else cur_time
            duration = _bin_center(dur_bin) if dur_bin != SENTINELS[2] else 0.25
            velocity = _vel_center(vel_bin) if vel_bin != SENTINELS[3] else 64
            pitch = midi_off + PITCH_MIN
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=cur_time,
                end=cur_time + max(duration, 0.01),
            )
            get_inst(voice).notes.append(note)
        elif stype in (STEP_PEDAL_DOWN, STEP_PEDAL_UP):
            voice = int(step[5])
            ptype_aux = int(step[6])
            if voice != SENTINELS[5] and ptype_aux < N_PEDAL_TYPES:
                ptype_name = list(PEDAL_TYPE_TO_AUX.keys())[ptype_aux]
                cc_num = next(k for k, v in PEDAL_CC_NUMBERS.items() if v == ptype_name)
                value = 127 if stype == STEP_PEDAL_DOWN else 0
                get_inst(voice).control_changes.append(
                    pretty_midi.ControlChange(number=cc_num, value=value, time=cur_time)
                )
        elif stype == STEP_CC:
            voice = int(step[5])
            aux = int(step[6])
            if voice != SENTINELS[5] and aux < N_CC_AUX:
                cc_idx = aux // N_CC_BINS
                cc_bin = aux % N_CC_BINS
                cc_name = list(CC_TYPES.values())[cc_idx]
                cc_num = next(k for k, v in CC_TYPES.items() if v == cc_name)
                get_inst(voice).control_changes.append(
                    pretty_midi.ControlChange(
                        number=cc_num, value=_cc_center(cc_bin), time=cur_time
                    )
                )
        elif stype == STEP_PB:
            voice = int(step[5])
            if voice != SENTINELS[5]:
                get_inst(voice).pitch_bends.append(
                    pretty_midi.PitchBend(pitch=_pb_center(int(step[6])), time=cur_time)
                )
        # CHORD_*, BAR_END, BOS, EOS, PAD: structural, no PrettyMIDI side effect

    for v in sorted(instruments):
        if instruments[v].notes or instruments[v].control_changes or instruments[v].pitch_bends:
            pm.instruments.append(instruments[v])
    return pm


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    print(f"Compound axes: {AXIS_NAMES}")
    print(f"Axis sizes:    {AXIS_SIZES}")
    print(f"Sentinels:     {SENTINELS}")

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    for p in [60, 64, 67, 72]:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=t, end=t + 0.5))
        t += 0.5
    pm.instruments.append(inst)

    steps = encode_compound(pm)
    print(f"\n{len(steps)} compound steps from {len(inst.notes)} notes:")
    for s in steps[:8]:
        print(f"  {s}")

    pm2 = decode_compound(steps)
    n_recon = sum(len(i.notes) for i in pm2.instruments)
    print(f"\nDecoded {n_recon} notes (orig {len(inst.notes)})")
