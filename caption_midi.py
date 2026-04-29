"""
caption_midi.py  v2
--------------------
Auto-labels MIDI files with rich, multi-sentence natural-language captions.
Extracts 14 musical features per file and assembles 4-sentence descriptions
covering: identity / instrumentation, tempo + rhythm, texture + articulation,
dynamic arc + dynamic range.

Usage:
    python caption_midi.py --midi_dir data/gigamidi/sample \
                           --output   data/captions.jsonl \
                           --limit    10000

    # Faster (skips music21 key analysis):
    python caption_midi.py --midi_dir ... --output ... --skip_music21

Output (one JSON per line):
    {"path": "...", "features": {...}, "caption": "..."}
"""

import os, json, random, argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# 1.  INSTRUMENT FAMILY TABLES  (needed by extract_features)
# ══════════════════════════════════════════════════════════════════════════════

FAMILY_MAP = {
    range(0,   8):  "piano",         range(8,  16): "chromatic perc",
    range(16, 24):  "organ",         range(24, 32): "guitar",
    range(32, 40):  "bass",          range(40, 48): "strings",
    range(48, 56):  "ensemble",      range(56, 64): "brass",
    range(64, 72):  "reed",          range(72, 80): "pipe",
    range(80, 88):  "synth lead",    range(88, 96): "synth pad",
    range(96, 104): "synth effects", range(104,112):"ethnic",
    range(112,120): "percussive",    range(120,128):"sound effects",
}

def program_to_family(p: int) -> str:
    for r, name in FAMILY_MAP.items():
        if p in r:
            return name
    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION  (14 features)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(midi_path: str, skip_music21: bool = False) -> Optional[dict]:
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None

    duration_sec = pm.get_end_time()
    if duration_sec < 4:
        return None

    all_notes = [n for inst in pm.instruments for n in inst.notes]
    if len(all_notes) < 20:
        return None

    pitches    = np.array([n.pitch    for n in all_notes])
    velocities = np.array([n.velocity for n in all_notes])
    durations  = np.array([n.end - n.start for n in all_notes])
    onsets     = np.array([n.start    for n in all_notes])

    # ── Tempo ─────────────────────────────────────────────────────────────────
    _, tempo_vals = pm.get_tempo_changes()
    bpm = float(tempo_vals.mean()) if len(tempo_vals) else 120.0
    tempo_change_count = len(tempo_vals)

    # ── Dynamics ──────────────────────────────────────────────────────────────
    avg_velocity = float(velocities.mean())
    velocity_std = float(velocities.std())

    # Dynamic arc: compare mean velocity of first vs last 25% of piece
    t25 = duration_sec * 0.25
    vel_early = velocities[onsets < t25]
    vel_late  = velocities[onsets > duration_sec * 0.75]
    dyn_delta = float(vel_late.mean() - vel_early.mean()) \
                if len(vel_early) > 3 and len(vel_late) > 3 else 0.0

    # ── Pitch ─────────────────────────────────────────────────────────────────
    pitch_min, pitch_max = int(pitches.min()), int(pitches.max())
    pitch_range = pitch_max - pitch_min

    # Pitch-class entropy (0 = one note, ~3.58 = completely even)
    pc_counts = np.bincount(pitches % 12, minlength=12).astype(float)
    pc_probs  = pc_counts / pc_counts.sum()
    pc_probs  = pc_probs[pc_probs > 0]
    pitch_entropy = float(-np.sum(pc_probs * np.log2(pc_probs)))

    # Melodic contour via top-voice pitch in three time windows
    t3 = duration_sec / 3.0
    def top_pitch(t0, t1):
        ps = [n.pitch for n in all_notes if t0 <= n.start < t1]
        return float(np.mean(sorted(ps)[-max(1, len(ps)//5):])) if ps else None

    p1, p2, p3 = top_pitch(0, t3), top_pitch(t3, 2*t3), top_pitch(2*t3, 3*t3)
    if p1 and p2 and p3:
        if   p3 > p1 + 2:             contour = "ascending"
        elif p3 < p1 - 2:             contour = "descending"
        elif p2 > max(p1, p3) + 2:   contour = "arch"
        elif p2 < min(p1, p3) - 2:   contour = "valley"
        else:                          contour = "stable"
    else:
        contour = "stable"

    # ── Articulation ──────────────────────────────────────────────────────────
    beat_dur  = 60.0 / bpm
    dur_ratio = float(durations.mean()) / beat_dur if beat_dur > 0 else 0.5

    # ── Rhythmic regularity ───────────────────────────────────────────────────
    sixteenth = beat_dur / 4
    if sixteenth > 0:
        bar_dur   = beat_dur * 4
        grid_pos  = (onsets % bar_dur) / sixteenth          # 0..15 = position in bar
        on_beat   = float(np.sum((grid_pos % 4) < 0.35)  / len(grid_pos))
        off_beat  = float(np.sum(((grid_pos % 2) > 0.65) & ((grid_pos % 2) < 1.35)) / len(grid_pos))
    else:
        on_beat, off_beat = 0.5, 0.1

    # ── Polyphony ─────────────────────────────────────────────────────────────
    piano_roll = pm.get_piano_roll(fs=10)
    polyphony  = float((piano_roll > 0).sum(axis=0).mean())

    # ── Time signature ────────────────────────────────────────────────────────
    ts_list  = pm.time_signature_changes
    time_sig = f"{ts_list[0].numerator}/{ts_list[0].denominator}" if ts_list else "4/4"

    # ── Instruments ───────────────────────────────────────────────────────────
    instruments = []
    for inst in pm.instruments:
        if not inst.is_drum:
            name   = inst.name.strip() or pretty_midi.program_to_instrument_name(inst.program)
            family = program_to_family(inst.program)
            instruments.append({"name": name, "program": inst.program, "family": family})
    is_drum = any(inst.is_drum for inst in pm.instruments)

    # ── Key (music21 — optional) ──────────────────────────────────────────────
    key_str = None
    if not skip_music21:
        try:
            import music21
            score   = music21.converter.parse(midi_path)
            k       = score.analyze("key")
            key_str = f"{k.tonic.name} {k.mode}"
        except Exception:
            pass

    return {
        "path":           midi_path,
        "bpm":            round(bpm, 1),
        "tempo_changes":  tempo_change_count,
        "duration_sec":   round(duration_sec, 1),
        "pitch_min":      pitch_min,
        "pitch_max":      pitch_max,
        "pitch_range":    pitch_range,
        "pitch_entropy":  round(pitch_entropy, 3),
        "contour":        contour,
        "avg_velocity":   round(avg_velocity, 1),
        "velocity_std":   round(velocity_std, 1),
        "dyn_delta":      round(dyn_delta, 1),
        "note_density":   round(len(all_notes) / duration_sec, 2),
        "polyphony":      round(polyphony, 2),
        "dur_ratio":      round(dur_ratio, 3),
        "on_beat_ratio":  round(on_beat, 3),
        "off_beat_ratio": round(off_beat, 3),
        "time_sig":       time_sig,
        "instruments":    instruments[:6],
        "drum":           is_drum,
        "key":            key_str,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  INSTRUMENT LABEL TABLES
# ══════════════════════════════════════════════════════════════════════════════

FAMILY_LABELS = {
    "piano":          ["piano",              "keyboard"],
    "guitar":         ["guitar",             "acoustic guitar",    "electric guitar"],
    "bass":           ["bass",               "bass guitar"],
    "strings":        ["strings",            "string section"],
    "ensemble":       ["ensemble",           "full ensemble"],
    "brass":          ["brass",              "brass section",      "horns"],
    "reed":           ["woodwinds",          "saxophone",          "reeds"],
    "organ":          ["organ",              "electric organ"],
    "pipe":           ["flute",              "wind instruments"],
    "synth lead":     ["synthesizer",        "lead synth"],
    "synth pad":      ["synth pads",         "ambient synthesizer"],
    "chromatic perc": ["mallet instruments", "vibraphone"],
    "ethnic":         ["ethnic instruments"],
    "percussive":     ["mallet percussion"],
    "synth effects":  ["synthesizer effects"],
    "sound effects":  ["sound design"],
}

NAME_FRAGMENTS = {
    "sax":"reed",  "oboe":"reed",   "clarinet":"reed",  "bassoon":"reed",
    "flute":"pipe","piccolo":"pipe",
    "trumpet":"brass","trombone":"brass","tuba":"brass","cornet":"brass",
    "violin":"strings","viola":"strings","cello":"strings","contrabass":"strings","harp":"strings",
    "harpsichord":"piano","clavinet":"piano",
    "synth":"synth lead","pad":"synth pad",
}

COMBO_PHRASES = {
    frozenset(["guitar","bass"]):               ["guitar and bass",           "guitar-bass arrangement"],
    frozenset(["guitar","bass","piano"]):        ["guitar, bass, and piano",   "rock/pop trio"],
    frozenset(["guitar","bass","ensemble"]):     ["full band",                 "guitar-led ensemble"],
    frozenset(["guitar","bass","brass"]):        ["horn-driven band",          "brass-accented rock ensemble"],
    frozenset(["guitar","bass","reed"]):         ["jazz-rock band",            "jazz-influenced ensemble"],
    frozenset(["organ","guitar","bass"]):        ["classic rock organ trio",   "organ-driven band"],
    frozenset(["piano","strings"]):              ["piano and strings",         "chamber ensemble"],
    frozenset(["piano","ensemble"]):             ["piano with orchestra",      "orchestral arrangement"],
    frozenset(["piano","strings","ensemble"]):   ["full orchestral arrangement"],
    frozenset(["strings","brass","ensemble"]):   ["full orchestral texture",   "symphonic arrangement"],
    frozenset(["brass","ensemble"]):             ["brass ensemble",            "big band-style arrangement"],
    frozenset(["reed","piano"]):                 ["jazz combo",                "woodwind and piano duo"],
    frozenset(["reed","piano","bass"]):          ["jazz trio",                 "jazz ensemble"],
    frozenset(["synth lead","synth pad"]):       ["electronic arrangement",    "synthesizer piece"],
    frozenset(["synth lead","bass"]):            ["synth-driven arrangement",  "electronic bass and synth"],
}

def pick(lst): return random.choice(lst)

def describe_instruments(instruments: list, has_drum: bool) -> str:
    families = set()
    for item in instruments:
        if isinstance(item, dict):
            fam = item.get("family", "other")
            if fam != "other":
                families.add(fam)
                continue
            name_l = item.get("name", "").lower()
        else:
            name_l = str(item).lower()
        for fam, labels in FAMILY_LABELS.items():
            if fam in name_l or any(lb in name_l for lb in labels):
                families.add(fam); break
        else:
            for frag, fam in NAME_FRAGMENTS.items():
                if frag in name_l:
                    families.add(fam); break

    families.discard("other")

    for combo in sorted(COMBO_PHRASES, key=len, reverse=True):
        if combo.issubset(families):
            base = pick(COMBO_PHRASES[combo])
            return base + (" with drums" if has_drum else "")

    if len(families) == 1:
        fam   = next(iter(families))
        label = pick(FAMILY_LABELS.get(fam, [fam]))
        if has_drum: return f"{label} with drums"
        return f"solo {label}" if fam in ("piano","guitar","organ") else label

    if len(families) <= 3:
        labels = [pick(FAMILY_LABELS.get(f, [f])) for f in sorted(families)]
        base   = ", ".join(labels[:-1]) + f" and {labels[-1]}"
        return base + (" with drums" if has_drum else "")

    return pick(["full band with drums","multi-instrument ensemble with drums"]) if has_drum \
           else pick(["multi-instrument ensemble","full ensemble"])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GENRE HINTING
# ══════════════════════════════════════════════════════════════════════════════

def infer_genre(families: set, bpm: float, is_minor: bool, has_drum: bool) -> Optional[str]:
    fast = bpm >= 120
    slow = bpm < 90

    if {"guitar","bass"}.issubset(families) and has_drum:
        if "brass" in families or "reed" in families:
            return pick(["funk","soul","horn-driven pop"])
        if fast and is_minor: return pick(["hard rock","minor-key rock"])
        if fast:              return pick(["rock","pop-rock","uptempo rock"])
        if slow and is_minor: return pick(["rock ballad","blues-rock"])
        return pick(["rock","pop","band music"])

    if {"organ","guitar","bass"}.issubset(families):
        return pick(["classic rock","blues-rock","organ-driven rock"])

    if {"reed","piano"}.issubset(families) or ({"reed","bass"}.issubset(families) and has_drum):
        return pick(["jazz ballad","cool jazz"]) if slow else pick(["jazz","swing jazz","bebop"])

    if {"brass","ensemble"}.issubset(families) and not {"guitar"}.issubset(families):
        return pick(["orchestral brass","cinematic orchestral music","symphonic music"])
    if {"brass","ensemble"}.issubset(families):
        return pick(["big band","swing","brass-forward arrangement"])

    if {"strings","ensemble"}.issubset(families) or ({"strings","piano"}.issubset(families) and not has_drum):
        return pick(["classical","orchestral","neo-classical"])

    if {"piano"}.issubset(families) and not has_drum and not {"guitar","bass"}.issubset(families):
        if slow and is_minor: return pick(["romantic piano","nocturne-style classical"])
        if fast:              return pick(["classical piano","virtuosic piano étude"])
        return pick(["solo piano music","contemporary classical piano"])

    if {"synth lead"}.issubset(families) and has_drum:
        return pick(["electronic dance music","synthwave","EDM"]) if fast \
               else pick(["ambient electronic","synth-pop"])

    if {"ethnic"}.issubset(families):
        return pick(["world music","folk-influenced music"])

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FEATURE → NATURAL LANGUAGE BUCKETS
# ══════════════════════════════════════════════════════════════════════════════

def b_tempo(bpm):
    if bpm < 55:  return "very slow"
    if bpm < 75:  return "slow"
    if bpm < 100: return "moderate"
    if bpm < 130: return "upbeat"
    if bpm < 160: return "fast"
    return "very fast"

def b_dynamics(vel):
    if vel < 42:  return "very soft and delicate"
    if vel < 62:  return "soft"
    if vel < 82:  return "moderately dynamic"
    if vel < 102: return "expressive and strong"
    return "loud and powerful"

def b_dyn_range(std):
    if std < 8:   return pick(["flat, uniform dynamics throughout",
                                "consistent, mechanical velocity"])
    if std < 18:  return pick(["gently varying dynamics",
                                "moderate dynamic variation"])
    if std < 28:  return pick(["wide dynamic contrasts",
                                "expressive dynamic range"])
    return pick(["dramatic swings between loud and soft",
                 "intense, highly contrasted dynamics"])

def b_dyn_arc(delta):
    if delta >  12: return pick(["builds progressively in intensity toward the end",
                                  "grows louder and more forceful as it develops",
                                  "swells to a powerful conclusion"])
    if delta < -12: return pick(["fades gradually toward a quiet close",
                                  "dissolves in intensity as it concludes",
                                  "retreats to a soft, restrained ending"])
    return pick(["maintains a consistent energy level throughout",
                 "sustains its dynamic character from opening to close",
                 "holds a steady intensity with no dramatic arc"])

def b_density(nd):
    if nd < 1.5: return "very sparse"
    if nd < 3.5: return "sparse"
    if nd < 6:   return "flowing"
    if nd < 10:  return "busy"
    return "dense"

def b_register(pmin, pmax):
    mid = (pmin + pmax) / 2
    if mid < 45:  return "deep low register"
    if mid < 57:  return "low-to-mid register"
    if mid < 69:  return "mid register"
    if mid < 81:  return "mid-to-high register"
    return "high register"

def b_range(pr):
    if pr < 12: return "a narrow, compact pitch range"
    if pr < 24: return "a moderate pitch range"
    if pr < 36: return "a wide pitch range"
    return "a very wide pitch range spanning multiple octaves"

def b_texture(poly):
    if poly < 1.4: return "a single melodic line"
    if poly < 2.2: return "a melody with light accompaniment"
    if poly < 3.5: return "a full chordal texture"
    return "a dense, layered texture"

def b_articulation(dur_ratio):
    if dur_ratio < 0.3: return pick(["crisp, staccato articulation",
                                      "short, punchy detached notes"])
    if dur_ratio < 0.65: return pick(["mixed articulation blending short and sustained notes",
                                       "varied note lengths ranging from staccato to held tones"])
    if dur_ratio < 1.1: return pick(["smooth, connected legato phrasing",
                                      "flowing, well-sustained lines"])
    return pick(["long, generously held tones with broad sustain",
                 "slow-moving, fully sustained notes"])

def b_rhythm(on_beat, off_beat, has_drum):
    if on_beat > 0.55:
        return pick(["a strong, metronomic on-the-beat pulse",
                     "a regular, grid-locked rhythmic feel",
                     "a steady, predictable rhythmic foundation"])
    if off_beat > 0.20:
        return pick(["syncopated, off-beat rhythmic phrasing",
                     "a push-and-pull syncopated groove",
                     "rhythmic displacement and offbeat accents"])
    if has_drum:
        return pick(["a driving rhythmic groove anchored by drums",
                     "a solid drum-driven rhythmic backbone"])
    return pick(["a free, loosely timed rhythmic feel",
                 "fluid, rubato-inflected phrasing without strict pulse"])

def b_contour(c):
    return {
        "ascending":  pick(["rises steadily to higher registers as it unfolds",
                             "climbs progressively toward higher pitches"]),
        "descending": pick(["descends gradually into lower registers",
                             "settles downward in pitch as the piece progresses"]),
        "arch":       pick(["arcs upward to a melodic peak then resolves downward",
                             "builds to a high point before descending to close"]),
        "valley":     pick(["dips to a lower register mid-piece before returning upward",
                             "descends to a low point then recovers to higher pitches"]),
        "stable":     pick(["maintains a stable, centred melodic register throughout",
                             "stays grounded in a consistent pitch range"]),
    }.get(c, "maintains a consistent register")

def b_tonal(entropy):
    if entropy < 2.0: return pick(["strongly tonal and key-centred",   "clearly diatonic"])
    if entropy < 2.8: return pick(["moderately tonal",                 "largely diatonic with some colour"])
    if entropy < 3.2: return pick(["mildly chromatic",                 "modally coloured"])
    return pick(["highly chromatic",  "tonally ambiguous"])

def b_tempo_stability(n):
    if n <= 1:  return pick(["a rock-steady, unwavering tempo", "a metronomic tempo throughout"])
    if n <= 4:  return pick(["a largely stable tempo with minor fluctuations",
                              "mostly consistent tempo with occasional shifts"])
    return pick(["frequent tempo changes that shift the feel", "a fluid, rubato-like approach to time"])

def b_time_sig(ts):
    return {
        "4/4":  pick(["common time (4/4)", "a 4/4 metre"]),
        "3/4":  pick(["waltz time (3/4)",  "a lilting 3/4 metre"]),
        "6/8":  pick(["compound 6/8 metre","a swinging 6/8 feel"]),
        "2/4":  pick(["a brisk 2/4 march metre"]),
        "5/4":  "an irregular 5/4 metre",
        "7/4":  "a complex 7/4 metre",
        "12/8": "a rolling 12/8 compound metre",
    }.get(ts, f"{ts} metre")

def b_duration(sec):
    if sec < 30:   return "A short"
    if sec < 90:   return "A medium-length"
    if sec < 180:  return "An extended"
    return "A long"

def _an(word: str) -> str:
    """Return 'An' if word starts with a vowel sound, else 'A'."""
    return "An" if word[0].lower() in "aeiou" else "A"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CAPTION ASSEMBLY  — 4 sentences
# ══════════════════════════════════════════════════════════════════════════════

MOOD_MINOR = ["melancholic","introspective","brooding","wistful","somber","bittersweet"]
MOOD_MAJOR = ["bright","cheerful","optimistic","uplifting","warm","lively","spirited"]

def build_caption(f: dict) -> str:
    minor       = "minor" in (f.get("key") or "").lower()
    key_phrase  = f["key"].lower() if f.get("key") else None
    mood        = pick(MOOD_MINOR if minor else MOOD_MAJOR) if key_phrase else None

    inst_phrase = describe_instruments(f["instruments"], f["drum"])

    # Collect families for genre
    families = set()
    for item in f["instruments"]:
        fam = item.get("family","other") if isinstance(item, dict) else "other"
        if fam != "other": families.add(fam)
    genre = infer_genre(families, f["bpm"], minor, f["drum"])

    # ── Sentence 1 : identity ──────────────────────────────────────────────
    key_cl   = f" in {key_phrase}" if key_phrase else ""
    mood_cl  = f" {mood} in character," if mood else ""
    genre_cl = f", in the style of {genre}," if (genre and mood) else \
               f", in the style of {genre}" if genre else ""
    tempo_w  = b_tempo(f["bpm"])
    tonal_w  = b_tonal(f["pitch_entropy"])
    dur_ph   = b_duration(f["duration_sec"])

    s1 = pick([
        f"{dur_ph} {genre or 'instrumental'} piece for {inst_phrase}{key_cl}{genre_cl}{mood_cl} "
        f"lasting {round(f['duration_sec'])} seconds.",

        f"{_an(tempo_w)} {tempo_w}, {tonal_w} composition for {inst_phrase}{key_cl}{genre_cl}. "
        f"The overall character is {mood or 'expressive'}.",

        f"{_an(genre or 'multi')} {genre or 'multi-instrument'} piece performed by {inst_phrase}{key_cl}. "
        f"It is {tonal_w} and {mood or 'expressive'} in feel.",
    ])

    # ── Sentence 2 : tempo + rhythm ────────────────────────────────────────
    s2 = pick([
        f"It moves at {round(f['bpm'])} BPM in {b_time_sig(f['time_sig'])}, "
        f"with {b_tempo_stability(f['tempo_changes'])} and {b_rhythm(f['on_beat_ratio'], f['off_beat_ratio'], f['drum'])}.",

        f"The tempo is {round(f['bpm'])} BPM ({b_tempo(f['bpm'])}), set in "
        f"{b_time_sig(f['time_sig'])}, driven by {b_rhythm(f['on_beat_ratio'], f['off_beat_ratio'], f['drum'])}.",

        f"Rhythmically it features {b_rhythm(f['on_beat_ratio'], f['off_beat_ratio'], f['drum'])} "
        f"at {round(f['bpm'])} BPM in {b_time_sig(f['time_sig'])}, "
        f"with {b_tempo_stability(f['tempo_changes'])}.",
    ])

    # ── Sentence 3 : texture + articulation + register + contour ──────────
    s3 = pick([
        f"The texture is {b_texture(f['polyphony'])} with {b_articulation(f['dur_ratio'])}, "
        f"a {b_density(f['note_density'])} note density across the {b_register(f['pitch_min'], f['pitch_max'])}. "
        f"The melodic line {b_contour(f['contour'])}.",

        f"The writing sits in the {b_register(f['pitch_min'], f['pitch_max'])} "
        f"with {b_range(f['pitch_range'])}, featuring {b_texture(f['polyphony'])} "
        f"and {b_articulation(f['dur_ratio'])}. The melody {b_contour(f['contour'])}.",

        f"Texturally it presents {b_texture(f['polyphony'])} and {b_articulation(f['dur_ratio'])} "
        f"— {b_density(f['note_density'])} overall — in the {b_register(f['pitch_min'], f['pitch_max'])}. "
        f"The melodic contour {b_contour(f['contour'])}.",
    ])

    # ── Sentence 4 : dynamic arc + range ──────────────────────────────────
    s4 = pick([
        f"Dynamically it is {b_dynamics(f['avg_velocity'])}, with {b_dyn_range(f['velocity_std'])}, "
        f"and {b_dyn_arc(f['dyn_delta'])}.",

        f"The performance is {b_dynamics(f['avg_velocity'])} throughout, "
        f"showing {b_dyn_range(f['velocity_std'])}. The piece {b_dyn_arc(f['dyn_delta'])}.",

        f"In terms of dynamics: {b_dynamics(f['avg_velocity'])}, {b_dyn_range(f['velocity_std'])}. "
        f"Overall the energy {b_dyn_arc(f['dyn_delta'])}.",
    ])

    return f"{s1} {s2} {s3} {s4}"


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def collect_midi_files(root: str) -> list[str]:
    paths = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".mid", ".midi")):
                paths.append(os.path.join(dp, fn))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir",     required=True)
    ap.add_argument("--output",       required=True)
    ap.add_argument("--limit",        type=int, default=10_000)
    ap.add_argument("--skip_music21", action="store_true",
                    help="Skip key detection (faster, no key/mood in captions)")
    args = ap.parse_args()

    paths = collect_midi_files(args.midi_dir)
    random.shuffle(paths)
    paths = paths[:args.limit * 3]
    print(f"Found {len(paths)} MIDI files (capped at {args.limit * 3})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(out_path, "w") as fout:
        for path in tqdm(paths, desc="Labelling"):
            if written >= args.limit:
                break
            try:
                feats = extract_features(path, skip_music21=args.skip_music21)
                if feats is None:
                    skipped += 1
                    continue
                caption = build_caption(feats)
                fout.write(json.dumps({"path": path, "features": feats, "caption": caption}) + "\n")
                written += 1
            except Exception:
                skipped += 1

    print(f"\nDone.  Written: {written}  Skipped/failed: {skipped}")
    print(f"Output: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  (run with no args)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        raise SystemExit

    CASES = [
        ("solo piano — nocturne", {
            "bpm":72.0,"tempo_changes":2,"duration_sec":210.0,
            "pitch_min":36,"pitch_max":84,"pitch_range":48,"pitch_entropy":2.1,"contour":"arch",
            "avg_velocity":52.0,"velocity_std":22.0,"dyn_delta":-14.0,
            "note_density":3.0,"polyphony":2.1,"dur_ratio":0.9,
            "on_beat_ratio":0.45,"off_beat_ratio":0.12,"time_sig":"4/4",
            "instruments":[{"name":"Acoustic Grand Piano","program":0,"family":"piano"}],
            "drum":False,"key":"D minor",
        }),
        ("rock band — guitar + bass + drums", {
            "bpm":132.0,"tempo_changes":1,"duration_sec":195.0,
            "pitch_min":38,"pitch_max":80,"pitch_range":42,"pitch_entropy":2.6,"contour":"stable",
            "avg_velocity":92.0,"velocity_std":14.0,"dyn_delta":8.0,
            "note_density":8.2,"polyphony":2.9,"dur_ratio":0.35,
            "on_beat_ratio":0.60,"off_beat_ratio":0.10,"time_sig":"4/4",
            "instruments":[{"name":"Electric Guitar","program":27,"family":"guitar"},
                           {"name":"Electric Bass","program":33,"family":"bass"}],
            "drum":True,"key":"E minor",
        }),
        ("jazz trio — sax + piano + bass + drums", {
            "bpm":92.0,"tempo_changes":3,"duration_sec":280.0,
            "pitch_min":45,"pitch_max":82,"pitch_range":37,"pitch_entropy":3.0,"contour":"ascending",
            "avg_velocity":68.0,"velocity_std":19.0,"dyn_delta":5.0,
            "note_density":5.5,"polyphony":2.4,"dur_ratio":0.6,
            "on_beat_ratio":0.38,"off_beat_ratio":0.25,"time_sig":"4/4",
            "instruments":[{"name":"Tenor Sax","program":66,"family":"reed"},
                           {"name":"Piano","program":0,"family":"piano"},
                           {"name":"Bass","program":32,"family":"bass"}],
            "drum":True,"key":"Bb major",
        }),
        ("electronic — synth + pad + drums", {
            "bpm":128.0,"tempo_changes":1,"duration_sec":240.0,
            "pitch_min":42,"pitch_max":86,"pitch_range":44,"pitch_entropy":2.8,"contour":"valley",
            "avg_velocity":88.0,"velocity_std":10.0,"dyn_delta":15.0,
            "note_density":9.0,"polyphony":3.2,"dur_ratio":0.8,
            "on_beat_ratio":0.62,"off_beat_ratio":0.08,"time_sig":"4/4",
            "instruments":[{"name":"Lead Synth","program":81,"family":"synth lead"},
                           {"name":"Pad","program":88,"family":"synth pad"}],
            "drum":True,"key":"A minor",
        }),
        ("orchestral — strings + brass + ensemble", {
            "bpm":78.0,"tempo_changes":6,"duration_sec":360.0,
            "pitch_min":28,"pitch_max":90,"pitch_range":62,"pitch_entropy":2.5,"contour":"ascending",
            "avg_velocity":75.0,"velocity_std":30.0,"dyn_delta":20.0,
            "note_density":4.5,"polyphony":5.8,"dur_ratio":1.1,
            "on_beat_ratio":0.50,"off_beat_ratio":0.08,"time_sig":"4/4",
            "instruments":[{"name":"Strings","program":48,"family":"strings"},
                           {"name":"French Horn","program":60,"family":"brass"},
                           {"name":"Ensemble","program":48,"family":"ensemble"}],
            "drum":False,"key":"C major",
        }),
        ("horn-driven funk — guitar + bass + brass + drums", {
            "bpm":112.0,"tempo_changes":1,"duration_sec":230.0,
            "pitch_min":40,"pitch_max":82,"pitch_range":42,"pitch_entropy":2.7,"contour":"stable",
            "avg_velocity":96.0,"velocity_std":16.0,"dyn_delta":4.0,
            "note_density":9.5,"polyphony":3.6,"dur_ratio":0.3,
            "on_beat_ratio":0.30,"off_beat_ratio":0.35,"time_sig":"4/4",
            "instruments":[{"name":"Electric Guitar","program":27,"family":"guitar"},
                           {"name":"Electric Bass","program":33,"family":"bass"},
                           {"name":"Brass Section","program":61,"family":"brass"}],
            "drum":True,"key":"G minor",
        }),
    ]

    print("=" * 72)
    for label, feats in CASES:
        print(f"\n[{label}]")
        print(build_caption(feats))
    print("\n" + "=" * 72)
    print("Smoke test passed.")
