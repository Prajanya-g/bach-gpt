"""Corpus statistics for bach-gpt check-in.

Samples from:
  - JSB chorales via music21 (probing corpus)
  - GigaMIDI (all-instruments-with-drums, training-V1.1-80% split;
    Metacreation Lab, same corpus MIDI-GPT trained on) as the
    pretraining corpus

For each sampled corpus, this script:
  - Runs the tokenizer round-trip test on every file
  - Computes sequence length distribution, top-K token frequencies,
    vocabulary utilization, and total tokens
  - For GigaMIDI, also emits a genre-distribution plot from the
    metadata CSV when labels are available
  - Projects full-corpus token budget
  - Writes figures to ./figures/
  - Writes a markdown summary to ./results/corpus_stats.md

Run: python3 src/corpus_stats.py
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

# Silence noisy music21 / pretty_midi warnings that swamp stdout.
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from tokenizer import encode, decode, round_trip_test, ID2TOKEN, VOCAB_SIZE

N_SAMPLES_JSB = 30
N_SAMPLES_GIGAMIDI = 100
RNG_SEED = 17
TOP_K = 20

# GigaMIDI paths — we sample from the all-instruments-with-drums training split.
GIGAMIDI_ROOT = "Final_GigaMIDI_V1.1_Final"
GIGAMIDI_SAMPLE_SUBDIR = "gigamidi/sample"   # populated by corpus_stats itself from the zip
GIGAMIDI_METADATA_CSV = "Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv"
GIGAMIDI_TRAINING_ZIP_REL = "training-V1.1-80%/all-instruments-with-drums.zip"

FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


def _gigamidi_zip_candidates() -> List[Path]:
    """v1.1 nested zip (local unzip) or flat zip (some Hub layouts)."""
    root = DATA_DIR / GIGAMIDI_ROOT
    return [
        root / GIGAMIDI_TRAINING_ZIP_REL,
        root / "all-instruments-with-drums.zip",
    ]


def _gigamidi_all_instruments_dir() -> Optional[Path]:
    """Directory of training MIDIs: nested v1.1, Hub flat under root, or fuzzy name."""
    root = DATA_DIR / GIGAMIDI_ROOT
    if not root.is_dir():
        return None
    candidates = [
        root / "training-V1.1-80%" / "all-instruments-with-drums",
        root / "all-instruments-with-drums",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    for child in root.iterdir():
        if child.is_dir() and child.name.strip() == "all-instruments-with-drums":
            return child
    return None


def _is_midi_suffix(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in (".mid", ".midi")


def _collect_midi_paths(source: Path) -> List[Path]:
    return [
        p
        for p in source.rglob("*")
        if _is_midi_suffix(p) and not p.name.startswith(".")
    ]


def _count_midi_zip_entries(zp: Path) -> int:
    import zipfile

    try:
        with zipfile.ZipFile(zp) as z:
            return sum(
                1
                for nm in z.namelist()
                if Path(nm).suffix.lower() in (".mid", ".midi")
                and "__MACOSX" not in nm
            )
    except (zipfile.BadZipFile, OSError):
        return 0


def _iter_shard_zips(source_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in source_dir.rglob("*.zip")
        if "__MACOSX" not in p.parts
    )


def _ensure_gigamidi_from_shard_zips(
    source_dir: Path, n: int, sample_dir: Path
) -> int:
    """Sample MIDIs from nested .zip shards (typical Hugging Face tree)."""
    import shutil
    import zipfile

    zips = _iter_shard_zips(source_dir)
    if not zips:
        return 0

    total_files = sum(_count_midi_zip_entries(zp) for zp in zips)
    if total_files == 0:
        return 0

    existing = list(sample_dir.glob("*.mid")) + list(
        sample_dir.glob("*.midi")
    )
    if len(existing) >= n:
        return total_files

    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RNG_SEED)
    needed = n - len(existing)
    copied = 0
    rng.shuffle(zips)
    for zp in zips:
        if copied >= needed:
            break
        try:
            with zipfile.ZipFile(zp) as z:
                members = [
                    m
                    for m in z.namelist()
                    if Path(m).suffix.lower() in (".mid", ".midi")
                    and "__MACOSX" not in m
                ]
                if not members:
                    continue
                rng.shuffle(members)
                for nm in members:
                    if copied >= needed:
                        break
                    dest = sample_dir / Path(nm).name
                    if dest.exists():
                        continue
                    with z.open(nm) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    copied += 1
        except (zipfile.BadZipFile, OSError):
            continue
    return total_files


# --- Corpus loaders -----------------------------------------------------------

def load_jsb(n: int) -> List[Tuple[str, pretty_midi.PrettyMIDI]]:
    """Use music21 chorale iterator; write each score to a temp .mid and
    re-load through pretty_midi."""
    from music21 import corpus
    it = corpus.chorales.Iterator(numberingSystem="bwv", returnType="stream")
    out: List[Tuple[str, pretty_midi.PrettyMIDI]] = []
    tmp = Path(tempfile.mkdtemp(prefix="jsb_"))
    pbar = tqdm(total=n, desc="JSB")
    for i, score in enumerate(it):
        if len(out) >= n:
            break
        try:
            path = tmp / f"chorale_{i:03d}.mid"
            score.write("midi", fp=str(path))
            pm = pretty_midi.PrettyMIDI(str(path))
            out.append((f"bwv_{i:03d}", pm))
            pbar.update(1)
        except Exception as e:
            continue
    pbar.close()
    return out


def load_maestro(n: int) -> List[Tuple[str, pretty_midi.PrettyMIDI]]:
    """Load a random sample of MAESTRO MIDI files.

    Searches DATA_DIR/maestro-v3.0.0/ recursively. If no MAESTRO files are
    found, falls back to a classical stand-in sampled from the music21
    corpus (Bach non-chorale cantatas + Beethoven + Mozart) so the
    pipeline can still be exercised. The fallback is clearly flagged in
    the returned label.
    """
    maestro_root = DATA_DIR / "maestro-v3.0.0"
    midis = list(maestro_root.rglob("*.midi")) + list(maestro_root.rglob("*.mid"))

    if midis:
        rng = random.Random(RNG_SEED)
        rng.shuffle(midis)
        out = []
        pbar = tqdm(total=min(n, len(midis)), desc="MAESTRO")
        for p in midis:
            if len(out) >= n:
                break
            try:
                pm = pretty_midi.PrettyMIDI(str(p))
                out.append((p.stem, pm))
                pbar.update(1)
            except Exception:
                continue
        pbar.close()
        return out

    # --- Fallback: classical stand-in ---------------------------------------
    print("[corpus_stats] MAESTRO not found at", maestro_root)
    print("[corpus_stats] Using music21 classical stand-in so pipeline runs.")
    print("[corpus_stats] Re-run after populating ./data/maestro-v3.0.0/ for real MAESTRO stats.")

    from music21 import corpus
    rng = random.Random(RNG_SEED)
    pool = []
    for composer in ("beethoven", "mozart", "bach"):
        for p in corpus.getComposer(composer):
            s = str(p)
            if s.endswith((".mxl", ".xml", ".krn")):
                pool.append(s)
    rng.shuffle(pool)

    out = []
    tmp = Path(tempfile.mkdtemp(prefix="classical_stand_in_"))
    pbar = tqdm(total=n, desc="CLASSICAL-STAND-IN")
    for i, src in enumerate(pool):
        if len(out) >= n:
            break
        try:
            from music21 import converter
            score = converter.parse(src)
            mid_path = tmp / f"piece_{i:04d}.mid"
            score.write("midi", fp=str(mid_path))
            pm = pretty_midi.PrettyMIDI(str(mid_path))
            if not any(inst.notes for inst in pm.instruments if not inst.is_drum):
                continue
            out.append((Path(src).stem, pm))
            pbar.update(1)
        except Exception:
            continue
    pbar.close()
    return out


def _ensure_gigamidi_sample(n: int, sample_dir: Path) -> int:
    """Ensure the GigaMIDI sample directory contains at least n MIDI files.

    Source (first match): training zip (nested or flat), else extracted
    all-instruments-with-drums/ tree (Hub snapshot or v1.1 unzip).

    Returns total MIDI count in the source (zip entries or files on disk).
    """
    import shutil
    import zipfile

    zip_path = next((p for p in _gigamidi_zip_candidates() if p.exists()), None)

    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as z:
            all_midi = [
                nm
                for nm in z.namelist()
                if Path(nm).suffix.lower() in (".mid", ".midi")
                and "__MACOSX" not in nm
            ]
            total_files = len(all_midi)

            existing = list(sample_dir.glob("*.mid")) + list(
                sample_dir.glob("*.midi")
            )
            if len(existing) >= n:
                return total_files

            sample_dir.mkdir(parents=True, exist_ok=True)
            rng = random.Random(RNG_SEED)
            rng.shuffle(all_midi)
            needed = n - len(existing)
            extracted = 0
            for nm in all_midi:
                if extracted >= needed:
                    break
                fname = Path(nm).name
                dest = sample_dir / fname
                if dest.exists():
                    continue
                with z.open(nm) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
        return total_files

    source_dir = _gigamidi_all_instruments_dir()
    if source_dir is None:
        return 0

    all_paths = _collect_midi_paths(source_dir)
    if all_paths:
        total_files = len(all_paths)
        existing = list(sample_dir.glob("*.mid")) + list(
            sample_dir.glob("*.midi")
        )
        if len(existing) >= n:
            return total_files

        sample_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(RNG_SEED)
        rng.shuffle(all_paths)
        needed = n - len(existing)
        copied = 0
        for src_path in all_paths:
            if copied >= needed:
                break
            dest = sample_dir / src_path.name
            if dest.exists():
                continue
            shutil.copy2(src_path, dest)
            copied += 1
        return total_files

    return _ensure_gigamidi_from_shard_zips(source_dir, n, sample_dir)


def load_gigamidi(n: int) -> Tuple[List[Tuple[str, pretty_midi.PrettyMIDI]], int]:
    """Load a random sample of GigaMIDI MIDI files from the
    all-instruments-with-drums training split.

    Returns (pairs, total_files_in_source_split) so the caller can project
    per-corpus token budgets. Will extract a fresh sample from the training
    zip if one isn't already present at data/gigamidi/sample/.
    """
    sample_dir = DATA_DIR / GIGAMIDI_SAMPLE_SUBDIR
    total_files = _ensure_gigamidi_sample(n, sample_dir)
    if total_files == 0:
        tried = ", ".join(str(p) for p in _gigamidi_zip_candidates())
        print(
            "[corpus_stats] GigaMIDI training source not found. "
            f"Tried zips: {tried}; "
            f"or directory {_gigamidi_all_instruments_dir() or '<none>'}"
        )
        return [], 0

    midis = sorted(sample_dir.glob("*.mid")) + sorted(sample_dir.glob("*.midi"))
    rng = random.Random(RNG_SEED + 7)
    rng.shuffle(midis)
    out = []
    pbar = tqdm(total=min(n, len(midis)), desc="GigaMIDI")
    for p in midis:
        if len(out) >= n:
            break
        try:
            pm = pretty_midi.PrettyMIDI(str(p))
            out.append((p.stem, pm))
            pbar.update(1)
        except Exception:
            continue
    pbar.close()
    return out, total_files


def load_gigamidi_metadata(sample_stems: List[str]) -> dict:
    """Join sampled file stems (md5 hashes) against the GigaMIDI metadata CSV.
    Returns {md5: row} for files present in the sample.
    """
    import csv as _csv, sys as _sys
    _csv.field_size_limit(_sys.maxsize)

    meta_path = DATA_DIR / GIGAMIDI_ROOT / GIGAMIDI_METADATA_CSV
    if not meta_path.exists():
        return {}
    wanted = set(sample_stems)
    out = {}
    with open(meta_path, newline="") as f:
        rdr = _csv.DictReader(f)
        for row in rdr:
            md5 = row.get("md5", "")
            if md5 in wanted:
                out[md5] = row
    return out


def extract_genres(meta_by_md5: dict) -> Tuple[Counter, int]:
    """Pull a single genre label per file, falling back through the five
    metadata columns in order of curation quality. Returns (counter, n_labeled)."""
    cols = ["music_styles_curated", "music_style_scraped",
            "music_style_audio_text_Discogs", "music_style_audio_text_Lastfm",
            "music_style_audio_text_Tagtraum"]
    ctr: Counter = Counter()
    n_labeled = 0
    for row in meta_by_md5.values():
        for col in cols:
            v = (row.get(col) or "").strip()
            if v:
                # Fields are sometimes python-style lists "['Rock','Pop']" -
                # take the first term, lowercase, strip quotes.
                head = v.strip("[]").split(",")[0].strip().strip("'").strip('"').lower()
                if head:
                    ctr[head] += 1
                    n_labeled += 1
                break
    return ctr, n_labeled


# --- Stats --------------------------------------------------------------------

def corpus_stats(name: str, pairs: List[Tuple[str, pretty_midi.PrettyMIDI]]):
    lengths = []
    token_counter: Counter = Counter()
    rt_pass, rt_fail = 0, 0
    rt_info_sizes = []

    for fid, pm in tqdm(pairs, desc=f"tokenize-{name}"):
        try:
            ids = encode(pm)
        except Exception as e:
            rt_fail += 1
            continue
        lengths.append(len(ids))
        token_counter.update(ids)
        ok, info = round_trip_test(pm)
        if ok:
            rt_pass += 1
            rt_info_sizes.append(info["n_orig"])
        else:
            rt_fail += 1

    total_tokens = sum(token_counter.values())
    vocab_used = len(token_counter)
    rt_rate = rt_pass / max(1, rt_pass + rt_fail)

    stats = {
        "corpus": name,
        "n_files": len(pairs),
        "round_trip_pass": rt_pass,
        "round_trip_fail": rt_fail,
        "round_trip_rate": rt_rate,
        "total_tokens": total_tokens,
        "mean_seq_len": float(np.mean(lengths)) if lengths else 0.0,
        "median_seq_len": float(np.median(lengths)) if lengths else 0.0,
        "min_seq_len": int(np.min(lengths)) if lengths else 0,
        "max_seq_len": int(np.max(lengths)) if lengths else 0,
        "vocab_used": vocab_used,
        "vocab_size": VOCAB_SIZE,
        "vocab_util": vocab_used / VOCAB_SIZE,
        "lengths": lengths,
        "top_tokens": token_counter.most_common(TOP_K),
    }
    return stats


# --- Plots --------------------------------------------------------------------

def plot_lengths(name: str, lengths: List[int]):
    fig, ax = plt.subplots(figsize=(6, 4))
    if lengths:
        ax.hist(lengths, bins=min(20, max(5, len(lengths) // 2)),
                color="#4878A6", edgecolor="white")
    ax.set_xlabel("tokens per sequence")
    ax.set_ylabel("files")
    ax.set_title(f"{name}: sequence length distribution (n={len(lengths)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"{name}_lengths.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_genres(name: str, genre_counter: Counter, total_sample: int):
    if not genre_counter:
        return None
    items = genre_counter.most_common(12)
    labels = [g for g, _ in items]
    counts = [c for _, c in items]
    labeled = sum(counts)
    unlabeled = max(0, total_sample - labeled)
    if unlabeled > 0:
        labels.append("(unlabeled)")
        counts.append(unlabeled)
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(labels))[::-1]
    colors = ["#4878A6"] * (len(labels) - (1 if unlabeled > 0 else 0))
    if unlabeled > 0:
        colors += ["#BBBBBB"]
    ax.barh(y, counts, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"files (sample of {total_sample})")
    ax.set_title(f"{name}: genre distribution from metadata")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG_DIR / f"{name}_genres.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_topk(name: str, top_tokens):
    labels = [ID2TOKEN.get(i, str(i)) for i, _ in top_tokens]
    counts = [c for _, c in top_tokens]
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, color="#C27A3F", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("frequency")
    ax.set_title(f"{name}: top-{len(labels)} tokens")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG_DIR / f"{name}_topk.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --- Reporting ----------------------------------------------------------------

def format_md(all_stats, corpus_sizes=None, genre_info=None):
    """corpus_sizes: dict {corpus_name: total_files_in_full_corpus} for projection.
    genre_info: dict {"counter": Counter, "labeled": int, "total": int} for GigaMIDI."""
    corpus_sizes = corpus_sizes or {}
    lines = ["# Corpus statistics", ""]
    # Summary table
    lines.append("| corpus | files | round-trip pass | tokens | mean len | median len | min | max | vocab used / total |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in all_stats:
        lines.append(
            f"| {s['corpus']} | {s['n_files']} | "
            f"{s['round_trip_pass']}/{s['round_trip_pass']+s['round_trip_fail']} "
            f"({100*s['round_trip_rate']:.1f}%) | "
            f"{s['total_tokens']:,} | "
            f"{s['mean_seq_len']:.0f} | {s['median_seq_len']:.0f} | "
            f"{s['min_seq_len']} | {s['max_seq_len']} | "
            f"{s['vocab_used']}/{s['vocab_size']} ({100*s['vocab_util']:.0f}%) |"
        )

    # Token budget
    if corpus_sizes:
        lines += ["", "## Token budget (projected from sample means)", ""]
        lines.append("Per-corpus projection = (mean tokens / sample file) \u00D7 (full-corpus file count). "
                     "GigaMIDI training split = the all-instruments-with-drums training-V1.1-80% split used as pretraining; "
                     "JSB Chorales is held out for zero-shot probing.")
        lines.append("")
        lines.append("| corpus | full-corpus files | sample size | mean tokens / file | projected total tokens | role |")
        lines.append("|---|---|---|---|---|---|")
        roles = {"JSB": "probing (held out)", "GigaMIDI": "pretraining"}
        for s in all_stats:
            full = corpus_sizes.get(s["corpus"], s["n_files"])
            mean_tok = s["mean_seq_len"]
            projected = int(round(mean_tok * full))
            role = roles.get(s["corpus"], "")
            lines.append(
                f"| {s['corpus']} | {full:,} | {s['n_files']} | {mean_tok:,.0f} | {projected:,} | {role} |"
            )

    if genre_info and genre_info.get("counter"):
        lines += ["", "## GigaMIDI genre distribution (sample)", ""]
        lines.append(f"{genre_info['labeled']} of {genre_info['total']} sampled files carry a genre label "
                     f"across the five metadata columns (curated, scraped, Discogs, Last.fm, Tagtraum). "
                     f"Top entries:")
        lines.append("")
        lines.append("| genre | count |")
        lines.append("|---|---|")
        for g, c in genre_info["counter"].most_common(20):
            lines.append(f"| {g} | {c} |")

    lines += ["", "## Top-20 tokens by corpus", ""]
    for s in all_stats:
        lines.append(f"### {s['corpus']}")
        lines.append("")
        lines.append("| rank | token | count | share |")
        lines.append("|---|---|---|---|")
        total = max(1, s["total_tokens"])
        for r, (tid, c) in enumerate(s["top_tokens"], 1):
            lines.append(f"| {r} | `{ID2TOKEN.get(tid, tid)}` | {c:,} | {100*c/total:.2f}% |")
        lines.append("")
    return "\n".join(lines)


def print_console_table(all_stats):
    rows = []
    for s in all_stats:
        rows.append({
            "corpus": s["corpus"],
            "files": s["n_files"],
            "rt_pass": f"{s['round_trip_pass']}/{s['round_trip_pass']+s['round_trip_fail']}",
            "rt_rate": f"{100*s['round_trip_rate']:.1f}%",
            "tokens": s["total_tokens"],
            "mean_len": round(s["mean_seq_len"]),
            "median_len": round(s["median_seq_len"]),
            "min_len": s["min_seq_len"],
            "max_len": s["max_seq_len"],
            "vocab_used": f"{s['vocab_used']}/{s['vocab_size']}",
        })
    df = pd.DataFrame(rows)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    for s in all_stats:
        print(f"\n--- Top-10 tokens [{s['corpus']}] ---")
        for r, (tid, c) in enumerate(s["top_tokens"][:10], 1):
            print(f"  {r:>2}. {ID2TOKEN.get(tid, tid):<16s}  {c:>8,}")


# --- Main ---------------------------------------------------------------------

def main():
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    print(f"[corpus_stats] vocab_size={VOCAB_SIZE}")

    jsb = load_jsb(N_SAMPLES_JSB)
    giga, giga_full = load_gigamidi(N_SAMPLES_GIGAMIDI)

    # GigaMIDI genre metadata for the sampled files.
    genre_info = None
    if giga:
        meta = load_gigamidi_metadata([stem for stem, _ in giga])
        ctr, n_labeled = extract_genres(meta)
        genre_info = {"counter": ctr, "labeled": n_labeled, "total": len(giga)}
        plot_genres("GigaMIDI", ctr, len(giga))

    corpus_sizes = {
        "JSB": 371,            # Bach chorales available via music21.corpus.chorales.Iterator()
        "GigaMIDI": giga_full, # all-instruments-with-drums training-V1.1-80% split
    }

    all_stats = []
    for name, pairs in [("JSB", jsb), ("GigaMIDI", giga)]:
        if not pairs:
            print(f"[corpus_stats] {name}: no files loaded, skipping.")
            continue
        s = corpus_stats(name, pairs)
        plot_lengths(name, s["lengths"])
        plot_topk(name, s["top_tokens"])
        all_stats.append(s)

    md = format_md(all_stats, corpus_sizes=corpus_sizes, genre_info=genre_info)
    (RES_DIR / "corpus_stats.md").write_text(md)
    print_console_table(all_stats)

    # Console token-budget summary
    print("\n=== Token budget (projected from sample means) ===")
    for s in all_stats:
        full = corpus_sizes.get(s["corpus"], s["n_files"])
        mean_tok = s["mean_seq_len"]
        projected = int(round(mean_tok * full))
        role = {"JSB": "probe", "GigaMIDI": "pretrain"}.get(s["corpus"], "")
        print(f"  {s['corpus']:10s} full_files={full:8d} sample_n={s['n_files']:4d} "
              f"mean_tok={mean_tok:9.0f} projected={projected:>13,}  [{role}]")

    if genre_info and genre_info["counter"]:
        print(f"\n=== GigaMIDI genre labels (sample) ===")
        print(f"  {genre_info['labeled']}/{genre_info['total']} sampled files carry any genre label")
        for g, c in genre_info["counter"].most_common(15):
            print(f"    {c:3d}  {g}")

    print(f"\n[corpus_stats] wrote {RES_DIR / 'corpus_stats.md'}")
    print(f"[corpus_stats] wrote figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
