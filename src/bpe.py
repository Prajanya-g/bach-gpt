"""Byte-pair encoding on top of the base MIDI tokenizer.

Trains greedy pair merges over sequences of base token ids and
exposes apply/unapply for use during dataset construction and decode.

Merge id space
--------------
Base ids occupy [0, base_vocab_size). Merge i (0-indexed) is assigned id
base_vocab_size + i, so the BPE-aware vocab size is base_vocab_size + n_merges.

Boundary protection
-------------------
Pairs where either side equals PAD or EOS are never merged, so the model
still sees explicit sequence terminators.

API
---
train_bpe(streams, n_merges, base_vocab_size, no_merge_ids) -> merges
apply_bpe(ids, merges)   -> List[int]
unapply_bpe(ids, merges) -> List[int]
save(merges, path), load(path)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# A merge table is a list of ((left, right), merged_id) entries in
# learning order. The order matters: earlier merges may participate in
# later merges. Stored as a JSON list of [left, right, merged_id].

Merge = Tuple[int, int, int]


def default_no_merge_ids() -> set:
    """Structural ids that must remain unmerged so the model sees clean
    sequence/phrase/bar boundaries."""
    from tokenizer import (
        BAR_END,
        BAR_START,
        EOS,
        PAD,
        PHRASE_END,
        PHRASE_START,
    )
    return {PAD, EOS, PHRASE_START, PHRASE_END, BAR_START, BAR_END}


def _count_pairs(
    streams: Sequence[Sequence[int]],
    no_merge_ids: set,
) -> Counter:
    counter: Counter = Counter()
    for s in streams:
        for a, b in zip(s, s[1:]):
            if a in no_merge_ids or b in no_merge_ids:
                continue
            counter[(a, b)] += 1
    return counter


def _replace_pair(
    seq: Sequence[int],
    pair: Tuple[int, int],
    new_id: int,
    dropout: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Replace adjacent occurrences of ``pair`` with ``new_id``.

    With ``dropout > 0`` each occurrence is independently skipped with that
    probability, leaving the original two tokens in place. This is the
    BPE-dropout regularization from Provilkov et al. 2020.
    """
    a, b = pair
    out: List[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            if dropout > 0.0 and (rng or random).random() < dropout:
                out.append(seq[i])
                i += 1
            else:
                out.append(new_id)
                i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(
    streams: Sequence[Sequence[int]],
    n_merges: int,
    base_vocab_size: int,
    no_merge_ids: Iterable[int] = (),
    min_pair_count: int = 2,
) -> List[Merge]:
    """Greedy BPE on base-id sequences. Returns the merge list."""
    no_merge = set(no_merge_ids)
    working: List[List[int]] = [list(s) for s in streams]
    merges: List[Merge] = []
    next_id = base_vocab_size

    for _ in range(n_merges):
        counter = _count_pairs(working, no_merge)
        if not counter:
            break
        (best_pair, best_count) = counter.most_common(1)[0]
        if best_count < min_pair_count:
            break
        merges.append((best_pair[0], best_pair[1], next_id))
        working = [_replace_pair(s, best_pair, next_id) for s in working]
        next_id += 1

    return merges


def apply_bpe(
    ids: Sequence[int],
    merges: Sequence[Merge],
    dropout: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Apply merges in learned order. O(M * N) per stream; fine offline.

    ``dropout`` enables BPE-dropout: each merge candidate is randomly
    skipped with this probability, exposing the model to multiple
    segmentations of the same underlying base sequence. Use 0.0 at
    inference time and roughly 0.1 during training.
    """
    out = list(ids)
    for left, right, merged in merges:
        out = _replace_pair(out, (left, right), merged, dropout=dropout, rng=rng)
    return out


def unapply_bpe(ids: Sequence[int], merges: Sequence[Merge]) -> List[int]:
    """Expand merged ids back to base. Walks merges in reverse order."""
    expand: Dict[int, Tuple[int, int]] = {m[2]: (m[0], m[1]) for m in merges}
    if not expand:
        return list(ids)

    out = list(ids)
    changed = True
    while changed:
        changed = False
        new_out: List[int] = []
        for tid in out:
            if tid in expand:
                a, b = expand[tid]
                new_out.append(a)
                new_out.append(b)
                changed = True
            else:
                new_out.append(tid)
        out = new_out
    return out


def save(merges: Sequence[Merge], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([list(m) for m in merges]))


def load(path: Path) -> List[Merge]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [(int(a), int(b), int(c)) for a, b, c in data]


def effective_vocab_size(base_vocab_size: int, merges: Sequence[Merge]) -> int:
    return base_vocab_size + len(merges)


# --- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path as _P

    _SRC = _P(__file__).resolve().parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    import pretty_midi  # noqa: E402

    from tokenizer import (  # noqa: E402
        BAR_END,
        BAR_START,
        EOS,
        PAD,
        PHRASE_END,
        PHRASE_START,
        VOCAB_SIZE,
        encode,
    )

    parser = argparse.ArgumentParser(description="Train BPE on encoded MIDI.")
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=str(_SRC.parent / "data" / "gigamidi" / "sample"),
    )
    parser.add_argument("--n-merges", type=int, default=2000)
    parser.add_argument(
        "--out",
        type=str,
        default=str(_SRC.parent / "data" / "bpe" / "merges.json"),
    )
    args = parser.parse_args()

    sample_dir = _P(args.sample_dir)
    midi_paths = (
        sorted(sample_dir.rglob("*.mid"))
        + sorted(sample_dir.rglob("*.midi"))
    )
    if not midi_paths:
        raise SystemExit(f"No MIDI files found under {sample_dir}")

    streams: List[List[int]] = []
    n_failed = 0
    for p in midi_paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(p))
            streams.append(encode(pm))
        except Exception:
            n_failed += 1

    n_base_tokens = sum(len(s) for s in streams)
    print(
        f"[bpe] streams={len(streams)} failed={n_failed} "
        f"base_tokens={n_base_tokens}"
    )

    merges = train_bpe(
        streams=streams,
        n_merges=args.n_merges,
        base_vocab_size=VOCAB_SIZE,
        no_merge_ids={PAD, EOS, PHRASE_START, PHRASE_END, BAR_START, BAR_END},
    )

    after = sum(len(apply_bpe(s, merges)) for s in streams)
    print(
        f"[bpe] learned {len(merges)} merges; "
        f"compression: {n_base_tokens} -> {after} "
        f"({(1 - after / max(n_base_tokens, 1)) * 100:.1f}% fewer tokens)"
    )

    out_path = _P(args.out)
    save(merges, out_path)
    print(f"[bpe] saved -> {out_path}")
