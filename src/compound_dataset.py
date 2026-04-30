"""Dataset utilities for the Octuple-style compound tokenizer.

Mirrors ``dataset.py`` but yields compound steps shaped ``(T, N_AXES)``
instead of flat token sequences. Each step is a tuple of feature ids
that the compound model embeds in parallel and sums.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pretty_midi
import torch
from torch.utils.data import DataLoader, Dataset

from compound import (
    AXIS_SIZES,
    N_AXES,
    SENTINELS,
    STEP_BOS,
    STEP_EOS,
    STEP_PAD,
    encode_compound,
)

DEFAULT_BLOCK_SIZE = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_SPLIT_RATIO = 0.9
DEFAULT_SEED = 17


@dataclass
class CompoundDatasetStats:
    n_files_seen: int
    n_files_encoded: int
    n_files_failed: int
    n_steps_total: int
    n_chunks_total: int
    n_train_chunks: int
    n_val_chunks: int


class CompoundChunkDataset(Dataset):
    """Returns (input, target) pairs of shape (block_size-1, N_AXES) each."""

    def __init__(self, chunks: Sequence[torch.Tensor]) -> None:
        self._chunks = list(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self._chunks[idx]
        return chunk[:-1], chunk[1:]


def _eos_pad_step() -> List[int]:
    """A PAD step used to fill block boundaries."""
    s = list(SENTINELS)
    s[0] = STEP_PAD
    return s


def _bos_step_separator() -> List[int]:
    """A BOS step inserted between concatenated piece sequences."""
    s = list(SENTINELS)
    s[0] = STEP_BOS
    return s


def load_encoded_compound_sequences(
    sample_dir: Path,
) -> tuple[List[List[List[int]]], int]:
    paths = sorted(sample_dir.rglob("*.mid")) + sorted(sample_dir.rglob("*.midi"))
    sequences: List[List[List[int]]] = []
    n_failed = 0
    for p in paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(p))
            sequences.append(encode_compound(pm))
        except Exception:
            n_failed += 1
    return sequences, n_failed


def concat_sequences(seqs: Sequence[Sequence[Sequence[int]]]) -> List[List[int]]:
    """Concatenate compound sequences with a BOS separator between pieces."""
    flat: List[List[int]] = []
    sep = _bos_step_separator()
    for i, seq in enumerate(seqs):
        flat.extend(seq)
        if i < len(seqs) - 1:
            flat.append(sep)
    return flat


def chunk_compound_stream(stream: Sequence[Sequence[int]], block_size: int) -> List[torch.Tensor]:
    n_chunks = len(stream) // block_size
    chunks: List[torch.Tensor] = []
    for i in range(n_chunks):
        block = stream[i * block_size:(i + 1) * block_size]
        chunks.append(torch.tensor(block, dtype=torch.long))
    return chunks


def split_chunks(
    chunks: Sequence[torch.Tensor],
    split_ratio: float,
    seed: int,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    rng = random.Random(seed)
    indices = list(range(len(chunks)))
    rng.shuffle(indices)
    n_train = int(len(indices) * split_ratio)
    train = [chunks[i] for i in indices[:n_train]]
    val = [chunks[i] for i in indices[n_train:]]
    return train, val


def build_compound_dataloaders(
    sample_dir: Path | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[DataLoader, DataLoader, CompoundDatasetStats]:
    if sample_dir is None:
        root = Path(__file__).resolve().parent.parent
        sample_dir = root / "data" / "gigamidi" / "sample"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample dir not found: {sample_dir}")

    midi_paths = sorted(sample_dir.rglob("*.mid")) + sorted(sample_dir.rglob("*.midi"))
    sequences, n_failed = load_encoded_compound_sequences(sample_dir)
    stream = concat_sequences(sequences)

    if stream:
        max_per_axis = [max(s[a] for s in stream) for a in range(N_AXES)]
        for a, mx in enumerate(max_per_axis):
            assert mx < AXIS_SIZES[a], (
                f"axis {a} value {mx} exceeds size {AXIS_SIZES[a]}"
            )

    chunks = chunk_compound_stream(stream, block_size)
    train_chunks, val_chunks = split_chunks(chunks, split_ratio, seed)

    train_ds = CompoundChunkDataset(train_chunks)
    val_ds = CompoundChunkDataset(val_chunks)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    stats = CompoundDatasetStats(
        n_files_seen=len(midi_paths),
        n_files_encoded=len(sequences),
        n_files_failed=n_failed,
        n_steps_total=len(stream),
        n_chunks_total=len(chunks),
        n_train_chunks=len(train_chunks),
        n_val_chunks=len(val_chunks),
    )
    return train_loader, val_loader, stats


if __name__ == "__main__":
    train_loader, val_loader, stats = build_compound_dataloaders()
    print(
        f"[compound_dataset] files seen/encoded/failed: "
        f"{stats.n_files_seen}/{stats.n_files_encoded}/{stats.n_files_failed}"
    )
    print(
        f"[compound_dataset] steps={stats.n_steps_total} "
        f"chunks={stats.n_chunks_total} train={stats.n_train_chunks} val={stats.n_val_chunks}"
    )
    print(f"[compound_dataset] axis sizes = {AXIS_SIZES}")
