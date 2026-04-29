"""Dataset utilities for GPT-style next-token training on GigaMIDI samples.

Pipeline:
1) Load/encode MIDI files from data/gigamidi/sample/
2) Concatenate all token ids with EOS separators
3) Chunk into fixed non-overlapping windows
4) Build (input, target) examples via one-token shift
5) Split into train/val and wrap in DataLoader
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pretty_midi
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import EOS, ID2TOKEN, VOCAB_SIZE, decode, encode

DEFAULT_BLOCK_SIZE = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_SPLIT_RATIO = 0.9
DEFAULT_SEED = 17


@dataclass
class DatasetStats:
    n_files_seen: int
    n_files_encoded: int
    n_files_failed: int
    n_sequences: int
    n_tokens_total: int
    n_chunks_total: int
    n_train_chunks: int
    n_val_chunks: int


class TokenChunkDataset(Dataset):
    """A dataset of fixed-size token chunks for next-token prediction."""

    def __init__(self, chunks: Sequence[torch.Tensor]) -> None:
        self._chunks = list(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self._chunks[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_encoded_sequences(sample_dir: Path) -> tuple[List[List[int]], int]:
    """Encode each MIDI in sample_dir into token-id sequences.

    Returns (sequences, n_failed).
    Files that fail parsing/encoding are skipped.
    """
    midi_paths = (
        sorted(sample_dir.glob("*.mid"))
        + sorted(sample_dir.glob("*.midi"))
    )
    sequences: List[List[int]] = []
    n_failed = 0
    for midi_path in midi_paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            ids = encode(pm)
            sequences.append(ids)
        except Exception:
            n_failed += 1
    return sequences, n_failed


def concat_with_eos(sequences: Sequence[Sequence[int]]) -> List[int]:
    """Join sequences into one stream with exactly one EOS between pieces."""
    if not sequences:
        return []

    flat: List[int] = []
    for i, seq in enumerate(sequences):
        piece = list(seq)
        while piece and piece[-1] == EOS:
            piece.pop()
        flat.extend(piece)
        if i < len(sequences) - 1:
            flat.append(EOS)
    return flat


def chunk_token_stream(
    token_stream: Sequence[int], block_size: int
) -> List[torch.Tensor]:
    """Split stream into non-overlapping fixed-size chunks."""
    if block_size < 2:
        raise ValueError("block_size must be >= 2")

    n_chunks = len(token_stream) // block_size
    usable = n_chunks * block_size
    chunks: List[torch.Tensor] = []
    for i in range(0, usable, block_size):
        chunk = torch.tensor(token_stream[i:i + block_size], dtype=torch.long)
        chunks.append(chunk)
    return chunks


def split_chunks(
    chunks: Sequence[torch.Tensor],
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Shuffle chunk indices once, then split into train/val by ratio."""
    if not chunks:
        return [], []
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1")

    rng = random.Random(seed)
    indices = list(range(len(chunks)))
    rng.shuffle(indices)

    n_train = int(len(indices) * split_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_chunks = [chunks[i] for i in train_idx]
    val_chunks = [chunks[i] for i in val_idx]
    return train_chunks, val_chunks


def build_dataloaders(
    sample_dir: Path | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[DataLoader, DataLoader, DatasetStats]:
    """Build train/val DataLoaders from local GigaMIDI sample files."""
    if sample_dir is None:
        root = Path(__file__).resolve().parent.parent
        sample_dir = root / "data" / "gigamidi" / "sample"

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    midi_paths = (
        sorted(sample_dir.glob("*.mid"))
        + sorted(sample_dir.glob("*.midi"))
    )
    sequences, n_failed = load_encoded_sequences(sample_dir=sample_dir)
    token_stream = concat_with_eos(sequences)

    # Sanity check: all ids must fall within tokenizer vocab.
    if token_stream:
        max_id = max(token_stream)
        assert (
            max_id < VOCAB_SIZE
        ), f"Found token id {max_id} but vocab size is {VOCAB_SIZE}"

    chunks = chunk_token_stream(token_stream=token_stream, block_size=block_size)
    train_chunks, val_chunks = split_chunks(
        chunks=chunks, split_ratio=split_ratio, seed=seed
    )

    train_ds = TokenChunkDataset(train_chunks)
    val_ds = TokenChunkDataset(val_chunks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    stats = DatasetStats(
        n_files_seen=len(midi_paths),
        n_files_encoded=len(sequences),
        n_files_failed=n_failed,
        n_sequences=len(sequences),
        n_tokens_total=len(token_stream),
        n_chunks_total=len(chunks),
        n_train_chunks=len(train_chunks),
        n_val_chunks=len(val_chunks),
    )
    return train_loader, val_loader, stats


def _print_decoded_batch_sanity(train_loader: DataLoader) -> None:
    """Decode one random sample from a random batch for quick sanity checks."""
    if len(train_loader.dataset) == 0:
        print("[dataset] No train samples available for sanity decode.")
        return

    batch = next(iter(train_loader))
    x, _ = batch
    sample_idx = random.randrange(x.shape[0])
    token_ids = x[sample_idx].tolist()

    decoded_pm = decode(token_ids)
    n_notes = sum(len(inst.notes) for inst in decoded_pm.instruments)
    token_preview = " ".join(
        ID2TOKEN.get(tid, f"UNK({tid})") for tid in token_ids[:40]
    )

    print("[dataset] Random decoded sample preview (first 40 tokens):")
    print(token_preview)
    print(f"[dataset] Decoded PrettyMIDI note count: {n_notes}")


if __name__ == "__main__":
    train_loader, val_loader, stats = build_dataloaders()

    print(
        "[dataset] Files seen/encoded/failed: "
        f"{stats.n_files_seen}/{stats.n_files_encoded}/{stats.n_files_failed}"
    )
    print(
        "[dataset] Tokens/chunks/train/val: "
        f"{stats.n_tokens_total}/{stats.n_chunks_total}/"
        f"{stats.n_train_chunks}/{stats.n_val_chunks}"
    )
    _print_decoded_batch_sanity(train_loader)
