"""Caption-conditioned MIDI dataset and dataloaders.

Reads records from captions_llm.jsonl and returns fixed-length MIDI token
windows plus text captions for conditioning.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pretty_midi
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import encode

DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_BATCH_SIZE = 64
DEFAULT_SPLIT_RATIO = 0.95
DEFAULT_SEED = 17
DEFAULT_NUM_WORKERS = 4


@dataclass
class CaptionDatasetStats:
    n_total_records: int
    n_train_records: int
    n_val_records: int


def _load_jsonl_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _caption_from_record(record: Dict[str, Any]) -> str:
    llm_caption = str(record.get("caption", "")).strip()
    if llm_caption:
        return llm_caption
    return str(record.get("caption_template", "")).strip()


class MidiCaptionDataset(Dataset):
    """Dataset yielding (midi_tokens, attention_mask, caption, path)."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        is_train: bool,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.records = records
        self.is_train = is_train
        self.max_seq_len = max_seq_len
        self._rng = random.Random(seed)
        self._token_cache: Dict[int, Optional[List[int]]] = {}
        self._valid_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize_record(self, idx: int) -> Optional[List[int]]:
        if idx in self._token_cache:
            return self._token_cache[idx]

        rec = self.records[idx]
        midi_path = Path(str(rec.get("path", "")))
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            ids = encode(pm)
        except Exception:
            self._token_cache[idx] = None
            return None

        # Keep only sequences that can produce a full fixed-length crop.
        if len(ids) < self.max_seq_len:
            self._token_cache[idx] = None
            return None

        self._token_cache[idx] = ids
        self._valid_indices.add(idx)
        return ids

    def _crop(self, ids: List[int]) -> List[int]:
        if self.is_train:
            max_start = len(ids) - self.max_seq_len
            start = 0 if max_start <= 0 else self._rng.randint(0, max_start)
            return ids[start:start + self.max_seq_len]
        return ids[: self.max_seq_len]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ids = self._tokenize_record(idx)

        # If this file cannot be parsed/tokenized, sample another valid item.
        if ids is None:
            if not self._valid_indices:
                # Warmup pass to discover valid indices lazily.
                for probe_idx in range(len(self.records)):
                    self._tokenize_record(probe_idx)
                    if self._valid_indices:
                        break
            if not self._valid_indices:
                raise RuntimeError(
                    "No valid MIDI records found that meet max_seq_len."
                )
            replacement_idx = self._rng.choice(list(self._valid_indices))
            ids = self._tokenize_record(replacement_idx)
            rec = self.records[replacement_idx]
        else:
            rec = self.records[idx]

        assert ids is not None
        window = self._crop(ids)
        caption = _caption_from_record(rec)

        input_ids = torch.tensor(window, dtype=torch.long)
        attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": caption,
            "path": str(rec.get("path", "")),
        }


def _collate_caption_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in items], dim=0),
        "attention_mask": torch.stack(
            [x["attention_mask"] for x in items], dim=0
        ),
        "captions": [x["caption"] for x in items],
        "paths": [x["path"] for x in items],
    }


def build_caption_dataloaders(
    jsonl_path: Path | str,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    seed: int = DEFAULT_SEED,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: Optional[bool] = None,
) -> tuple[DataLoader, DataLoader, CaptionDatasetStats]:
    """Build train/val DataLoaders from captions_llm.jsonl records.

    - 95/5 split with fixed random seed
    - train random crop to max_seq_len
    - val deterministic first-window crop
    - drop_last=True for train to keep fixed contrastive matrix shape
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1.")

    records = _load_jsonl_records(path)
    if not records:
        raise ValueError(f"No records found in {path}")

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_train = int(len(indices) * split_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds = MidiCaptionDataset(
        records=train_records,
        is_train=True,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    val_ds = MidiCaptionDataset(
        records=val_records,
        is_train=False,
        max_seq_len=max_seq_len,
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_caption_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_caption_batch,
    )

    stats = CaptionDatasetStats(
        n_total_records=len(records),
        n_train_records=len(train_records),
        n_val_records=len(val_records),
    )
    return train_loader, val_loader, stats
