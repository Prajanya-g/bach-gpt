"""Minimal decoder-only GPT for MIDI token LM (interpretability-friendly).

Architecture (Pre-LN, GPT-2 style):
  tok_emb + pos_emb
  → repeat: x += attn(LN(x)); x += mlp(LN(x))
  → LN → logits

Causal self-attention is implemented explicitly so attention weights can be
returned for probing (``return_attn=True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import (
    ID2TOKEN,
    N_POS_BINS,
    PITCH_MAX,
    PITCH_MIN,
    VOCAB_SIZE,
)


SCALE_DEGREE_NONE = 12   # sentinel for "no key context / not a pitch"


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 1024
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    # Compound-embedding axes (additive on top of token embedding).
    use_pitch_class_embed: bool = True   # adds 13 (pc 0..11 + sentinel)
    use_octave_embed: bool = True        # adds 9  (oct 0..7 + sentinel)
    use_interval_embed: bool = True      # adds 27 (-13..13 + sentinel) for melodic interval
    use_beat_cyclic_embed: bool = True   # adds N_POS_BINS+1 for beat-within-bar
    use_scale_degree_embed: bool = True  # adds 13 (chromatic 0..11 + sentinel) relative to current key


def default_gpt_config() -> GPTConfig:
    """Recommended starter config (~10M params with weight tying)."""
    return GPTConfig()


# --- Static per-token feature lookups -----------------------------------------

# Sentinel index for "no pitch class / octave applies."
PC_NONE = 12
OCT_NONE = 8


def _build_token_pitch_feature_tables() -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token-id buffers giving (pitch_class, octave) for pitch tokens
    and sentinels otherwise.
    """
    pc = torch.full((VOCAB_SIZE,), PC_NONE, dtype=torch.long)
    oct_ = torch.full((VOCAB_SIZE,), OCT_NONE, dtype=torch.long)
    for tid, name in ID2TOKEN.items():
        if (
            name.startswith("P")
            and not name.startswith("POS")
            and name[1:].isdigit()
        ):
            midi = int(name[1:])
            if PITCH_MIN <= midi <= PITCH_MAX:
                pc[tid] = midi % 12
                oct_[tid] = max(0, min(7, midi // 12 - 1))
    return pc, oct_


def _is_pitch_token_mask() -> torch.Tensor:
    """Boolean mask of length VOCAB_SIZE: True for pitch tokens."""
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    for tid, name in ID2TOKEN.items():
        if (
            name.startswith("P")
            and not name.startswith("POS")
            and name[1:].isdigit()
        ):
            mask[tid] = True
    return mask


def _midi_for_pitch_token(tid: int) -> int:
    """MIDI number for a pitch token id, or -1 if not a pitch token."""
    name = ID2TOKEN.get(tid, "")
    if name.startswith("P") and not name.startswith("POS") and name[1:].isdigit():
        return int(name[1:])
    return -1


def _build_pitch_to_midi() -> torch.Tensor:
    """Per-token-id MIDI value for pitch tokens, -1 elsewhere."""
    arr = torch.full((VOCAB_SIZE,), -1, dtype=torch.long)
    for tid in range(VOCAB_SIZE):
        arr[tid] = _midi_for_pitch_token(tid)
    return arr


def _build_pos_token_value() -> torch.Tensor:
    """For each token id, the POS bin value if it's a POS token, else -1."""
    arr = torch.full((VOCAB_SIZE,), -1, dtype=torch.long)
    for tid, name in ID2TOKEN.items():
        if name.startswith("POS") and name[3:].isdigit():
            arr[tid] = int(name[3:])
    return arr


def _build_key_token_root() -> torch.Tensor:
    """For each token id, the root pitch class for KEY tokens (0..11),
    else -1. KEY_0..11 are major keys C..B; KEY_12..23 are minor keys C..B.
    """
    arr = torch.full((VOCAB_SIZE,), -1, dtype=torch.long)
    for tid, name in ID2TOKEN.items():
        if name.startswith("KEY_") and name[4:].isdigit():
            arr[tid] = int(name[4:]) % 12
    return arr


# Interval embedding: clipped to [-13..13] with 27 = sentinel (no interval).
INTERVAL_RANGE = 13
INTERVAL_NONE = 2 * INTERVAL_RANGE + 1   # = 27


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional weight return."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # [1, 1, block_size, block_size] boolean mask: 1 = allowed
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, block_size, block_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        B, Tq, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, Tq, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        Tk = k.size(2)
        past_len = Tk - Tq
        key_pos = torch.arange(Tk, device=x.device).unsqueeze(0)
        query_pos = (
            torch.arange(Tq, device=x.device).unsqueeze(1) + past_len
        )
        causal = key_pos <= query_pos
        att = att.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        att_weights = F.softmax(att, dim=-1)
        att_weights = self.attn_drop(att_weights)

        out = att_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, Tq, C)
        out = self.resid_drop(self.proj(out))
        present = (k, v) if use_cache else None

        if return_attn:
            return out, att_weights, present
        return out, None, present


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            block_size=config.block_size,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        h, attn_w, present = self.attn(
            self.ln1(x),
            return_attn=return_attn,
            use_cache=use_cache,
            past_kv=past_kv,
        )
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, attn_w, present


class GPT(nn.Module):
    """Decoder-only transformer LM with optional attention outputs."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # --- Compound (2D) embedding axes ------------------------------------
        # Static per-token-id feature lookups (computed from tokenizer vocab).
        pc_tab, oct_tab = _build_token_pitch_feature_tables()
        self.register_buffer("tok_to_pc", pc_tab, persistent=False)
        self.register_buffer("tok_to_octave", oct_tab, persistent=False)
        self.register_buffer(
            "tok_to_midi", _build_pitch_to_midi(), persistent=False
        )
        self.register_buffer(
            "tok_to_pos_value", _build_pos_token_value(), persistent=False
        )
        self.register_buffer(
            "tok_to_key_root", _build_key_token_root(), persistent=False
        )
        self.register_buffer(
            "tok_is_pitch", _is_pitch_token_mask(), persistent=False
        )

        if config.use_pitch_class_embed:
            self.embed_pc = nn.Embedding(PC_NONE + 1, config.d_model)
        if config.use_octave_embed:
            self.embed_octave = nn.Embedding(OCT_NONE + 1, config.d_model)
        if config.use_interval_embed:
            self.embed_interval = nn.Embedding(INTERVAL_NONE + 1, config.d_model)
        if config.use_beat_cyclic_embed:
            # N_POS_BINS bins + sentinel for "no bar context".
            self.embed_beat = nn.Embedding(N_POS_BINS + 1, config.d_model)
        if config.use_scale_degree_embed:
            self.embed_scale_degree = nn.Embedding(SCALE_DEGREE_NONE + 1, config.d_model)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[
            List[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]],
        Tuple[
            torch.Tensor,
            List[torch.Tensor],
            List[Tuple[torch.Tensor, torch.Tensor]],
        ],
    ]:
        """Compute logits for token inputs.

        Provide exactly one of:
        - ``idx``: token ids of shape (B, T), or
        - ``inputs_embeds``: precomputed embeddings of shape (B, T, d_model)

        If ``return_attn`` is True, also returns a list of attention weight
        tensors, one per layer, each shaped (B, n_heads, T, T) after softmax.
        """
        if (idx is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of idx or inputs_embeds.")

        if inputs_embeds is not None:
            B, T, C = inputs_embeds.shape
            if C != self.config.d_model:
                raise ValueError(
                    "inputs_embeds last dim "
                    f"{C} != d_model {self.config.d_model}"
                )
        else:
            assert idx is not None
            B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        if position_ids is None:
            if idx is not None:
                device = idx.device
            else:
                assert inputs_embeds is not None
                device = inputs_embeds.device
            pos = torch.arange(0, T, device=device, dtype=torch.long)
            pos_e = self.wpe(pos).unsqueeze(0)
        else:
            if position_ids.shape[-1] != T:
                raise ValueError(
                    "position_ids length must match sequence length."
                )
            pos_e = self.wpe(position_ids)
            if pos_e.dim() == 2:
                pos_e = pos_e.unsqueeze(0)
            if pos_e.shape[0] == 1 and B > 1:
                pos_e = pos_e.expand(B, -1, -1)

        tok = self.wte(idx) if idx is not None else inputs_embeds
        x = tok + pos_e
        if idx is not None:
            x = x + self._compound_embeds(idx)
        x = self.drop(x)

        attn_list: List[torch.Tensor] = []
        present_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            block_idx = len(present_key_values)
            past_kv = None
            if past_key_values is not None and block_idx < len(past_key_values):
                past_kv = past_key_values[block_idx]
            x, aw, present = block(
                x,
                return_attn=return_attn,
                use_cache=use_cache,
                past_kv=past_kv,
            )
            if aw is not None:
                attn_list.append(aw)
            if present is not None:
                present_key_values.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_attn and use_cache:
            return logits, attn_list, present_key_values
        if return_attn:
            return logits, attn_list
        if use_cache:
            return logits, present_key_values
        return logits

    def _compound_embeds(self, idx: torch.Tensor) -> torch.Tensor:
        """Sum of all enabled compound (2D) embedding axes for a batch of
        token ids. Returns a tensor of shape (B, T, d_model) — zero if all
        axes are disabled."""
        B, T = idx.shape
        out = torch.zeros(B, T, self.config.d_model, device=idx.device, dtype=self.wte.weight.dtype)

        if self.config.use_pitch_class_embed:
            out = out + self.embed_pc(self.tok_to_pc[idx])
        if self.config.use_octave_embed:
            out = out + self.embed_octave(self.tok_to_octave[idx])
        if self.config.use_interval_embed:
            out = out + self.embed_interval(self._compute_interval_ids(idx))
        if self.config.use_beat_cyclic_embed:
            out = out + self.embed_beat(self._compute_beat_ids(idx))
        if self.config.use_scale_degree_embed:
            out = out + self.embed_scale_degree(self._compute_scale_degree_ids(idx))
        return out

    def _compute_scale_degree_ids(self, idx: torch.Tensor) -> torch.Tensor:
        """For each pitch position, the chromatic scale degree
        (pitch_class - current_key_root) mod 12 — where current key is the
        most recent KEY token seen. Non-pitch positions and positions
        before the first KEY token get the sentinel.
        """
        B, T = idx.shape
        pc = self.tok_to_pc[idx]                              # (B, T) PC_NONE if not pitch
        key_root = self.tok_to_key_root[idx]                  # (B, T) -1 if not KEY
        arange = torch.arange(T, device=idx.device).expand(B, T)
        cand = torch.where(key_root >= 0, arange, torch.full_like(arange, -1))
        last_key_idx = cand.cummax(dim=1).values
        safe_idx = last_key_idx.clamp(min=0)
        cur_root = torch.gather(key_root, 1, safe_idx)
        # Compute (pc - root) mod 12 for pitch positions with a known key.
        is_pitch = pc != PC_NONE
        sd = (pc - cur_root) % 12
        valid = is_pitch & (last_key_idx >= 0)
        return torch.where(
            valid, sd, torch.full_like(sd, SCALE_DEGREE_NONE)
        )

    def _compute_interval_ids(self, idx: torch.Tensor) -> torch.Tensor:
        """For each pitch-token position, the clipped melodic interval to the
        previous pitch token in the same row. Non-pitch positions and the
        first pitch get the sentinel INTERVAL_NONE. Vectorized via cummax.
        """
        B, T = idx.shape
        midi = self.tok_to_midi[idx]                          # (B, T) -1 if not pitch
        is_pitch = midi >= 0                                   # (B, T)
        arange = torch.arange(T, device=idx.device).expand(B, T)
        cand = torch.where(is_pitch, arange, torch.full_like(arange, -1))
        # Shift right by 1: previous-pitch-up-to-t-1
        shifted = torch.cat(
            [torch.full_like(cand[:, :1], -1), cand[:, :-1]], dim=1
        )
        last_idx = shifted.cummax(dim=1).values                # (B, T)
        safe_idx = last_idx.clamp(min=0)
        prev_midi = torch.gather(midi, 1, safe_idx)
        delta = (midi - prev_midi).clamp(-INTERVAL_RANGE, INTERVAL_RANGE) + INTERVAL_RANGE
        valid = is_pitch & (last_idx >= 0)
        return torch.where(valid, delta, torch.full_like(delta, INTERVAL_NONE))

    def _compute_beat_ids(self, idx: torch.Tensor) -> torch.Tensor:
        """For each position, the most recent POS<n> bin value seen so far,
        or N_POS_BINS (sentinel) if no POS token has been emitted yet.
        Vectorized via cummax over POS positions."""
        B, T = idx.shape
        pos_val = self.tok_to_pos_value[idx]                   # (B, T) -1 if not POS
        arange = torch.arange(T, device=idx.device).expand(B, T)
        cand = torch.where(pos_val >= 0, arange, torch.full_like(arange, -1))
        last_idx = cand.cummax(dim=1).values                   # (B, T)
        safe_idx = last_idx.clamp(min=0)
        gathered = torch.gather(pos_val, 1, safe_idx)
        return torch.where(
            last_idx >= 0, gathered, torch.full_like(gathered, N_POS_BINS)
        )

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    cfg = default_gpt_config()
    model = GPT(cfg)
    n_params = model.count_parameters()
    print(f"Config: {cfg}")
    print(f"Parameter count: {n_params:,} (~{n_params / 1e6:.2f}M)")

    x = torch.randint(0, cfg.vocab_size, (2, min(64, cfg.block_size)))
    logits = model(x)
    assert logits.shape == (2, x.shape[1], cfg.vocab_size)

    logits2, attn = model(x, return_attn=True)
    assert len(attn) == cfg.n_layers
    assert attn[0].shape == (2, cfg.n_heads, x.shape[1], x.shape[1])
    print("Forward + return_attn smoke test OK.")
