"""Minimal decoder-only GPT for MIDI token LM (interpretability-friendly).

Architecture (Pre-LN, GPT-2 style):
  tok_emb + pos_emb
  → repeat: x += attn(LN(x)); x += mlp(LN(x))
  → LN → logits

Causal self-attention is implemented explicitly so attention weights can be
returned for probing (``return_attn=True``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tokenizer import VOCAB_SIZE


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 1024
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1


def default_gpt_config() -> GPTConfig:
    """Recommended starter config (~10M params with weight tying)."""
    return GPTConfig()


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention; exposes softmax weights when needed."""

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
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * self.scale
        causal = self.causal_mask[:, :, :T, :T]
        att = att.masked_fill(~causal, float("-inf"))
        att_weights = F.softmax(att, dim=-1)
        att_weights = self.attn_drop(att_weights)

        out = att_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))

        if return_attn:
            return out, att_weights
        return out, None


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
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h, attn_w = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, attn_w


class GPT(nn.Module):
    """Decoder-only transformer LM with optional per-layer attention outputs."""

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
        idx: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Compute logits for token indices ``idx`` of shape (B, T).

        If ``return_attn`` is True, also returns a list of attention weight
        tensors, one per layer, each shaped (B, n_heads, T, T) after softmax.
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        tok = self.wte(idx)
        pos_e = self.wpe(pos)
        x = self.drop(tok + pos_e)

        attn_list: List[torch.Tensor] = []
        for block in self.blocks:
            x, aw = block(x, return_attn=return_attn)
            if aw is not None:
                attn_list.append(aw)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_attn:
            return logits, attn_list
        return logits

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
