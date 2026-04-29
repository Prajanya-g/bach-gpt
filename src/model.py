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
        x = self.drop(tok + pos_e)

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
