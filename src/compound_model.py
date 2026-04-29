"""Compound (Octuple-style) GPT for bach-gpt.

Each input position carries N_AXES feature ids that are embedded in
parallel and summed before the transformer. Output is N_AXES separate
classification heads with their own softmaxes; the training loss is the
(optionally weighted) sum of per-axis cross-entropies.

Usage
-----
    from compound import AXIS_SIZES, AXIS_NAMES, STEP_PAD
    from compound_model import CompoundGPT, CompoundGPTConfig, compound_loss

    cfg = CompoundGPTConfig()              # axis_sizes default to compound.AXIS_SIZES
    model = CompoundGPT(cfg)
    # idx: (B, T, N_AXES) long; targets: same shape, shifted by one step
    logits = model(idx)                    # list of (B, T, axis_size_a) tensors
    loss = compound_loss(logits, targets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compound import AXIS_NAMES, AXIS_SIZES, N_AXES, STEP_PAD
from model import GPT, GPTConfig, TransformerBlock


@dataclass
class CompoundGPTConfig:
    axis_sizes: Tuple[int, ...] = field(default_factory=lambda: tuple(AXIS_SIZES))
    block_size: int = 1024
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    # Optional per-axis loss weighting at training time. None = uniform.
    axis_loss_weights: Optional[Tuple[float, ...]] = None


def default_compound_config() -> CompoundGPTConfig:
    return CompoundGPTConfig()


class CompoundGPT(nn.Module):
    """Decoder-only transformer over compound (multi-axis) inputs."""

    def __init__(self, config: CompoundGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_axes = len(config.axis_sizes)
        if self.n_axes != N_AXES:
            raise ValueError(
                f"axis_sizes has {self.n_axes} entries, expected {N_AXES}"
            )

        # Per-axis input embeddings, summed across axes.
        self.input_embeds = nn.ModuleList(
            nn.Embedding(s, config.d_model) for s in config.axis_sizes
        )
        self.wpe = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Reuse the regular transformer blocks. vocab_size in this fake
        # GPTConfig is unused by TransformerBlock; we only need the
        # attention/MLP shape parameters.
        block_cfg = GPTConfig(
            vocab_size=1,
            block_size=config.block_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(block_cfg) for _ in range(config.n_layers)
        )
        self.ln_f = nn.LayerNorm(config.d_model)

        # Per-axis output heads. No weight tying — each axis has its own
        # classifier over a small vocabulary.
        self.heads = nn.ModuleList(
            nn.Linear(config.d_model, s, bias=False) for s in config.axis_sizes
        )

        self.apply(GPT._init_weights)

    def forward(
        self,
        idx: torch.Tensor,
        return_attn: bool = False,
        return_hidden: bool = False,
    ) -> List[torch.Tensor] | Tuple[List[torch.Tensor], List[torch.Tensor]] | torch.Tensor:
        """idx: (B, T, N_AXES) of long feature ids.

        - return_hidden=False, return_attn=False: list of N_AXES logits.
        - return_hidden=True: post-LN hidden states (B, T, d_model). Used by
          the contrastive MIDI encoder for pooling. No classification heads run.
        - return_attn=True: (logits, attn_list).
        """
        if idx.dim() != 3 or idx.shape[-1] != self.n_axes:
            raise ValueError(
                f"Expected idx of shape (B, T, {self.n_axes}); got {tuple(idx.shape)}"
            )
        B, T, _ = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        x = self.input_embeds[0](idx[..., 0])
        for a in range(1, self.n_axes):
            x = x + self.input_embeds[a](idx[..., a])

        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.drop(x + self.wpe(pos).unsqueeze(0))

        attn_list: List[torch.Tensor] = []
        for block in self.blocks:
            x, aw, _ = block(x, return_attn=return_attn)
            if aw is not None:
                attn_list.append(aw)
        x = self.ln_f(x)

        if return_hidden:
            return x

        logits_per_axis = [head(x) for head in self.heads]
        if return_attn:
            return logits_per_axis, attn_list
        return logits_per_axis

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compound_loss(
    logits_per_axis: List[torch.Tensor],
    targets: torch.Tensor,
    axis_weights: Optional[Tuple[float, ...]] = None,
    pad_step_value: int = STEP_PAD,
    ignore_pad_steps: bool = True,
) -> torch.Tensor:
    """Sum of per-axis cross-entropies. ``targets`` shape (B, T, N_AXES).

    If ``ignore_pad_steps`` is True, positions whose step-axis target is
    ``pad_step_value`` contribute zero loss (standard padding mask).
    """
    if targets.dim() != 3:
        raise ValueError(
            f"targets must be (B, T, N_AXES); got {tuple(targets.shape)}"
        )
    n_axes = len(logits_per_axis)
    if axis_weights is None:
        axis_weights = tuple([1.0] * n_axes)
    if len(axis_weights) != n_axes:
        raise ValueError(
            f"axis_weights length {len(axis_weights)} != {n_axes}"
        )

    step_targets = targets[..., 0]              # (B, T)
    valid = step_targets != pad_step_value if ignore_pad_steps else None

    total = torch.zeros((), device=targets.device, dtype=torch.float32)
    for a in range(n_axes):
        logits = logits_per_axis[a]             # (B, T, A_a)
        tgt = targets[..., a]                   # (B, T)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_tgt = tgt.reshape(-1)
        if valid is not None:
            flat_mask = valid.reshape(-1)
            if flat_mask.sum() == 0:
                continue
            loss_a = F.cross_entropy(
                flat_logits[flat_mask], flat_tgt[flat_mask], reduction="mean"
            )
        else:
            loss_a = F.cross_entropy(flat_logits, flat_tgt, reduction="mean")
        total = total + axis_weights[a] * loss_a
    return total


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    cfg = default_compound_config()
    cfg.block_size = 64
    cfg.n_layers = 2
    cfg.d_model = 128
    cfg.d_ff = 512
    model = CompoundGPT(cfg)
    print(f"Compound axes ({N_AXES}): {AXIS_NAMES}")
    print(f"Axis sizes: {AXIS_SIZES}")
    print(
        f"Parameters: {model.count_parameters():,} "
        f"(~{model.count_parameters()/1e6:.2f}M)"
    )

    B, T = 2, 32
    idx = torch.stack([
        torch.randint(0, AXIS_SIZES[a], (B, T)) for a in range(N_AXES)
    ], dim=-1).long()
    logits = model(idx)
    assert len(logits) == N_AXES
    for a, l in enumerate(logits):
        assert l.shape == (B, T, AXIS_SIZES[a]), (a, l.shape, AXIS_SIZES[a])

    loss = compound_loss(logits, idx)
    print(f"Forward + per-axis loss OK. loss={loss.item():.4f}")
