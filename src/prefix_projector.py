"""Phase 3 prefix-conditioning scaffolding.

This module enforces the Phase 3 discipline:
- Frozen: CLAP model and MIDI GPT
- Trainable: prefix projector only
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from contrastive_model import (
    CompoundMidiTextContrastiveModel,
    MidiTextContrastiveModel,
)
from compound import SENTINELS, STEP_PAD
from compound_model import (
    CompoundGPT,
    CompoundGPTConfig,
    compound_loss,
    default_compound_config,
)
from model import GPT, GPTConfig, default_gpt_config


def clap_text_for_prefix_projector(
    clap_model: MidiTextContrastiveModel,
    captions: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Text features in the same 256-d space as contrastive training (not raw ST dim).

    ``encode_text`` returns sentence-transformer size (e.g. 384 for MiniLM);
    ``text_projection`` maps to ``clap_model.embed_dim`` (256), which
    ``PrefixProjector`` was built for.
    """
    raw = clap_model.encode_text(captions, device=device)
    proj = clap_model.text_projection(raw)
    return F.normalize(proj, p=2, dim=-1)


class PrefixProjector(nn.Module):
    """Map a 256-d CLAP projected text embedding to K prefix vectors in GPT d_model space."""

    def __init__(
        self,
        clap_embed_dim: int,
        gpt_d_model: int,
        n_prefix_tokens: int,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.clap_embed_dim = clap_embed_dim
        self.gpt_d_model = gpt_d_model
        self.n_prefix_tokens = n_prefix_tokens
        hidden_dim = gpt_d_model * 2
        self.fc1 = nn.Linear(clap_embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_prefix_tokens * gpt_d_model)
        self.out_ln = nn.LayerNorm(gpt_d_model)

        # Critical for stable prefix tuning: start near-zero so the frozen GPT
        # first behaves like unconditional decoding, then learns prefix usage.
        with torch.no_grad():
            self.fc2.weight.mul_(0.01)
            if self.fc2.bias is not None:
                self.fc2.bias.mul_(0.01)

    def forward(self, clap_embedding: torch.Tensor) -> torch.Tensor:
        # clap_embedding: (B, 256) -> prefix: (B, K, d_model)
        out = self.fc1(clap_embedding)
        out = self.act(out)
        out = self.drop(out)
        out = self.ln(out)
        out = self.fc2(out)
        out = out.view(
            clap_embedding.size(0), self.n_prefix_tokens, self.gpt_d_model
        )
        return self.out_ln(out)


@dataclass
class Phase3FrozenTrainableCounts:
    n_clap_params: int
    n_gpt_params: int
    n_projector_params: int
    n_total_trainable: int


def _extract_gpt_config_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(GPTConfig.__dataclass_fields__.keys())
    return {k: raw[k] for k in keys if k in raw}


def _extract_compound_gpt_config_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(CompoundGPTConfig.__dataclass_fields__.keys())
    return {k: raw[k] for k in keys if k in raw}


def _load_gpt_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = default_gpt_config()
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(ckpt_cfg, dict):
        for k, v in _extract_gpt_config_dict(ckpt_cfg).items():
            setattr(cfg, k, v)
    gpt = GPT(cfg).to(device)
    state = (
        ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    )
    gpt.load_state_dict(state)
    gpt.eval()
    return gpt


def _load_compound_gpt_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> CompoundGPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = default_compound_config()
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(ckpt_cfg, dict):
        for k, v in _extract_compound_gpt_config_dict(ckpt_cfg).items():
            setattr(cfg, k, v)
    compound_gpt = CompoundGPT(cfg).to(device)
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
    if state is None:
        state = ckpt
    compound_gpt.load_state_dict(state, strict=False)
    compound_gpt.eval()
    return compound_gpt


def _load_clap_model(
    clap_checkpoint_path: Path,
    gpt: GPT,
    device: torch.device,
) -> MidiTextContrastiveModel:
    ckpt = torch.load(
        clap_checkpoint_path, map_location=device, weights_only=True
    )
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    clap = MidiTextContrastiveModel(
        midi_gpt=gpt,
        text_model_name=args.get(
            "text_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embed_dim=args.get("embed_dim", 256),
        init_temperature=args.get("init_temperature", 0.07),
        min_temperature=args.get("min_temperature", 0.01),
        max_temperature=args.get("max_temperature", 1.0),
        device=device,
    )
    state = None
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("model")
    if state is None:
        state = ckpt
    clap.load_state_dict(state, strict=False)
    clap.eval()
    return clap


def _load_compound_clap_model(
    clap_checkpoint_path: Path,
    compound_gpt: CompoundGPT,
    device: torch.device,
) -> CompoundMidiTextContrastiveModel:
    ckpt = torch.load(
        clap_checkpoint_path, map_location=device, weights_only=True
    )
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    clap = CompoundMidiTextContrastiveModel(
        midi_compound_gpt=compound_gpt,
        text_model_name=args.get(
            "text_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embed_dim=args.get("embed_dim", 256),
        init_temperature=args.get("init_temperature", 0.07),
        min_temperature=args.get("min_temperature", 0.01),
        max_temperature=args.get("max_temperature", 1.0),
        device=device,
    )
    state = None
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("model")
    if state is None:
        state = ckpt
    clap.load_state_dict(state, strict=False)
    clap.eval()
    return clap


def freeze_all_except_prefix_projector(
    clap_model: MidiTextContrastiveModel,
    midi_gpt: GPT,
    prefix_projector: PrefixProjector,
) -> Phase3FrozenTrainableCounts:
    for p in clap_model.parameters():
        p.requires_grad = False
    for p in midi_gpt.parameters():
        p.requires_grad = False
    for p in prefix_projector.parameters():
        p.requires_grad = True

    clap_model.eval()
    midi_gpt.eval()
    prefix_projector.train()

    n_clap = sum(p.numel() for p in clap_model.parameters())
    n_gpt = sum(p.numel() for p in midi_gpt.parameters())
    n_proj = sum(p.numel() for p in prefix_projector.parameters())
    n_trainable = sum(
        p.numel()
        for p in list(clap_model.parameters())
        + list(midi_gpt.parameters())
        + list(prefix_projector.parameters())
        if p.requires_grad
    )

    # Hard guardrail: only projector is trainable in Phase 3.
    if n_trainable != n_proj:
        raise RuntimeError(
            "Phase 3 freeze check failed: non-projector parameters are "
            "trainable."
        )

    return Phase3FrozenTrainableCounts(
        n_clap_params=n_clap,
        n_gpt_params=n_gpt,
        n_projector_params=n_proj,
        n_total_trainable=n_trainable,
    )


def load_phase3_components(
    midi_checkpoint: str,
    clap_checkpoint: str,
    n_prefix_tokens: int,
    device: torch.device,
) -> Tuple[MidiTextContrastiveModel, GPT, PrefixProjector, Phase3FrozenTrainableCounts]:
    """Load CLAP+GPT, create projector, and enforce Phase 3 freeze policy."""
    midi_gpt = _load_gpt_from_checkpoint(Path(midi_checkpoint), device=device)
    clap_model = _load_clap_model(
        clap_checkpoint_path=Path(clap_checkpoint),
        gpt=midi_gpt,
        device=device,
    )
    projector = PrefixProjector(
        clap_embed_dim=clap_model.embed_dim,
        gpt_d_model=midi_gpt.config.d_model,
        n_prefix_tokens=n_prefix_tokens,
    ).to(device)
    counts = freeze_all_except_prefix_projector(
        clap_model=clap_model,
        midi_gpt=midi_gpt,
        prefix_projector=projector,
    )
    return clap_model, midi_gpt, projector, counts


def load_phase3_compound_components(
    compound_midi_checkpoint: str,
    compound_clap_checkpoint: str,
    n_prefix_tokens: int,
    device: torch.device,
) -> Tuple[
    CompoundMidiTextContrastiveModel,
    CompoundGPT,
    PrefixProjector,
    Phase3FrozenTrainableCounts,
]:
    """Load compound CLAP+CompoundGPT, create projector, and enforce freeze policy."""
    compound_gpt = _load_compound_gpt_from_checkpoint(
        Path(compound_midi_checkpoint), device=device
    )
    compound_clap = _load_compound_clap_model(
        clap_checkpoint_path=Path(compound_clap_checkpoint),
        compound_gpt=compound_gpt,
        device=device,
    )
    projector = PrefixProjector(
        clap_embed_dim=compound_clap.embed_dim,
        gpt_d_model=compound_gpt.config.d_model,
        n_prefix_tokens=n_prefix_tokens,
    ).to(device)
    counts = freeze_all_except_prefix_projector(
        clap_model=compound_clap,  # type: ignore[arg-type]
        midi_gpt=compound_gpt,  # type: ignore[arg-type]
        prefix_projector=projector,
    )
    return compound_clap, compound_gpt, projector, counts


def forward_prefix_conditioned_logits(
    clap_model: MidiTextContrastiveModel,
    midi_gpt: GPT,
    prefix_projector: PrefixProjector,
    input_ids: torch.Tensor,
    captions: list[str],
) -> torch.Tensor:
    """Run Phase 3 prefix injection and return MIDI-position logits.

    Steps:
    1) Encode text with frozen CLAP text tower (no grad)
    2) Project to soft prefix embeddings (grad flows only here)
    3) Lookup token embeddings with frozen GPT embedding table (no grad)
    4) Concatenate prefix + token embeddings
    5) Run GPT via inputs_embeds (bypassing token-id embedding path)
    6) Slice off prefix positions; return logits for original MIDI tokens
    """
    device = input_ids.device

    with torch.no_grad():
        text_emb = clap_text_for_prefix_projector(
            clap_model, captions, device
        )
        token_embeds = midi_gpt.wte(input_ids)

    prefix_embeds = prefix_projector(text_emb)
    k = prefix_embeds.size(1)

    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
    seq_len = inputs_embeds.size(1)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)

    logits = midi_gpt(inputs_embeds=inputs_embeds, position_ids=position_ids)
    return logits[:, k:, :]


def build_phase3_lm_labels(
    input_ids: torch.Tensor,
    n_prefix_tokens: int,
) -> torch.Tensor:
    """Build full-sequence labels with ignored prefix and shifted MIDI targets.

    Returns labels of shape (B, K+T), initialized to -100.
    We supervise only standard next-token MIDI positions:
      labels[:, K+1:K+T] = input_ids[:, 1:]
    """
    bsz, seq_len = input_ids.shape
    labels = torch.full(
        (bsz, n_prefix_tokens + seq_len),
        fill_value=-100,
        dtype=torch.long,
        device=input_ids.device,
    )
    if seq_len > 1:
        labels[:, n_prefix_tokens + 1 :] = input_ids[:, 1:]
    return labels


def phase3_prefix_lm_loss(
    clap_model: MidiTextContrastiveModel,
    midi_gpt: GPT,
    prefix_projector: PrefixProjector,
    input_ids: torch.Tensor,
    captions: list[str],
    prefix_attn_reg_weight: float = 0.0,
    prefix_attn_min_mean: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Phase 3 LM loss on MIDI token positions only.

    - Prefix outputs are ignored with label -100.
    - Loss is next-token cross-entropy with standard shift.
    """
    device = input_ids.device

    with torch.no_grad():
        text_emb = clap_text_for_prefix_projector(
            clap_model, captions, device
        )
        token_embeds = midi_gpt.wte(input_ids)

    prefix_embeds = prefix_projector(text_emb)
    k = prefix_embeds.size(1)
    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

    full_len = inputs_embeds.size(1)
    attention_mask = torch.ones(
        input_ids.size(0),
        full_len,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(full_len, device=device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)

    # attention_mask is intentionally all ones so MIDI tokens can attend to prefix.
    _ = attention_mask
    use_attn_reg = prefix_attn_reg_weight > 0.0
    if use_attn_reg:
        logits_full, attn_list = midi_gpt(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            return_attn=True,
        )
    else:
        logits_full = midi_gpt(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        attn_list = []
    labels_full = build_phase3_lm_labels(input_ids=input_ids, n_prefix_tokens=k)

    loss_main = F.cross_entropy(
        logits_full[:, :-1, :].reshape(-1, logits_full.size(-1)),
        labels_full[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    if not use_attn_reg:
        return loss_main, logits_full

    # Regularize attention to ensure MIDI query positions attend to prefix keys.
    # attn: (B, H, S, S). Queries from K..S-1, prefix keys are 0..K-1.
    prefix_attn_means = []
    for attn in attn_list:
        if attn.size(-1) <= k:
            continue
        prefix_mass = attn[:, :, k:, :k].mean()
        prefix_attn_means.append(prefix_mass)
    if prefix_attn_means:
        mean_prefix_attn = torch.stack(prefix_attn_means).mean()
        attn_penalty = torch.relu(
            torch.tensor(prefix_attn_min_mean, device=mean_prefix_attn.device)
            - mean_prefix_attn
        )
        loss = loss_main + prefix_attn_reg_weight * attn_penalty
    else:
        loss = loss_main
    return loss, logits_full


def phase3_compound_prefix_lm_loss(
    clap_model: CompoundMidiTextContrastiveModel,
    midi_compound_gpt: CompoundGPT,
    prefix_projector: PrefixProjector,
    compound_input: torch.Tensor,
    captions: list[str],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Phase 3 loss for compound path using inputs_embeds + compound_loss.

    - Prefix positions are ignored by setting their target step-axis to STEP_PAD.
    - Uses next-step supervision (shifted by one), mirroring phase3_prefix_lm_loss.
    """
    device = compound_input.device

    with torch.no_grad():
        text_emb = clap_text_for_prefix_projector(
            clap_model, captions, device  # type: ignore[arg-type]
        )
        token_embeds = midi_compound_gpt.input_embeds[0](compound_input[..., 0])
        for a in range(1, midi_compound_gpt.n_axes):
            token_embeds = token_embeds + midi_compound_gpt.input_embeds[a](
                compound_input[..., a]
            )

    prefix_embeds = prefix_projector(text_emb)
    k = prefix_embeds.size(1)
    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

    full_len = inputs_embeds.size(1)
    position_ids = torch.arange(full_len, device=device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(compound_input.size(0), -1)

    logits_full = midi_compound_gpt(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
    )

    bsz, seq_len, n_axes = compound_input.shape
    if n_axes != midi_compound_gpt.n_axes:
        raise ValueError(
            f"Expected n_axes={midi_compound_gpt.n_axes}; got {n_axes}"
        )
    pad_step = torch.tensor(
        [STEP_PAD] + SENTINELS[1:],
        device=device,
        dtype=torch.long,
    ).view(1, 1, n_axes)
    labels_full = pad_step.expand(bsz, k + seq_len, n_axes).clone()
    if seq_len > 1:
        labels_full[:, k + 1 :, :] = compound_input[:, 1:, :]

    logits_shift = [logits[:, :-1, :] for logits in logits_full]
    labels_shift = labels_full[:, 1:, :]
    loss = compound_loss(
        logits_per_axis=logits_shift,
        targets=labels_shift,
        pad_step_value=STEP_PAD,
        ignore_pad_steps=True,
    )
    return loss, logits_full
