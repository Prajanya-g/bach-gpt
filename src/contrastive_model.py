"""Contrastive MIDI-text model architecture.

Components:
1) Frozen MIDI GPT encoder + masked mean pooling
2) Frozen sentence-transformers MiniLM text encoder
3) Two trainable projection heads to a shared embedding space
4) Learnable log-temperature scalar
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from model import GPT
# CompoundGPT lives in compound_model; only imported for the compound
# variant below to avoid a circular dependency when only GPT is used.


def symmetric_infonce_loss(
    midi_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute symmetric InfoNCE loss for paired MIDI/text batches.

    Args:
        midi_embeds: L2-normalized MIDI embeddings, shape (N, D).
        text_embeds: L2-normalized text embeddings, shape (N, D).
        temperature: Scalar temperature tensor.

    Returns:
        Dictionary containing:
            - logits_midi_to_text: (N, N)
            - logits_text_to_midi: (N, N)
            - loss_midi_to_text: scalar
            - loss_text_to_midi: scalar
            - loss: symmetric average of both directions
            - acc_midi_to_text: top-1 retrieval accuracy for rows
            - acc_text_to_midi: top-1 retrieval accuracy for cols
    """
    if midi_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError("midi_embeds and text_embeds must both be rank-2 tensors.")
    if midi_embeds.shape != text_embeds.shape:
        raise ValueError("midi_embeds and text_embeds must have identical shape.")

    logits = midi_embeds @ text_embeds.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_m2t = F.cross_entropy(logits, labels)
    loss_t2m = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_m2t + loss_t2m)

    with torch.no_grad():
        pred_m2t = torch.argmax(logits, dim=1)
        pred_t2m = torch.argmax(logits.t(), dim=1)
        acc_m2t = (pred_m2t == labels).float().mean()
        acc_t2m = (pred_t2m == labels).float().mean()

    return {
        "logits_midi_to_text": logits,
        "logits_text_to_midi": logits.t(),
        "loss_midi_to_text": loss_m2t,
        "loss_text_to_midi": loss_t2m,
        "loss": loss,
        "acc_midi_to_text": acc_m2t,
        "acc_text_to_midi": acc_t2m,
    }


class ProjectionHead(nn.Module):
    """Linear -> GELU -> LayerNorm -> Linear projection MLP."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 512, out_dim: int = 256
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MidiTextContrastiveModel(nn.Module):
    """MIDI-text contrastive architecture with frozen base encoders."""

    def __init__(
        self,
        midi_gpt: GPT,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_dim: int = 256,
        init_temperature: float = 0.07,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.midi_encoder = midi_gpt
        self.text_encoder = SentenceTransformer(text_model_name)
        self.embed_dim = embed_dim
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self._last_hidden: Optional[torch.Tensor] = None

        self.midi_projection = ProjectionHead(
            input_dim=midi_gpt.config.d_model,
            hidden_dim=512,
            out_dim=embed_dim,
        )
        self.text_projection = ProjectionHead(
            input_dim=384,
            hidden_dim=512,
            out_dim=embed_dim,
        )

        # Learnable log-temperature (CLIP-style init at log(0.07)).
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(init_temperature))
        )

        if device is not None:
            self.to(device)

        self.freeze_midi_encoder()
        self.freeze_text_encoder()

    def freeze_midi_encoder(self) -> None:
        for p in self.midi_encoder.parameters():
            p.requires_grad = False
        self.midi_encoder.eval()

    def freeze_text_encoder(self) -> None:
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

    def unfreeze_text_encoder(self) -> None:
        for p in self.text_encoder.parameters():
            p.requires_grad = True
        self.text_encoder.train()

    def _capture_last_hidden_hook(self, _module, _inputs, output) -> None:
        x = output[0] if isinstance(output, tuple) else output
        self._last_hidden = x

    def _extract_midi_last_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Preferred route: try native hidden-state support if available.
        try:
            out = self.midi_encoder(input_ids, output_hidden_states=True)
            if isinstance(out, dict) and "hidden_states" in out:
                hs = out["hidden_states"]
                if isinstance(hs, (list, tuple)) and hs:
                    return hs[-1]
            if hasattr(out, "hidden_states") and out.hidden_states:
                return out.hidden_states[-1]
        except TypeError:
            # Expected for this repo's custom GPT; use hook fallback below.
            pass

        # Hook fallback uses instance state and is not concurrency-safe across
        # overlapping forward passes on the same model instance.
        self._last_hidden = None
        hook = self.midi_encoder.blocks[-1].register_forward_hook(
            self._capture_last_hidden_hook
        )
        try:
            _ = self.midi_encoder(input_ids)
        finally:
            hook.remove()

        if self._last_hidden is None:
            raise RuntimeError("Failed to capture MIDI last hidden states.")
        return self.midi_encoder.ln_f(self._last_hidden)

    @staticmethod
    def _masked_mean_pool(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states: (B, T, D), attention_mask: (B, T)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def encode_midi(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            hidden = self._extract_midi_last_hidden(input_ids)
            pooled = self._masked_mean_pool(hidden, attention_mask)
        return pooled

    def encode_text(
        self, captions: List[str], device: torch.device
    ) -> torch.Tensor:
        text_trainable = any(p.requires_grad for p in self.text_encoder.parameters())
        if text_trainable:
            features = self.text_encoder.tokenize(captions)
            features = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in features.items()
            }
            out = self.text_encoder(features)
            emb = out.get("sentence_embedding")
            if emb is None:
                emb = out.get("sentence_embeddings")
            if emb is None:
                emb = next(iter(out.values()))
            return emb

        with torch.no_grad():
            return self.text_encoder.encode(
                captions,
                convert_to_tensor=True,
                device=str(device),
                normalize_embeddings=False,
            )

    def get_temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(
            self.min_temperature, self.max_temperature
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        captions: List[str],
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device

        midi_features = self.encode_midi(input_ids, attention_mask)
        text_features = self.encode_text(captions, device=device)

        midi_proj = self.midi_projection(midi_features)
        text_proj = self.text_projection(text_features)

        midi_embeds = F.normalize(midi_proj, p=2, dim=-1)
        text_embeds = F.normalize(text_proj, p=2, dim=-1)

        temperature = self.get_temperature()
        loss_out = symmetric_infonce_loss(
            midi_embeds=midi_embeds,
            text_embeds=text_embeds,
            temperature=temperature,
        )

        return {
            "midi_embeds": midi_embeds,
            "text_embeds": text_embeds,
            "temperature": temperature,
            **loss_out,
        }

    def trainable_parameters(self):
        # Stage A: only projection heads + temperature are trainable.
        return list(self.midi_projection.parameters()) + list(
            self.text_projection.parameters()
        ) + [self.log_temperature]

    def make_optimizer_param_groups(
        self,
        proj_lr: float,
        weight_decay: float = 0.0,
        temperature_lr_scale: float = 0.1,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "params": self.midi_projection.parameters(),
                "lr": proj_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": self.text_projection.parameters(),
                "lr": proj_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [self.log_temperature],
                "lr": proj_lr * temperature_lr_scale,
                "weight_decay": 0.0,
            },
        ]


# --- Compound (Octuple) variant ----------------------------------------------

class CompoundMidiTextContrastiveModel(nn.Module):
    """CLIP-style contrastive model with a CompoundGPT MIDI encoder.

    This is the "Option C" hybrid: the contrastive encoder uses compound
    (per-axis) MIDI inputs to learn structured musical representations,
    while the autoregressive decoder side of the project remains the 1D +
    BPE GPT (loaded separately at prefix-tuning time).

    Inputs to ``forward`` differ from ``MidiTextContrastiveModel``:
    - ``compound_input`` of shape (B, T, N_AXES) instead of (input_ids, attention_mask)
    - The pad mask is derived from the step-type axis (== STEP_PAD).
    """

    def __init__(
        self,
        midi_compound_gpt: "CompoundGPT",
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_dim: int = 256,
        init_temperature: float = 0.07,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        from compound import STEP_PAD          # local import to avoid cycles
        self._step_pad = STEP_PAD

        self.midi_encoder = midi_compound_gpt
        self.text_encoder = SentenceTransformer(text_model_name)
        self.embed_dim = embed_dim
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        self.midi_projection = ProjectionHead(
            input_dim=midi_compound_gpt.config.d_model,
            hidden_dim=512,
            out_dim=embed_dim,
        )
        self.text_projection = ProjectionHead(
            input_dim=384,
            hidden_dim=512,
            out_dim=embed_dim,
        )

        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(init_temperature))
        )

        if device is not None:
            self.to(device)

        self.freeze_midi_encoder()
        self.freeze_text_encoder()

    def freeze_midi_encoder(self) -> None:
        for p in self.midi_encoder.parameters():
            p.requires_grad = False
        self.midi_encoder.eval()

    def freeze_text_encoder(self) -> None:
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

    def unfreeze_text_encoder(self) -> None:
        for p in self.text_encoder.parameters():
            p.requires_grad = True
        self.text_encoder.train()

    def encode_midi(self, compound_input: torch.Tensor) -> torch.Tensor:
        """compound_input: (B, T, N_AXES) long. Returns pooled (B, d_model).
        Pad steps (step-axis == STEP_PAD) are excluded from the mean pool."""
        with torch.no_grad():
            hidden = self.midi_encoder(compound_input, return_hidden=True)
            mask = (compound_input[..., 0] != self._step_pad).to(hidden.dtype)
            mask = mask.unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom
        return pooled

    def encode_text(
        self, captions: List[str], device: torch.device
    ) -> torch.Tensor:
        text_trainable = any(p.requires_grad for p in self.text_encoder.parameters())
        if text_trainable:
            features = self.text_encoder.tokenize(captions)
            features = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in features.items()
            }
            out = self.text_encoder(features)
            emb = out.get("sentence_embedding")
            if emb is None:
                emb = out.get("sentence_embeddings")
            if emb is None:
                emb = next(iter(out.values()))
            return emb

        with torch.no_grad():
            return self.text_encoder.encode(
                captions,
                convert_to_tensor=True,
                device=str(device),
                normalize_embeddings=False,
            )

    def get_temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(
            self.min_temperature, self.max_temperature
        )

    def forward(
        self,
        compound_input: torch.Tensor,
        captions: List[str],
    ) -> Dict[str, torch.Tensor]:
        device = compound_input.device

        midi_features = self.encode_midi(compound_input)
        text_features = self.encode_text(captions, device=device)

        midi_proj = self.midi_projection(midi_features)
        text_proj = self.text_projection(text_features)

        midi_embeds = F.normalize(midi_proj, p=2, dim=-1)
        text_embeds = F.normalize(text_proj, p=2, dim=-1)

        temperature = self.get_temperature()
        loss_out = symmetric_infonce_loss(
            midi_embeds=midi_embeds,
            text_embeds=text_embeds,
            temperature=temperature,
        )

        return {
            "midi_embeds": midi_embeds,
            "text_embeds": text_embeds,
            "temperature": temperature,
            **loss_out,
        }

    def trainable_parameters(self):
        return (
            list(self.midi_projection.parameters())
            + list(self.text_projection.parameters())
            + [self.log_temperature]
        )

    def make_optimizer_param_groups(
        self,
        proj_lr: float,
        weight_decay: float = 0.0,
        temperature_lr_scale: float = 0.1,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "params": self.midi_projection.parameters(),
                "lr": proj_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": self.text_projection.parameters(),
                "lr": proj_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [self.log_temperature],
                "lr": proj_lr * temperature_lr_scale,
                "weight_decay": 0.0,
            },
        ]
