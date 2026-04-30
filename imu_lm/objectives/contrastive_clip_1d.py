"""CLIP-style contrastive objective: IMU ↔ text (UniMTS-inspired).

Architecture
------------
    IMU branch:   ViT1DEncoder (mean-pooled, [B, D_enc])
                      └→ ContrastiveCLIPHead.projection  → L2-norm → [B, D_clip]

    Text branch:  CLIPTextEncoder (frozen by default, [B, D_clip])
                      └→ L2-norm

Loss
----
    Symmetric InfoNCE (CLIP paper, Eq. 1):

        L = ½ · (CE(S, labels) + CE(Sᵀ, labels))

    where S = exp(logit_scale) · z_imu @ z_text.T   [B, B]
    and labels = [0, 1, 2, ..., B-1].

    logit_scale is a learnable scalar in ContrastiveCLIPHead, initialised
    from log(1/0.07) = 2.6593 (same init as CLIP).

Modules
-------
    ContrastiveCLIPHead   — projection head + logit_scale (trained)
    CLIPTextEncoder       — HuggingFace CLIP text encoder wrapper (frozen by default)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# CLIP default: temperature τ = 0.07  →  logit_scale = log(1/τ) ≈ 2.6593
_CLIP_INIT_LOGIT_SCALE = math.log(1.0 / 0.07)
# Clamp to avoid numerical explosions (CLIP paper uses max 100)
_MAX_LOGIT_SCALE = math.log(100.0)


# --------------------------------------------------------------------------- #
# IMU → CLIP-space projection head                                            #
# --------------------------------------------------------------------------- #

class ContrastiveCLIPHead(nn.Module):
    """Linear projection: [B, D_enc] → [B, D_clip] + learnable temperature.

    Architecture (no bias, then L2-norm in forward — matches CLIP):
        Linear(D_enc, D_clip, bias=False)

    The L2 normalisation is applied in ``forward`` so the output lives on the
    unit hypersphere, consistent with the CLIP text embedding.

    Args:
        imu_embed_dim: Encoder output size (D_enc).
        clip_embed_dim: CLIP embedding dimension (D_clip, default 512).
        init_logit_scale: Initial value of the log-temperature scalar.
    """

    def __init__(
        self,
        imu_embed_dim: int,
        clip_embed_dim: int = 512,
        init_logit_scale: float = _CLIP_INIT_LOGIT_SCALE,
    ):
        super().__init__()
        self.projection = nn.Linear(imu_embed_dim, clip_embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

        # Weight initialisation: small normal (similar to CLIP's text projection)
        nn.init.normal_(self.projection.weight, std=imu_embed_dim ** -0.5)

    def forward(self, imu_emb: torch.Tensor) -> torch.Tensor:
        """[B, D_enc] → [B, D_clip] (L2-normalised)."""
        z = self.projection(imu_emb)   # [B, D_clip]
        return F.normalize(z, dim=-1)


# --------------------------------------------------------------------------- #
# Frozen CLIP text encoder wrapper                                             #
# --------------------------------------------------------------------------- #

class CLIPTextEncoder(nn.Module):
    """HuggingFace CLIP text encoder wrapper.

    Tokenises a list of strings on-the-fly and returns L2-normalised text
    embeddings from the [EOS] token (matching the original CLIP paper).

    Args:
        model_name: HuggingFace model identifier (default: openai/clip-vit-base-patch32).
        freeze: If True (default), all CLIP parameters are frozen — only the
                IMU projection head (ContrastiveCLIPHead) is trained.
        max_length: Maximum tokenisation length (default 77, same as CLIP).
        local_files_only: If True, do not fetch from the hub (useful on airgapped clusters).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
        max_length: int = 77,
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import CLIPTokenizer, CLIPTextModel

        logger.info("[CLIPTextEncoder] loading %s (freeze=%s)", model_name, freeze)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = CLIPTextModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.max_length = max_length

        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()
        self._frozen = freeze

    def train(self, mode: bool = True):
        # Keep the text model in eval when frozen to avoid BN / dropout issues
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    @torch.no_grad()
    def _encode_frozen(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Fast path when the model is fully frozen."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # CLIP uses the [EOS] token (pooler_output) as the text representation
        z = outputs.pooler_output        # [B, D_clip]
        return F.normalize(z, dim=-1)

    def _encode_trainable(self, texts: List[str], device: torch.device) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        z = outputs.pooler_output        # [B, D_clip]
        return F.normalize(z, dim=-1)

    @property
    def device(self) -> torch.device:
        """Device where the text model parameters live."""
        return next(self.model.parameters()).device

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Return L2-normalised text embeddings [B, D_clip] on *device*.

        The forward pass always runs on the text model's own device; the
        resulting embedding is then moved to *device* (e.g. the GPU where the
        IMU encoder lives) before returning.
        """
        model_device = self.device
        if self._frozen:
            z = self._encode_frozen(texts, model_device)
        else:
            z = self._encode_trainable(texts, model_device)
        return z.to(device)


# --------------------------------------------------------------------------- #
# Trainable text-side MLP projection                                          #
# --------------------------------------------------------------------------- #

class TextProjectionMLP(nn.Module):
    """2-layer MLP applied on top of (frozen) CLIP text embeddings.

    Keeps the CLIP text encoder frozen but adds a small trainable network on
    top so the text branch can adapt to the IMU domain without paying the cost
    of fine-tuning the full 63 M-parameter text model.

    Architecture:  LayerNorm → Linear(D, hidden) → GELU → Linear(hidden, D)
    The output is L2-normalised so it stays on the unit hypersphere, consistent
    with the contrastive loss geometry.

    Args:
        embed_dim:  CLIP embedding dimension (default 512).
        hidden_dim: Width of the hidden layer (defaults to embed_dim).
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Output projection: start near-identity so training is stable at init
        nn.init.zeros_(self.net[-1].bias)
        if embed_dim == hidden_dim:
            nn.init.eye_(self.net[-1].weight)
        else:
            nn.init.normal_(self.net[-1].weight, std=embed_dim ** -0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[B, D] → [B, D]  (L2-normalised)."""
        return F.normalize(self.net(z), dim=-1)


# --------------------------------------------------------------------------- #
# Contrastive loss                                                             #
# --------------------------------------------------------------------------- #

def info_nce_loss(
    z_imu: torch.Tensor,
    z_text: torch.Tensor,
    logit_scale: nn.Parameter,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Symmetric InfoNCE (CLIP-style).

    Args:
        z_imu:        [B, D]  L2-normalised IMU embeddings
        z_text:       [B, D]  L2-normalised text embeddings
        logit_scale:  scalar Parameter (log of the temperature inverse)

    Returns:
        loss:  scalar
        logs:  dict with loss, logit_scale, and retrieval accuracy
    """
    B    = z_imu.shape[0]
    scale = logit_scale.exp().clamp(max=math.exp(_MAX_LOGIT_SCALE))
    labels = torch.arange(B, device=z_imu.device)

    # logit matrices [B, B]
    logits_per_imu  = scale * z_imu @ z_text.T
    logits_per_text = logits_per_imu.T

    loss_i2t = F.cross_entropy(logits_per_imu,  labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    loss     = (loss_i2t + loss_t2i) / 2.0

    # Top-1 retrieval accuracy (diagonal = correct pair)
    with torch.no_grad():
        acc_i2t = (logits_per_imu.argmax(dim=1)  == labels).float().mean().item()
        acc_t2i = (logits_per_text.argmax(dim=1) == labels).float().mean().item()

    return loss, {
        "loss":            float(loss.item()),
        "loss_i2t":        float(loss_i2t.item()),
        "loss_t2i":        float(loss_t2i.item()),
        "logit_scale":     float(scale.item()),
        "acc_imu2text":    acc_i2t,
        "acc_text2imu":    acc_t2i,
    }


# --------------------------------------------------------------------------- #
# Objective entry point                                                        #
# --------------------------------------------------------------------------- #

def forward_loss(
    batch: Tuple[torch.Tensor, Union[List[str], Tuple[str, ...], torch.Tensor]],
    encoder: nn.Module,
    clip_head: ContrastiveCLIPHead,
    text_encoder: Optional[CLIPTextEncoder],
    cfg: Any,
    text_mlp: Optional[TextProjectionMLP] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CLIP-style contrastive forward pass.

    Accepts two forms of ``batch[1]``:

    * **List / tuple of strings** — captions are tokenised and encoded on-the-fly
      by ``text_encoder`` (original behaviour; ``text_encoder`` must not be None).
    * **FloatTensor [B, D_clip]** — pre-computed, L2-normalised text embeddings
      produced by ``scripts/cache_clip_text_embeddings.py`` and looked up inside
      ``HumanML3DIMUVarLenDataset``.  In this mode ``text_encoder`` is not called
      and can be ``None``, which means the CLIP model does not need to be loaded
      into memory at training time.

    Args:
        batch:        (x [B, 6, T],  texts_or_embs)
        encoder:      ViT1DEncoder   → [B, D_enc]  (mean-pooled)
        clip_head:    ContrastiveCLIPHead  → [B, D_clip]
        text_encoder: CLIPTextEncoder or None (None is allowed when batch contains
                      pre-cached embedding tensors)
        cfg:          config dict (unused currently; reserved for future options)
        text_mlp:     Optional TextProjectionMLP applied on top of the (frozen)
                      CLIP text embeddings.  Gives the text branch trainable
                      capacity without unfreezing the large CLIP model.

    Returns:
        (loss, logs)
    """
    x, texts_or_embs = batch          # x: [B, 6, T]

    # IMU branch: encode → project → L2-norm
    z_enc  = encoder(x)          # [B, D_enc]  (ViT1D mean pool)
    z_imu  = clip_head(z_enc)    # [B, D_clip] L2-normed

    # Text branch: either cached tensor or on-the-fly CLIP encoding
    if isinstance(texts_or_embs, torch.Tensor):
        # Fast path: pre-cached, L2-normalised embeddings [B, D_clip]
        z_text = texts_or_embs.to(x.device)
    else:
        if text_encoder is None:
            raise RuntimeError(
                "text_encoder is None but the batch contains caption strings.  "
                "Either pass a CLIPTextEncoder or use a pre-cached embedding file "
                "(scripts/cache_clip_text_embeddings.py + text_cache_path in config)."
            )
        texts  = list(texts_or_embs)
        z_text = text_encoder(texts, device=x.device)  # [B, D_clip] L2-normed

    # Optional trainable MLP on the text side
    if text_mlp is not None:
        z_text = text_mlp(z_text)   # [B, D_clip] re-normalised inside MLP

    return info_nce_loss(z_imu, z_text, clip_head.logit_scale)
