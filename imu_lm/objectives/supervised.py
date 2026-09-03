"""Supervised classification objective for fully-supervised training.

Architecture-agnostic: works with any encoder that outputs [B, embed_dim].
All current encoders (CNN1D, ViT1D, TSTransformer1D, ViT2D) handle pooling
internally, so this objective just needs encoder output → linear head → loss.

The classification head (LinearHead) is created by each run.py and passed in.
Reuses compute_metrics from utils/metrics.py for comprehensive evaluation.

Contract:
    forward_loss(batch, encoder, head, cfg) -> (loss, logs)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.utils.metrics import compute_metrics
from imu_lm.utils.training import remap_labels


def forward_loss(
    batch: Tuple[torch.Tensor, torch.Tensor],
    encoder: nn.Module,
    head: nn.Module,
    cfg: Any,
    raw_to_idx: Optional[Dict[int, int]] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute supervised classification loss with comprehensive metrics.
    
    Args:
        batch: (x, y) where x is input tensor, y is raw class labels [B]
        encoder: Any encoder that maps input → [B, embed_dim]
        head: Classification head that maps [B, embed_dim] → [B, num_classes]
        cfg: Config dict with objective.* params
        raw_to_idx: Mapping from raw dataset labels to contiguous [0, N) indices.
            If None, assumes labels are already contiguous.
        
    Returns:
        loss: Scalar cross-entropy loss
        logs: Dict with loss and metrics
    """
    x, y = batch
    
    # Remap raw labels to contiguous indices
    if raw_to_idx is not None:
        y = remap_labels(y, raw_to_idx)
    
    # Drop samples with unmapped labels (-100)
    valid = y >= 0
    if not valid.any():
        loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        return loss, {"loss": 0.0, "acc": 0.0, "bal_acc": 0.0, "macro_f1": 0.0, "macro_prec": 0.0, "macro_rec": 0.0}
    x, y = x[valid], y[valid]
    
    # Encode input → embedding
    z = encoder(x)  # [B, embed_dim]
    
    # Classify
    logits = head(z)  # [B, num_classes]
    
    # Cross-entropy loss (with optional class weights for imbalanced datasets)
    w = class_weights.to(x.device) if class_weights is not None else None
    loss = F.cross_entropy(logits, y, weight=w)
    
    # Predictions
    preds = logits.argmax(dim=1)
    
    # Use centralized metrics computation (same as probe)
    y_true = y.detach().cpu().tolist()
    y_pred = preds.detach().cpu().tolist()
    metrics = compute_metrics(y_true, y_pred)
    
    # Return flat metrics for trainer logging (exclude per_class and confusion for brevity)
    return loss, {
        "loss": float(loss.detach().item()),
        "acc": float((y.detach().cpu() == preds.detach().cpu()).float().mean()),
        "bal_acc": metrics["bal_acc"],
        "macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        # Consumed by Trainer._eval to compute pooled (not per-batch-averaged)
        # macro_f1/precision/recall/bal_acc over the whole validation pass --
        # see that method's docstring for why per-batch averaging is wrong
        # for these metrics. Harmless/ignored during training (only read in
        # eval mode); underscore prefix keeps it out of the scalar log path.
        "_y_true": y_true,
        "_y_pred": y_pred,
    }
