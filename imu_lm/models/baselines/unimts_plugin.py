"""UniMTS (Zhang et al., NeurIPS 2024) released encoder as an IMU-Bench plugin encoder.

The released checkpoint (HF xiyuanz/UniMTS, checkpoint/UniMTS.pth) holds the motion encoder `acc.*`, an ST-GCN
over the 22 SMPL joints, and a CLIP ViT-B/32 text tower; only the encoder is used (extracted once into
UniMTS_acc_encoder_only.pt). The model code is the authors' model.py (github.com/xiyuanzh/UniMTS), from the clone
under baselines/language_alignment/UniMTS.

The released encoder is ACCELEROMETER-ONLY (data_bn has 22 x 3 inputs), pretrained at 20 Hz on IMU simulated at
every joint of every HumanML3D motion. One wrist is fed as UniMTS does it: the right-wrist joint (SMPL index 21)
carries the signal and the other 21 joints are zero, as for the unselected joints in pretraining.

Tokens: the last ST-GCN feature map at the wrist joint (512-d, one step per 4 input frames = 0.2 s at 20 Hz),
placed on the harness grid by the plugin wrapper. Pooled: the model's own output (average over time and joints).

CONTAMINATION: UniMTS pretrains on all HumanML3D motions, including the official test split, and the benchmark's
retrieval task uses the same simulated signals. Its IMU-text retrieval cells are therefore marked contaminated on the
leaderboard; its other tasks are unaffected.
"""
from __future__ import annotations

import sys
from typing import Optional

import torch
import torch.nn as nn

from imu_lm.models.baselines._vendor import baselines_dir

REPO = baselines_dir("language_alignment", "UniMTS")
WRIST = 21


class UniMTSEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from model import ST_GCN_18                       # the authors' model.py
        self.acc = ST_GCN_18(in_channels=3)
        self.embed_dim = 512

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape                                  # [B, 3, T] accelerometer, m/s^2, 20 Hz
        g = x.new_zeros(B, 3, T, 22, 1)
        g[:, :, :, WRIST, 0] = x[:, :3]
        net = self.acc
        N, C_, T_, V, M = g.size()
        h = g.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C_, T_)
        h = net.data_bn(h).view(N, M, V, C_, T_).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_, T_, V)
        for gcn, importance in zip(net.st_gcn_networks, net.edge_importance):
            h, _ = gcn(h, net.A * importance)              # [B, 512, T/4, 22]
        pooled = h.mean(dim=(2, 3))                        # == F.avg_pool2d over (t, joint)
        tokens = h[:, :, :, WRIST].transpose(1, 2)         # [B, T/4, 512]
        return {"tokens": tokens, "pooled": pooled}


def build(weights: Optional[str] = None, **_) -> nn.Module:
    m = UniMTSEncoder()
    if weights:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        res = m.acc.load_state_dict(sd, strict=True)
    return m
