"""SSL baseline factory: SimMTM / LIMU-BERT / FOCAL for the unified launcher.

Each baseline's authentic model + objective is vendored self-contained from its
official repo into a dedicated module:

    limu_bert  -> imu_lm/models/baselines/limubert_ssl.py   (LIMU-BERT-Public)
    simmtm     -> imu_lm/models/baselines/simmtm_ssl.py      (SimMTM_Classification)
    focal      -> imu_lm/models/baselines/focal_ssl.py       (FOCAL repo)
    moment     -> imu_lm/models/baselines/moment_ssl.py      (MOMENT reconstruction)

Every wrapper exposes the uniform interface used by scripts/run_ssl_baseline.py:

    forward_loss(batch, device) -> (loss, logs_dict)   # batch is [B, T, C]
    encode(x)                   -> [B, D]               # frozen-encoder probing

All train on the exact same balanced OCDetect+realdata windows as the v0 MAE run.
"""
from __future__ import annotations

import torch.nn as nn


def build_ssl_model(name: str, cfg: dict) -> nn.Module:
    """Instantiate an SSL baseline wrapper by name from a flat model-config dict."""
    name = name.lower()
    if name == "limu_bert":
        from imu_lm.models.baselines.limubert_ssl import LIMUBertSSL
        return LIMUBertSSL(cfg)
    if name == "simmtm":
        from imu_lm.models.baselines.simmtm_ssl import SimMTMSSL
        return SimMTMSSL(cfg)
    if name == "focal":
        from imu_lm.models.baselines.focal_ssl import FOCALSSL
        return FOCALSSL(cfg)
    if name == "moment":
        from imu_lm.models.baselines.moment_ssl import MOMENTSSL
        return MOMENTSSL(cfg)
    raise ValueError(f"Unknown SSL baseline '{name}' (expected limu_bert / simmtm / focal / moment)")
