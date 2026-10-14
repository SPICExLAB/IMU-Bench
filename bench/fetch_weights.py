#!/usr/bin/env python3
"""Download the released weights (and pinned upstream code) of the open-track models.

    imu-bench fetch-weights                      # every model below
    imu-bench fetch-weights harnet10 imagebind_imu

Each file lands where its encoder card expects it: the card's `weights` path, rewritten by IMUBENCH_DATA_ROOT
(imu_lm/utils/data_root.py), and upstream code under the baselines directory (IMUBENCH_BASELINES_DIR, else
<repo>/baselines). Large multi-modal checkpoints are reduced to the encoder's keys once, so a cell job never loads
more than it needs. Weights are read with torch.load(weights_only=True); nothing in a download is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from typing import Callable, Dict, List, Optional

from bench.registry_io import load_encoders
from imu_lm.models.baselines._vendor import baselines_dir
from imu_lm.utils.data_root import data_path


def _download(url: str, dst: str, size: Optional[int] = None) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and (size is None or os.path.getsize(dst) == size):
        print(f"[fetch] have {dst}")
        return dst
    print(f"[fetch] {url}\n     -> {dst}")
    tmp = dst + ".part"
    subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "5", "-o", tmp, url], check=True)
    if size is not None and os.path.getsize(tmp) != size:
        raise RuntimeError(f"{tmp}: {os.path.getsize(tmp)} bytes, expected {size}")
    os.replace(tmp, dst)
    return dst


def _clone(url: str, commit: str, *parts: str) -> None:
    dst = baselines_dir(*parts)
    if os.path.isdir(os.path.join(dst, ".git")):
        print(f"[fetch] have {dst}")
        return
    print(f"[fetch] git clone {url} @ {commit[:7]} -> {dst}")
    subprocess.run(["git", "clone", "-q", url, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", commit], check=True)


def _extract(src: str, dst: str, select: Callable[[str], Optional[str]], inner: Optional[str] = None) -> None:
    if os.path.exists(dst):
        print(f"[fetch] have {dst}")
        return
    import torch
    sd = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    if inner and isinstance(sd, dict) and inner in sd:
        sd = sd[inner]
    out = {}
    for k, v in sd.items():
        nk = select(k)
        if nk is not None:
            out[nk] = v.clone()
    if not out:
        raise RuntimeError(f"{src}: no keys selected")
    torch.save(out, dst)
    print(f"[fetch] wrote {dst} ({len(out)} tensors)")


def _card_weights(enc: str) -> str:
    return data_path(load_encoders()[enc]["weights"])


SSLW_COMMIT = "150550ea5d41800229c95e36f88f5bf0d2e7cf04"     # OxWearables/ssl-wearables


def fetch_harnet(enc: str, fname: str, size: int) -> None:
    code = baselines_dir("SSL", "ssl_wearables")
    _download(f"https://raw.githubusercontent.com/OxWearables/ssl-wearables/{SSLW_COMMIT}/sslearning/models/accNet.py",
              os.path.join(code, "accNet.py"))
    open(os.path.join(code, "__init__.py"), "a").close()
    _download(f"https://github.com/OxWearables/ssl-wearables/raw/{SSLW_COMMIT}/model_check_point/{fname}",
              _card_weights(enc), size)


def fetch_imagebind() -> None:
    _clone("https://github.com/facebookresearch/ImageBind.git", "53680b02d7e37b19b124fa37bae4b6c98c38f5be",
           "SSL", "ImageBind", "ImageBind")
    dst = _card_weights("imagebind_imu")
    full = os.path.join(os.path.dirname(dst), "imagebind_huge.pth")
    if not os.path.exists(dst):
        _download("https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth", full, 4803584173)
    _extract(full, dst, lambda k: k if ".imu." in k else None)


def fetch_unimts() -> None:
    _clone("https://github.com/xiyuanzh/UniMTS.git", "656fcdab7eee17d5b4158acf98fa348f74ce29a6",
           "language_alignment", "UniMTS")
    dst = _card_weights("unimts_release")
    full = os.path.join(os.path.dirname(dst), "UniMTS.pth")
    if not os.path.exists(dst):
        _download("https://huggingface.co/xiyuanz/UniMTS/resolve/main/checkpoint/UniMTS.pth", full, 274583082)
    _extract(full, dst, lambda k: k[len("acc."):] if k.startswith("acc.") else None)


def fetch_primus() -> None:
    dst = _card_weights("primus")
    full = os.path.join(os.path.dirname(dst), "best_model.ckpt")
    if not os.path.exists(dst):
        _download("https://zenodo.org/records/15147513/files/best_model.ckpt?download=1", full, 621911316)
    _extract(full, dst, lambda k: k[len("imu_encoder."):] if k.startswith("imu_encoder.backbone.") else None,
             inner="state_dict")


def fetch_normwear() -> None:
    _clone("https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear.git",
           "07517fcb13def8c89cb586128359cec02f86ec8d", "SSL", "NormWear")
    _download("https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/download/v1.0.0-alpha/"
              "normwear_pretrain_ckpt.pth", _card_weights("normwear_base"))


def fetch_inrtlwrist() -> None:
    from huggingface_hub import hf_hub_download
    print("[fetch]", hf_hub_download("YiquanLi/InrtlWrist", "inrtlwrist_mae_5src_step144k.pt"))


FETCHERS: Dict[str, Callable[[], None]] = {
    "inrtlwrist": fetch_inrtlwrist,
    "harnet5": lambda: fetch_harnet("harnet5", "mtl_5_best.mdl", 17037714),
    "harnet10": lambda: fetch_harnet("harnet10", "mtl_best.mdl", 42014098),
    "normwear_base": fetch_normwear,
    "imagebind_imu": fetch_imagebind,
    "unimts_release": fetch_unimts,
    "primus": fetch_primus,
}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="imu-bench fetch-weights", description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="*", default=list(FETCHERS), help=f"any of {', '.join(FETCHERS)}")
    a = ap.parse_args(argv)
    bad = [m for m in a.models if m not in FETCHERS]
    if bad:
        sys.exit(f"unknown model(s) {bad}; choose from {list(FETCHERS)}")
    for m in a.models:
        FETCHERS[m]()


if __name__ == "__main__":
    main()
