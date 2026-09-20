"""Single source of truth for the volume x diversity factorial (EXPERIMENTS.md 11m).

Every other script in this directory imports the pool geometry and the arm
table from here, so the index files, the pretrain configs, the bench encoder
registry entries and the harvest all agree by construction.

Pool = the leak-free 4-source pool, at FULL size (no sub-sampling of any
source):

    ocdetect     1,253,077 windows  1,782 h
    eatingspeed    287,678 windows    409 h
    motionx         50,395 windows     72 h
    sonic          146,944 windows    209 h
    -------------------------------------------
    total        1,738,094 windows  2,472 h

The concat index space is ALL_SOURCES order filtered to the enabled sources
(imu_lm/data/combined_mae_multidata_loader.py:93), i.e. ocdetect, eatingspeed,
motionx, sonic -- asserted against the live loader in 01_embed_pool.py.
"""
from __future__ import annotations

import os

WORK_DIR = "/projects/b53017/vbc1591/volume_diversity"
IDX_DIR = os.path.join(WORK_DIR, "idx")
CACHE_DIR = "/projects/b53017/vbc1591/mae_multidata_cache_raw"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "volume_diversity", "out")

BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "base.yaml")
POOL_CONFIG = os.path.join(REPO_ROOT, "configs", "nhanes_scale", "mae_4src_z.yaml")
POOL_CKPT = os.path.join(REPO_ROOT, "runs", "mae_4src_z", "checkpoints", "best.pt")

SOURCES = ["ocdetect", "eatingspeed", "motionx", "sonic"]
SOURCE_SIZES = {"ocdetect": 1253077, "eatingspeed": 287678, "motionx": 50395, "sonic": 146944}
POOL_N = sum(SOURCE_SIZES.values())          # 1,738,094
WINDOW_SECONDS = 5.12                        # 10.24 s window @ hop 0.5 -> 5.12 h of unique time per window,
                                             # the hour convention used everywhere else in this repo
SEED = 42

K_TYPES = 1000        # k-means motion types, both definitions
PCA_ENC = 64          # PCA dim for the encoder embedding
PCA_PHYS = 32         # PCA dim for the physical feature vector

# --- the factorial -------------------------------------------------------
# volume: number of DISTINCT windows the arm may train on (compute is held
# fixed at STEPS gradient steps for every arm, so a small-N arm simply revisits
# its windows more often -- that is the control for "does more data help at
# matched compute", and it is stated as such in EXPERIMENTS.md 11m).
VOLUMES = [("n12k", 12_500), ("n50k", 50_000), ("n200k", 200_000), ("n800k", 800_000)]

# diversity: four levels, ordered a priori low -> high. Which of them is
# actually more diverse is MEASURED per cell (04_build_arms.py) rather than
# assumed -- the x-axis of every plot is the measured statistic, not the label.
#   nar  single source (ocdetect), narrowest feasible motion-type band
#   one  single source (ocdetect), uniform random  -> natural type mix, 1 source
#   nat  all 4 sources, uniform random over the pool -> the paper's own curve
#   bal  all 4 sources, source-balanced + type-balanced -> max diversity
DIVERSITIES = ["nar", "one", "nat", "bal"]

STEPS = 50_000        # equal fixed budget for every arm, incl. the anchors
BATCH = 128

DIV_LABEL = {
    "nar": "single-source, narrow motion band",
    "one": "single-source, uniform random",
    "nat": "4-source, natural proportions",
    "bal": "4-source, source- and type-balanced",
}


def arm_name(div: str, vol_tag: str) -> str:
    return f"vd_{div}_{vol_tag}"


def grid_arms():
    """The 16 index-file cells of the factorial."""
    return [(arm_name(d, t), d, t, n) for d in DIVERSITIES for (t, n) in VOLUMES]


def anchor_arms():
    """N = 100 % of the pool. `nar`/`one` do not exist at 100 % by definition
    (100 % of the pool IS full diversity), so the top volume level carries only
    the two 4-source arms -- the same principled triangularity as 11l."""
    return [("vd_nat_nfull", "nat", "nfull", POOL_N),
            ("vd_bal_nfull", "bal", "nfull", POOL_N)]


def all_arms():
    return grid_arms() + anchor_arms()


def source_ranges():
    lo, out = 0, {}
    for s in SOURCES:
        out[s] = (lo, lo + SOURCE_SIZES[s])
        lo += SOURCE_SIZES[s]
    return out


def idx_path(arm: str) -> str:
    return os.path.join(IDX_DIR, f"{arm}.npy")


def manifest_arms(manifest):
    """The 18 factorial cells plus any extra arms a later stage added to
    out/arms.json (the phase-3 subject ladder from 09_subject_arms.py), so the
    config generator, the harvest and the analysis all pick those up without
    needing to know about them in advance."""
    out = list(all_arms())
    known = {a for a, _, _, _ in out}
    for arm, info in (manifest.get("arms") or {}).items():
        if arm in known:
            continue
        out.append((arm, info.get("diversity", "?"), info.get("volume_tag", "?"),
                    int(info.get("n_windows", 0))))
    return out
