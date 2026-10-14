"""Derive the three 'native preprocessing' pretrain configs from their z-score twins.

Generated rather than hand-written so that every key except the documented
normalisation changes is provably identical to the row it is compared against.
"""
import copy, yaml, os, sys

REPO = "/gpfs/home/vbc1591/IMU-SSL"
RAW = "/projects/b53017/vbc1591/mae_multidata_cache_v3raw"
IDX = "/projects/b53017/vbc1591/diversity_saturation/corpus_ablation/idx_ref_v3raw.npy"
OUTDIR = os.path.join(REPO, "configs/nativenorm")
os.makedirs(OUTDIR, exist_ok=True)

PHYSICAL = {"enabled": True, "method": "physical", "eps": 1.0e-06,
            "acc_scale": 9.81, "gyro_scale": 5.0,
            "clip_acc": 8.0, "clip_gyro": 8.0, "n_acc": 3}

HEADERS = {
"simmtm": """# SimMTM with its OWN normalisation instead of the benchmark's per-window z-score.
#
# Derived key-for-key from configs/pretrain5src/simmtm_5src.yaml (the leaderboard row it is
# compared against) by launch/nativenorm/gen_native_cfgs.py. Everything -- backbone, corpus,
# 160k cap, patience 10, lr, augmentation, window length -- is identical. What changes:
#
#   1. preprocess.normalize.method: zscore -> physical. The dataloader no longer removes each
#      window's channel mean and amplitude; input reaches the model in fixed physical units
#      (acc / 9.81 m s^-2, gyro / 5 rad s^-1, clipped at +-8).
#   2. objective.visible_point_instance_norm: true. SimMTM normalises INSIDE the model, over the
#      VISIBLE points only, and de-normalises the reconstruction before the MSE
#      (SimMTM_Forecasting/models/SimMTM.py:172-178, 205-207). That is the normalisation this
#      method actually specifies, and it only becomes a real operation once (1) stops pre-empting it.
#   3. objective.recon_on_original_only: true. SimMTM reconstructs only the un-masked copy against
#      the original series (SimMTM.py:208-211); this port took the MSE over every masked copy
#      against a target that is zero at the masked points. Harmless in z-space, an amplitude-
#      weighted artefact in physical units, so the faithful arm restores the original.
#
# Because the model's own instance norm makes the encoder input ~(0,1) again, downstream probes
# feed z-scored input (encoder card normalize_method: zscore) -- for an unmasked window SimMTM's
# instance norm IS a per-window z-score, which is exactly why the two agree.
#
# Corpus: the un-normalised cache twin mae_multidata_cache_v3raw, subset by idx_ref_v3raw.npy to
# the 5-source composition of mae_multidata_cache (196,659 of 197,094 windows; the 0.22 % gap is
# the mocap spike sanitiser and the HAR re-windowing, see launch/design_ablation/make_idx_ref_v3raw.py).
""",
"focal": """# FOCAL with its OWN input representation instead of the benchmark's per-window z-score.
#
# Derived key-for-key from configs/pretrain5src/focal_5src.yaml by launch/nativenorm/gen_native_cfgs.py.
# Everything -- backbone, corpus, 160k cap, patience 10, lr, augmentation, window length, all six
# FOCAL loss weights -- is identical. What changes:
#
#   1. preprocess.normalize.method: zscore -> physical. FOCAL's public pipeline applies no
#      per-window z-norm; the repo ships a constant-amplitude rescale helper that the main path
#      never calls (src/general_utils/tensor_utils.py).
#   2. vit1d.input_transform: fft. FOCAL's encoder does not see the time series at all: each
#      sub-interval is FFT'd and the real and imaginary parts are handed to the model as channels
#      (src/data_augmenter/Augmenter.py:141-157, torch.fft.fft + torch.view_as_real). Here the
#      sub-interval is the patch, so each patch becomes its own 2*patch_size spectrum and the
#      token count is unchanged. NOTE this is an input REPRESENTATION, not a normalisation -- the
#      arm answers "what does FOCAL do natively", not "what does dropping z-norm do".
#
# The transform must also reach every downstream probe, so the encoder card carries it as a
# vit1d_override; a frequency-domain encoder cannot be probed on time-domain input.
#
# Corpus: as for the SimMTM arm -- see configs/nativenorm/simmtm_5src_native.yaml.
""",
"relcon": """# RelCon with its OWN normalisation instead of the benchmark's per-window z-score.
#
# Derived key-for-key from configs/pretrain5src/relcon_5src.yaml by launch/nativenorm/gen_native_cfgs.py.
# Everything -- backbone, corpus, 160k cap, patience 10, lr, augmentation, window length -- is
# identical. What changes:
#
#   1. preprocess.normalize.method: zscore -> physical. RelCon's SSL pretraining reads raw .npy
#      windows and applies NO input normalisation whatsoever
#      (baselines/SSL/relcon/relcon/data/Base_Dataset.py:50); scale is handled inside the model.
#      Fixed physical constants are a unit convention, not a data-dependent normalisation, so this
#      is the faithful stand-in (our corpus is m s^-2 / rad s^-1, theirs is g).
#   2. motifdist_run -> relcon_motifdist_5src_native: the stage-1 distance network is retrained on
#      the same un-normalised input. Its RevIN (nets/CrossAttn/CrossAttn_Net.py:34) is the only
#      normalisation RelCon actually specifies and it stays exactly where the authors put it.
#
# Note the StandardScaler in relcon/eval/Classify/linear_probe.py:53 is fit on the ENCODER'S
# EMBEDDINGS, not on the input signal, so it is not an input-normalisation convention at all.
#
# Corpus: as for the SimMTM arm -- see configs/nativenorm/simmtm_5src_native.yaml.
""",
}

VAL_IDX = "/projects/b53017/vbc1591/diversity_saturation/nativenorm"
VAL_SOURCES = ["realdata", "ocdetect", "eatingspeed", "motionx", "sonic"]


def _set_val_idx(mm):
    """Match the production cache's val COUNT per source (launch/nativenorm/make_val_idx.py).

    scaling_mode=index_file leaves the val split untouched, and the raw twin was built
    without the production cache's per-source val caps -- 290,638 windows, 64 % OCDetect,
    against production's 27,463 at 4 %. val_loss drives early stopping and best.pt, so the
    arm has to validate on the same mixture as the row it is compared against.
    """
    for s_ in VAL_SOURCES:
        mm["sources"][s_]["subset_index_file_val"] = f"{VAL_IDX}/val_idx_{s_}.npy"


SRC = {"simmtm": "configs/pretrain5src/simmtm_5src.yaml",
       "focal":  "configs/pretrain5src/focal_5src.yaml",
       "relcon": "configs/pretrain5src/relcon_5src.yaml"}

for meth, rel in SRC.items():
    cfg = yaml.safe_load(open(os.path.join(REPO, rel)))
    mm = cfg["objective"]["mae_multidata_varlen_altattn_flu"]

    # --- corpus: un-normalised cache twin, composition-matched to the z-score row ---
    mm["cache_dir"] = RAW
    mm["scaling_mode"] = "index_file"
    mm["subset_indices_file"] = IDX
    cfg["objective"]["cache_is_normalized"] = False
    _set_val_idx(mm)

    # --- the shared change: no per-window z-score ---
    cfg["preprocess"]["normalize"] = copy.deepcopy(PHYSICAL)

    # --- the per-method native bits ---
    if meth == "simmtm":
        cfg["objective"]["visible_point_instance_norm"] = True
        cfg["objective"]["recon_on_original_only"] = True
    elif meth == "focal":
        cfg["vit1d"]["input_transform"] = "fft"
    elif meth == "relcon":
        assert cfg["objective"]["motifdist_run"] == "relcon_motifdist_5src", cfg["objective"]["motifdist_run"]
        cfg["objective"]["motifdist_run"] = "relcon_motifdist_5src_native"

    out = os.path.join(OUTDIR, f"{meth}_5src_native.yaml")
    with open(out, "w") as f:
        f.write(HEADERS[meth])
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print("wrote", out)

# --- stage 1 for RelCon: the same distance network on the same un-normalised input ---
md = yaml.safe_load(open(os.path.join(REPO, "configs/pretrain5src/relcon_motifdist_5src.yaml")))
mm = md["objective"]["mae_multidata_varlen_altattn_flu"]
mm["cache_dir"] = RAW
mm["scaling_mode"] = "index_file"
mm["subset_indices_file"] = IDX
md["objective"]["cache_is_normalized"] = False
_set_val_idx(mm)
md["preprocess"]["normalize"] = copy.deepcopy(PHYSICAL)
out = os.path.join(OUTDIR, "relcon_motifdist_5src_native.yaml")
with open(out, "w") as f:
    f.write("# Stage 1 of the RelCon native-normalisation arm: configs/pretrain5src/relcon_motifdist_5src.yaml\n"
            "# on the un-normalised cache twin, physical units. The distance network's own RevIN\n"
            "# (baselines/SSL/relcon/relcon/nets/CrossAttn/CrossAttn_Net.py:34) is untouched -- it is the\n"
            "# normalisation RelCon specifies, and it is the reason this stage does not need a z-scored input.\n"
            "# Consumed by configs/nativenorm/relcon_5src_native.yaml via objective.motifdist_run.\n")
    yaml.safe_dump(md, f, sort_keys=False, default_flow_style=False)
print("wrote", out)
