import sys, yaml, numpy as np, torch; sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from imu_lm.data.loaders import make_loaders
from imu_lm.models.baselines.inertia1_encoder import build_encoder
from imu_lm.models.ViT1D.run import _load_encoder_state_dict
from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn

def feats(run, cfg_path, n=512):
    cfg = yaml.safe_load(open(cfg_path)); cfg.setdefault("data", {})["num_workers"] = 0
    enc = ViT1DEncoderAltAttn(cfg)
    st = torch.load(f"runs/{run}/checkpoints/best.pt", map_location="cpu", weights_only=False)
    _load_encoder_state_dict(enc, st); enc.eval()
    dl = make_loaders(cfg)["val_loader"]; X, Z = [], []
    with torch.no_grad():
        for b in dl:
            if b is None: continue
            x = (b[0] if isinstance(b, (list, tuple)) else b).float()
            Z.append(enc(x)); X.append(x)
            if sum(len(z) for z in Z) >= n: break
    Z = torch.cat(Z)[:n].numpy(); X = torch.cat(X)[:n].numpy()
    dc = X[:, :3].mean(-1)                                   # per-window gravity vector [n,3]
    Zc = Z - Z.mean(0); ev = np.linalg.svd(Zc, compute_uv=False) ** 2
    pr = ev.sum() ** 2 / (ev ** 2).sum()                     # participation ratio = effective #dims
    # variance of embedding explained by a linear map of the DC vector
    A = np.c_[dc, np.ones(len(dc))]
    pred = A @ np.linalg.lstsq(A, Zc, rcond=None)[0]
    r2_dc = 1 - ((Zc - pred) ** 2).sum() / (Zc ** 2).sum()
    print(f"{run:22s} dim={Z.shape[1]} eff-rank(PR)={pr:6.1f}  feat std={Z.std():.3f}  "
          f"var explained by gravity(DC)={r2_dc*100:5.1f}%", flush=True)

for run, cfg in [("mae_4src_physnp", "bench/generated/configs/bench_har_wisdm__mae_4src_physnp__probe__s42.yaml"),
                 ("mae_4src_dc",   "bench/generated/configs/bench_har_wisdm__mae_4src_dc__probe__s42.yaml"),
                 ("mae_div60_dc",  "bench/generated/configs/bench_har_wisdm__mae_div60_dc__probe__s42.yaml"),
                 ("vit1d_ssl_mae_varlen_altattn_multidata", "bench/generated/configs/bench_har_wisdm__mae_11src__probe__s42.yaml"),
                 ("vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic", "bench/generated/configs/bench_har_wisdm__mae_5src__probe__s42.yaml")]:
    try: feats(run, cfg)
    except Exception as e: print(run, "ERR", type(e).__name__, str(e)[:100], flush=True)
