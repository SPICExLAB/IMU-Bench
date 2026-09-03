import sys, yaml, numpy as np, torch; sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from imu_lm.data.loaders import make_loaders
from imu_lm.models.ViT1D.run import _load_encoder_state_dict
from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn

def diag(run, cfg_path, n=1024):
    cfg = yaml.safe_load(open(cfg_path)); cfg.setdefault("data", {})["num_workers"] = 0
    enc = ViT1DEncoderAltAttn(cfg)
    _load_encoder_state_dict(enc, torch.load(f"runs/{run}/checkpoints/best.pt", map_location="cpu", weights_only=False))
    enc.eval()
    Z, X, Y = [], [], []
    with torch.no_grad():
        for b in make_loaders(cfg)["val_loader"]:
            if b is None: continue
            x = b[0].float(); Z.append(enc(x)); X.append(x); Y.append(b[1])
            if sum(len(z) for z in Z) >= n: break
    Z = torch.cat(Z)[:n].numpy(); X = torch.cat(X)[:n].numpy(); Y = torch.cat(Y)[:n].numpy()
    Zc = Z - Z.mean(0)
    ev = np.linalg.svd(Zc, compute_uv=False) ** 2; pr_raw = ev.sum()**2 / (ev**2).sum()
    Zs = Zc / (Zc.std(0) + 1e-8)                                   # per-dimension standardised -> correlation-space
    ev2 = np.linalg.svd(Zs, compute_uv=False) ** 2; pr_std = ev2.sum()**2 / (ev2**2).sum()
    dc = X[:, :3].mean(-1); A = np.c_[dc, np.ones(len(dc))]
    resid = Zc - A @ np.linalg.lstsq(A, Zc, rcond=None)[0]
    r2 = 1 - (resid**2).sum() / (Zc**2).sum()
    # is the gravity-free residual still class-discriminative? (between-class / total variance)
    lab = np.unique(Y); bet = sum(len(resid[Y==c]) * ((resid[Y==c].mean(0) - resid.mean(0))**2).sum() for c in lab)
    fisher = bet / max((resid**2).sum(), 1e-9)
    print(f"{run[:40]:40s} PR_raw={pr_raw:6.1f}  PR_std={pr_std:6.1f}  gravity_R2={r2*100:5.1f}%  "
          f"class-signal in gravity-free residual={fisher*100:5.1f}%", flush=True)

for run, cfg in [("mae_4src_physnp", "bench/generated/configs/bench_har_wisdm__mae_4src_physnp__probe__s42.yaml"),
                 ("mae_4src_phys",   "bench/generated/configs/bench_har_wisdm__mae_4src_phys__probe__s42.yaml"),
                 ("mae_4src_dc",     "bench/generated/configs/bench_har_wisdm__mae_4src_dc__probe__s42.yaml"),
                 ("vit1d_ssl_mae_varlen_altattn_multidata", "bench/generated/configs/bench_har_wisdm__mae_11src__probe__s42.yaml")]:
    try: diag(run, cfg)
    except Exception as e: print(run, "ERR", type(e).__name__, str(e)[:90], flush=True)
