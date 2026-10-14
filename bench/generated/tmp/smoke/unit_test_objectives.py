# offline unit test of the three objectives on a tiny encoder (CPU, no data loader, no runs/ dir)
import yaml, torch, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives import dino_ts, jepa_ts, ssl_wearables_mtl as sw
def merge(a, b):
    for k, v in b.items(): a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a
MOD = {"dino": dino_ts, "jepa": jepa_ts, "sslwear": sw}
for m in ("dino", "jepa", "sslwear"):
    cfg = merge(yaml.safe_load(open("configs/base.yaml")), yaml.safe_load(open(f"bench/generated/tmp/smoke/{m}_cpu.yaml")))
    cfg["data"]["sensor_columns"] = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    enc = ViT1DEncoder(cfg)
    st = dino_ts.DINOState(enc, cfg) if m == "dino" else jepa_ts.JEPAState(enc, cfg) if m == "jepa" else sw.SSLWearablesHead(enc.embed_dim)
    for L in (64, 512, 256):
        enc.train(); st.train()
        x = torch.randn(8, 6, L); batch = (x, torch.empty(0), torch.ones(8, 6, dtype=torch.bool))
        loss, logs = MOD[m].forward_loss(batch, enc, st, cfg); loss.backward()
        enc.eval(); st.eval()
        with torch.no_grad():
            l1, _ = MOD[m].forward_loss(batch, enc, st, cfg); l2, _ = MOD[m].forward_loss(batch, enc, st, cfg)
        print(f"{m} L={L}: loss {float(loss):.4f} | eval deterministic {abs(float(l1)-float(l2))<1e-6} | "
              + ", ".join(f"{k}={v:.3g}" for k, v in list(logs.items())[1:6]), flush=True)
    sd = st.state_dict()
    st2 = type(st)(enc, cfg) if m != "sslwear" else sw.SSLWearablesHead(enc.embed_dim)
    st2.load_state_dict(sd); print(f"  {m}: state_dict round-trip ok ({len(sd)} tensors; step={int(sd['step']) if 'step' in sd else '-'})", flush=True)
# DINO-specific: prototypes frozen for the first steps -> weight_v grad must be zero; centre must have moved
cfg = merge(yaml.safe_load(open("configs/base.yaml")), yaml.safe_load(open("bench/generated/tmp/smoke/dino_cpu.yaml")))
cfg["data"]["sensor_columns"] = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
enc = ViT1DEncoder(cfg); st = dino_ts.DINOState(enc, cfg); enc.train(); st.train()
x = torch.randn(8, 6, 256); loss, logs = dino_ts.forward_loss((x, torch.empty(0)), enc, st, cfg); loss.backward()
print("dino: last-layer grad zero while frozen:", float(st.student_head.last_layer.weight_v.grad.abs().max()) == 0.0,
      "| centre moved:", float(st.center.abs().max()) > 0, "| ema momentum", round(st.ema_momentum(), 5), "| teacher temp", round(st.teacher_temp(), 4))
# SSL-Wearables transforms: reversal is exact, permutation is a permutation, time warp keeps the range
import numpy as np
rng = np.random.default_rng(0); x = torch.randn(4, 6, 128); sel = torch.tensor([True, False, True, True])
f = sw.flip_time(x, sel); assert torch.equal(f[0], x[0].flip(-1)) and torch.equal(f[1], x[1])
p = sw.permute_segments(x, sel, rng); assert torch.allclose(p[0].sort(-1).values, x[0].sort(-1).values) and torch.equal(p[1], x[1]) and not torch.equal(p[0], x[0])
w = sw.time_warp(x, sel, rng); assert torch.equal(w[1], x[1]) and w[0].min() >= x[0].min() - 1e-5 and w[0].max() <= x[0].max() + 1e-5 and not torch.allclose(w[0], x[0])
print("sslwear transforms: flip exact, permutation is a permutation, warp bounded -- ok")
