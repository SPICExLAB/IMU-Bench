import sys, copy, logging, torch
sys.path.insert(0,"analysis/mae_scaling_diagnostics")
logging.basicConfig(level=logging.INFO, format="%(message)s")
from lib.common import load_merged_cfg
from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn, ViT1DDecoder
from imu_lm.models.ViT1D.run import _load_encoder_state_dict
import imu_lm.objectives.masked_1d_freq_channel_mask as obj

CKPT = "runs/vit1d_ssl_mae_recipe_100pct/checkpoints/best.pt"
cfg = load_merged_cfg("configs/vit1d_ssl_mae_stage2_accgyro_flu.yaml")
mm  = cfg["objective"]["mae_multidata_varlen_altattn_flu"]
print(f"stage2 keys: pretrained_run={mm.get('pretrained_run')} ckpt={mm.get('pretrained_ckpt')} "
      f"load_decoder={mm.get('load_decoder')}", flush=True)

enc = ViT1DEncoderAltAttn(cfg)
dec = ViT1DDecoder(embed_dim=enc.embed_dim, out_channels=enc.in_channels, target_T=0, cfg=cfg)
fe, fd = copy.deepcopy(enc.state_dict()), copy.deepcopy(dec.state_dict())
st = torch.load(CKPT, map_location="cpu", weights_only=False)
_load_encoder_state_dict(enc, st)
dec.load_state_dict(st["decoder"], strict=True)
ec = sum(1 for k in fe if not torch.equal(fe[k], enc.state_dict()[k]))
dc = sum(1 for k in fd if not torch.equal(fd[k], dec.state_dict()[k]))
print(f"warm-start changed encoder {ec}/{len(fe)} tensors, decoder {dc}/{len(fd)} tensors", flush=True)
assert ec>0 and dc>0

x = torch.randn(2,6,512)
for lbl, gy in [("stage2 all-6ch-valid", True), ("stage1 gyro-masked", False)]:
    enc.zero_grad(); dec.zero_grad()
    cvm = torch.ones(2,6,dtype=torch.bool)
    if not gy: cvm[:,3:] = False
    l,_ = obj.forward_loss((x,None,cvm), enc, dec, cfg); l.backward()
    ng = sum(1 for p in enc.parameters() if p.grad is not None and p.grad.abs().sum()>0)
    tot= sum(1 for _ in enc.parameters())
    print(f"{lbl:<24s}: loss={l.item():.4f} finite={bool(torch.isfinite(l))} enc_grads={ng}/{tot}", flush=True)
print("\nVERIFIED: warm-start loads; both channel regimes run forward+backward")
