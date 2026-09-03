import sys; sys.path.insert(0,"/gpfs/home/vbc1591/IMU-SSL")
import torch, yaml
from imu_lm.utils.helpers import deep_update
from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn, ViT1DDecoder
from imu_lm.objectives.masked_1d_freq_channel_mask import forward_loss
cfg=deep_update(yaml.safe_load(open("configs/base.yaml")), yaml.safe_load(open("configs/nhanes_scale/mae_recipe_R0.yaml")))
enc=ViT1DEncoderAltAttn(cfg); B,C,T=4,6,512
dec=ViT1DDecoder(embed_dim=enc.embed_dim, out_channels=C, target_T=T, cfg=cfg)
x=torch.randn(B,C,T); cm=torch.ones(B,C,dtype=torch.bool)
tm=torch.ones(B,T,dtype=torch.bool); tm[0,128:]=False; tm[1,256:]=False; x[0,:,128:]=0; x[1,:,256:]=0
l3,g3=forward_loss((x,torch.empty(0),cm), enc, dec, cfg)
l4,g4=forward_loss((x,torch.empty(0),cm,tm), enc, dec, cfg)
print(f"3-tuple: loss {l3.item():.4f}  padded_patch_frac {g3.get('padded_patch_frac')}")
print(f"4-tuple: loss {l4.item():.4f}  padded_patch_frac {g4['padded_patch_frac']:.3f}  expect (0.75+0.5)/4=0.312")
l4.backward(); print("backward OK — gradients flow with padded patches excluded")
