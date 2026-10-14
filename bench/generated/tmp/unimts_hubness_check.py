# Hubness check for the UniMTS wrist-only reimplementation on the HumanML3D VAL split (1,197 items; small, CPU).
import os, sys, importlib.util, numpy as np, torch
R="/gpfs/home/vbc1591/IMU-SSL"; sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,"scripts"))
spec=importlib.util.spec_from_file_location("era", os.path.join(R,"scripts","eval_retrieval_alignment.py")); era=importlib.util.module_from_spec(spec); spec.loader.exec_module(era)
torch.set_num_threads(8); dev=torch.device("cpu")
model=era.load_unimts(os.path.join(R,"baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_jointmask_100ep/last.pth"),dev)
root="/projects/b53017/vbc1591/HumanML3D"
raw=era.enumerate_humanml3d_val_unimts_raw(os.path.join(root,"UniMTS_data","data"),os.path.join(root,"HumanML3D","val.txt"))
flats=[era.unimts_flat_from_grid(g,True) for _,g,_ in raw]; caps=[era._clean_caps(c)[0] for _,_,c in raw]
zi,_=era.encode_unimts(model,flats,[],dev,128); _,zt=era.encode_unimts(model,[],caps,dev,128)
zi=torch.nn.functional.normalize(zi.float(),dim=-1); zt=torch.nn.functional.normalize(zt.float(),dim=-1)
S=(zi@zt.T).numpy(); n=S.shape[0]
print("n",n,"| IMU emb: mean pairwise cos %.3f | text emb: mean pairwise cos %.3f"%(((zi@zi.T).sum()-n)/(n*n-n), ((zt@zt.T).sum()-n)/(n*n-n)))
row_off=S.mean(1); col_off=S.mean(0)     # per-IMU-item and per-text offsets
print("std of per-IMU-item mean similarity %.4f | std of per-text mean similarity %.4f | std of S within a row (median) %.4f"%(row_off.std(), col_off.std(), np.median(S.std(1))))
top_t2i=S.argmax(0); u,c=np.unique(top_t2i,return_counts=True); c=np.sort(c)[::-1]
print("t2i: %d text queries -> %d distinct top-1 IMU items; top-10 hubs take %.1f%% of all queries (counts %s)"%(n,len(u),100*c[:10].sum()/n,c[:10].tolist()))
top_i2t=S.argmax(1); u2,c2=np.unique(top_i2t,return_counts=True); c2=np.sort(c2)[::-1]
print("i2t: %d IMU queries  -> %d distinct top-1 texts;     top-10 take %.1f%%"%(n,len(u2),100*c2[:10].sum()/n))
# remove the per-IMU-item offset (centre each row) and redo t2i
Sc=S-row_off[:,None]; r1=(Sc.argmax(0)==np.arange(n)).mean()*100; r1raw=(S.argmax(0)==np.arange(n)).mean()*100
print("t2i R@1 (strict index match, full gallery): raw %.2f%% -> after removing each IMU item's mean similarity %.2f%%"%(r1raw,r1))
print("i2t R@1 (strict index match): %.2f%%"%((S.argmax(1)==np.arange(n)).mean()*100))
