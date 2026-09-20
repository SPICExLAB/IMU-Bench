"""Dense-query version: every HumanML3D test clip (incl. M-mirrors), 2 s snippets at hop 1 frame,
vs all Motion-X++ (7 subsets) snippets at hop 20. Sensitivity (inject_result_dense.json):
an exact copy at any offset is found at median 0.1 mm / max 3.6 cm, 99.8 % < 2 cm."""
import numpy as np, io, zipfile, json, sys, time, torch
import importlib.util
spec=importlib.util.spec_from_file_location("d2","dupcheck2.py"); d2=importlib.util.module_from_spec(spec); spec.loader.exec_module(d2)
torch.set_num_threads(8)
t0=time.time()
D=[]; meta=[]
for cat in ["animation","haa500","humman","idea400","kungfu","music","perform"]:
    zf=zipfile.ZipFile(f"{d2.MX}/{cat}.zip")
    for n in [x for x in zf.namelist() if x.endswith(".npy")]:
        j=np.load(io.BytesIO(zf.read(n)))
        if j.ndim!=3 or j.shape[1]<d2.J: continue
        f=d2.snippets(d2.resample(j[:, :d2.J].astype(np.float64),30.0,20.0), d2.HOP)
        f=f[np.isfinite(f).all(1)]
        if len(f): D.append(f); meta+=[(cat,n.split("/")[-1])]*len(f)
D=torch.from_numpy(np.concatenate(D)); dn=(D*D).sum(1)
print("db",D.shape,f"{time.time()-t0:.0f}s",flush=True)
ids=[l.strip() for l in open(f"{d2.R}/HumanML3D/test.txt") if l.strip()]
npts=D.shape[1]//3
res=[]
for k,i in enumerate(ids):
    p=f"HumanML3D/new_joints/{i}.npy"
    if p not in d2._HZN: continue
    j=np.load(io.BytesIO(d2._HZ.read(p)))
    if j.ndim!=3 or j.shape[0]<d2.L: continue
    Q=d2.snippets(j[:, :d2.J],1); Q=Q[np.isfinite(Q).all(1)]
    if not len(Q): continue
    Qt=torch.from_numpy(Q); best=[]; arg=[]
    for a in range(0,len(Qt),256):
        q=Qt[a:a+256]; d2m=(q*q).sum(1,keepdim=True)+dn[None]-2*(q@D.T)
        m,ix=d2m.min(1); best.append(m.clamp_min(0)); arg.append(ix)
    best=torch.sqrt(torch.cat(best)/npts).numpy(); arg=torch.cat(arg).numpy()
    b=int(best.argmin())
    res.append({"id":i,"n_snip":len(Q),"min_m":float(best.min()),"median_m":float(np.median(best)),
                "frac_lt_2cm":float((best<0.02).mean()),"closest":meta[arg[b]]})
    if k%500==0: print(k,f"{time.time()-t0:.0f}s",flush=True)
mins=np.array([r["min_m"] for r in res]); meds=np.array([r["median_m"] for r in res])
summ={"n_clips":len(res),"n_query_snippets":int(sum(r["n_snip"] for r in res)),"db_snippets":int(D.shape[0]),
 "clip_min_quantiles_m":{q:float(np.quantile(mins,q)) for q in [0,0.001,0.01,0.05,0.5]},
 "clips_min_lt_1cm":int((mins<0.01).sum()),"clips_min_lt_2cm":int((mins<0.02).sum()),"clips_min_lt_3.6cm":int((mins<0.036).sum()),
 "clips_median_lt_2cm":int((meds<0.02).sum()),"clips_any_frac_lt_2cm_gt_0":int(sum(r["frac_lt_2cm"]>0 for r in res)),
 "closest10":sorted(res,key=lambda r:r["min_m"])[:10]}
print(json.dumps(summ,indent=1)); json.dump({"summary":summ,"per_clip":res},open("dupcheck_dense_result.json","w"))
print(f"done {time.time()-t0:.0f}s")
