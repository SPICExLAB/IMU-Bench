"""Sensitivity of the dupcheck scheme: inject each of 500 random HumanML3D test clips as a
'duplicate' into a DB at a random start offset (0-19 frames, DB hop 20) after a 20->30->20 fps
round-trip (mimicking a 30 fps Motion-X++ copy resampled to 20 fps), then query with the clip's
own query snippets (hop 10). Report the clip-level min-NN RMS distance distribution."""
import numpy as np, io, zipfile, json, sys
sys.argv=[sys.argv[0]]
import importlib.util
spec=importlib.util.spec_from_file_location("d2","dupcheck2.py"); d2=importlib.util.module_from_spec(spec); spec.loader.exec_module(d2)
R=d2.R
ids=[l.strip() for l in open(f"{R}/HumanML3D/test.txt") if l.strip()]
rng=np.random.default_rng(0); pick=rng.choice(len(ids),500,replace=False)
mins=[]; mins_nors=[]
for k in pick:
    p=f"HumanML3D/new_joints/{ids[k]}.npy"
    if p not in d2._HZN: continue
    j=np.load(io.BytesIO(d2._HZ.read(p)))[:, :d2.J].astype(np.float64)
    if j.shape[0] < d2.L+20 or not np.isfinite(j).all(): continue
    Q=d2.snippets(j, 1)
    off=int(rng.integers(0,20))
    j30=d2.resample(j,20.0,30.0); jr=d2.resample(j30,30.0,20.0)   # fps round trip
    D=d2.snippets(jr[off:], d2.HOP); D0=d2.snippets(j[off:], d2.HOP)
    if len(D)==0: continue
    for DB,store in [(D,mins),(D0,mins_nors)]:
        dd=((Q[:,None,:]-DB[None,:,:])**2).sum(-1).min()
        store.append(float(np.sqrt(dd/(Q.shape[1]//3))))
mins=np.array(mins); m0=np.array(mins_nors)
out={"n":len(mins),
 "with_fps_roundtrip":{"median_m":float(np.median(mins)),"p95_m":float(np.quantile(mins,0.95)),"max_m":float(mins.max()),"frac_lt_2cm":float((mins<0.02).mean()),"frac_lt_3.6cm":float((mins<0.036).mean())},
 "no_resample":{"median_m":float(np.median(m0)),"p95_m":float(np.quantile(m0,0.95)),"max_m":float(m0.max()),"frac_lt_2cm":float((m0<0.02).mean()),"frac_lt_3.6cm":float((m0<0.036).mean())}}
print(json.dumps(out,indent=1)); json.dump(out,open("inject_result_dense.json","w"),indent=1)
