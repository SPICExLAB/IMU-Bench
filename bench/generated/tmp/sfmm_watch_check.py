import os, re, numpy as np, pandas as pd
R="/projects/b53017/vbc1591/SmartFallMM-Dataset/young"; PAT=re.compile(r"S(\d+)A(\d+)T(\d+)\.csv$")
def load(p):
    try: df=pd.read_csv(p,header=None,dtype=str,on_bad_lines="skip")
    except Exception: return None
    if df.shape[1]<4: return None
    t=pd.to_datetime(df[0],errors="coerce"); x=df[[1,2,3]].apply(pd.to_numeric,errors="coerce"); ok=t.notna()&x.notna().all(axis=1)
    if ok.sum()<5: return None
    t=t[ok].astype("int64").values/1e9; x=x[ok].values.astype(float); o=np.argsort(t,kind="stable"); return t[o],x[o]
A=set(os.listdir(R+"/accelerometer/watch")); G=set(os.listdir(R+"/gyroscope/watch")); both=sorted(f for f in A&G if PAT.search(f))
rows=[]
for f in both:
    a=load(f"{R}/accelerometer/watch/{f}"); g=load(f"{R}/gyroscope/watch/{f}")
    if a is None or g is None: continue
    (ta,xa),(tg,xg)=a,g; ov=max(0,min(ta[-1],tg[-1])-max(ta[0],tg[0])); act=int(PAT.search(f).group(2))
    rows.append(dict(s=int(PAT.search(f).group(1)),act=act,fall=act>=10,ov=ov,ra=(len(ta)-1)/max(ta[-1]-ta[0],1e-6),rg=(len(tg)-1)/max(tg[-1]-tg[0],1e-6),
                     amax=np.abs(xa).max(),gmax=np.abs(xg).max(),anorm=np.median(np.linalg.norm(xa,axis=1)),dta=np.median(np.diff(ta))*1000,dtg=np.median(np.diff(tg))*1000))
d=pd.DataFrame(rows); ok=d[d.ov>=2.56]
print(f"watch (left wrist), young: paired={len(d)} usable(ov>=2.56s)={len(ok)} subjects={ok.s.nunique()} falls={int(ok.fall.sum())} ADL={int((~ok.fall).sum())} zero-overlap={(d.ov<=0).mean()*100:.1f}%")
print(f"  acc rate med {d.ra.median():.1f} Hz (p5 {d.ra.quantile(.05):.1f} p95 {d.ra.quantile(.95):.1f}) dt med {d.dta.median():.0f} ms | gyro rate med {d.rg.median():.1f} Hz (p5 {d.rg.quantile(.05):.1f} p95 {d.rg.quantile(.95):.1f}) dt med {d.dtg.median():.0f} ms")
print(f"  |acc| median {d.anorm.median():.2f} m/s^2 | acc abs max: p50 {d.amax.median():.1f} p99 {d.amax.quantile(.99):.1f} max {d.amax.max():.1f} | gyro abs max (rad/s): p50 {d.gmax.median():.2f} p99 {d.gmax.quantile(.99):.2f} max {d.gmax.max():.2f}")
print("  falls: acc abs max p50", round(d[d.fall].amax.median(),1), "| gyro abs max p50", round(d[d.fall].gmax.median(),2))
