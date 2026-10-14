import numpy as np, hashlib, sys, time
raw="/projects/b53017/vbc1591/mae_multidata_cache_v3raw"; v3="/projects/b53017/vbc1591/mae_multidata_cache_v3"
def z(x, eps=1e-6):
    m=x.mean(-1,keepdims=True); s=x.std(-1,keepdims=True); return (x-m)/(s+eps)
def hashes(arr, do_z, chunk=4096):
    out=[]
    for s in range(0,len(arr),chunk):
        x=np.asarray(arr[s:s+chunk],dtype=np.float32)
        if do_z: x=z(x)
        x=np.round(x[:, :, :64],2)              # first 64 samples of every channel, 2 decimals
        out+= [hashlib.md5(r.tobytes()).hexdigest() for r in x]
    return out
for src in sys.argv[1:]:
    t0=time.time()
    a=np.load(f"{raw}/{src}_train_x.npy",mmap_mode="r"); b=np.load(f"{v3}/{src}_train_x.npy",mmap_mode="r")
    ha=hashes(a,True); hb=hashes(b,False)
    sa,sb=set(ha),set(hb)
    print(f"{src}: raw {len(a)} (unique {len(sa)}), v3 {len(b)} (unique {len(sb)}), v3 rows found in raw: {sum(h in sa for h in hb)} ({100*sum(h in sa for h in hb)/len(hb):.1f} %), raw rows found in v3: {sum(h in sb for h in ha)}  [{time.time()-t0:.0f}s]", flush=True)
