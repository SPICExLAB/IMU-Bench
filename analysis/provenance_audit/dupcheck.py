"""Read-only near-duplicate check: HumanML3D test motions vs Motion-X++ (all 7 subsets)
in the shared HumanML3D `new_joints` representation (uniform skeleton, Y-up).

Snippets: 2 s (40 frames @ 20 fps), subsampled to 10 frames, 22 body joints,
canonicalised (pelvis xz at frame 0 -> origin, heading via hip vector -> +X).
Distance = RMS per-joint position error (m) over the 10x22 points.

Positive control: HumanML3D test vs HumanML3D train+val (437 non-mirror test clips
share an overlapping frame range of the same AMASS file with a train/val clip).
"""
import os, io, csv, zipfile, json, sys, time
import numpy as np
import torch

torch.set_num_threads(8)
R = "/projects/b53017/vbc1591/HumanML3D"
MX = "/projects/b53017/vbc1591/Motion-Xplusplus/motion/motion_generation/new_joint"
OUT = os.path.dirname(os.path.abspath(__file__))
L, SUB, HOP = 40, 4, 20   # 2 s window, every 4th frame, 1 s hop (queries use HOP_Q)
HOP_Q = 10
J = 22


def resample(x, src_hz, dst_hz):
    T = x.shape[0]
    n = int(np.floor((T - 1) * dst_hz / src_hz)) + 1
    t_new = np.arange(n) * src_hz / dst_hz
    i0 = np.clip(np.floor(t_new).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    w = (t_new - i0)[:, None, None]
    return (1 - w) * x[i0] + w * x[i1]


def snippets(j, hop):
    """j: [T,22,3] at 20 fps -> [N, 10*22*3] canonicalised snippets."""
    T = j.shape[0]
    if T < L:
        return np.zeros((0, (L // SUB) * J * 3), np.float32)
    out = []
    for s in range(0, T - L + 1, hop):
        w = j[s:s + L:SUB].astype(np.float64).copy()   # [10,22,3]
        w[:, :, 0] -= w[0, 0, 0]
        w[:, :, 2] -= w[0, 0, 2]
        hv = w[0, 2] - w[0, 1]          # right hip - left hip
        ang = np.arctan2(hv[2], hv[0])  # angle in xz plane
        c, s_ = np.cos(ang), np.sin(ang)
        x, z = w[..., 0].copy(), w[..., 2].copy()
        w[..., 0] = c * x + s_ * z
        w[..., 2] = -s_ * x + c * z
        out.append(w.reshape(-1))
    return np.asarray(out, np.float32)


def load_hml(ids, hop=HOP_Q):
    feats, owner = [], []
    for k, i in enumerate(ids):
        p = f"{R}/HumanML3D/new_joints/{i}.npy"
        if not os.path.exists(p):
            continue
        j = np.load(p)
        if j.ndim != 3 or j.shape[0] < L:
            continue
        f = snippets(j[:, :J], hop)
        f = f[np.isfinite(f).all(1)]
        feats.append(f); owner += [k] * len(f)
    return np.concatenate(feats), np.asarray(owner)


class RunningNN:
    """Running nearest-neighbour over DB chunks (low memory)."""
    def __init__(self, Q):
        self.Qt = torch.from_numpy(Q); self.qn = (self.Qt * self.Qt).sum(1, keepdim=True)
        self.best = np.full(len(Q), np.inf, np.float32); self.arg = np.full(len(Q), -1, np.int64)
        self.offset = 0
    def add(self, D, bs=256):
        Dt = torch.from_numpy(D); dn = (Dt * Dt).sum(1)
        for a in range(0, len(self.Qt), bs):
            q = self.Qt[a:a + bs]
            d2 = self.qn[a:a + bs] + dn[None] - 2 * (q @ Dt.T)
            m, ix = d2.min(1)
            m = m.clamp_min(0).numpy(); ix = ix.numpy() + self.offset
            upd = m < self.best[a:a + bs]
            self.best[a:a + bs][upd] = m[upd]; self.arg[a:a + bs][upd] = ix[upd]
        self.offset += len(D)
    def result(self):
        npts = self.Qt.shape[1] // 3
        return np.sqrt(self.best / npts), self.arg


def summarize(tag, d, owner, nclips, extra=None):
    per_clip_med = np.array([np.median(d[owner == k]) for k in np.unique(owner)])
    per_clip_min = np.array([np.min(d[owner == k]) for k in np.unique(owner)])
    qs = [0.001, 0.01, 0.05, 0.5]
    res = {
        "tag": tag, "n_snippets": int(len(d)), "n_clips": int(len(per_clip_med)),
        "snippet_nn_rms_m_quantiles": {str(q): float(np.quantile(d, q)) for q in qs},
        "snippet_frac_lt_1cm": float((d < 0.01).mean()),
        "snippet_frac_lt_2cm": float((d < 0.02).mean()),
        "clip_median_nn_lt_1cm": int((per_clip_med < 0.01).sum()),
        "clip_median_nn_lt_2cm": int((per_clip_med < 0.02).sum()),
        "clip_min_nn_lt_1cm": int((per_clip_min < 0.01).sum()),
        "clip_min_nn_lt_2cm": int((per_clip_min < 0.02).sum()),
    }
    if extra: res.update(extra)
    print(json.dumps(res, indent=1), flush=True)
    return res, per_clip_med, per_clip_min


def main():
    t0 = time.time()
    ids = lambda s: [l.strip() for l in open(f"{R}/HumanML3D/{s}.txt") if l.strip()]
    test_ids = ids("test")
    Q, qown = load_hml(test_ids)
    print("queries", Q.shape, "clips", len(np.unique(qown)), f"{time.time()-t0:.0f}s", flush=True)

    # --- Motion-X++ database (all 7 subsets the pretraining loader enumerates)
    nnm = RunningNN(Q); dmeta = []
    cov = {}
    for cat in ["animation", "haa500", "humman", "idea400", "kungfu", "music", "perform"]:
        zf = zipfile.ZipFile(f"{MX}/{cat}.zip")
        names = [n for n in zf.namelist() if n.endswith(".npy")]
        nfr = nused = 0
        Dc = []
        for n in names:
            j = np.load(io.BytesIO(zf.read(n)))
            if j.ndim != 3 or j.shape[1] < J:
                continue
            j = resample(j[:, :J].astype(np.float64), 30.0, 20.0)
            nfr += len(j)
            f = snippets(j, HOP)
            f = f[np.isfinite(f).all(1)]
            if len(f):
                nused += len(j)
                Dc.append(f); dmeta += [(cat, n.split("/")[-1])] * len(f)
        nnm.add(np.concatenate(Dc)); del Dc
        cov[cat] = {"clips": len(names), "frames20": nfr, "frames_in_ge2s_clips": nused}
        print(cat, cov[cat], f"{time.time()-t0:.0f}s", flush=True)
    print("motionx db", nnm.offset, flush=True)
    d, arg = nnm.result()
    res_mx, med_mx, min_mx = summarize("hml_test_vs_motionxpp", d, qown, len(test_ids), {"motionx_coverage": cov})
    # closest matches for inspection
    order = np.argsort(d)[:15]
    closest = [{"hml_id": test_ids[qown[i]], "mx": dmeta[arg[i]], "rms_m": float(d[i])} for i in order]
    print("closest", json.dumps(closest, indent=0), flush=True)

    # --- positive control: HumanML3D train+val
    tv_ids = ids("train") + ids("val")
    nnt = RunningNN(Q)
    for c0 in range(0, len(tv_ids), 3000):
        Dtv, _ = load_hml(tv_ids[c0:c0 + 3000], hop=HOP)
        nnt.add(Dtv); del Dtv
    print("train+val db", nnt.offset, flush=True)
    d2, arg2 = nnt.result()
    # which test clips (incl. mirrors) share an overlapping frame range with a train/val clip
    idx = {r["new_name"][:-4]: r for r in csv.DictReader(open(f"{R}/index.csv"))}
    def key(i):
        m = i.startswith("M"); b = i[1:] if m else i
        r = idx[b]; return m, r["source_path"], int(r["start_frame"]), int(r["end_frame"])
    tv_src = {}
    for i in tv_ids:
        m, s, a, b = key(i); tv_src.setdefault((m, s), []).append((a, b))
    overlap = []
    for i in test_ids:
        m, s, a, b = key(i)
        ov = any(b == -1 or d_ == -1 or (a < d_ and c < b) for c, d_ in tv_src.get((m, s), []))
        overlap.append(ov)
    overlap = np.asarray(overlap)
    res_tv, med_tv, min_tv = summarize("hml_test_vs_hml_trainval (positive control)", d2, qown, len(test_ids))
    uq = np.unique(qown)
    ov_mask = overlap[uq]
    ctrl = {
        "test_clips_with_frame_overlap": int(ov_mask.sum()),
        "of_those_min_nn_lt_1cm": int((min_tv[ov_mask] < 0.01).sum()),
        "of_those_min_nn_lt_2cm": int((min_tv[ov_mask] < 0.02).sum()),
        "non_overlap_clips_min_nn_lt_1cm": int((min_tv[~ov_mask] < 0.01).sum()),
        "non_overlap_clips": int((~ov_mask).sum()),
        "motionx_min_nn_quantiles_per_clip": {str(q): float(np.quantile(min_mx, q)) for q in [0.0, 0.01, 0.05, 0.5]},
        "trainval_min_nn_quantiles_per_clip_nonoverlap": {str(q): float(np.quantile(min_tv[~ov_mask], q)) for q in [0.0, 0.01, 0.05, 0.5]},
    }
    print(json.dumps(ctrl, indent=1), flush=True)
    np.savez(f"{OUT}/dupcheck_perclip.npz", min_mx=min_mx, med_mx=med_mx, min_tv=min_tv, med_tv=med_tv, ov_mask=ov_mask)
    json.dump({"motionx": res_mx, "trainval": res_tv, "control": ctrl, "closest_motionx": closest},
              open(f"{OUT}/dupcheck_result.json", "w"), indent=1)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
