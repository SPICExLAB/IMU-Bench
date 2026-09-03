"""Trivial 'window statistics' baseline: logistic regression on per-window mean and std of the
raw acc (and optionally gyro) channels -- i.e. exactly the information the per-window z-score
removes. If this alone approaches Inertia-1's HAR score, the gap is input information."""
import sys, json, yaml, numpy as np, torch
sys.path.insert(0, ".")
from imu_lm.data.loaders import make_loaders
from imu_lm.utils.training import remap_labels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score
ds = sys.argv[1]
cfg = yaml.safe_load(open(f"configs/diag_har_norm/diag_har_{ds}__scratch__nonorm.yaml"))   # normalize OFF, filter ON
cfg["data"]["num_workers"] = 4
L = make_loaders(cfg)
def feats(loader):
    X, Y = [], []
    for b in loader:
        if b is None: continue
        x, y = b                                   # x [B, C, T] raw m/s^2 (acc) + rad/s (gyro)
        x = x.float()
        f = torch.cat([x.mean(-1), x.std(-1)], dim=1)   # [B, 2C]
        X.append(f.numpy()); Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y)
Xtr, Ytr = feats(L["train_loader"]); Xva, Yva = feats(L["val_loader"])
labs = sorted(set(Ytr.tolist())); m = {l:i for i,l in enumerate(labs)}
Ytr = np.array([m[v] for v in Ytr]); keep = np.array([v in m for v in Yva]); Xva, Yva = Xva[keep], np.array([m[v] for v in Yva[keep]])
out = {}
for name, cols in [("acc mean+std (6 feats)", list(range(0,3))+list(range(Xtr.shape[1]//2, Xtr.shape[1]//2+3))),
                   ("acc+gyro mean+std (12 feats)", list(range(Xtr.shape[1]))),
                   ("acc mean only (3 feats = gravity dir)", list(range(0,3)))]:
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=None).fit(Xtr[:, cols], Ytr)
    p = clf.predict(Xva[:, cols])
    out[name] = {"macro_f1": float(f1_score(Yva, p, average="macro")), "bal_acc": float(balanced_accuracy_score(Yva, p)), "acc": float((p==Yva).mean())}
    print(ds, name, out[name], flush=True)
json.dump(out, open(f"bench/generated/diag_gravity_{ds}.json","w"), indent=1)
