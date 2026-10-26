"""Phase 8 S2: out-of-sample carrier information overlap (protocol section 4). Mappings fitted on development items, evaluated on fresh confirm items.
Also the descriptive note that the answer-letter probe is non-discriminative (random rank-4 baseline)."""
import argparse, json
import numpy as np, torch
from sklearn.linear_model import Ridge
from causal_subspaces.phase7.stage1 import cluster_boot
from causal_subspaces.phase7.stage1b import build_1b
from causal_subspaces.phase8.s2lib import *

ap = argparse.ArgumentParser(); ap.add_argument("--hook", type=int, default=25); a = ap.parse_args()
w = World(a.hook); h = a.hook
d1b = build_1b("M", w.tok)["letters"]; rng = np.random.default_rng(0)
tr_items = [d1b["train"][i] for i in rng.choice(len(d1b["train"]), 600, replace=False)]
fit_items = tr_items + w.dev
Xd, _ = resid_matrices(w, fit_items, all_positions=False); Xc, _ = resid_matrices(w, w.conf, all_positions=False)
Xd, Xc = Xd.cpu(), Xc.cpu()
gd = np.array([i.gold for i in fit_items]); gc = np.array([i.gold for i in w.conf]); grp = np.array([i.group for i in w.conf])
oh = lambda g: np.eye(4)[g]


def r2_dims(yt, yh):
    return 1 - ((yt - yh) ** 2).sum(0) / ((yt - yt.mean(0)) ** 2).sum(0)


def r2_stats(yt, yh):
    d = r2_dims(yt, yh); return float(d.mean()), float(1 - ((yt - yh) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())


def fit_pred(Ptr, Ytr, Pte):
    return Ridge(alpha=1.0).fit(Ptr, Ytr).predict(Pte)


def heldout_cca(Ad, Bd, Ac, Bc, reg=1e-3):
    """Canonical weights fitted on dev, correlations measured on confirm."""
    Am, Bm = Ad.mean(0), Bd.mean(0); A0, B0 = Ad - Am, Bd - Bm
    Ca = A0.T @ A0 / len(A0) + reg * np.eye(A0.shape[1]); Cb = B0.T @ B0 / len(B0) + reg * np.eye(B0.shape[1]); Cab = A0.T @ B0 / len(A0)
    La, Lb = np.linalg.cholesky(Ca), np.linalg.cholesky(Cb)
    M = np.linalg.inv(La) @ Cab @ np.linalg.inv(Lb).T
    u, s, vt = np.linalg.svd(M); wa = np.linalg.inv(La).T @ u; wb = np.linalg.inv(Lb).T @ vt.T
    za, zb = (Ac - Am) @ wa, (Bc - Bm) @ wb
    corr = [float(np.corrcoef(za[:, i], zb[:, i])[0, 1]) for i in range(wa.shape[1])]
    return float(s[0]), corr[0]


res = dict(hook=h, chains={})
Rnd = [haar(4096, 4, 4242 + i).cpu() for i in range(6)]
for s, Us in w.chains.items():
    K = len(Us); Cd = [(Xd @ U.cpu()).numpy() for U in Us]; Cc = [(Xc @ U.cpu()).numpy() for U in Us]
    rows = dict(pairs={}, baseline_random={}, baseline_class={})
    for i in range(K):
        # baselines: coordinates of carrier i predicted from random rank-4 coordinates, and from the answer letter only
        rb = [r2_stats(Cc[i], fit_pred((Xd @ R).numpy(), Cd[i], (Xc @ R).numpy())) for R in Rnd]
        rows["baseline_random"][i + 1] = [float(np.mean([r[0] for r in rb])), float(np.mean([r[1] for r in rb]))]
        rows["baseline_class"][i + 1] = list(r2_stats(Cc[i], fit_pred(oh(gd), Cd[i], oh(gc))))
        for j in range(K):
            if i == j:
                continue
            yh = fit_pred(Cd[j], Cd[i], Cc[j]); pm, pv = r2_stats(Cc[i], yh)
            lo, hi = cluster_boot(lambda ix: r2_stats(Cc[i][ix], yh[ix])[0], grp, 200, 0)
            rows["pairs"][f"{i+1}<-{j+1}"] = dict(r2_dim_mean=pm, r2_var_weighted=pv, ci=[lo, hi])
    cca = {}
    for i in range(K):
        for j in range(i + 1, K):
            cca[f"{i+1}-{j+1}"] = heldout_cca(Cd[i], Cd[j], Cc[i], Cc[j])
    rows["heldout_top_cca"] = {k: dict(dev_corr=v[0], confirm_corr=v[1]) for k, v in cca.items()}
    pm = [v["r2_dim_mean"] for v in rows["pairs"].values()]; pv = [v["r2_var_weighted"] for v in rows["pairs"].values()]
    rows["summary"] = dict(median_r2_dim_mean=float(np.median(pm)), median_r2_var_weighted=float(np.median(pv)),
                           median_baseline_random_dim=float(np.median([v[0] for v in rows["baseline_random"].values()])),
                           median_baseline_class_dim=float(np.median([v[0] for v in rows["baseline_class"].values()])),
                           median_cca_confirm=float(np.median([v["confirm_corr"] for v in rows["heldout_top_cca"].values()])),
                           median_cca_dev=float(np.median([v["dev_corr"] for v in rows["heldout_top_cca"].values()])))
    res["chains"][s] = rows
    print(s, rows["summary"], flush=True)
json.dump(res, open(R2 / f"overlap_oos_h{h}.json", "w"), indent=1, default=float)
print("S2 OVERLAP DONE", h)
