"""Phase 0.6 Part A: is logical validity decodable from the residual at the hook location after removing the frozen consensus
verdict direction? Linear probes trained on the optimisation pool, tested on held-out items/families/nonwords and on other
answer formats; random direction-removal and MVS-span controls; INLP. Usage: python scripts/phase06_decodability.py"""
import json

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5, phase06 as P6
from causal_subspaces.data.controls import build_controls as get_controls
from causal_subspaces.experiment import get_reference, get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.checkpoint import load_checkpoint
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

OUT = ROOT / "results" / "phase06"
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
FMTS = list(P5.FORMATS)
model, tok, rev = load_model(cfg["model"], dev)
splits = get_splits(cfg)
ref = get_reference(model, tok, cfg, rev, dev)
d_c = P6.load_consensus()
rng = np.random.default_rng(0)
tr = [splits["train"][i] for i in rng.choice(len(splits["train"]), 1000, replace=False)]
data = {"train": tr, "val": splits["val"], "items": splits["items"], "families": splits["families"], "nonwords": splits["nonwords"]}

# --- features -------------------------------------------------------------------------------------------------------
feats, ys, yl = {}, {}, {}
for fmt in FMTS:
    for sp, ex in data.items():
        f = P6.collect_residuals(model, tok, [P5.build_prompt(fmt, e.syllogism) for e in ex], idx, device=dev, batch_size=32)
        for pos in ("last", "concl"):
            feats[(pos, fmt, sp)] = torch.tensor(f[pos], device=dev)
        v = np.array([e.valid for e in ex])
        ys[(fmt, sp)] = v
        # literal-first-label target: is the CORRECT answer the label listed first in the instruction?
        lit_first_correct = (P5.FORMATS[fmt]["valid"] == P5.FORMATS[fmt]["lit_first"])
        yl[(fmt, sp)] = v if lit_first_correct else ~v
    print("collected", fmt, flush=True)
ctrl = {s: get_controls(s, 128, ("high_school_biology",), 4) for s in (101, 0)}
cf, cy = {}, {}
for name in ("language_localizer", "prontoqa"):
    for s in (101, 0):
        items = ctrl[s][name]["items"]
        cf[(name, s)] = torch.tensor(P6.collect_residuals(model, tok, [t for t, _ in items], idx, device=dev)["last"], device=dev)
        cy[(name, s)] = np.array([g == "1" for _, g in items])
print("controls collected", flush=True)
del model; torch.cuda.empty_cache()

# --- directions to remove -------------------------------------------------------------------------------------------
Us = {s: load_checkpoint(ROOT / f"results/mvs_rank2_seed{s}/checkpoints/step_1600.pt")["U"].to(dev) for s in (0, 1, 2)}
Qm = ref["Qm"].to(dev)
dcv = d_c.to(dev)[:, None]
union = torch.linalg.qr(torch.cat(list(Us.values()), 1))[0]
def rand_dirs(k, n, seed0):
    return [P6.random_orth_subspace(Qm.cpu(), torch.zeros(Qm.shape[0]), k, seed0 + i).to(dev) for i in range(n)]
# random controls here are NOT forced orthogonal to d_c (d is the zero vector => no projection): "random direction removal"
TRANSFORMS = {"full": [None], "rm_dc": [dcv], "rm_rand1": rand_dirs(1, 20, 100), "rm_topPC1": [Qm[:, :1]],
              "rm_mvs_s0": [Us[0]], "rm_rand2": rand_dirs(2, 10, 200), "rm_dc+rand1": [torch.linalg.qr(torch.cat([dcv, r], 1))[0] for r in rand_dirs(1, 10, 300)],
              "rm_mvs_union(6)": [union], "rm_rand6": rand_dirs(6, 5, 400)}
for k, v in TRANSFORMS.items():
    print(k, [tuple(x.shape) if x is not None else None for x in v][:1], len(v))

def XY(pos, fmt, sp, D):
    return P6.project_out(feats[(pos, fmt, sp)], D)

# l2 chosen once on validation (full features, num_orig)
best, grid = None, {}
for l2 in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
    p = P6.fit_probe(feats[("last", "num_orig", "train")], torch.tensor(ys[("num_orig", "train")], device=dev), l2)
    grid[l2] = P6.probe_eval(p, feats[("last", "num_orig", "val")], ys[("num_orig", "val")])["acc"]
l2 = max(grid, key=lambda k: grid[k]); print("l2 grid (val acc)", grid, "->", l2, flush=True)

results = {"l2_grid_val_acc": {str(k): v for k, v in grid.items()}, "l2": l2, "probes": {}, "n_train": {"validity": 1000}}
SPL = ("items", "families", "nonwords")
for pos in ("last", "concl"):
    for regime in ("train_num_orig", "train_pooled"):
        for tname, Ds in TRANSFORMS.items():
            acc = {}
            for D in Ds:
                if regime == "train_num_orig":
                    Xtr, ytr = XY(pos, "num_orig", "train", D), torch.tensor(ys[("num_orig", "train")], device=dev)
                else:
                    Xtr = torch.cat([XY(pos, f, "train", D) for f in FMTS]); ytr = torch.tensor(np.concatenate([ys[(f, "train")] for f in FMTS]), device=dev)
                p = P6.fit_probe(Xtr, ytr, l2)
                for sp in SPL:
                    for fmt in FMTS:
                        acc.setdefault(f"{sp}|{fmt}", []).append(P6.probe_eval(p, XY(pos, fmt, sp, D), ys[(fmt, sp)]))
            results["probes"][f"{pos}|{regime}|{tname}"] = {k: dict(acc=float(np.mean([r["acc"] for r in v])), acc_sd=float(np.std([r["acc"] for r in v])),
                                                                     auc=float(np.mean([r["auc"] for r in v])), n=len(v)) for k, v in acc.items()}
        print("probes done", pos, regime, flush=True)

# literal-answer-token target (pooled training): does the residual (minus d_c) encode WHICH TOKEN to emit, or validity?
lit = {}
for tname in ("full", "rm_dc"):
    D = TRANSFORMS[tname][0]
    Xtr = torch.cat([XY("last", f, "train", D) for f in FMTS]); ytr = torch.tensor(np.concatenate([yl[(f, "train")] for f in FMTS]), device=dev)
    pv = P6.fit_probe(Xtr, torch.tensor(np.concatenate([ys[(f, "train")] for f in FMTS]), device=dev), l2)
    pl = P6.fit_probe(Xtr, ytr, l2)
    lit[tname] = {f"{sp}|{f}": dict(validity_probe=P6.probe_eval(pv, XY("last", f, sp, D), ys[(f, sp)]), literal_probe=P6.probe_eval(pl, XY("last", f, sp, D), yl[(f, sp)]),
                                    validity_probe_on_literal_target=P6.probe_eval(pv, XY("last", f, sp, D), yl[(f, sp)])) for sp in ("families",) for f in FMTS}
results["literal_vs_validity"] = lit

# d_c coordinate alone as a score
coord = {}
for fmt in FMTS:
    for sp in SPL:
        s = (feats[("last", fmt, sp)] @ dcv).squeeze(1).cpu().numpy()
        coord[f"{sp}|{fmt}"] = P5.auc(s, ys[(fmt, sp)])
for name in ("language_localizer", "prontoqa"):
    coord[f"{name}"] = P5.auc((cf[(name, 0)] @ dcv).squeeze(1).cpu().numpy(), cy[(name, 0)])
results["dc_coordinate_auc"] = coord

# controls: localizer & PrOntoQA decodability (train on seed-101 items, test on seed-0 items)
cres = {}
for name in ("language_localizer", "prontoqa"):
    for tname in ("full", "rm_dc", "rm_rand1", "rm_mvs_s0"):
        accs = []
        for D in TRANSFORMS[tname]:
            p = P6.fit_probe(P6.project_out(cf[(name, 101)], D), torch.tensor(cy[(name, 101)], device=dev), l2)
            accs.append(P6.probe_eval(p, P6.project_out(cf[(name, 0)], D), cy[(name, 0)]))
        cres[f"{name}|{tname}"] = dict(acc=float(np.mean([a["acc"] for a in accs])), auc=float(np.mean([a["auc"] for a in accs])))
results["control_probes"] = cres

# INLP on validity (train num_orig, last position)
tests = {f"{sp}|{f}": (feats[("last", f, sp)], ys[(f, sp)]) for sp in SPL for f in ("num_orig", "num_rev", "yn")}
results["inlp_validity"] = P6.inlp(feats[("last", "num_orig", "train")], torch.tensor(ys[("num_orig", "train")], device=dev), tests, l2, 12)
results["inlp_localizer"] = P6.inlp(cf[("language_localizer", 101)], torch.tensor(cy[("language_localizer", 101)], device=dev),
                                    {"localizer": (cf[("language_localizer", 0)], cy[("language_localizer", 0)])}, l2, 8)
json.dump(results, open(OUT / "decodability.json", "w"), indent=1)
print("saved")
