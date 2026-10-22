"""Shared Phase-4 setup, job builders, raw-array IO and curve analysis (docs/phase4_plan.md)."""
import hashlib
import json
import time

import numpy as np
import torch
import yaml

from causal_subspaces.data.logic import HELDOUT_FAMILIES, TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.upstream import stats as S
from causal_subspaces.upstream.data import build_trials
from causal_subspaces.upstream.runner import Ctx, run_jobs
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

OUT = ROOT / "results/phase4"
FAMS = TRAIN_FAMILIES + HELDOUT_FAMILIES
IDX_ALL = list(range(24))
FLOORS = dict(margin=10.0, c=2.0, rj=1.0, r22=0.5)
STAGE2_CONDS = ["G1", "G2", "G1G2", "Self", "G1G2Self", "Rule", "Fact", "Concl", "NonSummary", "AllMinusG1G2", "AllMinusG1G2Self", "rand0", "rand1", "rand2"]
SPARSE_ALL = [0, 4, 8, 12, 16, 20, 23]
NOOP_IDX = [0, 8, 16, 22]
FREEZES = ["S8", "nonS8", "late", "all", "onlyS8", "S8late", "rnd0", "rnd1", "rnd2", "rnd3", "rnd4"]


def setup():
    cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
    seed_everything(0)
    dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
    model, tok, rev = load_model(cfg["model"], dev)
    d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
    S8 = [(s["layer"], s["head"]) for s in json.load(open(ROOT / "results/phase2/selected_heads.json"))["S8"]]
    return dict(cfg=cfg, tok=tok, ctx=Ctx(model, tok, d_c, S8, dev, idx_hook), rev=rev)


def trial_sets(P, which: str, fmt: str):
    cfg, tok = P["cfg"], P["tok"]
    splits = get_splits(cfg)
    if which == "A":
        return build_trials(tok, build_contents("ordinary"), sorted({e.content_id for e in splits["items"]}), FAMS, fmt, tag="phase4A")
    if which == "val":
        return build_trials(tok, build_contents("ordinary"), sorted({e.content_id for e in splits["val"]}), FAMS, fmt, tag="phase4V")
    if which == "B":
        seed = cfg.get("data_seed", cfg["seed"])
        return build_trials(tok, build_contents("nonword", seed=seed), list(range(48)), FAMS, fmt, max_pairs=200, seed=1, tag="phase4B")
    raise ValueError(which)


def boot_for(trials, nboot=2000):
    return S.Boot([t["pair"] for t in trials], [t["content"] for t in trials], nboot=nboot, seed=7)


def sigma_of(trials):
    return np.array([t["sigma"] for t in trials])


def getters():
    return dict(margin=lambda d: d["margin"], c=lambda d: d["c"], rj=lambda d: d["r"].sum(1), r22=lambda d: d["r"][:, 3])


def terms(trials, base_t, base_s, r, key):
    g = getters()[key]
    return S.transfer_terms(sigma_of(trials), g(r), g(base_t), g(base_s))


def summarize(boot, trials, base_t, base_s, r, keys=("margin", "c", "rj", "r22")):
    out = {}
    sig = sigma_of(trials)
    for k in keys:
        num, den = terms(trials, base_t, base_s, r, k)
        rt = boot.ratio(num, den); gap = float(den.mean())
        out[k] = dict(T=rt["point"], pair_ci=boot.ci(rt["rep_pair"]), content_ci=boot.ci(rt["rep_content"]), loco=[float(rt["loco"].min()), float(rt["loco"].max())],
                      gap=gap, undefined=bool(abs(gap) < FLOORS[k]))
    out["flip"] = float(np.mean(np.sign(r["margin"]) == sig))
    for f in ("nrm", "dc_frac", "npos"):
        if f in r: out[f] = float(np.nanmean(r[f])) if np.isfinite(r[f]).any() else None
    if "verify" in r: out["verify_max"] = float(r["verify"])
    return out


def save_raw(path, trials, base_t, base_s, res):
    a = dict(sigma=sigma_of(trials), pair=np.array([t["pair"] for t in trials]), content=np.array([t["content"] for t in trials]))
    for nm, d in (("tgt", base_t), ("src", base_s)):
        for k, v in d.items(): a[f"__{nm}|{k}"] = v
    for jn, d in res.items():
        for k, v in d.items(): a[f"{jn}|{k}"] = v
    np.savez_compressed(path, **a)


def load_raw(path):
    z = np.load(path, allow_pickle=True); trials = dict(sigma=z["sigma"], pair=z["pair"], content=z["content"])
    bt = {k.split("|")[1]: z[k] for k in z.files if k.startswith("__tgt|")}; bs = {k.split("|")[1]: z[k] for k in z.files if k.startswith("__src|")}
    jobs = {}
    for k in z.files:
        if k.startswith("__") or k in trials: continue
        jn, f = k.rsplit("|", 1); jobs.setdefault(jn, {})[f] = z[k]
    return trials, bt, bs, jobs


def curve(boot, sig, bt, bs, jobs, cond, key, indices=IDX_ALL, prefix="patch"):
    """T(i) with pair and content bounds for one condition. jobs keys 'patch|{cond}|{i}'."""
    g = getters()[key]; pts, lp, lc, up, uc = [], [], [], [], []
    for i in indices:
        r = jobs[f"{prefix}|{cond}|{i}"]; num, den = S.transfer_terms(sig, g(r), g(bt), g(bs))
        rt = boot.ratio(num, den); pts.append(rt["point"]); lp.append(np.percentile(rt["rep_pair"], 2.5)); lc.append(np.percentile(rt["rep_content"], 2.5))
        up.append(np.percentile(rt["rep_pair"], 97.5)); uc.append(np.percentile(rt["rep_content"], 97.5))
    return dict(indices=list(indices), T=np.array(pts), lo_pair=np.array(lp), lo_content=np.array(lc), hi_pair=np.array(up), hi_content=np.array(uc))


def onset_of(cv):
    full = lambda a: np.concatenate([a, np.zeros(24 - len(a))]) if len(a) < 24 else a
    idx = cv["indices"]
    pad = lambda a: np.array([a[idx.index(i)] if i in idx else np.nan for i in range(24)])
    # grid may be sparse (replication B): interpolation is NOT used; missing indices are skipped by the onset rules
    T, lp, lc = pad(cv["T"]), pad(cv["lo_pair"]), pad(cv["lo_content"])
    ok = ~np.isnan(T)
    plat_idx = [i for i in (19, 20, 21, 22) if ok[i]]
    Tm, lpm, lcm = np.where(ok, T, -9.0), np.where(ok, lp, -9.0), np.where(ok, lc, -9.0)
    plat = float(np.median(Tm[plat_idx])); thr = 0.5 * plat
    grid = [i for i in range(23) if ok[i]]
    first = next((i for i in grid if Tm[i] >= thr), None)
    def sus(lo):
        for k, i in enumerate(grid):
            if all(lo[j] >= thr for j in grid[k:]): return i
        return None
    sp, sc = sus(lpm), sus(lcm)
    return dict(plateau=plat, threshold=thr, first_crossing=first, sustained_pair=sp, sustained_content=sc, sustained_both=None if sp is None or sc is None else max(sp, sc))


def job_list_patch(conds, indices, extra=None):
    return [dict(name=f"patch|{c}|{i}", kind="patch", index=i, cond=c, **(extra or {})) for c in conds for i in indices]


def file_hashes(paths):
    return {str(p): hashlib.sha256(open(ROOT / p, "rb").read()).hexdigest() for p in paths}


FROZEN_FILES = ["src/causal_subspaces/upstream/patching.py", "src/causal_subspaces/upstream/stats.py", "src/causal_subspaces/upstream/data.py", "src/causal_subspaces/upstream/runner.py",
                "src/causal_subspaces/upstream/pipeline.py", "scripts/phase4_run.py", "docs/phase4_plan.md"]
