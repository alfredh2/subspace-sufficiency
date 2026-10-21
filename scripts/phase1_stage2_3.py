"""Phase 1 Stages 2-3 on the layers selected in Stage 1: native vs null concentration (|Delta| only), bootstrap / split-half checks,
cross-task rankings, signed top-k reconstruction, cancellation. Nulls are kept as separate families:
  true_haar (QR), haar_equiv_sampler (exact two-frame sampler), structured (block-permute-Haar; NOT Haar), shuffled_alignment.
Usage: python scripts/phase1_stage2_3.py"""
import json
import time

import numpy as np
import torch
import yaml
from scipy.stats import ks_2samp, spearmanr

from causal_subspaces import phase05 as P5
from causal_subspaces.data.controls import build_controls
from causal_subspaces.experiment import get_controls, get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.neurons import concentration as C
from causal_subspaces.neurons.decompose import layerwise_pass, unit_gain
from causal_subspaces.rotations import haar as H
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

T0 = time.time()
OUT = ROOT / "results" / "phase1"
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
S1 = json.load(open(OUT / "stage1.json"))
LS = S1["selection"]["layers"]
dirs = torch.load(OUT / "directions.pt")
N_HAAR, N_STRUCT, N_SAMPLER, N_SHUF, N_BOOT, K_GRID = 20, 20, 1000, 1000, 200, [4, 16, 64, 256, 512, 2458]
m = 12288

# ---------------- collect last-position / conclusion-end activations for the selected layers ----------------------------
model, tok, rev = load_model(cfg["model"], dev)
splits, controls0 = get_splits(cfg), get_controls(cfg)
rng = np.random.default_rng(0)
tr = splits["train"]
val, inv = [e for e in tr if e.valid], [e for e in tr if not e.valid]
pick = [val[i] for i in rng.choice(len(val), 500, replace=False)] + [inv[i] for i in rng.choice(len(inv), 500, replace=False)]   # identical to Stage 1
ctrl101 = build_controls(101, 256, ("anatomy", "philosophy"), 16)
ho = [e for sp in ("items", "families", "nonwords") for e in splits[sp]]
sets = {
    "logic": dict(tr=([P5.build_prompt("num_orig", e.syllogism) for e in pick], np.array([e.valid for e in pick])),
                  te=([P5.build_prompt("num_orig", e.syllogism) for e in ho], np.array([e.valid for e in ho]))),
}
for name, key in (("localizer", "language_localizer"), ("prontoqa", "prontoqa")):
    sets[name] = dict(tr=([t for t, _ in ctrl101[key]["items"]], np.array([g == "1" for _, g in ctrl101[key]["items"]])),
                      te=([t for t, _ in controls0[key]["items"]], np.array([g == "1" for _, g in controls0[key]["items"]])))
dcons = dirs["consensus"].to(dev)
acts, accs = {}, {}
for name, sp in sets.items():
    for part in ("tr", "te"):
        out = layerwise_pass(model, tok, sp[part][0], sp[part][1], dcons, idx_hook, device=dev, keep_acts_layers=LS)
        acts[(name, part)] = {l: out["acts"][l].float().to(dev) for l in LS}; accs[(name, part)] = torch.tensor(sp[part][1], device=dev)
for part in ("tr", "te"):  # secondary position (logic only)
    out = layerwise_pass(model, tok, sets["logic"][part][0], sets["logic"][part][1], dcons, idx_hook, device=dev, keep_acts_layers=LS, position="concl")
    acts[("logic_concl", part)] = {l: out["acts"][l].float().to(dev) for l in LS}; accs[("logic_concl", part)] = torch.tensor(sets["logic"][part][1], device=dev)
G = {dn: {l: unit_gain(model, l, d) for l in LS} for dn, d in dirs.items()}
del model; torch.cuda.empty_cache()
print(f"activations collected ({time.time() - T0:.0f}s)", flush=True)

def cdelta(A, y):
    return A[y].mean(0) - A[~y].mean(0)

DELTA = {k: {l: cdelta(acts[k + ("tr",)[1:]][l] if False else acts[(k, "tr")][l], accs[(k, "tr")]) for l in LS} for k in ("logic", "localizer", "prontoqa", "logic_concl")}
ANALYSES = {"cons/logic": ("logic", "consensus"), "cons/localizer": ("localizer", "consensus"), "cons/prontoqa": ("prontoqa", "consensus"),
            "s0/logic": ("logic", "shared_s0"), "s1/logic": ("logic", "shared_s1"), "s2/logic": ("logic", "shared_s2"), "cons/logic_concl": ("logic_concl", "consensus")}
DK = ["logic", "localizer", "prontoqa", "logic_concl"]; GK = ["consensus", "shared_s0", "shared_s1", "shared_s2"]
V = {l: torch.stack([DELTA[k][l] for k in DK] + [G[g][l] for g in GK], 1) for l in LS}            # (m, 8)
cols = {a: (DK.index(dk), len(DK) + GK.index(gk)) for a, (dk, gk) in ANALYSES.items()}

def mb(D):  # D: (n, M) pooled; returns dict of lists
    s = C.concentration_summary(D)
    return {k: [float(x) for x in v.reshape(-1).cpu()] for k, v in s.items()}

def per_layer_mb(Dl):  # Dl: {layer: (n, m)}
    out = {}
    for l, D in Dl.items():
        s = C.concentration_summary(D)
        out[str(l)] = {k: [float(x) for x in v.reshape(-1).cpu()] for k, v in s.items()}
    return out

res = {"layers": LS, "analyses": {}}
for a, (ci, gi) in cols.items():
    nat = {l: (V[l][:, ci] * V[l][:, gi]).unsqueeze(0) for l in LS}
    res["analyses"][a] = dict(native=dict(pooled=mb(torch.cat(list(nat.values()), 1)), per_layer=per_layer_mb(nat),
                                          marginal_g=mb(torch.cat([V[l][:, gi].abs().unsqueeze(0) for l in LS], 1)),
                                          marginal_delta=mb(torch.cat([V[l][:, ci].abs().unsqueeze(0) for l in LS], 1))), nulls={})

# ---------------- nulls ---------------------------------------------------------------------------------------------------
def collect_null(fam, per_draw):   # per_draw: list over draws of {analysis: {layer: (m,)}}
    for a in cols:
        Dl = {l: torch.stack([d[a][l] for d in per_draw]) for l in LS}
        res["analyses"][a]["nulls"][fam] = dict(pooled=mb(torch.cat(list(Dl.values()), 1)), per_layer=per_layer_mb(Dl), n=len(per_draw))

t = time.time()
draws = [dict((a, {}) for a in cols) for _ in range(N_HAAR)]
for li, l in enumerate(LS):
    for r in range(N_HAAR):
        R = H.haar_orthogonal(m, 10_000 * (l + 1) + r, dev)
        RV = R @ V[l]
        for a, (ci, gi) in cols.items():
            draws[r][a][l] = RV[:, ci] * RV[:, gi]
        del R
collect_null("true_haar", draws); print(f"true Haar done ({time.time() - t:.0f}s)", flush=True)

t = time.time()
draws = [dict((a, {}) for a in cols) for _ in range(N_STRUCT)]
for l in LS:
    for r in range(N_STRUCT):
        S = H.StructuredRotation(m, 20_000 * (l + 1) + r, device=dev)
        RV = S.apply(V[l].T).T
        for a, (ci, gi) in cols.items():
            draws[r][a][l] = RV[:, ci] * RV[:, gi]
collect_null("structured_block_permute_haar", draws); print(f"structured done ({time.time() - t:.0f}s)", flush=True)

t = time.time()
for a, (ci, gi) in cols.items():
    Dl, Ds = {}, {}
    for l in LS:
        Rd, Rg = H.two_frame_pair(V[l][:, ci], V[l][:, gi], N_SAMPLER, seed=30_000 + l)
        Dl[l] = Rd * Rg
        gen = torch.Generator(device=dev).manual_seed(40_000 + l)
        perm = torch.rand(N_SHUF, m, generator=gen, device=dev).argsort(1)
        Ds[l] = V[l][:, ci].unsqueeze(0) * V[l][:, gi][perm]
    res["analyses"][a]["nulls"]["haar_equivalent_sampler"] = dict(pooled=mb(torch.cat(list(Dl.values()), 1)), per_layer=per_layer_mb(Dl), n=N_SAMPLER)
    res["analyses"][a]["nulls"]["shuffled_alignment"] = dict(pooled=mb(torch.cat(list(Ds.values()), 1)), per_layer=per_layer_mb(Ds), n=N_SHUF)
print(f"sampler + shuffle done ({time.time() - t:.0f}s)", flush=True)

# sampler validation against true Haar (single layer, 100 true draws)
l0 = LS[0]; ci, gi = cols["cons/logic"]
tru = []
for r in range(100):
    R = H.haar_orthogonal(m, 50_000 + r, dev); RV = R @ V[l0]
    tru.append(float(C.topk_abs_share(RV[:, ci] * RV[:, gi], 123)))
Rd, Rg = H.two_frame_pair(V[l0][:, ci], V[l0][:, gi], N_SAMPLER, seed=7)
smp = C.topk_abs_share(Rd * Rg, 123).cpu().numpy()
ks = ks_2samp(tru, smp)
res["sampler_validation"] = dict(layer=l0, ks_stat=float(ks.statistic), ks_p=float(ks.pvalue), true_mean=float(np.mean(tru)), sampler_mean=float(smp.mean()), n_true=100)
print("sampler validation", res["sampler_validation"], flush=True)

# ---------------- bootstrap and split-half (logic, consensus) ---------------------------------------------------------------
A_tr, y_tr = acts[("logic", "tr")], accs[("logic", "tr")]
g_c = G["consensus"]
def pooled_native(Adict, y, idx_rows=None):
    return torch.cat([cdelta(Adict[l] if idx_rows is None else Adict[l][idx_rows], y if idx_rows is None else y[idx_rows]) * g_c[l] for l in LS])
bg = torch.Generator(device=dev).manual_seed(0)
pos_i, neg_i = y_tr.nonzero()[:, 0], (~y_tr).nonzero()[:, 0]
boots = {"top1%": [], "n_eff": []}
for b in range(N_BOOT):
    rows = torch.cat([pos_i[torch.randint(len(pos_i), (len(pos_i),), generator=bg, device=dev)], neg_i[torch.randint(len(neg_i), (len(neg_i),), generator=bg, device=dev)]])
    s = C.concentration_summary(pooled_native(A_tr, y_tr, rows).unsqueeze(0))
    boots["top1%"].append(float(s["top1%"])); boots["n_eff"].append(float(s["n_eff"]))
res["bootstrap_native_logic"] = {k: dict(mean=float(np.mean(v)), lo=float(np.quantile(v, .025)), hi=float(np.quantile(v, .975))) for k, v in boots.items()}
perm = torch.randperm(len(pos_i), generator=torch.Generator().manual_seed(1)).to(dev), torch.randperm(len(neg_i), generator=torch.Generator().manual_seed(2)).to(dev)
hA = torch.cat([pos_i[perm[0][:len(pos_i) // 2]], neg_i[perm[1][:len(neg_i) // 2]]]); hB = torch.cat([pos_i[perm[0][len(pos_i) // 2:]], neg_i[perm[1][len(neg_i) // 2:]]])
DA, DB = pooled_native(A_tr, y_tr, hA), pooled_native(A_tr, y_tr, hB)
k1 = int(round(0.01 * DA.numel()))
topA = DA.abs().topk(k1).indices
res["split_half_logic"] = dict(k=k1, share_of_B_mass_in_A_topk=float(DB.abs()[topA].sum() / DB.abs().sum()), share_of_B_mass_in_B_topk=float(DB.abs().topk(k1).values.sum() / DB.abs().sum()),
                               random_expectation=k1 / DA.numel(), rank_spearman_A_B=float(spearmanr(DA.abs().cpu(), DB.abs().cpu()).correlation))

# ---------------- cross-task rankings ------------------------------------------------------------------------------------------
pool = {k: torch.cat([DELTA[k][l] * g_c[l] for l in LS]) for k in ("logic", "localizer", "prontoqa")}
res["cross_task"] = {}
for a_, b_ in (("logic", "localizer"), ("logic", "prontoqa"), ("localizer", "prontoqa")):
    top = lambda x: set(x.abs().topk(k1).indices.tolist())
    inter = len(top(pool[a_]) & top(pool[b_]))
    res["cross_task"][f"{a_}~{b_}"] = dict(spearman_signed=float(spearmanr(pool[a_].cpu(), pool[b_].cpu()).correlation),
                                           spearman_abs=float(spearmanr(pool[a_].abs().cpu(), pool[b_].abs().cpu()).correlation),
                                           top1pct_overlap=inter, top1pct_jaccard=inter / (2 * k1 - inter), expected_overlap_independent=k1 * k1 / pool[a_].numel())

# ---------------- Stage 3: signed reconstruction / cancellation -----------------------------------------------------------------
def S_of(Adict, gvec):  # per-example unit contributions, pooled over layers: (n, M)
    return torch.cat([Adict[l] * gvec[l] for l in LS], 1)

res["stage3"] = {}
for task in ("logic", "localizer", "prontoqa"):
    Str, Ste = S_of(acts[(task, "tr")], g_c), S_of(acts[(task, "te")], g_c)
    Dtr = pooled_native(acts[(task, "tr")], accs[(task, "tr")])
    Dte = pooled_native(acts[(task, "te")], accs[(task, "te")])
    order = Dtr.abs().argsort(descending=True)
    out = {"net_D_train": float(Dtr.sum()), "net_D_test": float(Dte.sum()), "cancellation_train": C.cancellation_index(Dtr), "cancellation_test": C.cancellation_index(Dte),
           "per_layer_cancellation_test": {str(l): C.cancellation_index(cdelta(acts[(task, "te")][l], accs[(task, "te")]) * g_c[l]) for l in LS},
           "mean_per_example_kappa_test": float(C.per_example_kappa(Ste).mean()), "native": {}, "random": {}, "true_haar_top": {}}
    gen = np.random.default_rng(5)
    for k in K_GRID:
        idx = order[:k]
        out["native"][str(k)] = dict(**C.signed_reconstruction(Dte, idx), **C.per_example_r2(Str, Ste, idx), abs_mass_train=float(Dtr.abs()[idx].sum() / Dtr.abs().sum()),
                                     abs_mass_test=float(Dte.abs()[idx].sum() / Dte.abs().sum()))
        rr = []
        for _ in range(20):
            ridx = torch.as_tensor(gen.choice(Dtr.numel(), k, replace=False), device=dev)
            rr.append((C.signed_reconstruction(Dte, ridx)["frac"], C.per_example_r2(Str, Ste, ridx)["r2"]))
        out["random"][str(k)] = dict(frac_mean=float(np.mean([x[0] for x in rr])), frac_max=float(np.max([x[0] for x in rr])), r2_mean=float(np.mean([x[1] for x in rr])), r2_max=float(np.max([x[1] for x in rr])), n=20)
    if task == "logic":   # rotated reconstruction (true Haar, 10 draws; the per-example version needs R applied to stored activations)
        accum = {str(k): [] for k in K_GRID}
        for r in range(10):
            Str_r, Ste_r, gl_tr = [], [], []
            for l in LS:
                R = H.haar_orthogonal(m, 10_000 * (l + 1) + r, dev); Rg = R @ g_c[l]
                Str_r.append((acts[("logic", "tr")][l] @ R.T) * Rg); Ste_r.append((acts[("logic", "te")][l] @ R.T) * Rg); del R
            Str_r, Ste_r = torch.cat(Str_r, 1), torch.cat(Ste_r, 1)
            ytr, yte = accs[("logic", "tr")], accs[("logic", "te")]
            Dtr_r = Str_r[ytr].mean(0) - Str_r[~ytr].mean(0); Dte_r = Ste_r[yte].mean(0) - Ste_r[~yte].mean(0)
            order_r = Dtr_r.abs().argsort(descending=True)
            for k in K_GRID:
                accum[str(k)].append((C.signed_reconstruction(Dte_r, order_r[:k])["frac"], C.per_example_r2(Str_r, Ste_r, order_r[:k])["r2"]))
            del Str_r, Ste_r
        out["true_haar_top"] = {k: dict(frac_mean=float(np.mean([x[0] for x in v])), frac_max=float(np.max([x[0] for x in v])), r2_mean=float(np.mean([x[1] for x in v])), r2_max=float(np.max([x[1] for x in v])), n=len(v)) for k, v in accum.items()}
    res["stage3"][task] = out
    print("stage3", task, {k: round(v["frac"], 3) for k, v in out["native"].items()}, flush=True)

res["seconds"] = time.time() - T0
json.dump(res, open(OUT / "stage2_3.json", "w"), indent=1)
print("saved", res["seconds"])
