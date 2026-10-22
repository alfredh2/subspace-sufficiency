"""Phase 2 Stage 1: head-wise decomposition of the attention writes into the verdict coordinate (layers 0-22, 736 heads), concentration vs layer-matched
nulls (alternative coordinate decompositions, NOT heads), held-out reconstruction, cancellation, cross-task overlap, gate S1, frozen selection of S8/S16.
Uses the model's own o_proj inputs only (no attention recompute). Usage: python scripts/phase2_stage1_heads.py"""
import json
import time

import numpy as np
import torch
import yaml
from scipy.stats import ks_2samp, spearmanr

from causal_subspaces import phase05 as P5
from causal_subspaces.data.controls import build_controls
from causal_subspaces.experiment import get_controls, get_splits
from causal_subspaces.heads import nulls as N
from causal_subspaces.heads.decompose import class_diff_heads, head_dims, head_pass
from causal_subspaces.models.loading import load_model
from causal_subspaces.neurons import concentration as C
from causal_subspaces.rotations.haar import haar_orthogonal
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

T0 = time.time()
OUT = ROOT / "results/phase2"
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
dirs = {k: v.to(dev) for k, v in torch.load(ROOT / "results/phase1/directions.pt").items()}
model, tok, rev = load_model(cfg["model"], dev)
H, HD, KV = head_dims(model)
LAYERS = list(range(idx_hook)); NL = len(LAYERS)
KS = [1, 2, 4, 8, 16, 32, 64]
splits, controls0 = get_splits(cfg), get_controls(cfg)
rng = np.random.default_rng(0)
tr = splits["train"]
val, inv = [e for e in tr if e.valid], [e for e in tr if not e.valid]
pick = [val[i] for i in rng.choice(len(val), 500, replace=False)] + [inv[i] for i in rng.choice(len(inv), 500, replace=False)]   # same 1000 training prompts as Phase 1
ctrl101 = build_controls(101, 256, ("anatomy", "philosophy"), 16)
ho = [e for sp in ("items", "families", "nonwords") for e in splits[sp]]
SETS = {"logic": dict(tr=([P5.build_prompt("num_orig", e.syllogism) for e in pick], np.array([e.valid for e in pick])),
                      te=([P5.build_prompt("num_orig", e.syllogism) for e in ho], np.array([e.valid for e in ho])))}
for name, key in (("localizer", "language_localizer"), ("prontoqa", "prontoqa")):
    SETS[name] = dict(tr=([t for t, _ in ctrl101[key]["items"]], np.array([g == "1" for _, g in ctrl101[key]["items"]])),
                      te=([t for t, _ in controls0[key]["items"]], np.array([g == "1" for _, g in controls0[key]["items"]])))

# ---- passes (consensus direction for all tasks; other directions for logic; conclusion-end position for logic) -------------------
P = {}
for name, sp in SETS.items():
    for part in ("tr", "te"):
        P[("cons", name, part)] = head_pass(model, tok, sp[part][0], sp[part][1], dirs["consensus"], idx_hook, device=dev, keep_z=(name == "logic"))
for dn in ("shared_s0", "shared_s1", "shared_s2"):
    for part in ("tr", "te"):
        P[(dn, "logic", part)] = head_pass(model, tok, SETS["logic"][part][0], SETS["logic"][part][1], dirs[dn], idx_hook, device=dev)
for part in ("tr", "te"):
    P[("cons", "logic_concl", part)] = head_pass(model, tok, SETS["logic"][part][0], SETS["logic"][part][1], dirs["consensus"], idx_hook, device=dev, position="concl")
ACC = {(n, p): SETS["logic" if n == "logic_concl" else n][p][1] for n in ("logic", "localizer", "prontoqa", "logic_concl") for p in ("tr", "te")}
print(f"passes done ({time.time() - T0:.0f}s)", flush=True)

res = {"layers": LAYERS, "H": H, "HD": HD}
# ---- identity ------------------------------------------------------------------------------------------------------------------
idn = {}
for k, o in P.items():
    err = np.abs(o["S"].sum(-1) - o["attn_dot"]).max()
    idn["/".join(k)] = dict(max_abs_err=float(err), attn_dot_std=float(o["attn_dot"].std()), rel=float(err / max(o["attn_dot"].std(), 1e-9)))
res["identity"] = idn
res["identity_ok"] = bool(all(v["rel"] <= 0.02 for v in idn.values()))
print("identity max rel", max(v["rel"] for v in idn.values()), flush=True)

# ---- native head class differences ---------------------------------------------------------------------------------------------
D = {k: class_diff_heads(P[k + ("tr",)]["S"], ACC[(k[1], "tr")]) for k in [("cons", n) for n in ("logic", "localizer", "prontoqa", "logic_concl")] + [(d_, "logic") for d_ in ("shared_s0", "shared_s1", "shared_s2")]}
Dte = {k: class_diff_heads(P[k + ("te",)]["S"], ACC[(k[1], "te")]) for k in D}
Dl = D[("cons", "logic")]                                         # (L, H)
res["native_logic"] = dict(delta=Dl.tolist(), abs=np.abs(Dl).tolist(), layer_signed=Dl.sum(1).tolist(), layer_abs_mass=np.abs(Dl).sum(1).tolist(),
                           total_signed=float(Dl.sum()), total_abs=float(np.abs(Dl).sum()), kappa=float(np.abs(Dl).sum() / abs(Dl.sum())),
                           held_out_total_signed=float(Dte[("cons", "logic")].sum()))
for k, v in D.items():
    res.setdefault("totals", {})["/".join(k)] = dict(signed=float(v.sum()), abs=float(np.abs(v).sum()), held_out_signed=float(Dte[k].sum()))
print("native logic: total signed", Dl.sum(), "abs", np.abs(Dl).sum(), "layer signed 19-22", Dl.sum(1)[19:], flush=True)

# ---- concentration: native vs nulls (layer-matched) -----------------------------------------------------------------------------
def pooled_stats(Dpool: torch.Tensor):    # (n, 736)
    s = C.topk_abs_share(Dpool, KS)
    return dict(topk=s.reshape(-1, len(KS)).T.tolist(), n_eff=[float(x) for x in C.n_eff(Dpool).reshape(-1)], gini=[float(x) for x in C.gini(Dpool).reshape(-1)])

def within_layer(Dlayers: torch.Tensor, native_mass):      # Dlayers: (n, L, H) -> mass-weighted mean over layers 19-22 of top-2 share and N_eff
    sel = list(range(19, 23)); w = torch.tensor(native_mass[sel], dtype=torch.float64); w = w / w.sum()
    t2 = C.topk_abs_share(Dlayers[:, sel], 2); ne = C.n_eff(Dlayers[:, sel])
    return dict(top2=[float(x) for x in (t2 * w).sum(-1)], n_eff=[float(x) for x in (ne * w).sum(-1)])

ANALYSES = {"cons/logic": ("cons", "logic"), "cons/localizer": ("cons", "localizer"), "cons/prontoqa": ("cons", "prontoqa"), "s0/logic": ("shared_s0", "logic"),
            "s1/logic": ("shared_s1", "logic"), "s2/logic": ("shared_s2", "logic"), "cons/logic_concl": ("cons", "logic_concl")}
res["concentration"] = {}
n_haar, n_samp = 20, 1000
for aname, (dn, task) in ANALYSES.items():
    dkey = ("cons" if dn == "cons" else dn, task)
    dirname = "consensus" if dn == "cons" else dn
    ocls = P[(dn, task, "tr")] if (dn, task, "tr") in P else P[("cons", task, "tr")]
    delta_l = (ocls["mean_acc"] - ocls["mean_rej"]).to(dev)               # (L, H*HD)
    nat = torch.tensor(D[dkey], dtype=torch.float32)                      # (L, H)
    mass = np.abs(D[dkey]).sum(1)
    fam = {}
    fam["haar_block_true"] = torch.zeros(n_haar, NL, H)
    fam["haar_block_sampler"] = torch.zeros(n_samp, NL, H)
    fam["permuted_block"] = torch.zeros(n_samp, NL, H)
    fam["head_shuffled"] = torch.zeros(n_samp, NL, H)
    for li, l in enumerate(LAYERS):
        g = ocls["gains"][l]
        for r in range(n_haar):
            fam["haar_block_true"][r, li] = N.haar_block_delta(delta_l[li], g, HD, 10_000 * (l + 1) + r, dev).cpu()
        fam["haar_block_sampler"][:, li] = N.haar_block_delta_sampler(delta_l[li], g, HD, n_samp, 30_000 + l).cpu()
        for r in range(n_samp):
            fam["permuted_block"][r, li] = N.permuted_block_delta(delta_l[li], g, HD, 50_000 + 1000 * l + r).cpu()
            fam["head_shuffled"][r, li] = N.head_shuffled_delta(delta_l[li], g, HD, 70_000 + 1000 * l + r).cpu()
    out = dict(native=dict(pooled=pooled_stats(nat.reshape(1, -1)), within=within_layer(nat.unsqueeze(0), mass), per_layer_top2=C.topk_abs_share(nat, 2).tolist(), per_layer_n_eff=C.n_eff(nat).tolist(), layer_mass=mass.tolist()), nulls={})
    for f, X in fam.items():
        matched = torch.stack([N.match_layer_mass(X[:, li], float(mass[li])) for li in range(NL)], 1)      # layer-mass matched
        out["nulls"][f] = dict(n=int(X.shape[0]), pooled_layer_matched=pooled_stats(matched.reshape(X.shape[0], -1)), pooled_unmatched=pooled_stats(X.reshape(X.shape[0], -1)),
                               within=within_layer(matched, mass), per_layer_top2_mean=C.topk_abs_share(matched, 2).mean(0).tolist(), per_layer_n_eff_mean=C.n_eff(matched).mean(0).tolist())
    # pooled over early (0-18) vs late (19-22) layers, layer-matched, true Haar only
    for nm, sel in (("layers0_18", list(range(0, 19))), ("layers19_22", list(range(19, 23)))):
        nat_s = nat[sel].reshape(1, -1); nl_s = torch.stack([N.match_layer_mass(fam["haar_block_true"][:, li], float(mass[li])) for li in sel], 1).reshape(n_haar, -1)
        out.setdefault("subsets", {})[nm] = dict(native=pooled_stats(nat_s), haar_block_true=pooled_stats(nl_s))
    res["concentration"][aname] = out
    nt = out["native"]["pooled"]["topk"][KS.index(8)][0]; ht = np.array(out["nulls"]["haar_block_true"]["pooled_layer_matched"]["topk"][KS.index(8)])
    print(f"{aname}: native top8={nt:.3f} N_eff={out['native']['pooled']['n_eff'][0]:.1f} | true-Haar-block matched top8 mean={ht.mean():.3f} max={ht.max():.3f}; within-layer top2 native={out['native']['within']['top2'][0]:.3f} null={np.mean(out['nulls']['haar_block_true']['within']['top2']):.3f}", flush=True)
k_s = ks_2samp(res["concentration"]["cons/logic"]["nulls"]["haar_block_true"]["pooled_layer_matched"]["topk"][KS.index(8)],
               res["concentration"]["cons/logic"]["nulls"]["haar_block_sampler"]["pooled_layer_matched"]["topk"][KS.index(8)])
res["sampler_validation"] = dict(ks_stat=float(k_s.statistic), ks_p=float(k_s.pvalue), n_true=n_haar, n_sampler=n_samp)

# ---- held-out reconstruction (consensus/logic): native heads, random heads, rotated blocks -------------------------------------
Str, Ste = P[("cons", "logic", "tr")]["S"].reshape(1000, -1), P[("cons", "logic", "te")]["S"].reshape(len(ho), -1)
Strt, Stet = torch.tensor(Str), torch.tensor(Ste)
ytr, yte = torch.tensor(ACC[("logic", "tr")]), torch.tensor(ACC[("logic", "te")])
Dtr_n = Strt[ytr].mean(0) - Strt[~ytr].mean(0); Dte_n = Stet[yte].mean(0) - Stet[~yte].mean(0)
order = Dtr_n.abs().argsort(descending=True)
rec = {"net_D_test_total": float(Dte_n.sum()), "native": {}, "random": {}, "haar_block": {}}
gen = np.random.default_rng(3)
for k in KS:
    idx = order[:k]
    rec["native"][str(k)] = dict(**C.signed_reconstruction(Dte_n, idx), **C.per_example_r2(Strt, Stet, idx), abs_mass_train=float(Dtr_n.abs()[idx].sum() / Dtr_n.abs().sum()), abs_mass_test=float(Dte_n.abs()[idx].sum() / Dte_n.abs().sum()))
    rr = [(C.signed_reconstruction(Dte_n, torch.as_tensor(gen.choice(Dtr_n.numel(), k, replace=False)))["frac"], C.per_example_r2(Strt, Stet, torch.as_tensor(gen.choice(Dtr_n.numel(), k, replace=False)))["r2"]) for _ in range(100)]
    rec["random"][str(k)] = dict(frac_mean=float(np.mean([x[0] for x in rr])), frac_max=float(np.max([x[0] for x in rr])), r2_mean=float(np.mean([x[1] for x in rr])), r2_max=float(np.max([x[1] for x in rr])), n=100)
ztr, zte = P[("cons", "logic", "tr")]["z"], P[("cons", "logic", "te")]["z"]
acc_h = {str(k): [] for k in KS}
for r in range(10):
    Btr, Bte = [], []
    for li, l in enumerate(LAYERS):
        R = haar_orthogonal(H * HD, 10_000 * (l + 1) + r, dev); g = P[("cons", "logic", "tr")]["gains"][l]
        Rg = R @ g
        Btr.append(N.block_sums((ztr[:, li].float().to(dev) @ R.T) * Rg, HD).cpu()); Bte.append(N.block_sums((zte[:, li].float().to(dev) @ R.T) * Rg, HD).cpu()); del R
    Btr, Bte = torch.cat(Btr, 1), torch.cat(Bte, 1)
    Dr_tr = Btr[ytr].mean(0) - Btr[~ytr].mean(0); Dr_te = Bte[yte].mean(0) - Bte[~yte].mean(0)
    od = Dr_tr.abs().argsort(descending=True)
    for k in KS:
        acc_h[str(k)].append((C.signed_reconstruction(Dr_te, od[:k])["frac"], C.per_example_r2(Btr, Bte, od[:k])["r2"]))
rec["haar_block"] = {k: dict(frac_mean=float(np.mean([x[0] for x in v])), frac_max=float(np.max([x[0] for x in v])), r2_mean=float(np.mean([x[1] for x in v])), r2_max=float(np.max([x[1] for x in v])), n=len(v)) for k, v in acc_h.items()}
# cancellation
rec["cancellation_train"] = C.cancellation_index(Dtr_n); rec["cancellation_test"] = C.cancellation_index(Dte_n)
rec["per_layer_cancellation_test"] = {str(l): C.cancellation_index(torch.tensor(Dte[("cons", "logic")][li])) for li, l in enumerate(LAYERS) if l >= 17}
rec["mean_per_example_kappa_test"] = float(C.per_example_kappa(Stet).mean())
res["reconstruction"] = rec
print("recon native", {k: round(v["frac"], 3) for k, v in rec["native"].items()}, "random max", {k: round(v["frac_max"], 3) for k, v in rec["random"].items()}, flush=True)

# ---- split-half and held-out mass; cross-task ----------------------------------------------------------------------------------
Sl = P[("cons", "logic", "tr")]["S"]; yl = ACC[("logic", "tr")]
pos_i, neg_i = np.where(yl)[0], np.where(~yl)[0]
pr, nr = np.random.default_rng(1).permutation(pos_i), np.random.default_rng(2).permutation(neg_i)
hA, hB = np.concatenate([pr[:len(pr) // 2], nr[:len(nr) // 2]]), np.concatenate([pr[len(pr) // 2:], nr[len(nr) // 2:]])
dA = (Sl[hA][yl[hA]].mean(0) - Sl[hA][~yl[hA]].mean(0)).reshape(-1); dB = (Sl[hB][yl[hB]].mean(0) - Sl[hB][~yl[hB]].mean(0)).reshape(-1)
topA = np.argsort(-np.abs(dA))[:8]
res["split_half"] = dict(spearman_abs=float(spearmanr(np.abs(dA), np.abs(dB)).correlation), top8_A_mass_in_B=float(np.abs(dB)[topA].sum() / np.abs(dB).sum()),
                         top8_B_own_mass=float(np.sort(np.abs(dB))[::-1][:8].sum() / np.abs(dB).sum()), random_expectation=8 / dA.size,
                         top8_overlap_A_B=int(len(set(topA) & set(np.argsort(-np.abs(dB))[:8]))))
res["cross_task"] = {}
pool = {t: D[("cons", t)].reshape(-1) for t in ("logic", "localizer", "prontoqa")}
pool_te = {t: Dte[("cons", t)].reshape(-1) for t in ("logic", "localizer", "prontoqa")}
for a_, b_ in (("logic", "localizer"), ("logic", "prontoqa"), ("localizer", "prontoqa")):
    out = {}
    for k in (8, 16):
        ta, tb = set(np.argsort(-np.abs(pool[a_]))[:k]), set(np.argsort(-np.abs(pool[b_]))[:k])
        out[f"top{k}"] = dict(overlap=len(ta & tb), expected=k * k / pool[a_].size, jaccard=len(ta & tb) / len(ta | tb))
    res["cross_task"][f"{a_}~{b_}"] = dict(spearman_signed=float(spearmanr(pool[a_], pool[b_]).correlation), spearman_abs=float(spearmanr(np.abs(pool[a_]), np.abs(pool[b_])).correlation), heldout_spearman_signed=float(spearmanr(pool_te[a_], pool_te[b_]).correlation), **out)
# held-out confirmation of the training rankings per task
res["heldout_confirmation"] = {t: dict(train_top8_heldout_mass=float(np.abs(pool_te[t])[np.argsort(-np.abs(pool[t]))[:8]].sum() / np.abs(pool_te[t]).sum()),
                                       train_top8_heldout_signed_share=float(pool_te[t][np.argsort(-np.abs(pool[t]))[:8]].sum() / pool_te[t].sum()),
                                       spearman_train_test_abs=float(spearmanr(np.abs(pool[t]), np.abs(pool_te[t])).correlation)) for t in pool}

# ---- gate S1 and frozen selection --------------------------------------------------------------------------------------------------
S1 = json.load(open(ROOT / "results/phase1/stage1.json"))["tables"]["consensus/logic"]
abs_attn, abs_mlp = float(np.sum(S1["attn_abs"])), float(np.sum(S1["mlp_abs"]))
share = abs_attn / (abs_attn + abs_mlp + abs(S1["emb"]))
res["gate_S1"] = dict(identity_ok=res["identity_ok"], split_half_spearman=res["split_half"]["spearman_abs"], absolute_attention_share=share,
                      passed=bool(res["identity_ok"] and res["split_half"]["spearman_abs"] >= 0.5 and share >= 0.15))
flat = np.abs(Dl).reshape(-1); order_h = np.argsort(-flat)
sel = lambda k: [dict(layer=int(i // H), head=int(i % H), delta=float(Dl.reshape(-1)[i]), abs_delta=float(flat[i])) for i in order_h[:k]]
res["selection"] = dict(S8=sel(8), S16=sel(16), coverage_S8_of_total_abs=float(flat[order_h[:8]].sum() / flat.sum()), coverage_S8_signed_of_net=float(Dl.reshape(-1)[order_h[:8]].sum() / Dl.sum()))
json.dump(dict(S8=res["selection"]["S8"], S16=res["selection"]["S16"], rule="pooled top-k by training |Delta_h| (consensus, logic, num_orig)"), open(OUT / "selected_heads.json", "w"), indent=1)
res["seconds"] = time.time() - T0
json.dump(res, open(OUT / "stage1.json", "w"), indent=1)
np.savez_compressed(OUT / "stage1_arrays.npz", delta_logic=Dl, delta_localizer=D[("cons", "localizer")], delta_prontoqa=D[("cons", "prontoqa")], delta_logic_te=Dte[("cons", "logic")],
                    delta_localizer_te=Dte[("cons", "localizer")], delta_prontoqa_te=Dte[("cons", "prontoqa")], mean_acc=P[("cons", "logic", "tr")]["mean_acc"].numpy(), mean_rej=P[("cons", "logic", "tr")]["mean_rej"].numpy())
print("S8:", [(s["layer"], s["head"], round(s["delta"], 3)) for s in res["selection"]["S8"]], "\nGATE S1", res["gate_S1"], f"({res['seconds']:.0f}s)")
