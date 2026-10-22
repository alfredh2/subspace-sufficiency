"""Phase 7 Stage-0 falsification pilot (docs/phase7_plan.md section 11): is native-basis concentration of the verdict-direction write specific to the behaviour,
or does it also arise for shuffled labels, random / irrelevant target directions and verdict-irrelevant contrasts? Logic data only; held-out evaluation.
Usage: python scripts/phase7_stage0.py"""
import json
import time

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5
from causal_subspaces.data.logic import FAMILIES
from causal_subspaces.experiment import get_splits
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, load_model
from causal_subspaces.neurons.decompose import layerwise_pass, unit_gain
from causal_subspaces.phase7 import concentration as C7
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

T0 = time.time(); OUT = ROOT / "results/phase7"; OUT.mkdir(exist_ok=True)
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text()); seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
S1 = json.load(open(ROOT / "results/phase1/stage1.json")); LS = S1["selection"]["layers"]; m = 12288; Q = round(0.01 * len(LS) * m)
model, tok, rev = load_model(cfg["model"], dev)
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev).float()
splits = get_splits(cfg); rng = np.random.default_rng(0); tr = splits["train"]
val, inv = [e for e in tr if e.valid], [e for e in tr if not e.valid]
pick = [val[i] for i in rng.choice(len(val), 500, replace=False)] + [inv[i] for i in rng.choice(len(inv), 500, replace=False)]   # identical to Phase 1
ho = [e for sp in ("items", "families", "nonwords") for e in splits[sp]]
texts = {"tr": [P5.build_prompt("num_orig", e.syllogism) for e in pick], "te": [P5.build_prompt("num_orig", e.syllogism) for e in ho]}
exs = {"tr": pick, "te": ho}
y = {k: torch.tensor([e.valid for e in v], device=dev) for k, v in exs.items()}
acts = {}
for part in ("tr", "te"):
    out = layerwise_pass(model, tok, texts[part], y[part].cpu().numpy(), d_c, idx_hook, device=dev, keep_acts_layers=LS)
    acts[part] = {l: out["acts"][l].float().to(dev) for l in LS}
print(f"activations collected ({time.time() - T0:.0f}s), train {len(pick)}, held-out {len(ho)}, q = {Q}", flush=True)

# residual vectors at the hook (train) for principal directions
@torch.no_grad()
def collect_resid(tx):
    saved = model.model.layers; model.model.layers = saved[:idx_hook]; res = []
    try:
        for s in range(0, len(tx), 16):
            ids, mask, pos = encode(tok, [format_prompt(tok, t, False) for t in tx[s:s + 16]], dev); sink = []
            with capture_residual(model, idx_hook, sink): model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
            res.append(sink[0][:, -1].float())
    finally:
        model.model.layers = saved
    return torch.cat(res)
H = collect_resid(texts["tr"]); Hc = H - H.mean(0); _, _, Vh = torch.linalg.svd(Hc, full_matrices=False); PCs = Vh[:5]
# target directions
g_all = lambda d: torch.stack([unit_gain(model, l, d) for l in LS])                                    # (L, m)
gen = torch.Generator(device=dev).manual_seed(11)
rand_dirs = torch.randn(50, d_c.numel(), generator=gen, device=dev); rand_dirs = rand_dirs / rand_dirs.norm(dim=1, keepdim=True)
WU = model.lm_head.weight.float(); rg = np.random.default_rng(5)
pairs = [tuple(rg.choice(np.arange(1000, 50000), 2, replace=False)) for _ in range(20)] + [(tok.encode("1", add_special_tokens=False)[0], tok.encode("2", add_special_tokens=False)[0])]
unemb = torch.stack([(WU[a] - WU[b]) / (WU[a] - WU[b]).norm() for a, b in pairs])
G = {"dc": g_all(d_c)[None], "rand": torch.stack([g_all(d) for d in rand_dirs]), "pc": torch.stack([g_all(d) for d in PCs]), "unemb": torch.stack([g_all(d) for d in unemb])}
Wdown = {l: model.model.layers[l].mlp.down_proj.weight.float() for l in LS}
del H, Hc; torch.cuda.empty_cache()

# contrasts (weights over examples); shuffles preserve class sizes; irrelevant partitions are verdict-balanced structural splits
def partitions(es):
    base = [e.form.split("_")[0] for e in es]
    return {"neg_consequent": torch.tensor([e.neg_consequent for e in es], device=dev),
            "fact_cites_antecedent": torch.tensor([b in ("MP", "DA") for b in base], device=dev),
            "family_parity": torch.tensor([FAMILIES.index(e.family) % 2 == 0 for e in es], device=dev)}
W = {}
for part in ("tr", "te"):
    n = len(exs[part]); gs = torch.Generator(device=dev).manual_seed(3 if part == "tr" else 4)
    perms = [y[part][torch.randperm(n, generator=gs, device=dev)] for _ in range(50)]
    W[part] = dict(true=C7.contrast_weights(y[part])[None], shuf=torch.stack([C7.contrast_weights(p) for p in perms]),
                   irrel=torch.stack([C7.contrast_weights(v) for v in partitions(exs[part]).values()]))
    print(part, "irrelevant-partition class sizes (true / total):", {k: (int(v.sum()), n) for k, v in partitions(exs[part]).items()}, flush=True)
DELTA = {part: {kind: {l: W[part][kind] @ acts[part][l] for l in LS} for kind in ("true", "shuf", "irrel")} for part in ("tr", "te")}   # (S, m) per layer

def pooled(kind, gi, g_set, si=None):     # contrasts[kind][si] x gains g_set[gi] -> pooled (L*m,) train/test contributions
    f = lambda part: torch.cat([DELTA[part][kind][l][si] * g_set[gi][li] for li, l in enumerate(LS)])
    return f("tr"), f("te")
def stat(tr_, te_): return C7.pooled_stats(tr_, te_, Q)
fam = {}
fam["a_native_dc_true"] = [stat(*pooled("true", 0, G["dc"], 0))]
fam["b_shuffled_labels_dc"] = [stat(*pooled("shuf", 0, G["dc"], i)) for i in range(50)]
fam["c_random_directions_true"] = [stat(*pooled("true", i, G["rand"], 0)) for i in range(50)]
fam["d_random_directions_shuffled"] = [stat(*pooled("shuf", i, G["rand"], i)) for i in range(50)]
fam["e_irrelevant_partitions_dc"] = [stat(*pooled("irrel", 0, G["dc"], i)) for i in range(3)]
fam["f_residual_PCs_true"] = [stat(*pooled("true", i, G["pc"], 0)) for i in range(5)]
fam["g_unembedding_pairs_true"] = [stat(*pooled("true", i, G["unemb"], 0)) for i in range(20)]
fam["g2_answer_pair_true"] = [stat(*pooled("true", 20, G["unemb"], 0))]
# (h) true Haar rotations of the post-gate activation basis, train-rank / test-evaluate
hs = []
for dr in range(20):
    Dtr, Dte = [], []
    for li, l in enumerate(LS):
        R = C7.haar_orthogonal(m, 1000 * dr + l, dev); g = G["dc"][0][li]
        Dtr.append((R @ DELTA["tr"]["true"][l][0]) * (R @ g)); Dte.append((R @ DELTA["te"]["true"][l][0]) * (R @ g)); del R
    hs.append(stat(torch.cat(Dtr), torch.cat(Dte)))
fam["h_haar_rotations_dc"] = hs
# (i) PCA basis of the training activations (random orthonormal completion)
Dtr, Dte = [], []
for li, l in enumerate(LS):
    Ac = acts["tr"][l] - acts["tr"][l].mean(0); _, S_, Vh = torch.linalg.svd(Ac, full_matrices=False); r = int((S_ > 1e-6 * S_[0]).sum())
    B = C7.completed_basis(Vh[:r].T.contiguous(), 77 + l, dev); g = G["dc"][0][li]
    Dtr.append((B.T @ DELTA["tr"]["true"][l][0]) * (B.T @ g)); Dte.append((B.T @ DELTA["te"]["true"][l][0]) * (B.T @ g)); del B
fam["i_pca_basis_dc"] = [stat(torch.cat(Dtr), torch.cat(Dte))]
# (j) attainable-sparsity benchmarks: rotations FITTED to gain vectors of K_fit directions, evaluated on d_c (K_fit=1 uses d_c itself)
bench = {}
for kfit in (1, 4, 16, 64, 256):
    for own in ((True, False) if kfit == 1 else (False,)):
        Dtr, Dte = [], []
        for li, l in enumerate(LS):
            if own: F = G["dc"][0][li][None].T
            else:
                gd = torch.Generator(device=dev).manual_seed(900 + kfit); dirs = torch.randn(kfit, d_c.numel(), generator=gd, device=dev)
                F = (Wdown[l].T @ dirs.T)      # gains of kfit random directions (m x kfit)
            Qf, _ = torch.linalg.qr(F)
            B = C7.completed_basis(Qf.contiguous(), 55 + l, dev); g = G["dc"][0][li]
            Dtr.append((B.T @ DELTA["tr"]["true"][l][0]) * (B.T @ g)); Dte.append((B.T @ DELTA["te"]["true"][l][0]) * (B.T @ g)); del B
        bench[f"fit_on_{'d_c_itself' if own else 'random_directions'}_K{kfit}"] = stat(torch.cat(Dtr), torch.cat(Dte))
fam["j_attainable_sparsity_benchmarks"] = list(bench.values()); fam["_benchmark_names"] = list(bench.keys())

# ---- summaries and the preregistered decision rule
def summ(lst, key):
    v = np.array([s[key] for s in lst if s.get(key) is not None and np.isfinite(s[key])]); return dict(mean=float(v.mean()), sd=float(v.std()), p95=float(np.percentile(v, 95)), max=float(v.max()), n=int(len(v))) if len(v) else None
S = {k: {key: summ(v, key) for key in ("share_same", "share_heldout", "neff_heldout", "recon_512", "net_over_abs")} for k, v in fam.items() if not k.startswith("_")}
nat = S["a_native_dc_true"]["share_heldout"]["mean"]
decision = {k: dict(mean_ratio=S[k]["share_heldout"]["mean"] / nat, p95_ratio=S[k]["share_heldout"]["p95"] / nat, max_ratio=S[k]["share_heldout"]["max"] / nat) for k in S if k[0] in "bcdefg"}
flag = {k: ("H-generic supported (>= 0.7)" if v["mean_ratio"] >= 0.7 else "partial (0.4-0.7)" if v["mean_ratio"] >= 0.4 else "not supported (< 0.4)") for k, v in decision.items()}
gate = dict(native_share_same=S["a_native_dc_true"]["share_same"]["mean"], phase1_value=0.490, share_same_ok=bool(abs(S["a_native_dc_true"]["share_same"]["mean"] - 0.490) <= 0.01),
            native_recon_512_heldout=S["a_native_dc_true"]["recon_512"]["mean"], phase1_recon=0.984, recon_ok=bool(abs(S["a_native_dc_true"]["recon_512"]["mean"] - 0.984) <= 0.02))
res = dict(q=Q, layers=LS, n_train=len(pick), n_heldout=len(ho), regression_gate=gate, families=fam, summary=S, decision_ratios=decision, decision_flags=flag, benchmarks=bench)
json.dump(res, open(OUT / "stage0_falsification.json", "w"), indent=1, default=float)
print(json.dumps(gate), flush=True)
print(f"{'family':36s} {'share_heldout mean [p95/max]':32s} {'N_eff':>9s} {'recon512':>8s} {'net/abs':>8s} ratio-to-native")
for k, v in S.items():
    sh, ne, rc, na = v["share_heldout"], v["neff_heldout"], v["recon_512"], v["net_over_abs"]
    print(f"{k:36s} {sh['mean']:.3f} [{sh['p95']:.3f}/{sh['max']:.3f}] {ne['mean']:9.1f} {(rc['mean'] if rc else float('nan')):8.3f} {na['mean']:8.3f}  {sh['mean'] / nat:.2f}")
print(json.dumps(flag, indent=1)); print("benchmarks (held-out share, recon512):", {k: (round(v['share_heldout'], 3), round(v['recon_512'], 3)) for k, v in bench.items()})
print(f"done {time.time() - T0:.0f}s")
