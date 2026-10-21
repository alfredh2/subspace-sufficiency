"""Phase 0.5, Experiment 3b (non-verdict control tasks):  counterfactual activation patching restricted to the learned MVS coordinates.
Matched pairs: identical rule/antecedent/consequent sentences and conclusion marker, valid vs invalid by swapping which
sentence is the fact and which is the conclusion (MP<->AC, MT<->DA; +/- negated consequent). Same-label/different-lexicon
controls, random-subspace controls, complement patching (everything EXCEPT the MVS), reversed/A-B label formats, and
unmatched non-logic pairs (localizer, PrOntoQA) with last-position patching. Usage: python scripts/phase05_patching.py"""
import json

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P
from causal_subspaces.data.logic import HELDOUT_FAMILIES, TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_controls, get_reference, get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.checkpoint import load_checkpoint
from causal_subspaces.mvs.projection import orthonormalize
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

SEEDS, OUT = [0, 1, 2], ROOT / "results" / "phase05"
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
splits, controls = get_splits(cfg), get_controls(cfg)
ref = get_reference(model, tok, cfg, rev, dev)
cks = {s: load_checkpoint(ROOT / f"results/mvs_rank2_seed{s}/checkpoints/step_1600.pt") for s in SEEDS}
Us = {s: cks[s]["U"] for s in SEEDS}
cons = P.consensus_directions(Us)
cons_dir = torch.nn.functional.normalize(sum(cons[s]["shared"] for s in SEEDS), dim=0)

def S(U, comp=False):
    return (U.to(dev), comp)
mvs = {f"mvs_s{s}": S(Us[s]) for s in SEEDS}
comp = {f"complement_s{s}": S(Us[s], True) for s in SEEDS}
rank1 = {**{f"s{s}_shared": S(cons[s]["shared"][:, None]) for s in SEEDS}, **{f"s{s}_specific": S(cons[s]["specific"][:, None]) for s in SEEDS},
         "consensus": S(cons_dir[:, None])}
def rand(n_each):
    d = {}
    for i in range(n_each):
        d[f"rand_gauss{i}"] = S(orthonormalize(init_V(ref["Qm"], 2, 2000 + i, "gaussian")))
        d[f"rand_topvar{i}"] = S(orthonormalize(init_V(ref["Qm"], 2, 3000 + i, "onmanifold")))
    return d
FULL = {**mvs, **comp, **rank1, **rand(4)}
REDUCED = {**mvs, **comp, **rand(2)}


summary, raw = {"groups": {}}, {}
def go(group, trials, aff, rej, subs, positions):
    res = P.run_trials(model, tok, trials, aff, rej, subs, idx, positions=positions, device=dev, batch_size=16)
    ta, sa = np.array([t["tgt_aff"] for t in trials]), np.array([t["src_aff"] for t in trials])
    summary["groups"][group] = dict(n_trials=len(trials), results={f"{n}|{p}": P.summarize_patching(res, ta, sa, n, p) for n in subs for p in positions})
    for k, v in res.items():
        raw[f"{group}|{k}"] = v
    raw[f"{group}|tgt_aff"], raw[f"{group}|src_aff"] = ta, sa
    r = summary["groups"][group]["results"]["mvs_s0|last"]
    print(group, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

rng = np.random.default_rng(0)
def unmatched(items_aff, items_rej, n=200):
    a = rng.choice(len(items_aff), n); b = rng.choice(len(items_rej), n)
    tr = []
    for i, j in zip(a, b):
        tr += [dict(tgt=items_rej[j], src=items_aff[i], tgt_aff=False, src_aff=True), dict(tgt=items_aff[i], src=items_rej[j], tgt_aff=True, src_aff=False)]
    return tr
for name in ("mmlu_binary", "stroop", "word_recall_8"):
    it = controls[name]["items"]
    go(f"unmatched_{name}", unmatched([x[0] for x in it if x[1] == "1"], [x[0] for x in it if x[1] == "2"]), "1", "2", REDUCED, ("last",))
json.dump(summary, open(OUT / "exp3b_nonverdict_patching.json", "w"), indent=1)
np.savez_compressed(OUT / "exp3b_arrays.npz", **raw)
print("saved")
