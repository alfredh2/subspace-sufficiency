"""Causal-specificity control: ablate UNOPTIMISED random rank-2 subspaces at the same residual location, same mu, same eval.
Two null families: 'gaussian' (uniform random directions in R^d) and 'topvar' (random directions inside the top-m
variance subspace used for MVS init — a stricter null as it matches where activations actually vary).
Usage: python scripts/random_subspace_control.py --n 10 [--run-name mvs_rank2_seed0]"""
import argparse
import json

import torch
import yaml

from causal_subspaces.experiment import evaluate_condition, get_controls, get_lm_passages, get_reference, get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.checkpoint import load_checkpoint
from causal_subspaces.mvs.projection import orthonormalize
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", default="mvs_rank2_seed0")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--base-seed", type=int, default=1000)
a = ap.parse_args()
rd = ROOT / "results" / a.run_name
cfg = yaml.safe_load((rd / "config.yaml").read_text())
seed_everything(cfg["seed"])
dev = cfg["hardware"]["device"]
model, tok, rev = load_model(cfg["model"], dev)
splits, controls, passages = get_splits(cfg), get_controls(cfg), None
passages = get_lm_passages(tok, cfg)
ck = load_checkpoint(rd / "checkpoints" / "step_1600.pt")
mu = ck["mu"].to(dev)
ref = get_reference(model, tok, cfg, rev, dev)
assert torch.allclose(ref["mu"], ck["mu"]), "reference mean differs from the checkpoint's"
out = {"run_name": a.run_name, "residual_index": cfg["mvs"]["residual_index"], "random": []}
def summ(m):
    return dict(logic={k: v["overall"]["acc"] for k, v in m["logic"].items()},
                logic_invalid={k: v["overall"]["invalid_rate"] for k, v in m["logic"].items()},
                controls={k: (v if k == "lm_kl" else v["overall"]["acc"]) for k, v in m["controls"].items()})
for kind in ("gaussian", "topvar"):
    for i in range(a.n):
        V = init_V(ref["Qm"], 2, a.base_seed + i, "gaussian" if kind == "gaussian" else "onmanifold")
        U = orthonormalize(V).to(dev)
        learned_overlap = torch.linalg.svdvals(U.double().T @ ck["U"].to(dev).double()).tolist()
        m, _ = evaluate_condition(model, tok, cfg, splits, controls, "ablate", U, mu, passages, dev)
        out["random"].append(dict(kind=kind, seed=a.base_seed + i, overlap_with_learned=learned_overlap, **summ(m)))
        print(kind, i, out["random"][-1]["logic"], flush=True)
(ROOT / "results" / "phase0_robustness").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results" / "phase0_robustness" / "random_ablation.json", "w"), indent=2)
