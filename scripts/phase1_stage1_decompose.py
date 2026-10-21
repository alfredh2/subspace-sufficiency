"""Phase 1 Stage 1: layerwise decomposition of the verdict coordinate c = d.h at the hook into embedding / attention / MLP writes (all layers < hook),
identity check, per-unit class-difference vectors, frozen layer selection and scope gate (docs/phase1_plan.md section 3).
Training prompts only. Usage: python scripts/phase1_stage1_decompose.py"""
import json
import time

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5, phase06 as P6
from causal_subspaces.data.controls import build_controls
from causal_subspaces.experiment import get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.neurons.decompose import class_difference_table, layerwise_pass, unit_gain
from causal_subspaces.mvs.checkpoint import load_checkpoint
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

t0 = time.time()
OUT = ROOT / "results" / "phase1"
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
splits = get_splits(cfg)

# directions: consensus (primary) + the three per-seed shared directions (replication)
Us = {s: load_checkpoint(ROOT / f"results/mvs_rank2_seed{s}/checkpoints/step_1600.pt")["U"] for s in (0, 1, 2)}
cons = P5.consensus_directions(Us)
dirs = {"consensus": P6.load_consensus(), **{f"shared_s{s}": cons[s]["shared"] / cons[s]["shared"].norm() for s in (0, 1, 2)}}
torch.save(dirs, OUT / "directions.pt")

rng = np.random.default_rng(0)
tr = splits["train"]
val, inv = [e for e in tr if e.valid], [e for e in tr if not e.valid]
pick = [val[i] for i in rng.choice(len(val), 500, replace=False)] + [inv[i] for i in rng.choice(len(inv), 500, replace=False)]
logic_texts = [P5.build_prompt("num_orig", e.syllogism) for e in pick]
logic_acc = np.array([e.valid for e in pick])
ctrl = build_controls(101, 256, ("anatomy", "philosophy"), 16)
tasks = {"logic": (logic_texts, logic_acc)}
for name in ("language_localizer", "prontoqa"):
    it = ctrl[name]["items"]
    tasks[name] = ([t for t, _ in it], np.array([g == "1" for _, g in it]))

res, vecs = {"task_n": {k: len(v[0]) for k, v in tasks.items()}, "tables": {}}, {}
for dname, d in dirs.items():
    for tname, (texts, acc) in tasks.items():
        if dname != "consensus" and tname != "logic":
            continue
        out = layerwise_pass(model, tok, texts, acc, d, idx, device=dev, batch_size=16)
        L = out["layers"]
        tab = class_difference_table(out["comps"], acc)
        delta = out["mean_acc"] - out["mean_rej"]                                     # (L, m)
        g = torch.stack([unit_gain(model, l, d) for l in L]).cpu()                    # (L, m)
        Delta = delta * g
        A = Delta.abs().sum(1).numpy(); Dn = Delta.sum(1).numpy()
        assert np.allclose(Dn, tab["mlp"], rtol=1e-3, atol=1e-3 * np.abs(tab["mlp"]).max()), "unit-level and layer-level MLP class differences disagree"
        kappa = A / np.maximum(np.abs(Dn), 1e-12)
        res["tables"][f"{dname}/{tname}"] = dict(
            layers=L, emb=tab["emb"], resid_class_diff=tab["resid"], identity_max_abs_err=tab["identity_max_abs_err"], resid_std=tab["resid_std"],
            mlp_signed=tab["mlp"].tolist(), mlp_abs=np.abs(tab["mlp"]).tolist(), mlp_sum_abs_unit=A.tolist(), mlp_kappa=kappa.tolist(),
            attn_signed=tab["attn"].tolist(), attn_abs=np.abs(tab["attn"]).tolist())
        if dname == "consensus":
            vecs[f"{tname}/mean_acc"], vecs[f"{tname}/mean_rej"] = out["mean_acc"].numpy(), out["mean_rej"].numpy()
        if tname == "logic":
            vecs[f"{dname}/g"] = g.numpy()
        print(f"{dname}/{tname}: identity err {tab['identity_max_abs_err']:.3f} (std c {tab['resid_std']:.2f}); class-diff of c {tab['resid']:+.2f} = emb {tab['emb']:+.2f} + attn {tab['attn'].sum():+.2f} + mlp {tab['mlp'].sum():+.2f}", flush=True)

# frozen selection and gate (consensus, logic)
t = res["tables"]["consensus/logic"]
mlp_abs, attn_abs = np.array(t["mlp_abs"]), np.array(t["attn_abs"])
order = np.lexsort((-np.array(t["mlp_sum_abs_unit"]), -mlp_abs))[:4]
share = float(mlp_abs.sum() / (mlp_abs.sum() + attn_abs.sum() + abs(t["emb"])))
res["selection"] = dict(layers=[int(i) for i in order], coverage_of_sum_abs_mlp=float(mlp_abs[order].sum() / mlp_abs.sum()), absolute_mlp_share=share,
                        gate="full pilot" if share >= 0.30 else ("scope-limited" if share >= 0.15 else "stop after Stage 2"),
                        identity_ok=bool(t["identity_max_abs_err"] <= 0.02 * t["resid_std"]))
res["seconds"] = time.time() - t0
json.dump(res, open(OUT / "stage1.json", "w"), indent=1)
np.savez_compressed(OUT / "stage1_vectors.npz", **vecs)
print(json.dumps(res["selection"], indent=1))
print("layer  mlp_signed  |mlp|   sum|Delta_j|  kappa  attn_signed")
for l in range(len(mlp_abs)):
    print(f"{l:3d} {t['mlp_signed'][l]:+9.2f} {mlp_abs[l]:7.2f} {t['mlp_sum_abs_unit'][l]:9.2f} {t['mlp_kappa'][l]:7.1f} {t['attn_signed'][l]:+9.2f}")
