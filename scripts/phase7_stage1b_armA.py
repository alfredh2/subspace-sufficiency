"""Stage 1b Arm A: dev-only screening grid (hooks x ranks per task, 2 seeds, 1600 steps) on the FIRST format of each task, 1b partition,
evaluated on VAL (<=256 items) only; writes armA/<task>_h<h>_r<r>_s<seed>.{json,pt}. Selection is a separate step (phase7_stage1b_select.py).
Usage: python scripts/phase7_stage1b_armA.py --tasks G N S L M A"""
import argparse, json, time
import numpy as np, torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.phase7.stage1 import retention_metrics, run_stats, primary_accuracy_vec
from causal_subspaces.phase7.stage1b import build_1b, cap_items, CELLS, FORMATS_1B, fit_b, curve_slope
from causal_subspaces.utils.config import ROOT, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="*", default=["G", "N", "S", "L", "M", "A"])
ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
ap.add_argument("--steps", type=int, default=1600)
a = ap.parse_args()
out = ROOT / "results/phase7/stage1b/armA"
out.mkdir(parents=True, exist_ok=True)
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
refs = {}
for h in (21, 23, 25, 27):
    cfg["mvs"]["residual_index"] = h
    t0 = time.time(); refs[h] = get_reference(model, tok, cfg, rev, "cuda"); print("reference", h, round(time.time() - t0), "s", flush=True)

for t in a.tasks:
    data = build_1b(t, tok)
    fmt0 = FORMATS_1B[t][0]
    d = data[fmt0]
    val = cap_items(d["val"], 256)
    intact = run_stats(model, tok, val)
    print("TASK", t, fmt0, "val n", len(val), "intact dev acc %.3f" % primary_accuracy_vec(intact, val).mean(), flush=True)
    for h, r in CELLS[t]:
        for seed in a.seeds:
            f = out / f"{t}_h{h}_r{r}_s{seed}.json"
            if f.exists():
                continue
            torch.manual_seed(seed)
            V0 = init_V(refs[h]["Qm"], r, seed, "onmanifold")
            U, summ = fit_b(model, tok, d["train"], refs[h]["mu"], V0, h, steps=a.steps, seed=seed)
            m, _, _ = retention_metrics(model, tok, val, h, U, refs[h]["mu"].cuda(), intact)
            red = m["gap_intact"] - m["gap_ablate"]
            row = dict(task=t, fmt=fmt0, hook=h, rank=r, seed=seed, train=summ, val=m, reduction_abs=red, reduction_rel=red / m["gap_intact"],
                       train_slope_last25=curve_slope(summ["kl_curve"]), n_val=len(val))
            json.dump(row, open(f, "w"), indent=1)
            torch.save(U.cpu(), f.with_suffix(".pt"))
            print(t, h, r, seed, "trainKL %.3f valKL %.4f slope %.2f accI/R/A %.2f/%.2f/%.2f gap %.1f->%.1f %.0fs" % (
                summ["kl_last5_median"], m["kl_retain"], row["train_slope_last25"], m["acc_intact"], m["acc_retain"], m["acc_ablate"],
                m["gap_intact"], m["gap_ablate"], summ["seconds"]), flush=True)
print("ARM A DONE", flush=True)
