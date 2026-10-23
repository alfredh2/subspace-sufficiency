"""EXPLORATORY, non-rescuing diagnostic (declared before running): for each task with no passing sweep configuration, refit the sweep
configuration with the lowest val retained-KL at the full 1600 steps (seed 0) and report VAL metrics. The Stage-1 verdict for these tasks
is NOT changed by the outcome (plan section 12: no rescue); this only tests whether failure is due to the 300-step budget."""
import json, glob
import torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.phase7.tasks import FORMATS, build_task
from causal_subspaces.phase7.stage1 import fit_mvs, retention_metrics, run_stats
from causal_subspaces.utils.config import ROOT, load_config

cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
res = {}
for t in "GNM":
    rows = [json.load(open(f)) for f in glob.glob(str(ROOT / f"results/phase7/stage1/sweep/{t}_h*_r*.json"))]
    best = min(rows, key=lambda r: r["val"]["kl_retain"])
    h, r = best["hook"], best["rank"]
    cfg["mvs"]["residual_index"] = h
    ref = get_reference(model, tok, cfg, rev, "cuda")
    d = build_task(t, tok)[FORMATS[t][0]]
    intact = run_stats(model, tok, d["val"])
    torch.manual_seed(0)
    U, summ = fit_mvs(model, tok, d["train"], ref["mu"], init_V(ref["Qm"], r, 0, "onmanifold"), h, steps=1600, seed=0)
    m, _, _ = retention_metrics(model, tok, d["val"], h, U, ref["mu"].cuda(), intact)
    res[t] = dict(hook=h, rank=r, sweep_kl=best["val"]["kl_retain"], val=m, train=summ)
    print(t, h, r, "sweep KL %.3f -> 1600-step KL %.3f; acc I/R/A %.3f/%.3f/%.3f; gap %.1f -> %.1f" % (
        best["val"]["kl_retain"], m["kl_retain"], m["acc_intact"], m["acc_retain"], m["acc_ablate"], m["gap_intact"], m["gap_ablate"]), flush=True)
json.dump(res, open(ROOT / "results/phase7/stage1/longcheck_val.json", "w"), indent=1)
