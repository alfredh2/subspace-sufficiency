"""Stage 1b: MVS sweep (hooks x ranks, 300 steps, seed 0) on the FIRST format of each task, evaluated on VAL only,
with the frozen choice rule of plan section 12 (+12.2). Usage: python scripts/phase7_stage1_sweep.py --tasks L S G N M"""
import argparse, json, time
import numpy as np, torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.phase7.tasks import FORMATS, build_task
from causal_subspaces.phase7.stage1 import fit_mvs, retention_metrics, run_stats, gold_array
from causal_subspaces.utils.config import ROOT, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="*", default=["L", "S", "G", "N", "M"])
ap.add_argument("--hooks", nargs="*", type=int, default=[19, 21, 23, 25])
ap.add_argument("--ranks", nargs="*", type=int, default=[1, 2, 4])
ap.add_argument("--steps", type=int, default=300)
a = ap.parse_args()
out_dir = ROOT / "results/phase7/stage1/sweep"
out_dir.mkdir(parents=True, exist_ok=True)
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
refs = {}
for h in a.hooks:
    cfg["mvs"]["residual_index"] = h
    t0 = time.time()
    refs[h] = get_reference(model, tok, cfg, rev, "cuda")
    print("reference", h, round(time.time() - t0), "s", flush=True)


def passes(m):
    red = m["gap_intact"] - m["gap_ablate"]
    lit = red >= 0.5 * m["gap_intact"] and red >= 10
    adj = red >= 0.5 * m["gap_intact"] and red >= min(10, 0.5 * m["gap_intact"])
    return dict(reduction=red, rel=red / m["gap_intact"], literal=bool(lit), adjusted=bool(adj), kl_ok=bool(m["kl_retain"] <= 0.05))


for t in a.tasks:
    d = build_task(t, tok)[FORMATS[t][0]]
    intact = run_stats(model, tok, d["val"])
    rows = []
    for h in a.hooks:
        for r in a.ranks:
            f = out_dir / f"{t}_h{h}_r{r}.json"
            if f.exists():
                rows.append(json.load(open(f))); continue
            torch.manual_seed(0)
            V0 = init_V(refs[h]["Qm"], r, 0, "onmanifold")
            U, summ = fit_mvs(model, tok, d["train"], refs[h]["mu"], V0, h, steps=a.steps, seed=0, micro=16)
            m, _, _ = retention_metrics(model, tok, d["val"], h, U, refs[h]["mu"].cuda(), intact)
            row = dict(task=t, fmt=FORMATS[t][0], hook=h, rank=r, train=summ, val=m, gate=passes(m))
            json.dump(row, open(f, "w"), indent=1)
            torch.save(U.cpu(), out_dir / f"{t}_h{h}_r{r}.pt")
            rows.append(row)
            print(t, h, r, "klret %.4f" % m["kl_retain"], "accI/R/A %.3f/%.3f/%.3f" % (m["acc_intact"], m["acc_retain"], m["acc_ablate"]),
                  "gap %.1f -> %.1f" % (m["gap_intact"], m["gap_ablate"]), row["gate"], "%.0fs" % summ["seconds"], flush=True)
    ok = [x for x in rows if x["gate"]["kl_ok"] and x["gate"]["adjusted"]]
    okl = [x for x in ok if x["gate"]["literal"]]
    pick = min(ok, key=lambda x: (x["rank"], x["hook"])) if ok else None
    json.dump(dict(task=t, choice=None if pick is None else dict(hook=pick["hook"], rank=pick["rank"]),
                   literal_choice=None if not okl else dict(zip(("hook", "rank"), min(((x["hook"], x["rank"]) for x in okl), key=lambda p: (p[1], p[0])))),
                   n_pass=len(ok), n_pass_literal=len(okl), n_total=len(rows)), open(out_dir / f"choice_{t}.json", "w"), indent=1)
    print("CHOICE", t, None if pick is None else (pick["hook"], pick["rank"]), flush=True)
