"""Stage 1c: final MVS per (task, format) x seeds 0-2 at the sweep-chosen hook/rank (1600 steps), then all gates on TEST.
Usage: python scripts/phase7_stage1_final.py --tasks L S G N M  (needs results/phase7/stage1/sweep/choice_<T>.json)"""
import argparse, json, time
import numpy as np, torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.phase05 import consensus_directions
from causal_subspaces.phase7.tasks import FORMATS, build_task
from causal_subspaces.phase7.stage1 import (fit_mvs, retention_metrics, run_stats, pair_trials, transplant, ratio_stat, haar_subspace,
                                             gold_array, cluster_boot, primary_accuracy_vec)
from causal_subspaces.utils.config import ROOT, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="*", default=["L", "S", "G", "N", "M"])
ap.add_argument("--steps", type=int, default=1600)
ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
ap.add_argument("--n-random", type=int, default=20)
a = ap.parse_args()
root = ROOT / "results/phase7/stage1/final"
root.mkdir(parents=True, exist_ok=True)
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
for t in a.tasks:
    ch = json.load(open(ROOT / f"results/phase7/stage1/sweep/choice_{t}.json"))["choice"]
    if ch is None:
        print("TASK", t, "has no passing sweep configuration: recorded as failed feature discovery"); continue
    h, r = ch["hook"], ch["rank"]
    cfg["mvs"]["residual_index"] = h
    ref = get_reference(model, tok, cfg, rev, "cuda")
    mu = ref["mu"].cuda()
    data = build_task(t, tok)
    for fmt in FORMATS[t]:
        d = data[fmt]
        f_json = root / f"{t}_{fmt}.json"
        if f_json.exists():
            continue
        intact = run_stats(model, tok, d["test"])
        Us, rows = {}, []
        for seed in a.seeds:
            torch.manual_seed(seed)
            V0 = init_V(ref["Qm"], r, seed, "onmanifold")
            U, summ = fit_mvs(model, tok, d["train"], ref["mu"], V0, h, steps=a.steps, seed=seed, micro=16)
            torch.save(U.cpu(), root / f"{t}_{fmt}_h{h}_r{r}_s{seed}.pt")
            m, ret, abl = retention_metrics(model, tok, d["test"], h, U, mu, intact)
            Us[seed] = U
            rows.append(dict(seed=seed, train=summ, test=m))
            print(t, fmt, "seed", seed, "klret %.4f accI/R/A %.3f/%.3f/%.3f gap %.1f->%.1f" % (
                m["kl_retain"], m["acc_intact"], m["acc_retain"], m["acc_ablate"], m["gap_intact"], m["gap_ablate"]), "%.0fs" % summ["seconds"], flush=True)
        # CIs (content-clustered) for the per-item quantities of seed rows
        g = gold_array(d["test"])
        groups = [i.group for i in d["test"]]
        for row, seed in zip(rows, a.seeds):
            U = Us[seed]
            ret = run_stats(model, tok, d["test"], h, "retain", U, mu, with_kl=True)
            abl = run_stats(model, tok, d["test"], h, "ablate", U, mu, with_kl=True)
            from causal_subspaces.phase7.stage1 import class_gap
            def red(ix):
                gi = class_gap(intact["cand_logits"][ix], g[ix]); ga = class_gap(abl["cand_logits"][ix], g[ix])
                return gi - ga
            row["ci_reduction"] = cluster_boot(red, groups, 300, seed)
            row["ci_rel_reduction"] = cluster_boot(lambda ix: red(ix) / class_gap(intact["cand_logits"][ix], g[ix]), groups, 300, seed)
            row["ci_kl_retain"] = cluster_boot(lambda ix: ret["kl"][ix].mean(), groups, 300, seed)
            a_i, a_r = primary_accuracy_vec(intact, d["test"]), primary_accuracy_vec(ret, d["test"])
            row["ci_acc_ratio"] = cluster_boot(lambda ix: a_r[ix].mean() / a_i[ix].mean(), groups, 300, seed)
        # consensus across seeds
        cons = consensus_directions({s: Us[s] for s in a.seeds})
        S = len(a.seeds)
        c_vals = {s: float(np.sqrt(max(cons[s]["eig_shared"], 0) / (S - 1))) for s in a.seeds}
        sh = torch.stack([cons[s]["shared"] for s in a.seeds])
        dtask = torch.nn.functional.normalize(sh.mean(0), dim=0)
        # transplantation: seeds + random equal-rank subspaces
        trials = pair_trials(d["pairs"])
        subs = {f"s{s}": Us[s] for s in a.seeds}
        for k in range(a.n_random):
            subs[f"rand{k}"] = haar_subspace(Us[a.seeds[0]].shape[0], r, 1000 + k)
        res = transplant(model, tok, trials, h, subs, bs=16)
        tr = {}
        for nm in subs:
            for pos in ("all", "last"):
                num = res[f"m|{nm}|{pos}"] - res["m_tgt"]; den = res["m_src"] - res["m_tgt"]
                tr[f"{nm}|{pos}"] = ratio_stat(num, den, res["group"], 300) if nm.startswith("s") else (float(num.sum() / den.sum()),)
        out = dict(task=t, fmt=fmt, hook=h, rank=r, seeds=a.seeds, rows=rows, consensus_cos=c_vals, consensus_eig=[cons[s]["eig_shared"] for s in a.seeds],
                   dtask=dtask.tolist(), cos_dtask_seed=[float(dtask @ cons[s]["shared"]) for s in a.seeds], transplant=tr,
                   n_trials=len(trials), m_gap_mean=float((res["m_src"] - res["m_tgt"]).mean()))
        json.dump(out, open(f_json, "w"), indent=1)
        torch.save(dict(dtask=dtask, shared=sh, U={s: Us[s].cpu() for s in a.seeds}), root / f"{t}_{fmt}_directions.pt")
        np.savez(root / f"{t}_{fmt}_transplant.npz", **{k: v for k, v in res.items()})
        print("DONE", t, fmt, "consensus", c_vals, "transplant all:", {k: round(v[0], 3) for k, v in tr.items() if k.endswith("|all") and k.startswith("s")},
              "rand mean all %.3f" % np.mean([tr[f"rand{k}|all"][0] for k in range(a.n_random)]), flush=True)
