"""Stage 1b: frozen dev-only selection rule + cell labels from Arm A (and extensions if present). Writes stage1b/selection.json and
selection_table.md. Rule (plan section 6 / A9): sufficient cell = val retained-KL <= 0.05 in BOTH screening seeds; choose the lowest rank,
ties -> highest median dev relative ablation reduction, then earliest hook. No sufficient cell -> task fails sufficiency and the
lowest-median-val-KL cell is the diagnostic 'best cell' (used for the extensions)."""
import json, glob
import numpy as np
from causal_subspaces.phase7.stage1b import CELLS, TASKS_1B
from causal_subspaces.utils.config import ROOT

R = ROOT / "results/phase7/stage1b"
sel, lines = {}, ["| task | hook | rank | seeds | train KL (med) | val KL (seeds) | train slope last 25% | red. rel (seeds) | red. abs (seeds) | labels |", "|---|---|---|---|---|---|---|---|---|---|"]
for t in TASKS_1B:
    cells = {}
    for h, r in CELLS[t]:
        rows = [json.load(open(f)) for f in sorted(glob.glob(str(R / f"armA/{t}_h{h}_r{r}_s*.json")))]
        if len(rows) < 2:
            continue
        tr = np.array([x["train"]["kl_last5_median"] for x in rows]); va = np.array([x["val"]["kl_retain"] for x in rows])
        slope = np.array([x["train_slope_last25"] for x in rows])
        c = dict(hook=h, rank=r, n_seeds=len(rows), train_kl=tr.tolist(), val_kl=va.tolist(), slope=slope.tolist(),
                 red_rel=[x["reduction_rel"] for x in rows], red_abs=[x["reduction_abs"] for x in rows],
                 acc_ret_ratio=[x["val"]["acc_retain"] / x["val"]["acc_intact"] for x in rows],
                 sufficient=bool((va <= 0.05).all()),
                 opt_flags=dict(not_converged=bool(np.median(slope) >= 0.05), seed_spread=bool(tr.max() - tr.min() > 0.2 * np.median(tr))),
                 gen_limited=bool(np.median(tr) <= 0.25 * np.median(va) and np.median(va) - np.median(tr) >= 0.1))
        cells[f"h{h}_r{r}"] = c
    if not cells:
        continue
    suff = [c for c in cells.values() if c["sufficient"]]
    pick = min(suff, key=lambda c: (c["rank"], -float(np.median(c["red_rel"])), c["hook"])) if suff else None
    best = min(cells.values(), key=lambda c: float(np.median(c["val_kl"])))
    conv_fail_r8 = all((not c["sufficient"]) and float(np.median(c["train_kl"])) > 0.05 for c in cells.values() if c["rank"] == 8)
    sel[t] = dict(complete=len(cells) == len(CELLS[t]), n_cells=len(cells), sufficient_cells=[f"h{c['hook']}_r{c['rank']}" for c in suff],
                  choice=None if pick is None else dict(hook=pick["hook"], rank=pick["rank"]),
                  best_cell=dict(hook=best["hook"], rank=best["rank"]), rank8_all_train_kl_above_0p05=bool(conv_fail_r8), cells=cells)
    for c in cells.values():
        lab = [k for k, v in c["opt_flags"].items() if v] + (["generalization-gap"] if c["gen_limited"] else []) + (["SUFFICIENT"] if c["sufficient"] else [])
        lines.append("| %s | %d | %d | %d | %.3f | %s | %s | %s | %s | %s |" % (t, c["hook"], c["rank"], c["n_seeds"], np.median(c["train_kl"]),
                     "/".join("%.3f" % x for x in c["val_kl"]), "/".join("%.2f" % x for x in c["slope"]), "/".join("%.2f" % x for x in c["red_rel"]),
                     "/".join("%.1f" % x for x in c["red_abs"]), ",".join(lab) or "-"))
n_suff = sum(1 for t in sel if sel[t]["choice"] is not None)
out = dict(tasks=sel, n_tasks_with_sufficient_cell=n_suff, early_stop_four_task_gate_unreachable=bool(n_suff < 4 and all(s["complete"] for s in sel.values()) and len(sel) == 6))
json.dump(out, open(R / "selection.json", "w"), indent=1)
open(R / "selection_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print({t: (s["choice"], s["best_cell"]) for t, s in sel.items()}); print("tasks with a sufficient cell:", n_suff, "early-stop flag:", out["early_stop_four_task_gate_unreachable"])
