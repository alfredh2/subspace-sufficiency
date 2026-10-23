"""Stage 1a: baseline (intact) accuracy per task/format. `--split val` is used for prompt debugging (allowed, val only);
`--split test` is the frozen feasibility screen (gate a). Natural-data baselines (SST-2, MMLU) are reported separately."""
import argparse, json, sys
import numpy as np
from causal_subspaces.models.loading import load_model
from causal_subspaces.phase7.tasks import FORMATS, build_task, data_hash
from causal_subspaces.phase7.stage1 import run_stats, accuracy_vec, primary_accuracy_vec, restricted_accuracy_vec, cluster_boot, gold_array, class_gap
from causal_subspaces.utils.config import ROOT, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--tasks", nargs="*", default=["L", "S", "G", "N", "M", "A"])
ap.add_argument("--out", default=None)
a = ap.parse_args()
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
res = {}
for t in a.tasks:
    d = build_task(t, tok)
    for fmt, sp in d.items():
        items = sp[a.split]
        st = run_stats(model, tok, items)
        acc, racc = primary_accuracy_vec(st, items), restricted_accuracy_vec(st, items)
        lo, hi = cluster_boot(lambda ix: acc[ix].mean(), [i.group for i in items], 500)
        g = gold_array(items)
        r = dict(n=len(items), gacc=float(accuracy_vec(st, items).mean()), acc=float(acc.mean()), ci=[lo, hi], racc=float(racc.mean()), mass=float(np.exp(st["cand_logits"] - st["lse"][:, None]).sum(1).mean()),
                 gap=class_gap(st["cand_logits"], g), by_class=[float(acc[g == c].mean()) for c in np.unique(g)], hash=data_hash(items))
        res[f"{t}/{fmt}"] = r
        print(t, fmt, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "ci"}, [round(x, 3) for x in r["ci"]], flush=True)
    if t == "S" and a.split == "val":
        from datasets import load_dataset
        from causal_subspaces.phase7.tasks import Item, s_prompt, _ids
        ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
        items = []
        for fmt in FORMATS["S"]:
            cand = {"words": ["positive", "negative"], "num_orig": ["1", "2"], "num_rev": ["2", "1"]}[fmt]
            items = [Item(f"sst{i}", "S", fmt, "sst2", s_prompt(fmt, x["sentence"].strip()), False, cand, _ids(tok, cand), 0 if x["label"] == 1 else 1, f"s{i}")
                     for i, x in enumerate(ds) if i % 3 == 0][:256]
            st = run_stats(model, tok, items)
            res[f"S_sst2/{fmt}"] = dict(n=len(items), acc=float(accuracy_vec(st, items).mean()), racc=float(restricted_accuracy_vec(st, items).mean()))
            print("SST2", fmt, res[f"S_sst2/{fmt}"], flush=True)
if a.out:
    json.dump(res, open(ROOT / a.out, "w"), indent=1)
