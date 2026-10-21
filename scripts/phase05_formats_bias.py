"""Phase 0.5, Experiments 1 and 4: label/format interventions and decision-bias diagnostics.
Frozen model + the three existing MVS checkpoints; nothing is retrained.
Usage: python scripts/phase05_formats_bias.py [--seeds 0 1 2]"""
import argparse
import json

import numpy as np
import yaml

from causal_subspaces import phase05 as P
from causal_subspaces.experiment import get_controls, get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.checkpoint import load_checkpoint
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
a = ap.parse_args()
OUT = ROOT / "results" / "phase05"
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx, bs = cfg["hardware"]["device"], cfg["mvs"]["residual_index"], cfg["hardware"]["eval_batch_size"]
model, tok, rev = load_model(cfg["model"], dev)
splits, controls = get_splits(cfg), get_controls(cfg)
ex = [e for sp in ("items", "families", "nonwords") for e in splits[sp]]
gold_valid = np.array([e.valid for e in ex])
split = np.array([e.split for e in ex])
cks = {s: load_checkpoint(ROOT / f"results/mvs_rank2_seed{s}/checkpoints/step_1600.pt") for s in a.seeds}
mu = cks[a.seeds[0]]["mu"].to(dev)
conds = {"intact": ("noop", None)}
for s in a.seeds:
    conds[f"retain_s{s}"] = ("retain", cks[s]["U"].to(dev))
    conds[f"ablate_s{s}"] = ("ablate", cks[s]["U"].to(dev))

stats, arrays = {}, {}
def run(task, texts, aff, rej, gold_aff, lit_first=None, sp=None):
    for cname, (mode, U) in conds.items():
        rec = P.collect(model, tok, texts, aff, rej, device=dev, batch_size=bs, residual_index=idx, mode=mode, U=U, mu=mu)
        stats[f"{task}/{cname}"] = P.answer_stats(tok, rec, gold_aff, aff, rej, lit_first, sp)
        for k, v in rec.items():
            arrays[f"{task}|{cname}|{k}"] = v
        st = stats[f"{task}/{cname}"]
        print(f"{task:22s} {cname:11s} acc={st['acc']:.3f} inv={st['invalid_rate']:.3f} pred_aff={st['pred_aff']:.2f} auc={st['auc']:.3f} "
              f"margin={st['margin_mean']:+.2f}", flush=True)

# Experiment 1: five formats on the same 768 logic examples (semantic ground truth identical in all)
for fmt, f in P.FORMATS.items():
    run(f"logic/{fmt}", [P.build_prompt(fmt, e.syllogism) for e in ex], f["valid"], f["invalid"], gold_valid, f["lit_first"], split)
# Experiment 4 extras: the two verdict-structured controls in their original 1/2 format ("1" = affirmative)
for name in ("language_localizer", "prontoqa"):
    items = controls[name]["items"]
    run(f"control/{name}", [t for t, _ in items], "1", "2", np.array([g == "1" for _, g in items]), "1")
# Reference verdict-free control with the same label tokens (is the effect specific to verdict structure?)
items = controls["mmlu_binary"]["items"]
run("control/mmlu_binary", [t for t, _ in items], "1", "2", np.array([g == "1" for _, g in items]), "1")

json.dump(dict(model_revision=rev, residual_index=idx, seeds=a.seeds, stats=stats), open(OUT / "exp1_exp4_stats.json", "w"), indent=1)
np.savez_compressed(OUT / "exp1_exp4_arrays.npz", **arrays)
print("saved")
