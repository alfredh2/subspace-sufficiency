"""Phase 8 S1: fresh-entity baseline screen (intact model only, no intervention) and data freeze manifest.
Writes results/phase8/stage1/data_manifest.json and fresh_items.pkl. Usage: python scripts/phase8_s1_data.py"""
import json, pickle
import numpy as np
from causal_subspaces.models.loading import load_model
from causal_subspaces.phase7.stage1 import run_stats, primary_accuracy_vec
from causal_subspaces.phase8.data import assign, build_items, check_disjoint, items_hash
from causal_subspaces.utils.config import ROOT, load_config

check_disjoint()
R = ROOT / "results/phase8/stage1"
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
ents = assign()
scr = build_items(tok, ents, reps=2, seed=0)                    # screening items: all assigned entities
keep = {r: {} for r in ents}
rep = {}
for sp, d in scr.items():
    st = run_stats(model, tok, d["items"])
    acc = primary_accuracy_vec(st, d["items"])
    per = {}
    for it, a_ in zip(d["items"], acc):
        per.setdefault(it.group[2:], []).append(a_)
    for r in ents:
        keep[r][sp] = [e for e in ents[r][sp] if np.mean(per[e]) >= 0.9]
    rep[sp] = dict(acc=float(acc.mean()), per_entity={e: float(np.mean(v)) for e, v in per.items()},
                   dropped=[e for r in ents for e in ents[r][sp] if e not in keep[r][sp]])
    print(sp, "screen acc %.3f" % acc.mean(), "dropped", rep[sp]["dropped"], flush=True)
final = build_items(tok, keep, reps=2, seed=1)                  # rebuilt from kept entities with a fresh stream
for sp, d in final.items():
    st = run_stats(model, tok, d["items"])
    rep[sp]["final_intact_acc"] = float(primary_accuracy_vec(st, d["items"]).mean())
    rep[sp]["n_items"] = len(d["items"]); rep[sp]["n_pairs"] = len(d["pairs"])
    rep[sp]["n_entities"] = {r: len(keep[r][sp]) for r in keep}
    rep[sp]["hash"] = items_hash(d["items"]); rep[sp]["pair_hash"] = items_hash([x for p in d["pairs"] for x in p])
manifest = dict(assign_seed=8001, assignment=ents, kept=keep, report=rep, build_seed_screen=0, build_seed_final=1,
                n_conf_entities=sum(len(keep[r]["conf"]) for r in keep), n_dev_entities=sum(len(keep[r]["devf"]) for r in keep))
json.dump(manifest, open(R / "data_manifest.json", "w"), indent=1)
pickle.dump(final, open(R / "fresh_items.pkl", "wb"))
print("confirm entities", manifest["n_conf_entities"], "dev-fresh", manifest["n_dev_entities"], flush=True)
