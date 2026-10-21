"""Intact-model baseline on the logic splits + controls. Usage: python scripts/evaluate_baseline.py --config CFG [--run-name N] [--set a.b=c]"""
import argparse

from causal_subspaces.experiment import evaluate_condition, get_controls, get_splits
from causal_subspaces.data.logic import dataset_hash
from causal_subspaces.models.loading import load_model
from causal_subspaces.utils.config import ROOT, load_config
from causal_subspaces.utils.logging import RunDir
from causal_subspaces.utils.reproducibility import environment_info, seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/experiments/mvs_rank2_l22.yaml")
ap.add_argument("--run-name", default="baseline")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(ROOT / a.config, a.set)
seed_everything(cfg["seed"])
run = RunDir(ROOT / cfg["hardware"]["results_root"], a.run_name)
run.write_config(cfg)
model, tok, rev = load_model(cfg["model"], cfg["hardware"]["device"])
splits, controls = get_splits(cfg), get_controls(cfg)
run.write_json("environment.json", environment_info(rev, {"dataset_hash": dataset_hash(splits)}))
metrics, preds = evaluate_condition(model, tok, cfg, splits, controls, "noop")
run.write_json("metrics.json", metrics)
w = run.jsonl("predictions.jsonl", "w")
for p in preds:
    w.write(**p)
w.close()
for k, v in metrics["logic"].items():
    o = v["overall"]
    print(f"logic/{k}: acc={o['acc']:.3f} restricted={o['acc_restricted']:.3f} invalid={o['invalid_rate']:.3f}")
for k, v in metrics["controls"].items():
    print(f"control/{k}:", v["overall"]["acc"] if isinstance(v, dict) else v)
