"""Evaluate intact / retain / ablate for a trained MVS run. Usage: python scripts/evaluate_mvs.py --run-name NAME [--ckpt step_1600.pt]"""
import argparse
import json

from causal_subspaces.data.logic import dataset_hash
from causal_subspaces.evaluation.metrics import recovery
from causal_subspaces.experiment import evaluate_condition, get_controls, get_lm_passages, get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.checkpoint import latest_checkpoint, load_checkpoint
from causal_subspaces.utils.config import ROOT, load_config
from causal_subspaces.utils.logging import RunDir
from causal_subspaces.utils.reproducibility import seed_everything

PAPER = {"intact": (0.99, 0.97, 0.97), "retain": (0.99, 0.96, 0.96), "ablate": (0.51, 0.50, 0.51)}  # Items/Families/Nonwords (Fig. 4 caption; 2-model mean, our reading)
ap = argparse.ArgumentParser()
ap.add_argument("--run-name", required=True)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--config", default=None)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
rd = ROOT / "results" / a.run_name
import yaml
cfg = load_config(ROOT / a.config, a.set) if a.config else yaml.safe_load((rd / "config.yaml").read_text())
seed_everything(cfg["seed"])
dev = cfg["hardware"]["device"]
ck = load_checkpoint(rd / "checkpoints" / a.ckpt if a.ckpt else latest_checkpoint(rd / "checkpoints"))
model, tok, rev = load_model(cfg["model"], dev)
splits, controls = get_splits(cfg), get_controls(cfg)
passages = get_lm_passages(tok, cfg)
U, mu = ck["U"].to(dev), ck["mu"].to(dev)
out = {"checkpoint_step": ck["step"], "dataset_hash": dataset_hash(splits), "conditions": {}}
w = RunDir(ROOT / "results", a.run_name, exist_ok=True).jsonl("eval_predictions.jsonl", "w")
for cond, mode in [("intact", "noop"), ("retain", "retain"), ("ablate", "ablate")]:
    m, preds = evaluate_condition(model, tok, cfg, splits, controls, mode, U, mu, passages, dev)
    out["conditions"][cond] = m
    for p in preds:
        w.write(**p)
w.close()
chance = 0.5
for cond, m in out["conditions"].items():
    out.setdefault("recovery", {})[cond] = {k: recovery(m["logic"][k]["overall"]["acc"], out["conditions"]["intact"]["logic"][k]["overall"]["acc"], chance) for k in m["logic"]}
out["paper_reference_items_families_nonwords"] = PAPER
(rd / "eval_metrics.json").write_text(json.dumps(out, indent=2, default=float))
print(f"{'cond':8s} " + " ".join(f"{k:>16s}" for k in ("items", "families", "nonwords")) + "   (acc [inv%])  | paper")
for cond, m in out["conditions"].items():
    print(f"{cond:8s} " + " ".join(f"{m['logic'][k]['overall']['acc']:.3f} [{100*m['logic'][k]['overall']['invalid_rate']:4.1f}%]".rjust(16) for k in ("items", "families", "nonwords")), "|", PAPER[cond])
print("\ncontrols (acc / lm_kl):")
for t in out["conditions"]["intact"]["controls"]:
    print(f"{t:24s}", *[f"{out['conditions'][c]['controls'][t]['overall']['acc'] if t != 'lm_kl' else out['conditions'][c]['controls'][t]:.3f}" for c in out["conditions"]])
