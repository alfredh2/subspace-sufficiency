"""Train an MVS. Usage: python scripts/train_mvs.py --config CFG --run-name NAME [--resume] [--set train.steps=20 ...]"""
import argparse
import time

import torch

from causal_subspaces.data.logic import dataset_hash
from causal_subspaces.experiment import get_reference, get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.optimization import train_mvs
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.utils.config import ROOT, load_config
from causal_subspaces.utils.logging import RunDir
from causal_subspaces.utils.reproducibility import environment_info, seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/experiments/mvs_rank2_l22.yaml")
ap.add_argument("--run-name", default=None)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(ROOT / a.config, a.set)
name = a.run_name or cfg["run_name"]
seed_everything(cfg["seed"])
run = RunDir(ROOT / cfg["hardware"]["results_root"], name, exist_ok=a.resume)
run.write_config(cfg)
dev = cfg["hardware"]["device"]
model, tok, rev = load_model(cfg["model"], dev)
splits = get_splits(cfg)
t0 = time.time()
ref = get_reference(model, tok, cfg, rev, dev)
ref_s = time.time() - t0
run.write_json("environment.json", environment_info(rev, dict(
    dataset_hash=dataset_hash(splits), residual_index=cfg["mvs"]["residual_index"], dtype=cfg["model"]["dtype"],
    ref_tokens=ref["n_tokens"], ref_seconds=ref_s, init_top_m_var_fraction=ref["top_m_var"] / ref["total_var"])))
V0 = init_V(ref["Qm"], cfg["mvs"]["rank"], cfg["seed"], cfg["mvs"]["init"])
U, summ = train_mvs(model, tok, splits, ref["mu"], V0, cfg, run, dev, resume=a.resume)
run.write_json("train_summary.json", summ)
print(summ)
