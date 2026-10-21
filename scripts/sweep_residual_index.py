"""Pre-registered convention sweep: short MVS fits at candidate residual indices, ranked by VALIDATION KL only
(no eval split, no ablation accuracy used for selection). Usage: python scripts/sweep_residual_index.py --indices 21 22 23 24 --steps 300"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--indices", nargs="+", type=int, default=[21, 22, 23, 24])
ap.add_argument("--steps", type=int, default=300)
a = ap.parse_args()
for i in a.indices:
    subprocess.run([sys.executable, "scripts/train_mvs.py", "--run-name", f"sweep_idx{i}_{a.steps}steps", "--set",
                    f"mvs.residual_index={i}", f"train.steps={a.steps}", "train.eval_every=100", "train.ckpt_every=100"], check=True)
