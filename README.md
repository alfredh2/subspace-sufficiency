# causal-subspaces
Phase 0: reproduce the rank-2 logic Minimal Viable Subspace (layer 22) of Qwen3-8B (Kean & Boix-Adserà, arXiv 2609.32907).
See `docs/paper_reproduction.md` for what is faithful vs approximated (the benchmark is a re-creation, not the paper's data).

```bash
conda activate ml            # or any env with torch, transformers, datasets, pyyaml, pandas, pytest
pip install -e ".[dev]"
pytest                                                              # unit tests (no 8B model needed)
python scripts/evaluate_baseline.py --run-name baseline             # intact baseline: logic splits + controls
python scripts/train_mvs.py --run-name mvs_rank2_seed0              # 1600-step rank-2 MVS (configs/experiments/mvs_rank2_l22.yaml; residual_index 23)
python scripts/evaluate_mvs.py --run-name mvs_rank2_seed0           # intact / retain / ablate + controls
python scripts/sweep_residual_index.py --indices 21 22 23 24        # layer-convention sweep (validation KL only)
```
Any config value can be overridden with `--set a.b=c`. Outputs go to `results/<run_name>/` (config, environment, checkpoints, metrics, predictions, training log).
Hardware settings live in `configs/hardware/`; nothing is cluster-specific.
