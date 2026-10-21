# Research overview
Project: *From logical subspaces to native neurons.* Question: is a task's minimal viable subspace (MVS) in the residual stream implemented by a
sparse set of specialised native MLP neurons or a distributed population? Chain: native neurons → residual logic subspace → logic behaviour.

Phases: **0** reproduce Qwen3-8B layer-22 rank-2 logic MVS (this repo, implemented) · 1 neuron write-contribution analysis · 2 random-rotation
control on post-activation MLP coordinates · 3 causal neuron ablations with matched controls · 4 circuit/semantic interpretation.
Only Phase 0 exists in code. Lessons carried over: causal importance ≠ semantic interpretability; verify causality before verbalising.
Future modules (`neurons/`, `rotations/`, `interventions/`, `interpretation/`) will reuse `models/`, `data/`, `evaluation/`, `mvs/checkpoint.py`.
