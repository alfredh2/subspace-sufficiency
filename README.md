# subspace-sufficiency

What does a fitted low-rank "minimal viable subspace" (MVS) actually license you to conclude?

The project started from a mechanistic question about Qwen3-8B: Kean & Boix-Adserà ([arXiv 2609.32907](https://arxiv.org/abs/2609.32907)) find a rank-2 residual-stream subspace at layer 22 that is sufficient for logical-validity judgements. Is that subspace implemented by a sparse set of specialised native MLP neurons? We reproduced it, then tested it with preregistered, randomly-controlled interventions across eight phases. The native-neuron hypothesis was **not supported**. What survived is a set of results about the subspace itself.

Scope: one model (Qwen3-8B, bf16), residual-stream hooks, re-created templated task data. Most phases ended MIXED or negative and are reported that way.

## Main findings

1. **Sufficient is not necessary or transplantable.** On a four-way QA task, a KL-fitted rank-4 subspace reproduces the model's output exactly (retention KL 0.000, 5/5 seeds). Ablating it at natural strength removes only about 5% of the answer margin, and swapping it between items moves the answer only about 9%. Ablating it at 4x strength removes 83-96%, so the effect is strongly dose-dependent. Orthogonal complements are each independently sufficient.
2. **The "logic" subspace is a verdict channel.** Retention is bound to the answer mapping: accuracy falls to 0.04 when the mapping is reversed. Ablation shifts the model to answering "valid" while the margin still ranks the classes (AUC 0.95-0.99). It also breaks a language localizer and PrOntoQA, and leaves verdict-free tasks alone. A search for a logic-specific subspace orthogonal to it found none.
3. **Retention, ablation and transplantation dissociate across answer formats.** One task gives a semantic-class handle, another a handle on the answer token. Only one direction of a rank-r subspace reproduces across seeds (top cosine about 0.8).
4. **Layer-19 mechanism.** Three heads (L19H10, H31, H17) copy value content from the conclusion into summary positions, and attention routing is not involved. "The d_c direction is inert" turned out to be a magnitude effect: at equal norm it matches its orthogonal remainder.
5. **Native-component concentration has a generic floor.** The top-1% share of the verdict write in the native MLP basis (0.49 against 0.13 for Haar) is partly generic: shuffled labels give 0.25. The decisive multi-task test of native-basis privilege could not be run (0 of 6 tasks passed the preregistered gate).

A recurring theme: effects at natural magnitude are threshold-like, so "component X is inert" is not a dose-independent claim.

The full evidence ranking, caveats, and what is and is not demonstrated are in [`docs/research_synthesis.md`](docs/research_synthesis.md).

## Layout

```
src/causal_subspaces/   library: MVS fit/interventions, hooks, data, per-phase modules
scripts/                one runner and one report script per phase
configs/                model, hardware, experiment configs
tests/                  unit tests (no 8B model needed)
docs/
  research_synthesis.md   start here
  methodology.md, experiment_log.md, paper_reproduction.md
  plans/                  preregistered plan per phase (written before the runs)
  reports/                report per phase
results/                committed summaries, verdicts and metrics per phase
```

| phase | question | verdict |
|---|---|---|
| 0, 0.5, 0.6 | reproduce the MVS; is it a logic variable? | verdict channel, no logic-selective target |
| 1-3 | native neurons / heads / mediation | MIXED |
| 4-6 | where the layer-19 value content comes from | MIXED |
| 7 | multi-task fixed basis, sufficiency vs necessity | gate not met; stopped |
| 8 | independent carriers, compensation (in progress) | no verdict yet |

## Reproducing

```bash
pip install -e ".[dev]"   # torch, transformers, datasets, pyyaml, pandas
pytest                    # unit tests, no GPU
python scripts/evaluate_baseline.py --run-name baseline
python scripts/train_mvs.py --run-name mvs_rank2_seed0     # rank-2 MVS, 1600 steps
python scripts/evaluate_mvs.py --run-name mvs_rank2_seed0
```

Later phases run through `scripts/phase*.py`; each plan in `docs/plans/` names its scripts and outputs. Config values can be overridden with `--set a.b=c`. Runs write to `results/<run_name>/`. Experiments used one RTX 5090 (32 GB); settings are in `configs/hardware/`.

Notes:
- The benchmark is a **re-creation**, not the paper's data; see `docs/paper_reproduction.md` for what is faithful and what is approximated.
- Five raw arrays over 5 MB are not tracked (`results/phase1/`, `results/phase2/`; see `.gitignore`). The phase scripts regenerate them.
- The Python package keeps its original import name, `causal_subspaces`.
