# Phase 7 Stage 1b report: diagnosing why low-dimensional MVS discovery failed or was unstable

Scope: Stage 1b only, as preregistered in `docs/phase7_stage1b_plan.md` (amendments A1–A14 committed before implementation or any fit; selection committed before any confirm content was evaluated). Stage 1 verdicts (L PASS, S INCONCLUSIVE, G/N/M FAIL, insufficient panel) are untouched; nothing below is a retroactive Stage 1 result. Stage 2–4 were not started. No pushes, no co-author trailers.

## Headline
* **Four-task gate: not met. 0 of 6 candidate tasks (L, S, G, N, M, A) are 1b-PASS at tier 1.** Per the preregistered rule, the recommendation is to **stop the multi-task fixed-basis programme and consolidate** (section 7).
* Per task: **L** sufficiency PASS, transplantation PASS, necessity FAIL (2/5 seeds); **S** primary format (`words`) sufficiency FAIL on the confirm contents (KL 0.13 vs 0.003 on dev), necessity INCONCLUSIVE; **G** and **N**: no sufficient cell in the 8-cell grid (best dev val KL 0.88 and 3.8); **M** sufficiency PASS (KL 0.000) but necessity FAIL and transplantation FAIL (0.09); **A** sufficiency FAIL (2/5 seeds), necessity and transplantation FAIL.
* **Main methodological findings (new evidence, descriptive):** (i) sufficiency of a fitted subspace is easy to obtain and is *not* evidence of a causal handle: for M three mutually orthogonal rank-2…8 subspaces each retain the output with KL ≈ 0, yet ablating the fitted subspace at natural strength removes ≤ 6 % of the margin gap while scaling the ablation 4× removes 83–96 % (a magnitude/threshold effect, not an absent representation); (ii) necessity falls as rank rises (L: Stage 1 rank-2 passed, 1b rank-4 fails); (iii) for L and S, ablating the union of the fitted subspace and an orthogonal complement fit (rank 4+8) removes 65 % / 86 % of the gap against 1–2 % for random subspaces (one seed): the information is spread over more than the fitted subspace; (iv) retention-sufficiency is answer-format-bound for L and S, but for L the transplantation effect transfers across formats and follows the *semantic* class, while for S it follows the *answer token*; (v) G and N failures are mostly not optimisation failures: more steps and a supervised initialisation move validation KL by ≤ 4 %, N's failure is dominated by the name-pool shift (validation KL 3.8 → 0.16 when names are shared across splits).

## 1. What was run, and deviations
* Arm A: dev-only grid, 8 cells (hooks {21,23,25,27} × ranks {4,8}; M 6 cells), 2 seeds, 1600 steps, first format, 1b partition (seed 1001), val ≤ 256 items. 92 fits. Selection rule frozen in `selection.json` (committed before any confirm evaluation).
* Confirm: tasks with a sufficient dev cell (L, S, M, A), seeds 0–4 at the frozen cell; seeds 0,1 of the primary format are the Arm A fits (reused; evaluated on confirm contents for the first time), seeds 2–4 are new fits. Second formats were fitted for L and S always, and for M and A only if the primary gates all passed (A5 ii): they did not, so M-`digits` and A-`plus` were **not run**, and G(ii) is "not run" for them (their 1b-G is FAIL through transplantation anyway).
* Extensions (A5 i) for G and N only; tier-2 diagnostics (candidate-restricted objective, N-fixedpool) never enter the gate.
* Arm B and Arm C as planned, with these reductions: Arm B contrast tasks (L, S) use **one seed and complement rank 8 only**; `M_h25_r2` uses 2 seeds and complement ranks 2 and 8; nested nulling is reduced to **one union level** (ablate [U, U2], ranks 6/8/12, not a chain to rank 16). Validation KL is measured at the end of fits (not every 200 steps); the dense train-KL curve (every 25 steps) is the convergence record. The `not_converged` flag uses the last-25 % relative decrease of that curve (≥ 5 %), which also fires for curves already at ~0.001, so it is only informative for tasks with train KL > 0.05 (G, N).
* Data: realised as the repartition of A13/A14 (finite lexicons; N 14 confirm / 6 val / 19 train names; Stage 1 had 11 training names), so 1b-confirm contents can coincide with Stage 1 contents and the L and S choice is not independent of Stage 1 knowledge. The grid did not contain rank 2, so **L's 1b cell is (23, rank 4), not Stage 1's (23, rank 2)**; the L necessity result at rank 4 is consistent with the Stage 1 sweep (h23 r4: reduction 0.20, 5.8 logits).
* **Budget overrun.** GPU time ≈ 7.6 h (grid ≈ 3.5 h; confirm, extensions, Arm C, Arm B ≈ 4.0 h) against the 4–6 h soft budget. Reason: L and M prompts are long (L fits 250–280 s, M 150–180 s at 1600 steps), so 92 screening fits and 5-seed confirm runs cost about 1.5× the plan's estimate. Seed coverage was not reduced; only the Arm B contrasts were trimmed (above).

## 2. Frozen dev-only selection (Arm A)
| task | hook | rank | seeds | train KL (med) | val KL (seeds) | train slope last 25% | red. rel (seeds) | red. abs (seeds) | labels |
|---|---|---|---|---|---|---|---|---|---|
| L | 21 | 4 | 2 | 0.003 | 0.030/0.036 | 0.52/0.64 | 0.37/0.37 | 10.6/10.6 | not_converged,seed_spread,SUFFICIENT |
| L | 21 | 8 | 2 | 0.003 | 0.028/0.037 | 0.55/0.60 | 0.38/0.46 | 10.9/13.2 | not_converged,seed_spread,SUFFICIENT |
| L | 23 | 4 | 2 | 0.002 | 0.006/0.005 | 0.63/0.51 | 0.59/0.61 | 16.9/17.4 | not_converged,seed_spread,SUFFICIENT |
| L | 23 | 8 | 2 | 0.001 | 0.008/0.011 | 0.63/0.79 | 0.71/0.57 | 20.2/16.5 | not_converged,seed_spread,SUFFICIENT |
| L | 25 | 4 | 2 | 0.001 | 0.001/0.001 | 0.54/0.42 | 0.39/0.61 | 11.2/17.5 | not_converged,seed_spread,SUFFICIENT |
| L | 25 | 8 | 2 | 0.000 | 0.001/0.001 | 0.58/0.46 | 0.62/0.57 | 17.6/16.3 | not_converged,seed_spread,SUFFICIENT |
| L | 27 | 4 | 2 | 0.001 | 0.001/0.000 | 0.21/0.31 | 0.51/0.46 | 14.7/13.2 | not_converged,seed_spread,SUFFICIENT |
| L | 27 | 8 | 2 | 0.001 | 0.001/0.000 | 0.49/0.29 | 0.69/0.68 | 19.6/19.5 | not_converged,seed_spread,SUFFICIENT |
| S | 21 | 4 | 2 | 0.004 | 0.009/0.036 | 0.31/0.39 | 0.16/0.09 | 4.7/2.6 | not_converged,seed_spread,SUFFICIENT |
| S | 21 | 8 | 2 | 0.004 | 0.008/0.022 | 0.46/0.44 | 0.51/0.32 | 15.4/9.6 | not_converged,seed_spread,SUFFICIENT |
| S | 23 | 4 | 2 | 0.003 | 0.009/0.007 | 0.53/0.61 | 0.52/0.53 | 15.6/16.0 | not_converged,seed_spread,SUFFICIENT |
| S | 23 | 8 | 2 | 0.002 | 0.006/0.006 | 0.63/0.68 | 0.67/0.71 | 20.0/21.3 | not_converged,seed_spread,SUFFICIENT |
| S | 25 | 4 | 2 | 0.001 | 0.004/0.003 | 0.52/0.35 | 0.55/0.57 | 16.4/16.9 | not_converged,seed_spread,SUFFICIENT |
| S | 25 | 8 | 2 | 0.000 | 0.003/0.003 | 0.53/0.40 | 0.75/0.74 | 22.4/22.1 | not_converged,seed_spread,SUFFICIENT |
| S | 27 | 4 | 2 | 0.001 | 0.003/0.003 | 0.58/0.22 | 0.68/0.40 | 20.3/11.9 | not_converged,seed_spread,SUFFICIENT |
| S | 27 | 8 | 2 | 0.000 | 0.003/0.003 | 0.67/0.21 | 0.64/0.67 | 19.2/20.1 | not_converged,seed_spread,SUFFICIENT |
| G | 21 | 4 | 2 | 0.506 | 1.032/1.069 | 0.10/0.11 | 0.03/0.09 | 0.3/0.8 | not_converged |
| G | 21 | 8 | 2 | 0.299 | 0.999/0.919 | 0.14/0.14 | 0.06/0.17 | 0.6/1.5 | not_converged,seed_spread |
| G | 23 | 4 | 2 | 0.474 | 1.047/1.013 | 0.09/0.10 | 0.06/0.11 | 0.5/1.0 | not_converged,seed_spread |
| G | 23 | 8 | 2 | 0.307 | 0.923/0.916 | 0.13/0.12 | 0.27/0.31 | 2.4/2.7 | not_converged,seed_spread |
| G | 25 | 4 | 2 | 0.486 | 1.078/1.023 | 0.09/0.10 | 0.09/0.08 | 0.8/0.7 | not_converged,seed_spread |
| G | 25 | 8 | 2 | 0.284 | 0.956/0.863 | 0.14/0.12 | 0.13/0.15 | 1.1/1.3 | not_converged,seed_spread |
| G | 27 | 4 | 2 | 0.481 | 1.085/1.045 | 0.09/0.10 | 0.13/0.14 | 1.2/1.2 | not_converged,seed_spread |
| G | 27 | 8 | 2 | 0.274 | 0.904/0.896 | 0.12/0.11 | 0.30/0.21 | 2.6/1.8 | not_converged,seed_spread |
| N | 21 | 4 | 2 | 0.200 | 4.815/5.631 | 0.20/0.27 | 0.24/0.10 | 4.6/2.0 | not_converged,seed_spread,generalization-gap |
| N | 21 | 8 | 2 | 0.180 | 3.913/5.216 | 0.24/0.52 | 0.28/0.15 | 5.3/2.8 | not_converged,seed_spread,generalization-gap |
| N | 23 | 4 | 2 | 0.252 | 4.675/5.808 | 0.42/0.44 | 0.17/0.47 | 3.2/9.1 | not_converged,generalization-gap |
| N | 23 | 8 | 2 | 0.066 | 3.806/3.754 | 0.24/0.20 | 0.63/0.68 | 12.1/13.2 | not_converged,generalization-gap |
| N | 25 | 4 | 2 | 0.297 | 6.157/4.669 | 0.32/0.26 | 0.16/0.28 | 3.1/5.4 | not_converged,seed_spread,generalization-gap |
| N | 25 | 8 | 2 | 0.058 | 4.101/3.570 | 0.22/0.17 | 0.33/0.42 | 6.4/8.0 | not_converged,generalization-gap |
| N | 27 | 4 | 2 | 0.311 | 5.364/4.891 | 0.44/0.31 | 0.05/0.27 | 1.0/5.1 | not_converged,seed_spread,generalization-gap |
| N | 27 | 8 | 2 | 0.060 | 4.281/4.118 | 0.19/0.15 | 0.35/0.27 | 6.8/5.2 | not_converged,seed_spread,generalization-gap |
| M | 21 | 4 | 2 | 0.000 | 0.000/0.000 | 0.21/0.14 | -0.14/0.14 | -7.2/6.9 | not_converged,seed_spread,SUFFICIENT |
| M | 21 | 8 | 2 | 0.000 | 0.000/0.000 | 0.21/0.25 | 0.22/-0.18 | 11.1/-9.1 | not_converged,SUFFICIENT |
| M | 25 | 4 | 2 | 0.000 | 0.000/0.000 | 0.18/0.20 | 0.02/0.07 | 0.8/3.4 | not_converged,seed_spread,SUFFICIENT |
| M | 25 | 8 | 2 | 0.000 | 0.000/0.000 | 0.21/0.22 | 0.15/0.19 | 7.6/9.6 | not_converged,seed_spread,SUFFICIENT |
| M | 27 | 4 | 2 | 0.000 | 0.000/0.000 | 0.18/0.13 | -0.08/-0.22 | -4.0/-11.2 | not_converged,SUFFICIENT |
| M | 27 | 8 | 2 | 0.000 | 0.000/0.000 | 0.12/0.04 | -0.11/-0.10 | -5.5/-5.4 | not_converged,seed_spread,SUFFICIENT |
| A | 21 | 4 | 2 | 0.000 | 6.390/3.348 | 0.26/0.25 | -0.01/0.01 | -0.5/0.3 | not_converged,generalization-gap |
| A | 21 | 8 | 2 | 0.000 | 4.134/4.237 | 0.24/0.35 | 0.03/0.33 | 1.1/12.7 | not_converged,seed_spread,generalization-gap |
| A | 23 | 4 | 2 | 0.001 | 2.578/0.400 | 0.73/0.46 | 0.17/0.14 | 6.4/5.2 | not_converged,generalization-gap |
| A | 23 | 8 | 2 | 0.000 | 2.576/5.889 | 0.32/0.31 | 0.36/0.20 | 13.9/7.5 | not_converged,seed_spread,generalization-gap |
| A | 25 | 4 | 2 | 0.000 | 0.741/0.066 | 0.27/0.25 | 0.02/0.11 | 0.8/4.4 | not_converged,seed_spread,generalization-gap |
| A | 25 | 8 | 2 | 0.000 | 1.227/2.986 | 0.27/0.10 | 0.08/0.20 | 3.0/7.8 | not_converged,seed_spread,generalization-gap |
| A | 27 | 4 | 2 | 0.000 | 0.004/0.001 | 0.18/0.26 | 0.07/0.17 | 2.9/6.5 | not_converged,seed_spread,SUFFICIENT |
| A | 27 | 8 | 2 | 0.000 | 0.000/1.281 | 0.21/0.07 | 0.24/0.17 | 9.0/6.6 | not_converged,seed_spread,generalization-gap |

Cells where train KL is ≈ 0 and validation KL is large (A, N) are generalization gaps (labels in the last column).

## 3. Task verdicts and confirm-set gates (5 seeds)
### Task verdicts (tier 1, confirm contents, 5 seeds)

| task | cell | intact acc (a) | 1b-S sufficiency | 1b-N necessity | 1b-G(i) transplant | 1b-G(ii) second format 1b-S | 1b-G | **1b verdict** |
|---|---|---|---|---|---|---|---|---|
| L | h23 r4 | ok (0.969) | PASS | FAIL | PASS | PASS | PASS | **1b-FAIL** |
| S | h25 r4 | ok (0.961) | FAIL | INCONCLUSIVE | PASS | PASS | PASS | **1b-FAIL** |
| G | none | – | no dev cell with val KL <= 0.05 in both seeds (best dev cell h27 r8) | not run | not run | not run | not run | **1b-FAIL (sufficiency)** |
| N | none | – | no dev cell with val KL <= 0.05 in both seeds (best dev cell h23 r8) | not run | not run | not run | not run | **1b-FAIL (sufficiency)** |
| M | h25 r4 | ok (1.000) | PASS | FAIL | FAIL | NOT RUN | FAIL | **1b-FAIL** |
| A | h27 r4 | ok (1.000) | FAIL | FAIL | FAIL | NOT RUN | FAIL | **1b-FAIL** |

**Tier-1 count: 0 of 6 candidate tasks are 1b-PASS.** Four-task gate: NOT MET -> stop the multi-task fixed-basis programme and consolidate.

### Confirm-set cell results (median over 5 seeds; intervals are hierarchical 95 %)

| task/format | hook,rank | intact acc | KL med [CI] (seeds ok) | S | rel red. med [CI] | abs red. med [CI] (logits) | literal-10 ok | seeds N ok | N | transfer med [CI] | rand max | seeds T ok | T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L/num_orig | 23,4 | 0.969 | 0.0060 [0.0040, 0.0098] (5/5) | PASS | 0.44 [0.36, 0.58] | 13.0 [10.7, 17.2] | False | 2/5 | FAIL | 0.628 [0.59, 0.68] | 0.002 | 5/5 | PASS |
| L/num_rev | 23,4 | 0.969 | 0.0084 [0.0066, 0.0114] (5/5) | PASS | 0.36 [0.24, 0.78] | 8.8 [6.0, 18.9] | False | 2/5 | FAIL | 0.668 [0.47, 0.68] | 0.002 | 5/5 | PASS |
| S/words | 25,4 | 0.961 | 0.1328 [0.0044, 0.2969] (0/5) | FAIL | 0.55 [0.34, 0.57] | 16.5 [10.1, 17.0] | True | 3/5 | INCONCLUSIVE | 0.619 [0.55, 0.73] | 0.003 | 5/5 | PASS |
| S/num_orig | 25,4 | 0.938 | 0.0029 [0.0015, 0.0038] (5/5) | PASS | 0.45 [0.37, 0.48] | 10.8 [8.8, 11.5] | False | 0/5 | FAIL | 0.706 [0.68, 0.72] | 0.002 | 5/5 | PASS |
| S/num_rev | 25,4 | 0.995 | 0.0021 [0.0013, 0.0029] (5/5) | PASS | 0.38 [0.36, 0.43] | 8.6 [8.2, 9.9] | False | 0/5 | FAIL | 0.682 [0.59, 0.77] | 0.003 | 5/5 | PASS |
| M/letters | 25,4 | 1.000 | 0.0000 [0.0000, 0.0000] (5/5) | PASS | 0.05 [0.01, 0.06] | 2.8 [0.4, 3.1] | False | 0/5 | FAIL | 0.093 [0.03, 0.11] | 0.001 | 0/5 | FAIL |
| A/digits | 27,4 | 1.000 | 0.1378 [0.0088, 1.3224] (2/5) | FAIL | 0.13 [0.08, 0.19] | 5.2 [3.2, 7.2] | False | 0/5 | FAIL | 0.049 [0.02, 0.07] | 0.001 | 0/5 | FAIL |

### Arm A diagnostics for tasks without a sufficient cell (dev data; plan section 3 labels)

| task | run | train KL | val KL | cand-KL | retained acc | gap intact -> ablated | note |
|---|---|---|---|---|---|---|---|
| G | grid best cell h27 r8 (1600 steps, 2 seeds) | 0.274 | 0.900 | – | – | – | train slope 0.11 |
| G | long4800 | 0.206 | 0.880 | 0.048 | 0.99 | 8.7 -> 6.5 | slope 0.12 |
| G | lda_init | 0.272 | 0.891 | 0.037 | 0.99 | 8.7 -> 7.2 | slope 0.11 |
| G | tier2_candobj (TIER-2, diagnostic) | 0.001 | 3.638 | 0.027 | 1.00 | 8.7 -> 5.7 | slope 0.42 |
| G | non-fitted LDA reference (rank 1) | – | 4.709 | 0.949 | 0.46 | – | |
| N | grid best cell h23 r8 (1600 steps, 2 seeds) | 0.066 | 3.780 | – | – | – | train slope 0.22 |
| N | long4800 | 0.026 | 3.901 | 0.345 | 0.83 | 19.3 -> 7.3 | slope 0.27 |
| N | lda_init | 0.077 | 3.634 | 0.319 | 0.86 | 19.3 -> 7.5 | slope 0.21 |
| N | tier2_candobj (TIER-2, diagnostic) | 0.002 | 10.475 | 0.928 | 0.66 | 19.3 -> 16.3 | slope 0.20 |
| N | non-fitted LDA reference (rank 1) | – | 11.041 | 0.934 | 0.48 | – | |
| N | N-fixedpool (TIER-2, own split with shared names) | 0.104 | 0.157 | 0.008 | 1.00 | 18.8 -> 7.4 | |

#### Three-way diagnostic labels (frozen rules, plan section 3 + A8)

* **G** (best cell h27 r8): OPT-LIMITED=True (not_converged=True, seed_spread=True, best extension gain on val KL=2.2%); GENERALIZATION-LIMITED=True (1600-step=False, 4800-step=True); CAPACITY-LIMITED=False.
* **N** (best cell h23 r8): OPT-LIMITED=True (not_converged=True, seed_spread=False, best extension gain on val KL=3.8%); GENERALIZATION-LIMITED=True (1600-step=True, 4800-step=True); CAPACITY-LIMITED=False.


Reading: `S/words` retained KL is 0.13 on confirm contents against 0.003 on dev with the same seeds (0/5 seeds pass), while `num_orig`/`num_rev` retain with KL 0.002–0.003; necessity is below threshold in the numeric formats (0.45/0.38; 0/5 seeds) and borderline in `words` (0.55, 3/5 seeds, interval crossing 0.5). L passes sufficiency and transplantation in both formats but only 2/5 seeds reach the 0.5 relative reduction in either (seed values 0.58, 0.58, 0.37, 0.37, 0.44 in `num_orig`). M retains perfectly but transplantation transfers only 0.09 (0/5 seeds) and ablation 0.05. A's validation of the cell is unstable (2/5 seeds sufficient; A has only 55 sums, train 30, confirm 17), so A is a weak test and is flagged as such. Random equal-rank subspaces: transplant ≤ 0.003, ablation reduction ≈ 0 throughout. The literal 10-logit column is False wherever the relative criterion also fails; it is True only for `S/words`.

## 4. Q1: optimization vs generalization vs capacity (G, N; dev data)
See the diagnostics table and labels in `results/phase7/stage1b/stage1b_report_tables.md`. Summary under the frozen label rules (the three labels may co-occur):
* **G** (best dev cell h27 r8, train KL 0.27, val KL 0.90): train KL still decreases (flag OPT-LIMITED fires), but 3× longer training (4800 steps) and an LDA-initialised start change validation KL by only 2.2 %; train KL falls to 0.21 while val KL stays 0.88, i.e. a train/val gap (GENERALIZATION-LIMITED at 4800 steps); CAPACITY-LIMITED is not assigned because the run is not converged. Tier-2: with a candidate-restricted objective the candidate-level KL on validation is 0.027 with accuracy 1.00 and ablation 8.7 → 5.7 logits, so the choice-level information is held by a rank-8 subspace while the full-vocabulary distribution (mass on "'s", "of", "the") is not. This tier-2 result does not count toward any gate.
* **N** (best dev cell h23 r8, train KL 0.07, val KL 3.8): GENERALIZATION-LIMITED (1600 and 4800 steps); extensions move val KL ≤ 4 %; the non-fitted LDA reference is poor (KL 11). **N-fixedpool (tier 2)** with names shared across splits gives val KL 0.157, candidate-KL 0.008, retained accuracy 1.00 and ablation 18.8 → 7.4 logits: unseen-name identity is the dominant cause of the Stage 1 and 1b N failure, and what remains (KL 0.16 > 0.05) is a smaller, separate shortfall. The candidate-restricted objective with unseen names is worse (candidate-KL 0.93).
* Seeds: across the 2 screening seeds the final train KL differs by more than 20 % for several G/N cells (flag `seed_spread`), so two seeds are not enough to rank those cells finely.
* Conclusion for Q1: for G and N, more optimization under the frozen recipe is not what is missing (≤ 4 % effect). N's failure is mostly a content-generalization failure of the objective on held-out names; G's is a mismatch between a full-vocabulary objective and a low-rank choice subspace plus a train/val gap. Neither shows that no low-dimensional representation exists.

## 5. Q2: M sufficiency without necessity (Arm B; confirm contents)
Subjects: M selected cell (h25 r4, 3 seeds) and the Stage 1 object (h25 r2, 2 seeds, refitted on the 1b partition). Numbers from `armB/*.json`.
* **Natural-strength ablation**: relative gap reduction 0.02–0.06 (r4) and −0.07…0.06 (r2); final-position-only ablation ≤ 0.06; random Haar / on-manifold subspaces of equal rank −0.03…0.05; the top-PCA 4-subspace gives 0.13 (it captures 16 % of final-position variance).
* **Dose-response** (h − α·UUᵀ(h−μ)): α = 0.5, 1, 2, 4 give gap reductions ≈ 0.0, 0.02–0.06, 0.12–0.24, **0.83–0.96** (r4); r2: 0.57 and 0.54 at α = 4. Ablation at natural strength is therefore ineffective but strongly dose-dependent: the representation is *causally active at larger magnitude*, not inert.
* **Magnitude diagnostics**: span(U) holds 9 % of the final-position variance of h − μ (Haar 0.3 %, on-manifold random 0.2 %, top-4 PCA 16.5 %); ablation perturbation norm at the final position median ≈ 61 (random subspaces ≈ 10–12); the massive coordinates (2276: mean |h−μ| ≈ 490; 233: ≈ 93) were identified, but U's loading on them was not computed (not reported). So U is not a low-energy subspace; the natural-strength ablation is simply not enough to change the readout.
* **Complement test**: U2 ⊥ U of rank 2, 4 and 8 (fitted on train items, each retained alone) reproduces the output on confirm contents with KL 0.000–0.0002 and accuracy 1.00 for every seed, so **orthogonal subspaces of rank 2 are each independently sufficient** (for L and S contrasts, one seed, complement rank 8: retained KL 0.009 / 0.21 with accuracy 0.98 / 0.96). This establishes independent sufficiency of orthogonal subspaces; it does **not** establish that the intact network actively uses redundant pathways (A11).
* **Union ablation**: ablating [U, U2] (ranks 6/8/12) removes only 0.02–0.15 of the M gap (random same-rank ≈ 0), whereas for the necessity contrasts the L union (rank 12) removes 0.65 and the S union 0.86 (random 0.014 / 0.002; single seed). M's answer information is thus not removed by deleting three sufficient subspaces at natural strength, but is removed by scaling a single one 4×.
* **Transplantation of span(U)** (never computed in Stage 1): M transfers only 0.03–0.11 (r4) and 0.03–0.08 (r2) of the source–target gap against 0.000 for random subspaces: U is sufficient in the retention regime (everything else projected out) but swapping it between items does not change the answer.
* Reading (hedged): the pattern fits an answer signal that is **distributed with substantial redundancy at hook 25 and read out additively**, so natural-strength removal of any single low-rank piece is compensated, while amplification overcomes the compensation. It does not support "U is an inert epiphenomenon" (dose-response is large) and does not prove an active redundant mechanism.

## 6. Q3: cross-format stability for L and S (Arm C; confirm contents; reported separately from same-format generalization)
* **Retention-sufficiency is format-bound.** L: a subspace fitted on `num_orig` evaluated on `num_rev` retains with KL 11.4 (and 9.6 in the other direction); S: cross-format KL 0.09–39.6 (numeric↔numeric 0.09/0.47; with `words` 20–40). Frozen label: **FORMAT-SPECIFIC for retention** (L and S).
* **L: causal transplantation transfers across formats and follows the semantic class.** Transplantation of the `num_orig` subspace into `num_rev` targets transfers 0.46 (and 0.59 the other way, matrix cells), and the cross-format transplant (source prompt in the other format) gives semantic transfer 0.39–0.63 and token-flip ≈ 0.01, random ≤ 0.002. So for L the subspace is **functionally shared in its causal effect** while retention does not transfer; geometry: cross-format principal-angle overlap 0.19 vs within-format cross-seed 0.22 (Haar 0.001); the top cosine 0.62 vs 0.80 within format. Label: "functionally shared (transplantation, ablation relative to diagonal ≥ 0.5×), geometrically similar to within-format variation, retention format-bound".
* **S: cross-format causal effect follows the answer token, not polarity.** Cross-format transplant: semantic transfer −0.05…−0.12, token-flip +0.16…+0.27. Geometry: `num_orig`↔`num_rev` overlap 0.25 (within 0.25–0.28), but `words`↔numeric overlap 0.03–0.05, top cosine 0.27–0.34 (no pair with a cosine above 0.7). Label: FORMAT-SPECIFIC; `words` is geometrically distinct, numeric formats share a direction and differ in how the answer token is read.
* **Rank > 1 is not a single shared direction**: within-format cross-seed top cosine 0.79–0.80 but only the first principal angle is large (mean sorted cosines L 0.80/0.44/0.15/0.03; S 0.79/0.57/0.24/0.05), so across seeds usually one direction is reproducible, not the 4-d subspace. The A4 overlap benchmark (O ≥ 0.5) is not met by any pair (best 0.28), reported separately and not part of the PASS rule.
* **Joint-format fit**: L's joint rank-4 fit (both mappings) stays on the ln 2 plateau (KL 0.64–0.67; fails); S's joint fit retains in the numeric formats (KL 0.002–0.003) and not in `words` (0.15), with necessity below 0.5 in all.

## 7. Stopping gate and recommendation
The preregistered four-task tier-1 gate is **not met (0 of 6)**. Per your instruction: **stop the multi-task fixed-basis programme** (no Stage 2–4 as designed) and consolidate. Consolidation candidates, all supported by stored artifacts:
1. The pilot (Stage 0 flag under its original definition, Stage M norm-matched result superseding the Phase 6 interpretation) and the Phase 0.5–6 logic results.
2. Methodological findings: sufficiency (retention KL) is cheap and non-diagnostic of causal control; necessity at natural strength decreases with rank, is seed-variable (0.37–0.58 across seeds at fixed cell), and depends on perturbation magnitude (M dose-response); the full-vocabulary KL objective is not suited to raw continuation tasks and to held-out token sets (N names); retention is answer-format-bound while transplantation can be semantic (L) or token-like (S); basis-vector cosines are not meaningful for rank > 1 and only one direction is reproducible across seeds.
3. Caveats to carry: small content pools, a repartition not independent of Stage 1 knowledge, tier-1 verdicts at one cell per task chosen from two screening seeds, A's 55-sum design, Arm B contrast with one seed, and G/N conclusions restricted to ≤ 8 dimensions, four hooks and the stated objective.
No pending experiment is needed to write this up; any follow-up (e.g. rank-2 grid cells for L/S to test whether necessity returns, a content-shared N, or larger polarity pools for S `words`) would be a new preregistered stage and would not change the tier-1 gate result above.

Artifacts: `results/phase7/stage1b/{manifest.json, selection.json, selection_table.md, armA/, ext/, final/, armB/, armC/, stage1b_verdicts.json, stage1b_report_tables.md, *.log}`; code `src/causal_subspaces/phase7/stage1b.py`, `scripts/phase7_stage1b_*.py`.
