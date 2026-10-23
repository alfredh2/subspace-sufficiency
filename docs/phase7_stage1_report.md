# Phase 7 Stage 1 report: task feasibility and target representations

Scope: Stage 1 only (plan sections 4, 5, 11, 12 and 12.1/12.2, all committed before the runs). Stages 2–4 have not been started and wait for review. Nothing earlier was changed (pilot flag, Stage M outcome, Phase 1–6 reports and verdicts are untouched). GPU use: baselines ≈ 0.1 h, sweep ≈ 1.2 h, final MVS (L 2 formats, S 3 formats, 3 seeds) ≈ 1.3 h, exploratory long-check ≈ 0.25 h; total ≈ 3 GPU-h of the ≈ 4 h planned. Commits: `62d33f2` (methodology amendments), `4e78ca5`, `16bfc99` (scoring and scale amendments before any ablation result), then sweep and final results. No pushes, no co-author trailers.

## Headline
| task | outcome | why |
|---|---|---|
| **L** logic | **PASS** (all frozen gates) | hook 23, rank 2 reproduces the Phase 0–6 configuration; retention, ablation, transplantation and cross-seed consensus hold in both formats, but see the format caveat below |
| **S** sentiment | **INCONCLUSIVE** | hook 25, rank 4; every gate passes by point estimate in the primary format, but the lower bound of the ablation reduction (median per-seed 0.47) crosses the 0.5 threshold; ablation fails in the other two formats |
| **G** agreement | **FAIL** (failed feature discovery) | no hook/rank retains the output distribution (val KL ≥ 1.04) and ablation closes ≤ 20 % of the margin gap |
| **N** name copying | **FAIL** | retained KL ≈ 6 nats and retained accuracy at chance in all 12 configurations |
| **M** four-way QA | **FAIL** (necessity gate) | hook 25, ranks 2 and 4 retain perfectly (KL 0.000 / 0.001) but ablation does nothing (reduction ≈ 0 / −4 %) |
| A (reserve) | not triggered | all five primary tasks passed gate (a) |

**Panel:** one passing task (L) and one inconclusive (S). That is below the ≥ 4-task requirement and below the "fewer than 3 tasks" stopping rule of plan section 8, so Stages 2–4 cannot support the generalisation claims (H1–H4) as designed; the label is **"insufficient panel"**. Failed discoveries are results: **3 of 5 tasks (G, N, M) had no stable and causally meaningful low-rank representation under the frozen procedure.** Nothing was relaxed after seeing results.

## Gate (a), feasibility screen (test contents, frozen)
All 11 planned (task, format) cells pass (L 0.945 / 0.938; S 0.979 / 0.927 / 0.992; G 0.990 / 0.987 [candidate-restricted scoring, section 12.1]; N 1.000 [restricted] / 0.977; M 1.000 / 1.000). Unrestricted greedy accuracy for the raw formats is much lower (G 0.47 / 0.46, N-cont 0.81) because the model puts most mass on "'s", "of", "the" etc.; candidate mass 0.19 / 0.21 (G), 0.71 (N-cont). S `num_orig` passes by point estimate only (0.927, interval 0.869–0.975). On natural SST-2 sentences (256 val items, baseline context only) accuracy is 0.934 / 0.887 / 0.906 (words / num_orig / num_rev), i.e. the templated task is easier than natural sentiment and `num_orig` is below 0.9 there.

## Sweep (val contents only, 300 steps, seed 0, first format of each task)
Full 60-row table: `results/phase7/stage1/sweep_table.md`. Passing configurations out of 12: L 1 (hook 23, rank 2), S 1 (hook 25, rank 4), G 0, N 0, M 0. Observations (descriptive; one seed, 300 steps):
* Rank 1 never retained anything at any hook for any task except M/hook 25 (KL 1.16): retained KL ≈ 0.52–0.68 = the paper's ln 2 plateau for binary tasks. Hooks 19 and 21 also sit on the plateau at 300 steps for L and S. **This is partly a training-budget limit of the sweep**, not necessarily absence of a subspace (Phase 0 trained 1600 steps).
* L: ranks 2 and 4 at hook 23 retain (KL 0.015); only rank 2 loses accuracy when ablated (0.98 → 0.51). With rank 4 or at hook 25 the model retains and *survives* ablation (gap 28.5 → 21–23): the information is available elsewhere in the residual (redundancy), a sufficiency-without-necessity pattern.
* M: the same pattern in the extreme: the rank-2 subspace at hook 25 reproduces the full output distribution (KL 0.000) but ablating it leaves the margin gap unchanged (51.6 → 51.8).
* N: even rank 4 at hook 25 leaves KL ≈ 6 and chance accuracy. A "pointer" to the answer name cannot live in a fixed low-rank subspace of an unmodified residual stream at these layers under this objective.

**Exploratory long-check (declared before running; cannot change verdicts).** Refitting the lowest-val-KL configuration of each failed task at 1600 steps: G (hook 25, rank 4) KL 1.04 → 0.82, retained accuracy 0.965, gap 8.6 → 7.7 (still ≈ 10 % reduction); N (hook 23, rank 2) KL 5.7 → 6.1, retained accuracy 0.63; M (hook 25, rank 2) KL 0.000, ablation gap 51.6 → 49.7. So the failures are not explained by the short sweep for G, N and M. (`longcheck_val.json`, val split.)

## Final MVS gates on TEST contents (3 seeds; seed-median; clustered intervals)

| task/format | hook,rank | KL ret | acc I / R / A | gap I | gap reduction (rel) | transplant all (3 seeds) | random all mean / max | transplant last | consensus cos mean / min | b | c | c literal | d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L/num_orig | 23,2 | 0.0062 | 0.945 / 0.941 / 0.500 | 28.0 | 20.9 (0.75; seeds 0.75, 0.77, 0.37) | 0.60, 0.63, 0.57 | 0.000 / 0.001 | 0.48, 0.50, 0.45 | 0.84 / 0.82 | True | True | True | True |
| L/num_rev | 23,2 | 0.0067 | 0.938 / 0.930 / 0.789 | 23.1 | 8.9 (0.39; seeds 0.39, 0.67, 0.35) | 0.57, 0.54, 0.50 | 0.001 / 0.002 | 0.41, 0.38, 0.35 | 0.83 / 0.80 | True | False | False | True |
| S/words | 25,4 | 0.0045 | 0.979 / 0.982 / 0.935 | 29.0 | 14.8 (0.51; seeds 0.59, 0.51, 0.38) | 0.60, 0.60, 0.67 | 0.001 / 0.003 | 0.57, 0.58, 0.65 | 0.74 / 0.73 | True | True | True | True |
| S/num_orig | 25,4 | 0.0025 | 0.927 / 0.930 / 0.990 | 23.0 | 10.5 (0.45; seeds 0.45, 0.30, 0.50) | 0.59, 0.71, 0.78 | 0.001 / 0.003 | 0.59, 0.70, 0.78 | 0.84 / 0.81 | True | False | False | True |
| S/num_rev | 25,4 | 0.0014 | 0.992 / 0.992 / 0.984 | 22.0 | 8.2 (0.37; seeds 0.51, 0.27, 0.37) | 0.65, 0.67, 0.64 | 0.001 / 0.003 | 0.65, 0.67, 0.64 | 0.79 / 0.77 | True | False | False | True |

## Task verdicts

| task | gate a | primary-format b,c,d | >=2 formats (b&d) incl. reversed, consensus>=0.7 | verdict |
|---|---|---|---|---|
| L | True | True, True, True | True (['num_orig', 'num_rev']) | PASS |
| S | True | True, True, True | True (['words', 'num_orig', 'num_rev']) | INCONCLUSIVE |
| G | True | - | - | FAIL: no sweep configuration passed (KL and/or ablation) |
| N | True | - | - | FAIL: no sweep configuration passed (KL and/or ablation) |
| M | True | - | - | FAIL: no sweep configuration passed (KL and/or ablation) |

## Task-direction geometry (cosines of consensus directions; only comparable at the same hook)

| | L/num_orig | L/num_rev | S/words | S/num_orig | S/num_rev |
|---|---|---|---|---|---|
| L/num_orig | 1.00 | -0.38 | n/a (hook 23 vs 25) | n/a (hook 23 vs 25) | n/a (hook 23 vs 25) |
| L/num_rev | -0.38 | 1.00 | n/a (hook 23 vs 25) | n/a (hook 23 vs 25) | n/a (hook 23 vs 25) |
| S/words | n/a (hook 25 vs 23) | n/a (hook 25 vs 23) | 1.00 | 0.38 | 0.10 |
| S/num_orig | n/a (hook 25 vs 23) | n/a (hook 25 vs 23) | 0.38 | 1.00 | -0.56 |
| S/num_rev | n/a (hook 25 vs 23) | n/a (hook 25 vs 23) | 0.10 | -0.56 | 1.00 |
## Reading the gates
* **L passes**, with the following qualifications. Ablation reduces the gap by 75 % in `num_orig` (seeds 0.75 / 0.77 / 0.37) but only 39 % in `num_rev` (0.39 / 0.67 / 0.35), i.e. gate (c) fails in the reversed format (the frozen task rule requires (c) in the primary format only, and (b)+(d) in both). The seed spread (one seed in each format removes far less) means the discovered rank-2 span is not equally necessary across fits.
* **Transplantation (d) is the strongest result:** the fitted subspace transfers 0.50–0.78 of the source–target margin gap at all positions in every cell and format (L and S), while 20 random equal-rank subspaces transfer 0.000–0.003 on average (max ≤ 0.003); last-position-only transfer is 0.35–0.78. Together with retention (KL ≤ 0.007) this says the subspaces are *sufficient* carriers of the class information, in all formats including reversed mappings.
* **Necessity (c) is weak outside L-`num_orig` and S-`words`**: reductions are 0.30–0.51 of the gap, with a seed spread of ≈ 0.2. A consistent picture is a sufficient-but-redundant representation.
* **S** is INCONCLUSIVE by the frozen rule (point estimates pass, the interval for the primary-format ablation reduction crosses 0.5; the other two formats fail (c) at 0.45 and 0.37). Its hook/rank (25, 4) differ from L's (23, 2), so L and S directions are not directly comparable by cosine.
* **Format dependence (descriptive).** Consensus directions fitted in different answer formats of the same task are only partly aligned: |cos| = 0.38 (L num_orig vs num_rev) and 0.38 / 0.10 / 0.56 among S formats (signs are arbitrary). With the near-identical transplant effect in every format, this is compatible with format-specific subspaces rather than one task-semantic direction (Phase 0.5 reached the same reading for L). Cross-format set transfer of one format's subspace was not tested in Stage 1.

## Caveats (all affect interpretation)
1. Intervals are percentile bootstraps over test contents, 300 resamples; L has only 8 test contents and S's clusters are polarity words. They are likely too narrow (few clusters) and the INCONCLUSIVE rule should be read with that in mind. Several gate values sit close to thresholds (S ablation 0.37–0.51).
2. The sweep uses one seed and 300 steps; rank/hook choice rests on one configuration passing per task (L, S) and is fragile: neighbouring configurations fail, with a seed-to-seed spread of ≈ 0.2 in the ablation fraction seen later.
3. The retained-KL gate is over the full next-token distribution, so it is harder for raw continuation formats (G, N-cont), where the distribution is broad ("'s", "of", "the", names), than for instructed one-token answers. G's failure is nonetheless supported by the ablation gate (≤ 20 %), which does not depend on KL; N fails with chance retained accuracy.
4. Task generators are our re-creations with small content pools (S: 8 test polarity words per class and 6 nouns; M: 22 test questions; G/N: 24/18 name or noun pools). Pair sets are small (S 48, M 66, L/N/G ≈ 160) and M's validation set has 10 questions.
5. S `num_orig` is below 0.9 on natural sentences, and the 10-logit floor was scale-adjusted (section 12.2) before any ablation result; under the literal floor L num_orig and S words still pass and G would fail trivially.
6. Nothing in Stage 1 shows that any of these subspaces are *native-neuron* structures; it only identifies target directions.

## Suitable task directions for later native-basis comparisons
* **L, hook 23, rank 2:** consensus direction per format (`final/L_<fmt>_directions.pt`), plus the Phase 0.5 `d_c` already used in Phases 1–6. Suitable.
* **S, hook 25, rank 4:** consensus directions in `final/S_<fmt>_directions.pt`; usable only as an **exploratory** second task (INCONCLUSIVE), at a different hook from L.
* **M (hook 25, rank 2, sufficiency only):** not a gate-passing task; excluded from the panel. It would be a useful negative/redundancy case (subspace carries a complete copy of the answer that the model does not need) but that is an add-on, not part of the frozen panel.
* G, N: no usable representation under these procedures.

## Consequence and decisions requested
With one passing task and one inconclusive, H1–H4 cannot be tested as preregistered over a ≥ 4-task panel. I did not widen the gates. Options for you (my recommendation first):
1. **Reduced-claim Stage 2 on L and S only** (labelled "insufficient panel"): the fixed-basis / leave-one-task-out design of D2 degenerates with two tasks, so it would be run as a two-task transfer test (fit on one, evaluate on the other) plus the Phase 0.5 `d_c` and the M subspace as a sufficiency-only third target; the revised verdict-irrelevant-direction analysis (D1) is unaffected.
2. A **pre-registered Stage 1b** with a different procedure for the failed tasks (e.g. longer sweeps, more seeds, ranks up to 8, or necessity measured on the subspace complement), explicitly labelled as new rather than a rescue; compute ≈ 3–4 GPU-h.
3. Stop the multi-task programme here and write up Stage 1 plus the pilot as the finding: *beyond the logic verdict channel, low-rank causal subspaces are only found for sentiment (partially), and not for agreement, name copying or QA under the frozen procedure*.

Artifacts: `results/phase7/stage1/{baseline_test.json, sweep/, sweep_table.md, final/, stage1_verdicts.json, stage1_report_tables.md, longcheck_val.json}`; code `src/causal_subspaces/phase7/{tasks,stage1}.py`, `scripts/phase7_stage1_*.py`.
