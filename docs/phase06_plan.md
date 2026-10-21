# Phase 0.6 plan and pre-registered decision criteria (written BEFORE any Phase-0.6 run)

Question: does logical validity have an identifiable representation / causal component beyond the shared accept-reject verdict direction found in Phase 0.5?
Do not assume yes. Frozen: Qwen3-8B, the three Phase-0 checkpoints, residual boundary `residual_index=23`, the Phase-0 benchmark (re-created), Phase-0/0.5 results.

## Parts
**A. Decodability** of validity at the hook location from (a) the full residual, (b) the residual with the frozen consensus rank-1 verdict direction `d_c` projected out,
(c) with random rank-1 directions / the top PC / the seed-0 rank-2 MVS projected out. Linear probes (L2 logistic, standardised features) trained on the optimisation
pool (`train` split, 11 train families, ordinary words) and tested on `items` / held-out `families` / `nonwords`; evaluated on 5 answer formats (train on one format and on all
formats). Positions: last prompt token (answer position) and the final token of the conclusion sentence. Also INLP (iterative removal of probe directions) to see how
distributed the information is. Same probes for localizer and PrOntoQA labels. Caveat that is built in: in this benchmark validity is a function of the argument form,
so a probe can use surface-form cues; held-out families/nonwords and cross-format transfer limit, but do not remove, this.

**B. Causal search** (small pilot): learn rank k∈{1,2,4} subspaces V ⟂ d_c (2 search seeds each) whose *ablation* (h ← h − VVᵀ(h−μ); everything else in the residual
stream stays intact; no retention-type intervention) removes validity discrimination without changing anything else. Objective (fixed in advance, no tuning of weights after seeing results):
 * `L_disc`: matched valid/invalid pairs (swap pairs MP↔AC, MT↔DA, ± negated consequent; train contents × 11 train families; formats {num_orig, ab_orig} sampled per pair): squared difference of
   the semantic margins m_valid−m_invalid of the ablated model, normalised by the intact mean gap. (Drives discrimination → 0.)
 * `L_bias`: Huber on the shift of each pair's mean margin relative to the intact model (forbids a global shift toward either verdict) and on the shift of the log-probability mass on the two label tokens (forbids off-format collapse).
 * `L_ctrl`: Huber on the shift of every control example's margin (language localizer, PrOntoQA, MMLU-binary; separate generator seed from evaluation) relative to intact (forbids damage to other verdict tasks and to verdict-free tasks).
 Weights 1 / 0.5 / 1. Adam, lr 1e-2 cosine → lr/20, 300 steps, rank-k basis = QR((I−d_c d_cᵀ)W). Selection among runs uses validation contents/controls only. Run-to-run reporting includes ALL runs.
 Matched baselines: random orthogonal-to-d_c subspaces of equal rank (uniform in the top-2048 variance subspace) and variance-matched ones; `d_c` itself; seed-0 rank-2 MVS.

**C. Evaluation (held-out only)**: Items / Families / Nonwords × 5 formats, language localizer, PrOntoQA, MMLU-binary, Stroop, word recall, WikiText KL, using AUC, class gap, mean margin shift, within-class SD, accuracy and invalid-output rate; matched counterfactual
patching of the candidate coordinates on test-content pairs (num_orig and reversed formats) and unmatched last-position patching on localizer/PrOntoQA/verdict-free tasks.

## Decision rule (fixed now)
A candidate is a **credible, causally selective logic target** only if ALL hold on held-out data:
1. Logic AUC drops by ≥ 0.15 (absolute) averaged over Items/Families/Nonwords in num_orig, and by ≥ 0.10 in at least two other formats including a format not used for search; matched-variance random subspaces of the same rank drop < 0.03.
2. It is not a bias shift: |Δ mean margin| ≤ 25 % of the intact class gap and P(valid) within ±0.15 of 0.5 (balanced data).
3. Selectivity: localizer AUC and PrOntoQA AUC each drop < 0.05 (absolute) and absolute margin shifts on verdict-free controls < 1 logit; WikiText KL < 0.02.
4. Matched patching of its coordinates transfers validity (transfer ≥ 0.2, random ≈ 0) and does not transfer on localizer / PrOntoQA / verdict-free tasks (|transfer| < 0.1).
5. Reproduces across the two search seeds (largest principal-angle cosine between the two same-rank solutions ≥ 0.5 or both independently satisfy 1–4).
If no candidate satisfies 1–4, we report that no logic-selective causal component was found at this layer under this search; the search is not extended.
Partial results (e.g. a candidate satisfying 1–3 but not 4) are reported as such.
