# Phase 0.6 report — does logical validity have a representation beyond the shared verdict direction?

**Bottom line: no credible, causally selective logic target was found at this layer.** Validity is linearly decodable from the layer-23 residual after removing the verdict direction (so *information* exists
outside it), but the causal search produced only format-specific, non-transplantable ablation effects, and no candidate satisfies the pre-registered rule (`docs/plans/phase06_plan.md`, decision rule fixed before any run).
Setting: frozen Qwen3-8B; Phase 0 / 0.5 checkpoints and results untouched; same benchmark (re-created; validity is a function of argument form, which limits what any probe/search can show); hook `residual_index=23`.
Code: `src/causal_subspaces/phase06.py`, `scripts/phase06_*.py`; tables: `results/phase06/report_tables*.md`, `decodability.json`, `evaluation*.json`; 56 tests pass (`tests/test_phase06.py` adds the orthogonality, probe/INLP, loss and collection tests).

## A. Decodability outside the verdict direction (held-out data)
Linear probes (L2 logistic, standardised) trained on 1000 optimisation-pool examples (11 train families, ordinary words) and tested on held-out Items / Families (3 unseen families) / Nonwords, in 5 answer formats (train on `num_orig` only, or pooled).
* Validity probe accuracy at the answer position is 0.94–0.98 on every held-out split and format from the **full residual**, and **unchanged (0.95–0.99) after projecting out** the consensus verdict direction `d_c`, the seed-0 MVS, the union of all three MVS spans (6 dims), the top principal component, or random 1/2/6-dim subspaces (20/10/5 draws). Differences between conditions are ≤0.02 (noise level). At the final token of the conclusion sentence accuracy is 0.99+, again unchanged.
* INLP (iteratively removing the probe's own direction): after **12** removed directions, held-out accuracy is still ≈0.95 (num_orig splits) and ≈0.97 on the reversed-mapping format; Yes/No transfer is unstable (0.59–0.97). Information is highly redundant/distributed at this location.
* Format-invariance: the validity probe transfers across formats (0.97 on reversed mappings) and does *not* transfer to the literal answer-token target there (0.03), i.e. it reads validity, not the output token; a separate literal-answer-token probe also works (0.93–0.97), so both quantities are linearly available, with or without `d_c`.
* `d_c`'s coordinate alone already separates validity with AUC 0.99–1.00, the localizer label with AUC 1.00 and PrOntoQA with 0.963 — a strong but non-exclusive, non-logic-specific readout. Language-localizer and PrOntoQA labels likewise remain decodable (1.00 / 0.94) after removing `d_c`.
* Caveats: validity is determined by argument form in this benchmark, so a probe may decode surface-structure features (which proposition the fact cites, polarity) rather than "validity" as such; held-out families and nonwords limit but cannot remove this. Decodable ≠ causally used.

## B. Causal search for logic-selective subspaces orthogonal to `d_c`
Objective (pre-registered, weights fixed): drive the matched-pair margin gap to 0 while preserving each pair's mean margin and label-token mass (no global shift / off-format collapse) and keeping localizer, PrOntoQA and MMLU-binary margins at their intact values. Intervention = ablation of the candidate coordinates only (residual otherwise intact). Ranks 1/2/4 × 2 search seeds, 300 steps, search controls from different generator seeds/MMLU subjects than evaluation.
* **Pilot 1** (search formats `num_orig`, `ab_orig`): every candidate closes the validation pair gap to 4–7% of intact with no control damage and no mean-margin shift. On held-out data: logic AUC drops in the two search formats (num_orig 0.994 → 0.70–0.83, ab_orig 0.993 → 0.70–0.89) but **not in the reversed-mapping formats** (num_rev 0.97–0.99, ab_rev 0.96–0.99, yn 0.95–0.98). Localizer AUC is unchanged (1.000), PrOntoQA AUC unchanged or slightly higher (0.871–0.883 vs 0.863), MMLU/Stroop/WikiText unaffected. **Matched patching of the candidates' coordinates transfers nothing** (|transfer| ≤ 0.018; consensus direction 0.71, random 0.00).
* **Design confound found and one corrective round run (Amendment 1):** in both pilot-1 search formats "valid" is the first-listed label, so the search could exploit a literal-position mapping. **Pilot 2** (search formats `num_orig`, `ab_rev`; criteria unchanged; the only corrective round) narrows but does not remove the gap: held-out AUC num_orig 0.73–0.91, ab_rev 0.76–0.95, untrained ab_orig 0.77–0.99, num_rev 0.91–0.99, yn 0.94–0.99; patching transfer ≤ 0.021; localizer unchanged.
* Rank 1 → 4 does not help consistently; search-seed reproducibility is moderate (principal-angle cosines of same-rank solutions: rank 1 0.39–0.44; rank 2 0.65/0.36–0.48; rank 4 0.71/0.55 and two near-orthogonal directions).
* Candidates carry small variance (4–256 on the corpus-variance scale vs. 381 for the seed-0 MVS), unlike `d_c`/MVS ablation, whose effect is a +15-logit shift toward "valid" with AUC kept ≥0.92. The two behave as a **double dissociation in the trained formats**: `d_c` → bias shift without loss of ranking; candidates → loss of ranking in the trained formats without bias shift.

## C. Pre-registered verdict (held-out data only)
| | Pilot 1 (6 candidates) | Pilot 2 (6 candidates) |
|---|---|---|
| 1 logic AUC drop (≥0.15 num_orig, ≥0.10 in two other formats incl. an untrained one, random <0.03) | 0 / 6 | 1 / 6 (rank-4 seed 1) |
| 2 not a bias shift (|Δ mean margin| ≤ 25 % of gap, P(valid) within 0.5 ± 0.15) | 0 / 6 (mean margin unchanged, but P(valid) 0.74–0.82) | 0 / 6 (P(valid) 0.76–0.93) |
| 3 selective (localizer, PrOntoQA ΔAUC < 0.05; verdict-free shifts < 1 logit; WikiText KL < 0.02) | 5 / 6 (rank-1 seed 0 fails on word-recall shift 1.6) | 5 / 6 (rank-1 seed 0: word-recall shift 1.2) |
| 4 patching (transfer ≥ 0.2; off-target < 0.1) | 0 / 6 | 0 / 6 |
| **all of 1–4** | **none** | **none** |
Criterion 5 (cross-seed reproducibility) is moot. Criterion 2 fails mostly because the pre-registered objective closes the gap at the intact pair-mean margin (≈+2), which sits on the "valid" side of the decision threshold, so predictions drift toward "valid" even though the mean margin does not move; this is a property of the objective, not a hidden bias shift, but it is a failure under the rule as written and is reported as one.

## Interpretation
1. **Information vs. causal use.** Validity is redundantly and linearly readable at the hook location and even more clearly at the conclusion's last token; removing the verdict direction (or the whole MVS span, or 12 INLP directions) does not remove it. The model's own use of it need not go through any low-rank bottleneck at this layer.
2. **The searched ablation effects are format/template-bound.** They disrupt validity ranking in the prompt formats seen during search (largely) and partly in related ones, not in the reversed-mapping or Yes/No formats, and their coordinates are not a transplantable validity variable. The most economical reading is that the optimiser found low-variance directions that sabotage a template-specific readout of the answer, not a content variable of "logic". This is an interpretation; we did not test it directly (e.g. by searching on all five formats jointly).
3. **What remains robust from earlier phases:** `d_c` is a reproducible, transplantable accept/reject verdict channel (Phase 0.5), and it is not logic-specific. Nothing in Phase 0.6 contradicts this; it adds that validity discrimination does not depend on `d_c` (ablation keeps AUC ≥ 0.92).

## Is there a credible, causally selective target for neuron-level analysis?
**No for "abstract logic".** Neither a validity-specific direction orthogonal to `d_c` (none passes), nor `d_c` itself (verdict, not logic) qualifies. If neuron-level work proceeds, the defensible targets are:
* **`d_c`, framed honestly as the verdict channel** (which native neurons write it; do they write it for localizer/PrOntoQA as well?), or
* a **different locus**: validity is already decodable at the conclusion's last token, upstream of the answer position, and distributed at layer 23; a search at earlier layers/positions, over components (attention heads, MLP outputs), or with sparse/multi-direction interventions would be needed. This is a new study, not an extension of this pilot.

## Limits and negative results
* Pilot-scale search (300 steps, ranks ≤ 4, 2 seeds, one layer); not extended after pilot 2 by design. A joint all-format search, higher ranks, or other layers were not tried.
* The benchmark is re-created and form-determined; held-out formats num_rev/ab_orig were also seen (for pilot-1 candidates) before pilot 2, so only `yn` and all splits/controls are untouched by both searches.
* Search-time controls reuse the same 8 base sentences for the localizer (items differ); they enter only as constraints.
* Multiplication (generation-based) was omitted from Phase 0.6 control evaluation; word recall (48), Stroop, MMLU-binary, localizer, PrOntoQA and WikiText KL were used.
* Candidate intervention is mean-replacement ablation; other ablation types (resample, patching-only) were evaluated only via patching.
