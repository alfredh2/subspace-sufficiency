# Phase 1 report — native MLP-unit implementation of the consensus *verdict direction*

**Scope and wording.** The object is the *verdict direction* `d_c` (consensus rank-1 direction of the three Phase-0 MVS checkpoints, hook `residual_index 23`). It is an accept/reject verdict channel, shared by logic, the language localizer and PrOntoQA; nothing here supports "logic neurons".
All decision rules were frozen in `docs/plans/phase1_plan.md` (committed before any result). Tables: `results/phase1/report_tables.md`; criteria: `results/phase1/criteria_verdicts.json`. Pilot cost ≈1.2 GPU-hours of the 5-hour ceiling; no exhaustive run was launched.
Code: `src/causal_subspaces/{neurons,rotations,interventions}`, `scripts/phase1_*.py`; 74 tests pass (decomposition identities, down-proj capture, ablation ≡ manual surgery, k=0 no-op, layer-ordering guard, direct-change identity, Haar orthogonality/reproducibility, function-preserving rotation, `Δ^rot=(Rδ)⊙(Rg)`, sampler-vs-Haar KS, structured rotation, rank-k rotated ablation ≡ explicit projection, concentration/reconstruction closed forms, matched-control samplers, empirical-p/bootstrap).

## Bottom line (frozen rule): **mixed / inconclusive — geometrically sparse, causally selective but incomplete**
* Geometry satisfies every preregistered sparse criterion (G1–G3, all four directions).
* The causal test satisfies C1, C2, C3, C5 and **fails C4** narrowly (verdict-free margin shift on Stroop +1.13 logits vs the 1.0 limit, word recall −1.03; KL 0.003). The frozen rule therefore returns C-mixed, hence an overall *mixed* label, not "sparse native implementation supported".
* Substantively: ~500 native units (≈1 % of the 49,152 units in layers 19–22, dominated by layer 22) carry the MLP-written part of the verdict direction and are *causally selective* relative to random, activation-matched, alignment-matched and rotated-coordinate controls, but they reproduce only ≈40–60 % of the answer-level effect of removing `d_c`, and the MLP is only ≈56 % of the absolute writing. The rest (attention writes, more units, indirect paths) is not accounted for.

## Stage 1 — where the direction is written (training prompts)
* Class difference (valid − invalid) of `c = d_c·h₂₃` = **+19.2** = attention **+8.5** + MLP **+10.7** + embedding 0.0; identity error 0.04 (std of c 10.6). Absolute MLP share 0.557 → gate "full pilot".
* All 23 layers reported (signed, absolute, Σ|Δ_j|, κ). Layers 0–17 contribute ≈0; layer 22: +10.08 (κ = 2.0), 21: +0.35, 20: +0.20, 19: +0.07. Selected (frozen rule): **[22, 21, 20, 19]** (99.5 % of Σ|D^mlp|). Attention writes are largest at layer 22 (+4.9), 21 (+2.3). Same picture for the localizer (attn +11.8 / MLP +6.8) and PrOntoQA (+2.7 / +3.6) and for the three shared directions.
* Consequence: the hook-level coordinate is mostly produced by the last block(s) before the hook; neurons in earlier layers matter only indirectly.

## Stage 2 — concentration of |Δ_j| vs nulls (pooled over layers 19–22)
| | native | true Haar (n=20) | structured block-permute (n=20, not Haar) | Haar-equivalent sampler (n=1000) | shuffled alignment (n=1000) |
|---|---|---|---|---|---|
| top-1 % share of Σ\|Δ\| | **0.490** (bootstrap 95 % CI 0.484–0.494) | 0.126 (max 0.130) | 0.125 | 0.126 | 0.215 (max 0.335) |
| effective number N_eff | **60** (CI 59–62) | 11,671 | 11,748 | 11,700 | 3,904 (min 39) |
Native exceeds every draw of every family; ratio 3.9× (Haar). True-Haar generation at m=12,288 costs 0.24 s / 2.3 GB, so 20 true-Haar rotations per layer were used; the exact two-vector sampler (validated against 100 true-Haar draws: KS p = 0.66) supplies the large-n null; the structured null and the shuffled-alignment null are reported separately and never pooled with Haar. Shuffling alignment keeps marginals (|g| top-1 % 0.072, |δ| top-1 % 0.150) but loses most concentration, so it is the *joint* alignment of large `g_j` and large class-difference activations that is concentrated, not either alone.
* Layer 22 dominates: top-1 % share 0.579, N_eff 21 (vs Haar 0.079); layers 19–21: 0.30–0.31, N_eff 460–660.
* Split-half: top set chosen on half A holds 0.488 of half-B |Δ| mass (B's own top set: 0.488; chance 0.010); rank Spearman 0.97.
* Replicates for shared_s0/s1/s2 (top-1 % 0.46–0.48, ratio 3.7–3.8, N_eff 62–72) and for the localizer (0.338, N_eff 446) and PrOntoQA (0.423, N_eff 83); at the conclusion-end position it is weaker (0.259, N_eff 2364) but still 3.6× the null.
* Cross-task: the top-1 % unit sets overlap massively (logic∩localizer 265, logic∩PrOntoQA 308 of 492; ≈5 expected by chance; Spearman of |Δ| 0.61–0.69). The same units carry the verdict signal across tasks — consistent with a shared verdict channel.

## Stage 3 — signed reconstruction and cancellation (selection on training prompts, held-out evaluation)
* Net held-out class difference of the layer 19–22 MLP writes: +10.64. **Top-512 (≈1 %) units reconstruct 98.4 %** of it (per-example R² 0.998); top-16: 73 %, top-4: 61 % (R² 0.85). Random 512 units: 0.4 % (max 6.5 %); true-Haar rotated top-512: 39.7 % (R² 0.64); 2458 rotated units: 77 %. Same for the three shared directions (97–99 %, R² 0.998, random max 2 %), localizer (103 %) and PrOntoQA (97 %).
* Cancellation is modest at the class-difference level (κ = 3.1: positive mass 21.8, negative 11.2) and negligible in layer 22 (κ = 2.0) but large in single examples (mean per-example κ_i ≈ 145), i.e. individual prompts involve large opposing unit contributions that cancel in the net.
* Absolute mass share of the top-512 is only 0.49: half of Σ|Δ| is spread over many units, so "sparse" holds for the signed net reconstruction, not for all absolute mass.

## Stage 4 — causal ablation (mean replacement, all positions; held-out subsets)
`E_gap` = fraction of the answer-margin class-gap loss produced by full `d_c` ablation that the neuron intervention reproduces. Logic denominators (23.2 / 15.5 logits) and localizer (26.5) are valid; **PrOntoQA's denominator is 4.6 logits (< 5), invalid under the frozen rule: PrOntoQA values are shown but used in no criterion.**
| set (consensus) | k | E_gap logic num_orig | num_rev | localizer | direct class-gap change of c | actual class-gap of c at hook (intact 19.3) |
|---|---|---|---|---|---|---|
| native top-k | 4 | 0.17 | 0.26 | 0.23 | −6.6 | 12.7 |
| | 64 | 0.38 | 0.53 | 0.31 | −9.9 | 10.1 |
| | **512 (1 %)** | **0.41** | **0.59** | 0.25 | −10.5 | 7.2 |
| | 2458 (5 %) | 0.63 | 0.62 | 0.47 | −8.5 | 5.5 |
| activation-only top-k | 512 | −0.12 | −0.20 | −0.08 | −0.1 | 20.0 |
Controls at k=512 (n=20 each for random, activation-matched, alignment-matched; n=10 rotated): random 0.007 [−0.06, 0.05], activation-matched −0.026 [−0.16, 0.03], alignment-matched 0.028 [0.00, 0.07], rotated-top 0.071 [0.05, 0.12], rotated-random −0.022. Native exceeds **all** draws of the three native-control families (empirical p = 1/21 = 0.048, the minimum with n=20) and the bootstrap 95 % CI of `E_native − mean(E_control)` excludes 0 (e.g. +0.37…+0.44 vs random, logic `num_orig`). Same pattern for `num_rev` and, at k=64, for all four tasks.
* **Direct vs actual vs answer-level (reported separately):** the ablated units' own writes remove 10.5 logits of the verdict-coordinate class gap (97 % of the layer-19–22 MLP class difference), the actual gap at the hook falls by 12.1 (so only ≈1.6 is indirect through later layers), and the answer-level class gap loss is 41–59 % of what ablating `d_c` does. About 44 % of the coordinate's class difference is attention-written and is untouched here, which fits E_gap saturating around 0.4–0.6.
* **Perturbation strength:** native k=512 changes the hook residual by 29.0 on average; random/matched sets of the same k change it by 9–11 and rotated-top by 12; activation-only top-512 by 22 (and top-2458 by 35 > 29) with ≈0 effect. Matched-strength interpolation is *clamped* for k=512 (control curves never reach 29 with k ≤ 2458 except rotated-top 26.7 / activation-only 35.3), so matched-strength evidence rests on (i) the k=64 comparison (native ‖Δh‖ 12.9 inside the control range: native 0.375 vs random 0.009, act-matched −0.025, rotated-top 0.077, rotated-random −0.022) and (ii) the activation-only k=2458 set, whose perturbation exceeds native's and which has no effect. Rotated-top at 5 % (‖Δh‖ 26.7) reaches 0.18 vs native 0.41 at 29.0 (0.63 at 50.6).
* **Side effects (C4 fails narrowly):** WikiText KL 0.003 (limit 0.02). Verdict-free margin shifts at k=512: MMLU −0.30, Stroop **+1.13**, word recall **−1.03** (limit 1.0). Control draws produce verdict-free shifts up to 0.85 (random), 1.06 (act-matched), 1.09 (align-matched) at k=512, so the native value is within the range that generic perturbations of that size produce; the frozen rule is still reported as failed.
* **Position / replacement:** last-position-only ablation keeps 0.27 / 0.42 / 0.15 (num_orig/num_rev/localizer); zero replacement gives the same as mean replacement within 0.05.
* **Direction replication:** for each per-seed shared direction (own gains, own top-512): E_gap logic = 0.46 / 0.60 / 0.40 vs control max 0.04 / 0.08 / 0.08.
* **Cross-task causal check:** the logic-ranked set affects the localizer (0.25) and PrOntoQA; the PrOntoQA-ranked set gives 0.50 / 0.82 / 0.40 on logic/logic-rev/localizer; the localizer-ranked set affects logic (0.43 / 0.55) but **not** the localizer itself (0.000) — the localizer's behaviour is robust to this ablation despite sharing the same units. Shared units carry the verdict signal across tasks causally for logic and PrOntoQA, only partly for the localizer.

## Frozen criteria applied
G1 ✓ G2 ✓ G3 ✓ (4/4 directions) → **G-sparse**. C1 ✓ (0.41 / 0.59 ≥ 0.30), C2 ✓, C3 ✓ (rotated-top 0.071 < 0.10; matched-strength comparisons ≥ 0.15 below native, partly clamped), **C4 ✗ (Stroop +1.13)**, C5 ✓ (direct and actual both reduce the gap) → **C-mixed** → **overall: mixed / inconclusive**.

## What this does and does not show
* Supports: the native post-gating basis is *privileged* for this direction — large concentration relative to Haar and structured rotations, reconstruction and causal effects that rotated coordinates (≈0.07–0.18) and activation/alignment-matched controls (≈0) do not reproduce, and the effect is not a generic size effect (activation-only sets). Only a few hundred units, mostly in layer 22, are needed for the MLP-written part.
* Does not show: a complete, sparse implementation of the verdict direction. Attention writes ≈44 % of the coordinate; ablating even 5 % of the units reproduces only ≈0.6 of the behavioural effect; the narrow C4 failure; PrOntoQA effect sizes are not interpretable; the localizer-ranked set fails on the localizer.

## Limitations
Direct write ≠ total causal importance (units also act downstream); one hook layer, one direction family, answer-position reading; rotation after the gate is a coordinate change of the same update, not independent neurons; the benchmark is re-created and form-determined; verdict direction is shared across tasks so "selectivity" means verdict tasks vs verdict-free tasks; control draws n=20 give p_min=0.048; matched-strength interpolation partly extrapolated; mean-ablation baseline uses the neutral-corpus mean; bf16 numerics; attention heads were accounted for but not analysed.

## Suggested next steps (not started)
(i) attention-write analysis of the remaining ≈44 % (head-level, with the same frozen-style rules); (ii) scale the unit ranking to all positions/layers only if (i) supports the "MLP + few heads" picture; (iii) test whether the 512-unit set also drives the shared direction when tasks are placed in other prompt formats. None of these were run.
