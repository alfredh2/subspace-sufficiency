# Phase 1 plan — native-MLP-neuron implementation of the consensus *verdict direction*

**Terminology (binding for all code, docs and reports):** the object studied is the *verdict direction* `d_c` (consensus rank-1 direction of the three Phase-0 MVS checkpoints, built exactly as in Phase 0.5/0.6, `phase06.load_consensus()`).
It is an accept/reject verdict channel, **not** a logic-specific direction. No claim about "logic neurons" or "abstract-logic circuits" will be made. No new search for logic subspaces; no downstream-reader tracing.
Frozen artifacts: Phase 0–0.6 checkpoints, datasets, results are read-only. Everything new goes under `results/phase1/`. Nothing is pushed.

**This document is committed before any Phase-1 result exists. Thresholds below are preregistered targets; changes after results are logged as dated amendments, never silent edits.**

## 1. Setting and notation
* Hook = `residual_index 23` = output of `layers[22]`. Writers into the hook coordinate `c = d_c·h₂₃` are exactly: embedding, `attn_0..22`, `mlp_0..22` (additive residual ⇒ exact identity `c = d·emb + Σₗ d·attnₗ + Σₗ d·mlpₗ`). Layers ≥ 23 are excluded by construction (enforced in code).
* Native unit: post-gating intermediate unit `a_j = silu(gate_j(x))·up_j(x)` (12,288 per layer), the input of `down_proj`. `g^l = W_downˡᵀ d_c` (12,288). Direct contribution of unit j to the verdict coordinate: `s_j = a_j·g_j`.
* Accept/reject labels are semantic: logic valid/invalid; language localizer English(1)/nonsense(2); PrOntoQA true/false. Class-difference contribution of unit j: `Δ_j = E[s_j | accept] − E[s_j | reject] = δ_j g_j` with `δ = E[a|accept] − E[a|reject]`. Primary position: last prompt token. Secondary: conclusion-end token (logic only).
* Primary prompt format for ranking: `num_orig`; `num_rev` is the cross-format replication.

## 2. Metrics (frozen definitions)
* **Concentration uses |Δ_j| only** (never signed shares): top-k absolute mass share (k = 0.1 %, 1 %, 5 %, 10 % of units), effective number `N_eff = 1/Σp²` and `exp(H(p))` with `p_j = |Δ_j|/Σ|Δ|`, Gini. Per layer and pooled over selected layers.
* **Signed** quantities, never used as concentration measures: net `D_l = ΣΔ_j`; top-k signed reconstruction `Σ_{S}Δ_j / D` and error `|D − Σ_SΔ_j|`; per-example reconstruction R² of `y_i = Σ_j s_ij` by `Σ_{j∈S} s_ij + mean_train(Σ_{j∉S}s)`; cancellation index `κ = Σ|Δ_j| / |ΣΔ_j|`, positive/negative mass; per-example `κ_i`.
* Top-k sets are chosen on **training** prompts and evaluated on held-out; also report held-out re-estimated mass (selection-bias control).
* **Causal effect measures (reported separately):** (a) *direct* change of the verdict coordinate from the ablated units' own writes (`−Σ_S (a_j−ā_j) g_j`, last position); (b) *actual* change of `c = d_c·h₂₃` at the hook (includes indirect effects); (c) *answer-level* semantic margin effects (accept-label logit − reject-label logit): mean-margin shift, class gap, AUC, P(accept), invalid rate.
* **Headline endpoint** `E_gap^T(S) = (gap_T(∅) − gap_T(S)) / (gap_T(∅) − gap_T(d_c-ablation))`, task T ∈ {logic num_orig, logic num_rev, localizer, PrOntoQA}: fraction of the answer-margin class-gap loss caused by full `d_c` ablation that the neuron intervention reproduces. Secondary `E_shift^T` (same with mean-margin shift). Denominators < 5 logits are invalid. Raw accuracy is secondary.
* **Perturbation strength:** `P(S)` = mean over eval examples of `‖h₂₃(S) − h₂₃(∅)‖₂` at the final position, and `|Δc|`. Equal-k native vs rotated is reported, and **comparison at matched `P`** is done by interpolating each control family's pooled `(P, E_gap)` curve at the native set's `P`.

## 3. Stages, data, gates
**Stage 0 — infrastructure + tests** (no results): `neurons/`, `rotations/`, `interventions/`, tests (§7), rotation-cost benchmark. Gate: all tests pass; identity checks pass on the tiny model.

**Stage 1 — layerwise decomposition (all 23 layers).** Training prompts only: logic 1000 (balanced, train contents, 11 train families, `num_orig`), localizer 256 and PrOntoQA 256 (generator seed 101). Report for every layer `l ≤ 22`: signed class-difference of `d·mlpₗ`, its absolute value `|D_l|`, `A_l=Σ|Δ_j|`, `κ_l`; the same for `attn_l` and the embedding; total class difference of `c`; identity check `c` vs component sum (bf16 tolerance ≤ 2 % of the std of `c`; exact on the fp32 tiny model).
 * **Layer selection (frozen):** the ≤ 4 MLP layers with the largest `|D_l|` on the **logic** training prompts (ties by `A_l`). Report their share of `Σ_l|D_l|`. Selection does not require any coverage threshold.
 * **Scope gate (frozen), on the absolute MLP share** `Σ_l|D^{mlp}_l| / (Σ|D^{mlp}_l| + Σ|D^{attn}_l| + |D^{emb}|)`: ≥ 0.30 → full pilot; 0.15–0.30 → run Stages 2–4 but every claim is scope-limited to "the MLP share"; < 0.15 → stop after Stage 2 and report. The gate limits the scope of the neuron analysis; it is not evidence for or against sparsity.

**Stage 2 — native vs null concentration (selected layers L\*).** `Δ` from training prompts (logic, localizer, PrOntoQA separately), bootstrap over prompts for CIs, split-half held-out check. Nulls, each reported as its own distribution, **never pooled**:
 * *True Haar* `R ∈ O(12288)` (QR of Gaussian, sign-corrected): cost benchmarked in Stage 0 (§5). Rotated contributions are `Δ^rot = (Rδ) ⊙ (Rg)` (class means are linear in `a`).
 * *Haar-equivalent two-vector sampler*: `(Rδ, Rg)` has the law of a Haar-random 2-frame scaled by `‖δ‖, ‖g⊥‖` plus the fixed Gram matrix; exact for this statistic, O(m) per draw, used for large-n nulls (n = 1000 per layer). Validated against true-Haar draws (two-sample KS on top-1 % share; must not reject at 0.01) before it is used.
 * *Structured block-permute-Haar* null (12 blocks × 1024 independent Haar, random permutations, 3 rounds): exactly orthogonal, **not** Haar; reported separately.
 * *Shuffled alignment*: random permutation of `g` relative to `δ` (same marginals, alignment destroyed) → separates "g concentrated", "δ concentrated" and joint alignment.
 * Reported marginals: concentration of `|g_j|`, of `|δ_j|`, and of `|Δ_j|`. Also the residual-basis concentration of `d_c` itself (participation ratio 1175/4096 already known, not a claim of sparsity).

**Stage 3 — signed reconstruction, cancellation (L\*).** Training-selected top-k (by `|Δ_j|`) evaluated on held-out items/families/nonwords (logic, `num_orig`) and seed-0 controls; k as in Stage 4. Native vs true-Haar rotated (≥ 10 draws; per-example R² needs actual `R` applied to stored last-position activations).

**Stage 4 — causal ablation (main test).** Intervention: replace `a_j` by `ā_j` (neutral-corpus mean over all token positions; WikiText-2 train windows, as the Phase-0 `μ`) in the chosen units, applied at **all positions** (primary) and last-position only (secondary), layers ≤ 22 only. Zero replacement is secondary (k = 64, 512). Ranking = `|Δ_j|` on logic training prompts, selected globally across L\* (and per-layer as a secondary analysis).
 * k grid {4, 16, 64, 256, 512 (≈ 1 % of the ≤ 49,152 units in L\*), 2458 (≈ 5 %)}.
 * Eval subsets (held-out; fixed seeds): logic 384 (128 per split Items/Families/Nonwords, balanced) × {`num_orig`, `num_rev`}; localizer 128, PrOntoQA 128 (seed 0); verdict-free MMLU-binary 96, Stroop 96, word-recall-48 96; WikiText KL on 32 passages.
 * Reference conditions: intact; **full `d_c` ablation** (Phase 0 operator); activation-only top-k (largest `E[a²]`).
 * **Controls, per k:** random units (same per-layer counts as the native set), activation-matched (each native unit replaced by a random unit of the same layer with nearest `E|a|` and `E[a²]`), alignment-matched (nearest `|g_j|` quantile, random activation), rotated-coordinate ablation: top-k rotated coordinates by `|Δ^rot|` and random rotated coordinates (rank-k projection `a ← a − ((a−ā)R_Sᵀ)R_S`, true Haar, no d×d materialisation beyond the one R).
 * **Draw counts (budget-aware, frozen):** k ∈ {64, 512}: 20 random + 20 activation-matched + 20 alignment-matched + 10 rotated-top + 10 rotated-random; other k: 6 + 6 + 0 + 4 + 4.
 * **Statistics:** a native result "exceeds controls" if it exceeds **all** draws of that family (empirical p ≤ 1/21 for n = 20) *and* the bootstrap (over eval examples) 95 % CI of `E_native − mean(E_control)` excludes 0. With n < 20 only effect sizes and intervals are reported, no percentile claims.
 * Early stop (frozen): after the native grid and the k ∈ {64, 512} controls, if `E_gap^{logic}(top-5 %) < 0.10` and not above the control maxima, skip direction replication and causal cross-task runs (cheap ranking-overlap analyses still run) and report "distributed (causal)".
 * Replication: the three per-seed shared directions (own `g`, own top-k) at k = 512 with 10 random + 10 activation-matched controls each.

**Stage 5 — cross-task analysis (item 9).** Per task (logic, localizer, PrOntoQA): `|Δ_j|` rankings over L\*; Spearman and top-1 % Jaccard overlap vs the hypergeometric expectation, bootstrap CIs; causal: logic-ranked set's `E_gap` on all tasks, localizer-/PrOntoQA-ranked sets on logic (k = 512). Labelled "shared verdict implementation" only if cross-task `E_gap ≥ 0.2` for both.

## 4. Preregistered targets and conclusion rule
Geometric evidence (consensus `d_c`; replicated on the 3 shared directions):
 * **G-sparse** requires: (G1) native top-1 % `|Δ|` share ≥ 2× the *true-Haar* null mean and above all true-Haar draws, and `N_eff` ≤ 0.5× the null; (G2) the same relation to the structured and shuffled nulls; (G3) training-selected top-1 % signed reconstruction ≥ 50 % of the held-out class difference and per-example R² ≥ 0.5, random-k < 10 %.
 * **G-distributed**: top-1 % share < 1.5× null **and** signed reconstruction at 1 % < 20 %. Otherwise G-mixed.
Causal evidence (consensus, k = 512 ≈ 1 %):
 * **C-sparse** requires all of: (C1) `E_gap` ≥ 0.30 on logic `num_orig` **and** `num_rev`; (C2) exceeds all draws of random, activation-matched, alignment-matched families + CI criterion; (C3) rotated-top equal-k `E_gap` < 0.10 and, at matched `P`, rotated/random curves below the native by ≥ 0.15; (C4) WikiText KL < 0.02 and verdict-free margin shifts < 1 logit; (C5) answer-level effect is in the direction of the verdict coordinate effect (direct and actual `Δc` of the same sign).
 * **C-distributed**: `E_gap`(5 %) < 0.10, or native not distinguishable from matched controls until k ≥ 5 %.
 * Otherwise C-mixed.
**Overall conclusion:** *sparse native implementation supported* only if G-sparse and C-sparse and ≥ 3 of the 4 directions replicate the geometric criteria; *distributed implementation supported* only if G-distributed and C-distributed; every other combination is reported as **mixed/inconclusive with the named pattern** (e.g. geometrically concentrated but not causal; causal but not concentrated; direct-write small but total effect large via downstream layers; layer-level but not neuron-level concentration; scope-limited by the MLP share). A binary verdict is not forced.

## 5. Rotation cost benchmark (Stage 0, before committing counts)
Measure true-Haar generation (QR, fp32, GPU) for m = 12288: time and peak memory, and orthogonality error. Frozen rule: if ≤ 20 s and ≤ 4 GB each → 20 true-Haar per selected layer for Stage 2/3 and 10 per layer for Stage 4; otherwise reduce to the largest feasible number, label it "small Haar reference", and rely on the Haar-equivalent sampler and the structured null for large-n distributions without merging them.

## 6. Budget, ordering and ceilings
Ceiling for the pilot: **5 GPU-hours**, tracked in `results/phase1/budget.json`. Order: Stage 0 → 1 → 2 → 3 → Stage 4 (native grid, `d_c` ablation, then k ∈ {64, 512} controls, then rotated, then the rest) → Stage 5 → direction replication. Stop at any gate above. No exhaustive/full-layer run is launched automatically.

## 7. Tests (required before results)
Decomposition identity (fp32 tiny model; bf16 tolerance on Qwen); `Σ_j s_j = d·mlp_out`; down-proj pre-hook captures the true input; ablation hook ≡ manual column surgery; `k=0` = baseline; mean replacement idempotent; layer > 22 rejected; direct change equals actual change for the last layer before the hook; Haar orthogonality, seed reproducibility; function-preserving rotation (`W Rᵀ(Ra)=Wa`, MLP output and logits); `Δ^rot=(Rδ)⊙(Rg)` equals explicit per-example averaging; two-vector sampler vs true Haar (KS); structured rotation orthogonal; rank-k rotated ablation ≡ explicit `R Sₖ Rᵀ`; metric functions on closed-form cases (Gini, N_eff, top-k share, reconstruction at k=m exact); matched-control samplers (same layer counts, disjoint from native set, matched-statistics sanity); bootstrap/empirical-p helper correctness.

## 8. Limitations (carried into reports)
Direct write ≠ total causal importance (units also act through layers l+1..22); one direction at one hook, read at one position (though trained on all); rotation after the gate is a coordinate change of the same update, not independent neurons; `d_c` is partly seed-dependent (shared-direction cosine 0.76–0.83); the benchmark is re-created and form-determined; verdict direction is shared across tasks, so "selective" means verdict-tasks vs verdict-free tasks; mean-vs-zero replacement shifts distributions differently; bf16 numerics; multiple comparisons across layers/k/controls (stated, not corrected beyond the empirical-p rule).
