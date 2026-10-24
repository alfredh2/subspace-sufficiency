# Phase 8 pilot plan (PROPOSAL, not approved, nothing implemented): causal multiplicity of sufficient low-rank subspaces

Status: proposal for review. No experiment has been run, no file under `results/` was created or modified, and no earlier report, verdict, checkpoint or dataset is touched. The stopped Phase 7 multi-task fixed-basis programme and its failed gate (0 of 6 tasks, Stage 1b) are **not resumed**; this is a separate, single-task mechanism pilot motivated by one Stage 1b observation. All thresholds below are proposed to be frozen by a commit **before** any Phase 8 model run; after approval they may not be changed retroactively. New outputs would go to `results/phase8/`; existing checkpoints are read-only inputs whose file hashes get recorded in a manifest.

## 0. Question and what the pilot decides

> How many distinct low-rank residual subspaces can independently support the same behaviour, and do they represent genuinely redundant causal mechanisms or merely alternative sufficient readouts?

Pilot task: four-way QA (M, `letters` format), hook 25, rank 4, Qwen3-8B, starting from the existing Stage 1b selected subspaces. The pilot decides whether this becomes a main project (section 9), and is deliberately small: one task, one primary hook, one earlier hook, one comparison task.

### Three claims that must stay separate (operational definitions)

| label | meaning | what is needed to assert it | what does **not** suffice |
|---|---|---|---|
| **MSR** multiple sufficient representations | several mutually orthogonal rank-4 subspaces each reproduce the output when everything else is replaced by μ | fresh-question retention of each (KL ≤ 0.05, accuracy ≥ 0.9 × intact) | says nothing about the intact network |
| **SCR** static causal redundancy | in the *intact* network the carriers are jointly used: single removal at natural strength is small, joint removal is large and super-additive, and swapping the union moves the answer | joint-necessity and union-transplant tests beating three control families | orthogonal sufficiency alone |
| **ACT** active compensation | after removal, other components *change in a helpful way and that change is causally responsible* for preserved performance | a response (downstream change beyond controls) **and** blocking the response lowers performance | increased activation alone; a recovering logit-lens trace alone |

A fourth label, **ASR** (alternative sufficient readouts), is the null-like reading of MSR: the carriers are sufficient in the retention regime but the intact network does not use them (joint ablation and union transplant ≈ 0; the information is elsewhere).

## 1. What I inspected (read-only)

* **Existing QA subspaces:** `results/phase7/stage1b/final/M_letters_h25_r4_s{2,3,4}.pt` and `armA/M_h25_r4_s{0,1}.pt` (U1 candidates, seeds 0–4, rank 4, hook 25; `final/M_letters.json` records 5-seed gates on confirm contents); Arm B complements `armB/M_sel_U2_r{2,4,8}_s{0,1,2}.pt` (orthogonality error ≈ 1e-8, retention KL 6e-7 to 7e-5, accuracy 1.00 on confirm contents); `M_h25_r2` (2 seeds). Hook-21 rank-4 M fits exist for seeds 0, 1 (`armA/M_h21_r4_s{0,1}.pt`, val KL ≈ 1e-5).
* **Fitter:** `fit_b(..., perp=U)` in `src/causal_subspaces/phase7/stage1b.py` constrains span(V) ⊥ span(perp) *before* QR (exactly orthogonal; `perp` may be the stacked union, so chains U1 → U2 ⊥ U1 → U3 ⊥ [U1,U2] are supported without code changes). Recipe: Adam, 1600 steps, 16 prompts/step, lr 1e-2 cosine to lr/20, full-vocabulary KL, on-manifold init projected off `perp`. Cost ≈ 150–180 s per M fit.
* **Interventions:** `residual_intervention` (retain/ablate, all positions), `dose` hook in `armB` (h − α·UUᵀ(h−μ), last-position or all-position), `capture_residual`, `transplant` (length-matched pairs, ratio of sums), `energy` diagnostics, Haar / on-manifold / top-PCA controls, and `phase5.patching.pinned_rows` (pins attention/MLP outputs at chosen rows to clean values — reusable for the compensation block after a unit test).
* **Data (M):** `build_M`: 59 capitals + 36 elements; Stage 1b partition train 50 / val 13 / confirm 32 entities; test = 384 items, 96 pairs. **All 95 lexicon entities are already used in train, val or confirm**, so "fresh held-out questions" requires new entities (section 3).
* **Reports re-read:** Stage 1 and 1b reports (M: retention KL 0.0000; ablation 0.05 [0.01, 0.06]; transplant 0.093 [0.03, 0.11]; α = 4 ablation 0.83–0.96; union [U,U₂] 0.02–0.15; U holds 9 % of final-position variance vs Haar 0.3 %), the research synthesis, and the Phase 4–6 pinning machinery.
* **Facts from the artifacts that shape the design:**
  1. Ablation is an orthogonal projection, so **the other carriers' hook-25 coordinates are unchanged by construction**; a "do the remaining carriers change?" test is only meaningful *downstream* (section 5), and the hook-level version is reported as a numerical identity, not a finding.
  2. Some ablations *increase* the gap (Arm A, hook 21 rank 4: −0.14 / +0.14 across two seeds), so single-carrier effects are noisy in sign; the effect threshold must be above the measured noise floor (Phase 4 measured a 1.0-logit batch-size floor; M's intact gap is ≈ 52 logits).
  3. The fit objective is KL to the teacher; a fitted U can *construct* a carrier the downstream layers decode (Makelov-type concern), which is why transplant and a readout-likeness check are in the plan.
  4. Transplanting U1 alone moves the answer only 0.03–0.11. That is exactly what an OR-structured redundancy would predict (the other carriers still hold the target's own answer) **and** what unused alternative readouts would predict. Union transplant discriminates them.

## 2. Prior work and what it implies (verification status stated)

I delegated the literature search and did not read the papers myself; statements below are the subagent's, with the verification it reported.

* **Conditional Co-Ablation, arXiv:2607.01940** (Gong et al., 2026; PDF v3 read by the subagent). Scores each candidate unit by how much its ablation effect *grows* once a supplied primary set is already ablated (growth of δz_{u|S} over δz_u). Head level, GPT-2-small IOI: backups (8 documented heads) recovered at 0.941 ROC-AUC; joint ablation of primaries + backups changes the IOI margin 1.30 vs 0.28 for primaries alone and vs 0.67 additive prediction; random sets are roughly additive. Argues backups are invisible to intact-state attribution. **Not** a subspace-level or non-uniqueness-of-rank result, and the primary set is supplied. *Unresolved:* the abstract page and the PDF text give different baseline numbers (best-baseline AUC 0.815 vs 0.665); cite the specific version.
* **Rushing & Nanda 2024, arXiv:2402.15390** (PDF read by the subagent). Self-repair after single-head ablation is imperfect and noisy and sometimes overcorrects; **final-LayerNorm rescaling explains roughly 30 % of direct-effect repair on average**; MLP "anti-erasure" is carried by sparse, prompt-dependent neurons. Direct effects only; ablations are off-distribution. *Implication:* part of any apparent compensation is passive; LN scale must be frozen as a control.
* **Not read by me or the subagent in full (search or memory only; verify before citing):** McGrath et al. 2023 (Hydra effect); Wang et al. 2023 (IOI backup name movers); Makelov, Lange, Nanda 2023 (subspace patching illusion via dormant directions); Sutter et al. 2025, arXiv:2507.08802 (unrestricted non-linear alignment maps trivialise sufficiency; linear maps avoid this); Méloux et al. 2025, arXiv:2502.20914 (multiple circuits/explanations reproduce the same behaviour).
* **Design implications adopted:** (i) restrict to linear, fixed-rank, exactly orthogonal subspaces and a mean-background retention regime (Sutter, Makelov); (ii) CoAx-style *conditional* measurements at the subspace level (effect of B alone vs with A already removed, for both ablation and transplant); (iii) separate passive repair (final-LN scale frozen) from component-level response; (iv) component-level work is exploratory-to-confirmatory with a dev-only candidate selection, because CoAx supplies its primary set by hand and we do not.
* **Novelty claim limited to:** a subspace-level, counted version of the question on one task, with sufficiency / necessity / transplant / compensation measured separately. It does not extend CoAx or self-repair findings to other models.

## 3. Data and splits (fresh questions)

* **Fit data:** existing M train items (50 entities, 1200 items), unchanged.
* **Dev data (all selection decisions: carrier count K*, control matching, compensation candidate set, hook-21 feasibility):** existing M val (13 entities, 208 items) **plus a new fresh-dev batch** of ≥ 16 entities.
* **Confirm data (all verdicts):** a **fresh confirm batch of ≥ 40 entities never present in any earlier lexicon or split** (new countries/capitals not among the 59, new elements not among the 36), each with all four answer positions × 2 fresh distractor draws (≥ 320 items) and 3 length-matched transplant pairs per entity (≥ 120 pairs). Entities are the clustering unit (≥ 40 clusters vs 32 in Stage 1b).
* **Freeze order:** entity lists, seeds, prompts, token checks and a hash manifest are committed *before* any carrier is evaluated on them; a baseline screen (intact accuracy ≥ 0.95, entity dropped before any intervention only if its intact accuracy < 0.9) mirrors Stage 1.
* **Reused data, labelled as such:** Stage 1b confirm (32 entities, already examined) is re-evaluated as a secondary replication and flagged "reused".
* **Optional descriptive item:** a small fresh-*relation* set (one new question template) to see whether carriers transfer; never part of a verdict.

## 4. Carrier chains and counting

* **Chains:** 3 chains (seeds 0, 1, 2). Carrier 1 = the existing Stage 1b U1 for that seed. Carrier k = `fit_b(perp = [U1…U_{k−1}])`, fresh init seed (100·k + seed), rank 4, same recipe. Cap **K = 8** (32 dimensions) in the pilot: 7 new fits per chain, 21 fits.
* **Order-independence check:** one extra *parallel* fit per chain, U2′ ⊥ U1 only, from a different init; report its overlap with the sequential U2 (mean cos² vs Haar-in-complement null). If parallel fits overlap strongly, the sufficient set in the complement is lower-dimensional than the count suggests; if not, the family is large. Count and dimension are reported separately.
* **Sufficient carrier** (dev rule, frozen): retained KL ≤ 0.05 and accuracy ≥ 0.9 × intact on dev, in this chain. **K\*** = number of consecutive sufficient carriers from k = 1 (capped at 8; "≥ 8" if the cap is reached). Extension, pre-justified: if all three chains reach 8, one chain continues to 16 (+8 fits ≈ 25 min), because the count would otherwise be a cap artefact.
* **Distinctness:** exact orthogonality (|UᵢᵀUⱼ|max ≤ 1e-6, verified in fp32) is necessary for "distinct", not sufficient for "different information". Descriptive: (a) linear predictability of each carrier's hook-25 coordinates from the other carriers' coordinates on dev (ridge R², CCA), (b) answer decodability from each carrier alone by a linear probe (dev-fitted, confirm-evaluated), (c) conditional decodability of the answer from carrier k given earlier carriers.

## 5. Measurements (all at natural strength unless marked; confirm data)

For each chain and k ≤ K (unions U_{1:k} = span of the first k carriers):

1. **Sufficiency S:** retain(U_k) alone, and retain(U_{1:k}) (KL, accuracy, candidate-restricted gap).
2. **Single-subspace necessity N1_k:** relative gap reduction and absolute logit reduction from ablating U_k.
3. **Joint necessity N_J(k):** cumulative union ablation k = 2, 3, 4, 6, 8, plus all pairwise unions for k ≤ 4 (6 pairs). Additivity excess A(k) = N_J(k) − Σ_{i≤k} N1_i. **Conditional growth** CG(B|A) = TE(A∪B) − TE(A) compared with TE(B) (CoAx analogue).
4. **Transplantability:** T_k (swap U_k between length-matched pairs, all positions, ratio of sums, entity-cluster bootstrap), union transplant T_J(k) for the same unions, and **conditional transplant** T(B | A ablated): transplant B in a model whose A has been removed.
5. **Controls for every ablation and transplant (same total rank 4k):**
   * Haar random; on-manifold random (existing machinery);
   * **energy-matched:** random subspaces whose mean fraction of ‖h−μ‖² at the final position (and over all positions) is within ±10 % of the carriers' (carriers hold 7–9 % at the final position vs 0.2–0.3 % for naive random), drawn by rejection from covariance-weighted random draws on dev;
   * **intervention-effect-matched:** random subspaces (same pool) whose mean displacement ‖Δh_L‖ of the final-layer pre-norm residual at the answer position under ablation, measured on dev, is within ±15 % of the carriers'; effect is matched on a quantity independent of the answer margin. Match quality is reported; if no draw reaches the tolerance, the control is reported as "unmatched" with the achieved ratio, not rescaled;
   * 10 draws per family per k (6 for k = 6, 8 if cost requires, declared in advance).
6. **Dose and norm-matched supplements (secondary, never primary):** α ∈ {2, 4} for unions, and random-subspace ablations rescaled to the carriers' perturbation norm.
7. **Last-position-only ablation** (secondary, as in Arm B).

### Direct tests of "do the remaining carriers or downstream readouts change after another carrier is removed?"

* **7a. Hook-level identity (validation, not a finding):** carriers B's hook-25 coordinates are identical before and after ablating A (orthogonal projection). Reported as a numerical check (max change ≤ 1e-5 fp32 equivalent; bf16 rounding noted).
* **7b. Layerwise answer trace:** logit-lens answer margin (final norm + unembedding rows of A/B/C/D) at layers 26, 27, 29, 31, 33, 35 for intact, A-ablated, unions, and controls. A dip-then-recover profile (CoAx / Hydra signature) is a *candidate* signature only.
* **7c. Downstream representational drift:** at the same layers, the 3-dimensional class-mean subspace (4 answer classes) of the intact vs ablated runs at the answer position; principal-angle distance between them, and the accuracy of an intact-trained probe applied to the ablated run vs a probe retrained on the ablated run. Same-answer-subspace → static; rotated but still decodable → re-expression; both compared with random/energy/effect-matched ablations (where drift is a nuisance).
* **7d. Component response:** direct-logit-attribution of each attention and MLP output (layers 26–35, answer position) onto the correct-vs-others margin: ΔDLA = DLA(ablated) − DLA(intact), with **final-LayerNorm scale frozen to the intact value** as the passive-repair control (Rushing & Nanda). Response is "beyond passive" only if it persists with the LN scale frozen.
* **7e. Conditional growth in causal effect** (3, 4 above): B's ablation/transplant effect with A absent vs present.

### Causal compensation test (conditional on a plausible pathway)

* **Gate C1 (dev only):** a candidate set C of n = 6 components (attention or MLP outputs, layers 26–35, answer position) selected by largest positive ΔDLA toward the correct answer under ablation of U_{1:4} (K_c = 4, the primary preserved-performance condition; K_c = 1 and 8 secondary), required to exceed the same statistic's maximum over all energy- and effect-matched control ablations by a frozen margin and to persist with LN scale frozen. If no set passes, the block is **not run** and ACT is recorded as "not detected at component granularity".
* **Block test (confirm):** in the ablated run, pin C's outputs at the answer position to their intact-run values (reuse `pinned_rows`; unit-tested to be an exact no-op on the intact run and to reproduce intact logits when everything is pinned). The estimand is BE(C) = [gap reduction with ablation and C pinned] − [gap reduction with ablation alone]. Controls: pinning 20 random component sets of equal size (same layers/types distribution), pinning C under energy- and effect-matched random ablation, and pinning C in the intact run (must be 0).
* **Reading and its limit:** pinning removes the *whole* change of C, i.e. both a helpful compensating response and any passive propagation of the lost carrier's information through C. A drop on pinning (BE > 0) is compensation-consistent; a rise is passive propagation; the sign is pre-registered and the passive/active ambiguity is stated in the report. Increased activation alone is never read as compensation.

## 6. Late-layer readout checks (item 7)

* **Readout-likeness per carrier:** accuracy and gap of the logit lens applied to retain(U_k) *at hook 25* (before any downstream block). If a carrier already decodes the answer there (≥ 0.9 accuracy), it is classed "readout-like"; sufficiency then partly reflects a late write rather than computation. Also the fraction of carrier energy in the span of the four letter unembedding rows, and the retention result for that 4-d unembedding span itself as a reference.
* **Earlier hook:** repeat the chain at **hook 21** (15 layers remain vs 11), the earliest hook where Stage 1/1b fits were sufficient (rank 4, val KL ≈ 1e-5 in both seeds; hooks 19 and 23 were not sufficient in the 300-step Stage 1 sweep, and hook 19/17 were not fitted at 1600 steps, so no earlier choice is supported by data). 2 chains (seeds 0, 1), K ≤ 4, same measurements 1–4 and 7b; carriers must be sufficient on dev at hook 21 or the arm is reported "not feasible".

## 7. Limited logic comparison (item 6)

* L (`num_orig`) at its frozen Stage 1b cell (hook 23, rank 4; per-seed necessity 0.37–0.58, median 0.44), 3 chains, K ≤ 4 (9 new fits, ≈ 0.7 GPU-h at 250–280 s per fit), on the existing 14 L confirm contents (reused; flagged). Same measurements 1–4 and controls; no compensation test unless L's gate C1 passes on dev.
* **Comparator-validity gate (frozen):** the contrast is meaningful only if L's single-carrier necessity median is ≥ 0.30 while M's is ≤ 0.10 on their respective confirm sets. If not, fall back to Stage 1's rank-2 L fits (hook 23, necessity 0.75 / 0.77 / 0.37 per seed) with rank-2 complements, declared before the L runs.
* Purpose: to see whether the QA pattern (large MSR count, small single necessity) is task-specific or a property of this procedure. The contrast is descriptive (3 chains).

## 8. Numerical validation (Gate 0, before any scientific run)

* Orthonormality and exact orthogonality of every carrier (|UᵀU − I| ≤ 1e-6; |UᵢᵀUⱼ| ≤ 1e-6), projector idempotence, retain + ablate reconstruction, fp32 projection vs bf16 hook difference on a fixed batch.
* No-op hook returns identical logits (max |Δ| = 0); padding / batch-size invariance (noise floor of the candidate-restricted gap at bs 32 vs 5; all effect thresholds must exceed 2× this floor, otherwise the thresholds are raised *before* the run).
* Union projector of stacked carriers equals sequential ablation; `perp` fits orthogonal to a stacked union.
* Pinning unit tests (exact no-op, full pin reproduces the intact logits, LN-scale freeze reproduces intact logits on the intact run).
* Control-matching diagnostics (achieved energy and displacement ratios) and fresh-data checks (disjointness from every earlier split by entity, token checks, intact accuracy).
* Determinism: each carrier fit re-run once at the same seed for one chain to measure fit-level reproducibility.

## 9. Preregistered criteria

Per chain unless stated; "supported" in a chain means that chain meets the criterion on confirm data; **seed support = at least 2 of 3 chains and median on the supporting side**; each criterion's verdict is PASS (support holds and the hierarchical 95 % bound is on the passing side), FAIL (support fails and the median is on the failing side), else INCONCLUSIVE (same logic as Stage 1b). Intervals: hierarchical bootstrap, chains then entity clusters.

* **MSR (multiplicity).** PASS: median K\*_confirm ≥ 4 and ≥ 2/3 chains with K\* ≥ 4 (carriers also sufficient on the fresh confirm data). FAIL: median K\* ≤ 2. Else INCONCLUSIVE. (Expected to pass given three orthogonal carriers already exist; it is stated to guard against a count that does not generalise to fresh questions.)
* **Single necessity absent (R1).** Median N1 ≤ 0.10 relative and upper bound ≤ 0.15, with random controls ≈ 0 and above the noise floor-based threshold.
* **Joint necessity (R2).** Exists k ≤ 8 with N_J(k) ≥ 0.30 relative **and** ≥ 10 logits, exceeding the maximum of each control family (Haar, energy-matched, effect-matched) by ≥ 0.15.
* **Super-additivity (R3).** A(k) ≥ 0.15 at that k, with the lower bound > 0.
* **Union use (R4, supporting).** T_J(k) ≥ 0.25 with single T_k ≤ 0.15, controls ≤ 0.05.
* **SCR.** PASS = R1 ∧ R2 ∧ R3. FAIL ("ASR") = N_J(8) < 0.10 **and** T_J(8) < 0.15 in ≥ 2 chains. Otherwise MIXED (e.g. R2 without R1, or necessity without super-additivity).
* **ACT.** C0 (response persists with LN scale frozen) ∧ C1 (dev gate) ∧ C2 (BE(C) ≥ 0.10 relative and ≥ 5 logits, lower bound > 0.05, random-set pins ≤ 0.03 in absolute value, pin of C in the intact run = 0, in ≥ 2/3 chains) = PASS. C1 true but C2 false = FAIL ("response without causal contribution": increased activation, not compensation). C1 false = NOT DETECTED (block not run). A rise on pinning is reported as "passive propagation".
* **Hook robustness.** The MSR / SCR / ACT labels at hook 21 (2 chains) equal those at hook 25 → robust; any difference → hook-dependent (reported, not averaged).
* **Readout artefact flag.** If ≥ 50 % of carriers are readout-like at hook 25 but fewer than half at hook 21, the hook-25 multiplicity is flagged as partly a late-readout effect.
* **Task contrast.** Descriptive: difference in median N1 and in K\* between M and L with bootstrap intervals; labelled "task-dependent" only if the comparator gate holds and the difference in N1 ≥ 0.25 with the interval excluding 0.

### Outcome table and decision for the project

| MSR | SCR | ACT | reading | decision |
|---|---|---|---|---|
| PASS | PASS | PASS | multiplicity is genuine redundancy with active compensation | GO: main project |
| PASS | PASS | FAIL / NOT DETECTED | static causal redundancy | GO: main project (static-redundancy thesis) |
| PASS | FAIL (ASR) | – | cheap sufficiency, unused alternative readouts | do not start a main project; fold into the consolidation paper as the explanation of the Stage 1b M result |
| PASS | MIXED | any | partial / dose-limited use | one preregistered follow-up question only if R1 and R2 disagree in a way a single variable explains; otherwise fold in |
| FAIL | – | – | the 3-carrier result does not generalise | STOP; record as negative |
| INCONCLUSIVE | – | – | seed-limited | extension within budget (below), then re-score once |

## 10. Compute

| item | estimate |
|---|---|
| Gate 0, data freeze, baseline | 0.15 GPU-h |
| M chains at hook 25: 21 sequential + 3 parallel fits | 1.1 |
| hook-21 chains: 2 × 3 new + checks | 0.35 |
| L chains: 3 × 3 new fits | 0.7 |
| Confirm measurements incl. controls (M ×3 chains, hook 21, L) | 1.4 |
| Direct tests 7b–7d and compensation block | 0.6 |
| **Soft total** | **≈ 4.3 GPU-h** |

Pre-justified extensions (each needs a one-line note in the log, none changes a threshold): (i) chain to K = 16 on one chain if all reach 8 (+0.5 h); (ii) chains 3 → 5 for M at hook 25 if MSR/SCR is INCONCLUSIVE because of 2/3 support (+1.0 h); (iii) a rank-2 L fallback (+0.6 h). Seed coverage is not reduced to fit the estimate; controls per family (10 → 6) and the pairwise-union grid are the first things trimmed, and any trim is reported.

## 11. Risks and honest limits

* One task, one model; the pilot cannot say whether multiplicity is a general property. The logic comparison is 3 chains.
* The fits are KL-optimised: "distinct" carriers may all be optimiser-built decoders of the same upstream information (the decodability and readout-likeness checks address this only partly).
* Natural-strength joint ablation sets everything in the union to μ at all positions, a large off-distribution intervention; controls match energy and displacement but not on-manifoldness exactly.
* The compensation block operates at component granularity at one position and cannot see distributed compensation across many small components; "not detected" is not "absent".
* Pinning removes the whole component change (helpful response plus passive propagation); sign and controls are the only handle on that.
* Dev→confirm drift was large in Stage 1b (L necessity 0.59 / 0.61 → 0.44; S `words` KL 0.003 → 0.133); fresh entities may shift results. The pre-registered handling is to report the dev/confirm gap, not to retune.
* The fresh entity pool has not been checked for tokenisation or model knowledge yet; entities that fail the baseline screen are dropped before any intervention.

## 12. Decisions requested

1. Approve the question framing, the MSR / SCR / ACT / ASR distinctions, and the section 9 criteria, or amend them.
2. Chains: 3 for the pilot with the pre-justified extensions, or 5 from the start (≈ +1 GPU-h)?
3. Earlier hook: hook 21 (the only earlier hook with data support), or also fit a hook-19 feasibility probe first (not recommended: no evidence it is sufficient)?
4. Stopping points: single review at the end, or additional stops after Gate 0 / data freeze and after the carrier chains (K\* known) before the measurements?
5. Soft budget ≈ 4.3 GPU-h acceptable?
