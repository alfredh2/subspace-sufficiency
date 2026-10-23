# Phase 7 Stage 1b plan (PREREGISTERED; proposal for review, nothing implemented or run)

**Status and standing.** Stage 1 (report `docs/phase7_stage1_report.md`, commits up to `9044f62`) is closed with its outcome intact: L PASS, S INCONCLUSIVE, G / N / M FAIL, panel "insufficient". Stage 1b is a **new, separately preregistered diagnostic stage**. It does not change, re-score or re-label any Stage 1 threshold, choice, verdict or artifact, and a Stage 1b success is new evidence, never a retroactive Stage 1 pass. Original Stage 2 is not started. Commits carry no co-author trailers; nothing is pushed.

## 0. What the Stage 1 artifacts actually show (read before designing; all numbers from `results/phase7/stage1/`)

| observation | source | implication for 1b |
|---|---|---|
| Sweep log has **3 points per run** (300 steps, one point per 100 steps), one seed. Stage 1 cannot say whether a run converged. | `sweep/*.json` `train.kl_curve` | Convergence has to be measured (dense curves, slopes), not assumed. |
| **G** train KL is still falling at the end of every run (e.g. h25 r4: 1.93 → 1.34 → 1.16 at 300 steps; 1.93 → … → 0.61 at 1600 steps in the long-check). Train and val KL agree (0.61 vs 0.82). | `sweep`, `longcheck_val.json` | Looks **optimization-limited / slow** rather than capacity- or generalization-limited. The 1600-step long-check was still decreasing, so Stage 1's "not explained by the short sweep" was too strong for G. |
| **N** train KL falls to 0.73 at 1600 steps (still falling; h23 r2) while **val KL is 6.08**. Train and val differ in the *name pool* (10 / 18 held-out names). | `longcheck_val.json`, `build_N` | The N failure is a **train/val generalization gap tied to name identity**, not a failure to fit. The full-vocabulary KL requires the subspace to reproduce probability on the specific name tokens, which cannot transfer to unseen names from a low-rank fit. This may be a feature of the objective plus split design rather than evidence that the pointer has no low-rank form. Not yet tested. |
| **S** passes at (h25, r4); (h21, r4) misses the 0.05 KL gate by 0.001 (val 0.051) with train 0.012. Rank 1 sits on the ln 2 plateau at every hook. | `sweep` | S is near the boundary; seeds and a rank-8 / hook-27 check can resolve whether the S cell is stable. |
| **M**: plateau at ln 4 = 1.386 for h19–h23 (ranks 1–2), h21 r4 val KL 0.059; h25 r2 KL ≈ 0.000 with **no ablation effect** (51.6 → 51.8, 1600 steps: 49.7). M val has only 10 questions. | `sweep`, `longcheck_val.json` | Sufficiency without necessity. Cause undetermined (redundancy, off-path copy, or an energy/amplification effect). Dev data are tiny. |
| L necessity varies by seed (0.75 / 0.77 / 0.37 in `num_orig`; 0.39 / 0.67 / 0.35 in `num_rev`); S 0.30–0.59. Test contents: L has 8, S has 8 polarity words per class. | `final/*.json` | Seed variance is as large as the gate margin, and clustered intervals rest on very few clusters. 1b needs ≥ 5 seeds and fresh, larger content pools. |
| Cross-format consensus |cos| (rank-2 / rank-4 subspaces compared through arbitrary basis vectors): 0.38 (L); 0.38 / 0.10 / 0.56 (S). Cross-format transfer was **not tested**; principal angles were not computed. | `stage1_report_tables.md` | Basis-vector cosines are not meaningful for r > 1. The format question is open. |
| Per-fit cost at 1600 steps: G 42 s, N 68 s, M 158 s (+ evaluation). | `longcheck_val.json` | Many seeds are affordable; the cost is mostly evaluation and cache building. |

## 1. Questions and the competing explanations each one separates

**Q1 (S, G, N): optimization failure vs representational limit vs split-specific failure.** Predictions below are written as measurable patterns on **development data only** (Stage 1 train/val contents).

**Q2 (M): why does a retained rank-2 subspace give KL ≈ 0 while ablating it does nothing?** Candidate explanations, not mutually exclusive: (R) *redundancy*: other directions carry equivalent information; (O) *off-path / threshold*: the subspace carries a complete copy that is sufficient only when everything else is removed; (E) *energy*: the subspace has small activation energy and downstream layers amplify it only in isolation. Ablation alone cannot establish R, so R is tested by constructing a **complementary subspace** and measuring whether it independently suffices (an operational statement about sufficiency outside U, not a proof of redundancy in the unablated model).

**Q3 (L, S): are the causal effects of a subspace fitted in one answer format preserved in other formats?** Includes principal-angle geometry, transfer of sufficiency, transfer of ablation and transplantation, and a jointly fitted subspace.

## 2. Data discipline (frozen now)

* **Dev data** = Stage 1 train and val contents for every task. All hook / rank / diagnostic decisions use only these. (They are already used; this is acceptable for diagnosis but means "dev" is reused for selection.)
* **Confirm data** = **new** held-out contents generated from the same generators with new seeds and **larger pools** (target: L ≥ 32 test contents; S ≥ 24 polarity words per class and ≥ 12 nouns; G and N ≥ 30 fresh names/nouns; M ≥ 60 fresh questions with a new fact list; each disjoint from every Stage 1 split, verified by a leakage test). Pool sizes and a data hash are written to the log before any model call on them. The Stage 1 test sets stay unused by 1b (so the Stage 1 report remains reproducible and Stage 1 test numbers are not re-optimized).
* Confirm data are touched **once**, after the 1b selection file is committed. A fresh-contents evaluation of the *frozen Stage 1 choices* (L h23 r2, S h25 r4, 3 stored seeds) is run alongside as a replication; it is reported separately and does not alter the Stage 1 verdicts.
* Seeds are the fit seed (initialisation + batch order). **Unit of inference**: task; within a task, hierarchical bootstrap over seeds, then over content clusters (resample seeds with replacement, then clusters within seed; 2000 resamples). With ≤ 8 seeds and few clusters the intervals are approximate; every gate additionally has a **seed-support rule** so a pass cannot rest on one lucky seed.

## 3. Arm A: convergence, seeds, rank, hook (S, G, N; M and A as tasks screened for sufficiency)

**Grid (justified, not open-ended).** Hooks {21, 23, 25, 27} × ranks {4, 8}: 8 cells per task.
* Hook 27 was never tested; Stage 1 shows KL improving with later hooks, and 27 is the last hook before the readout region (hooks above ~29 approach logit-lens territory, where "representation" and "readout" blur; excluded and flagged).
* Hook 21 is kept only at rank 4 and 8 because (21, 4) nearly passed for S, L and M; hooks 19 and rank ≤ 2 remain on the ln-2/ln-4 plateau and are not re-run. Rank 8 is the upper limit (≤ 0.2 % of d = 4096).
* G: 8 cells; N: 8 cells; S: 8 cells; M: only hooks {21, 25, 27} × ranks {4, 8} (6 cells; h25 r2 already known); A (reserve candidate, see §7) uses the same 8 cells after its baseline screen.
* Screening: **2 seeds per cell**, 1600 steps (the Phase 0 recipe, unchanged: on-manifold init, QR, Adam, lr 1e-2 cosine), KL logged every 25 steps on train and every 200 steps on val.
* The chosen cell per task then gets **5 seeds** (0–4; seeds 0–2 reuse nothing from Stage 1: refit).

**Diagnostic labels (per cell, dev data only; computed by code, not by hand).**
1. *Converged* = relative change of the train-KL moving average (window 100 steps) over the last 25 % of steps < 5 % **and** the 4800-step extension (below) lowers dev KL by < 10 %.
2. *OPT-LIMITED*: not converged at 1600 steps, **or** across-seed range of final train KL > 0.2 × median, **or** the extension (3× steps) or alternative initialization (below) reduces dev KL by ≥ 30 % relative or moves a cell across the 0.05 gate.
3. *CAPACITY-LIMITED*: converged, seeds agree (range ≤ 0.2 × median), train KL > 0.05 at the cell, and at rank 8 the train KL still > 0.05 at every hook in the grid.
4. *GENERALIZATION-LIMITED*: train KL ≤ 0.25 × dev-val KL and dev-val KL − train KL ≥ 0.1 (content shift between pools).
5. *SUFFICIENT*: dev-val KL ≤ 0.05 in both screening seeds.
Labels may co-occur (N is expected to be both OPT- and GENERALIZATION-limited). A task with no SUFFICIENT cell and label CAPACITY-LIMITED at rank 8 is recorded as "no ≤ 8-dimensional sufficient subspace found at these hooks" (a negative result, not a proof of non-existence).

**Extensions inside Arm A (all pre-declared, all only for the best-KL cell per task):**
* *Longer training*: 4800 steps, 2 seeds (same schedule shape).
* *Alternative initialization*: initialise V from the top-r directions of the class-mean-difference / LDA subspace computed on **train** items plus the on-manifold recipe's covariance eigenvectors (one variant only, declared here), to test plateau escape. The frozen recipe remains the primary; the variant is diagnostic.
* *Non-optimised reference subspace*: the rank-r LDA/class-mean subspace retained directly with no fitting, giving a reference for how much of the KL is explained by simple supervised geometry.
* *Full-rank sanity*: retaining everything must give KL ≈ 0 (code check).

**N-specific diagnostics (labelled diagnostic-only; they do not count towards the panel gate at tier 1).**
* *N-fixedpool*: same templates, but the train/val/confirm splits share the name pool and differ in context combinations (place, object, template). Tests whether the failure is the unseen-name shift.
* *Candidate-restricted objective* (for G and N-cont only): KL over the renormalised distribution on the two candidate tokens, plus the standard full-vocab KL as a reported (not trained) metric. Tests whether the failure is the full-vocabulary objective on a broad raw-continuation distribution.
Passes under either variant are reported as **tier-2** ("choice-level sufficiency") and are never merged with tier-1 (the Stage 1 full-vocabulary definition).

## 4. Arm B: M sufficiency–necessity discrepancy

Subject: M with the best sufficient cell (expected h25, r2 and r4; also h21 r8 and h27 if sufficient), 5 seeds; contrast tasks with necessity: L (h23 r2) and S (h25 r4), same code path. Evaluated on dev (val + a rebuilt larger M-dev from train questions held out of fitting) first, then confirm contents for the frozen cell.

1. **Intervention family on U**: retain; ablate; *position scope* (all positions vs final position only); *dose*: h − α·UUᵀ(h−μ), α ∈ {0.5, 1, 2, 4}; *transplant* (source→target U-component, matched pairs, as in Stage 1) giving a causal-sufficiency measure for M that Stage 1 never computed.
2. **Controls matched on perturbation size and on-manifold energy**: 20 Haar random subspaces of equal rank; 20 random subspaces drawn inside the top-256 WikiText-covariance principal span (on-manifold control); top-PCA subspaces of equal rank; each reported with its energy fraction.
3. **Activation-magnitude diagnostics**: E‖UUᵀ(h−μ)‖² / E‖h−μ‖² at the answer position and across all positions; per-coordinate energy of U on the known massive coordinates (e.g. 2276, 233) and on the top-10 variance coordinates; norm of the ablation perturbation ‖ΔH‖ per item (distribution, not mean only).
4. **Complementary-subspace test**: fit U₂ of rank {2, 4, 8} constrained to span(U₂) ⊥ span(U) (V₂ projected off span(U) before QR every step), objective = the standard KL with only span(U₂) retained. Report KL(U₂). Also fit the same on the ablated model (identical by construction) and, as a **negative control**, repeat for L and S where ablation matters.
5. **Nested nulling (INLP-style)**: cumulative ablation of span(U), span(U₂), … up to total rank 16; gap reduction vs cumulative rank, against random nested controls of the same total rank and energy.
6. **Decision table (frozen):**
   | pattern | reading (hedged) |
   |---|---|
   | KL(U₂, r ≤ 8) ≤ 0.05 and nested ablation reaches ≥ 50 % gap reduction at total rank ≤ 16, controls ≈ 0 | consistent with **redundant copies** (R); necessity exists only for the union |
   | KL(U₂, r ≤ 8) > 0.05 for all r, ablation null at all cumulative ranks, transplant of U transfers ≥ 0.25 | sufficiency without necessity not explained by a second low-rank copy: **off-path / amplification** (O or E) remain open; check energy diagnostics |
   | U energy fraction below the 5th percentile of on-manifold random subspaces and dose-response monotone only for α ≥ 2 | **energy / threshold** (E) |
   | transplant also null | U is sufficient only in the retention regime (everything else removed), not as a causal handle |
   Any outcome is reported as the pattern observed. The words "redundancy proven" are not used.

## 5. Arm C: cross-format stability (L and S)

Reuses all stored Stage 1 subspaces (L: 2 formats × 3 seeds; S: 3 formats × 3 seeds) and adds seeds 3–4 for each, so the matrix has 5 seeds per (task, training format). New fits: **joint-format subspaces** (union of formats, same rank, 5 seeds). Evaluated on **confirm** contents.

1. **Transfer matrix**: for each ordered pair (train format F, eval format E): retention KL and retained accuracy; ablation gap reduction; transplant transfer with 20 random subspaces. Off-diagonal values are compared with the diagonal (same-format) value from the same seed. All evaluated with the Stage 1 definitions.
2. **Principal angles**: for each pair of rank-r subspaces, cosines of the r principal angles (SVD of U_aᵀU_b) and the overlap O = mean cos². Reference distributions: (i) *within-format cross-seed* (reproducibility ceiling, 10 seed pairs), (ii) *cross-format* pairs, (iii) 1000 Haar-random pairs (expectation r/d). Reported per task and with a subspace-level bootstrap over seed pairs.
3. **Cross-format transplantation** where token lengths match (num_orig vs num_rev by construction; checked in code): source from format F with answer token a, target from format E; distinguishes *semantic* coding (flip follows validity/polarity) from *output-token* coding (flip follows the answer digit). The `words` formats are excluded from this test if lengths cannot be matched.
4. **Joint fit** vs the span of per-format fits: does one rank-r subspace retain all formats (KL ≤ 0.05 in each) and keep the ablation effect?
5. **Frozen labels per task:**
   * FORMAT-GENERAL: every off-diagonal cell has KL ≤ 0.05, ablation reduction ≥ 0.5 × its diagonal value, transplant ≥ 0.25 with randoms ≤ 0.05, in the seed-median and in ≥ 4 of 5 seeds.
   * FORMAT-SPECIFIC: at least one off-diagonal cell fails sufficiency (KL > 0.05 or transplant < 0.25) in the seed-median with the hierarchical interval wholly on the failing side.
   * INCONCLUSIVE otherwise.
   Geometry is labelled separately: GEOMETRICALLY SHARED if O_cross ≥ 0.5 × O_within (and above the Haar level by a factor ≥ 10), otherwise GEOMETRICALLY DISTINCT. "Functionally shared but geometrically distinct" is an allowed, reportable combination.

## 6. Frozen Stage 1b task criteria ("1b-PASS"), counted per behaviourally distinct task

Computed at the cell chosen on dev data by the rule: *lowest rank ≤ 8 whose dev-val KL ≤ 0.05 in both screening seeds; ties → highest median dev ablation reduction; then earliest hook*. If none is sufficient the task is 1b-FAIL (sufficiency) with its diagnostic label. Evaluated on confirm contents, 5 seeds, primary format, hierarchical intervals.

| gate | criterion (primary format unless stated) |
|---|---|
| **1b-S sufficiency** | seed-median retained KL ≤ 0.05; ≥ 4/5 seeds individually ≤ 0.05; seed-median retained accuracy ≥ 0.9 × intact |
| **1b-N causal necessity** | seed-median ablation gap reduction ≥ 0.5 and ≥ min(10 logits, 0.5 × intact gap) (the §12.2 form, with the literal version also reported); ≥ 3/5 seeds individually ≥ 0.5; random equal-rank controls < 0.1 |
| **1b-G generalization** | (i) transplant transfer ≥ 0.25 (seed-median) with 20 random subspaces ≤ 0.05; (ii) 1b-S also holds in a second answer format of the same task (own fit or joint fit); (iii) cross-seed subspace overlap O ≥ 0.5 (mean cos² over seed pairs; the Haar level is ≈ r/d, so this is a strong requirement); (iv) held-out contents are fresh by construction |
Verdicts per gate: PASS, FAIL, INCONCLUSIVE (hierarchical 95 % bound crosses the threshold, or seed-support disagrees with the median). A task is **1b-PASS** only if all three gates PASS; 1b-FAIL if any gate FAILS; otherwise 1b-INCONCLUSIVE. Tier-2 variants (§3) are labelled "tier-2" and counted in a separate column.

**Behaviourally distinct tasks**: L, S, G, N, M, A are different input→output functions; formats of one task are never counted twice. S and L share the verdict-like binary readout and count separately (as in Stage 1) but are flagged as "same family" in the report.

## 7. Candidate panel and the stopping gate

* Candidates: L, S, G, N, M, **plus A (single-digit addition), the previously reserved task**. Rationale: with five candidates a ≥ 4 gate needs four of five to pass, which Stage 1 evidence makes close to unreachable; A is the only extra candidate already defined in `tasks.py` (`build_A`). A would first run the frozen Stage 1 feasibility screen (baseline accuracy ≥ 0.9, §4 of the plan) and is dropped if it fails. You may veto A; the gate then stays at "≥ 4 of 5".
* **Gate for proceeding (to a *proposal* for a multi-task fixed-basis comparison, which still needs your approval): at least four behaviourally distinct tasks are 1b-PASS at tier 1.** Tier-2-only passes, "sufficiency only" tasks, and INCONCLUSIVE tasks do not count.
* **Early-stop rule (saves compute, declared now):** after Arm A, if fewer than four of the candidate tasks have a SUFFICIENT cell on dev data, the four-task gate cannot be met; the confirm-set necessity and generalization evaluations for the remaining tasks are then run only for the tasks with a SUFFICIENT cell (so that their results are still reported), no further search is done, and the report recommends stopping the multi-task basis experiment.
* **If the panel remains insufficient** (the expected outcome by my estimate; I put the chance of reaching 4 tier-1 passes at roughly 10–15 %: L is likely, S is plausible, M has a necessity problem, G and N need a mechanism change, A is unscreened): the recommendation is to **stop the multi-task native-basis experiment** and consolidate the methodological findings (MVS necessity vs sufficiency, format dependence, objective/split sensitivity, seed variance, the pilot's native-basis result and the norm-matched Stage M reinterpretation of Phase 6).
* Arms B and C are scientific questions in their own right and run to completion regardless of the gate (they are cheap).

## 8. Preregistered handling of interpretation risks

* Fits are selected on dev data, so any hook × rank search has a forking-paths cost; the grid is fixed (8 cells), the choice rule is fixed, and the confirm set is touched once.
* Rank 8 vs rank 2: a larger subspace trivially makes retention easier; therefore the necessity gate (1b-N) and the random-subspace controls scale with rank, and a task passing only at rank 8 is flagged "higher-rank pass".
* A pass at hook 27 is flagged "near-readout" (the subspace may mostly reflect answer routing).
* The norm-matched, dose-response and energy diagnostics of Arm B address the same magnitude confound that Stage M found for Phase 6. Phase 6's report is not edited; the Stage M note stays in the Phase 7 pilot report.
* No claim of native-neuron structure is made in 1b; it only identifies target subspaces.

## 9. Compute (soft budget ≈ 3–4 GPU-h, with justified extensions)

| item | fits / evals | est. GPU-h |
|---|---|---|
| Reference caches for new hooks (21, 27), fresh data sets, A baseline | – | 0.3 |
| Arm A grid: S, G, N, A (8 cells × 2 seeds × ~1.2 min incl. dev eval) + M (6 cells) | ~ 70 fits | 1.4 |
| Arm A extensions (4800 steps × 2 seeds, init variant, LDA reference) on the best cell per task | ~ 40 fits | 0.5 |
| Arm B (complement fits, nested nulling, controls, energy diagnostics, transplant) | ~ 40 fits + evals | 0.6 |
| Arm C (transfer matrix evaluation, 4 extra seeds, joint fits, cross-format transplant) | ~ 30 fits + 120 evals | 0.7 |
| Confirm runs: 5 seeds for chosen cells of tasks that reach it (≤ 6 tasks) + frozen Stage 1 L/S replication | ~ 35 fits | 0.7 |
| **Nominal total** | | **≈ 4.2 h** |
Justified extension (needs a written reason in the report): up to +1.5 h for additional seeds on cells whose intervals cross a threshold. Hard cap 5.7 h. Anything beyond requires your approval.

## 10. Staging and stopping points

1. **1b-0 (implementation, no model runs on confirm data):** fresh generators with larger pools + leakage tests; dense logging; hierarchical bootstrap; principal-angle utilities; complement-subspace fitter; unit tests. Commit; hash the confirm sets.
2. **1b-A** (screening, dev only) → commit the selection file (cells, labels, early-stop decision) → short checkpoint report (no review stop unless the early-stop rule fires; if it fires I stop and report).
3. **1b-B, 1b-C** (dev first, then confirm for frozen cells).
4. **1b-confirm** and **Stage 1b report**: per-task 1b verdicts, the Q1–Q3 answers, the gate decision. **Stop for your review** before anything else.

## 11. What is not claimed, and open choices for you

* 1b cannot show that no low-dimensional representation exists for G, N or M; it can show that none was found within ≤ 8 dimensions, four hooks, five seeds and the stated objective.
* Choices where I would like your veto or edit: (a) inclusion of A as a sixth candidate; (b) the rank-8 cap and hook set {21, 23, 25, 27}; (c) N-fixedpool and candidate-restricted objective as diagnostic-only, tier-2; (d) the overlap threshold O ≥ 0.5; (e) the budget cap.
