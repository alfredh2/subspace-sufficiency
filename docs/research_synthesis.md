# Research synthesis: what the Phase 0–7 evidence supports

Status: consolidation document, written after the preregistered four-task gate of Stage 1b was **not met (0 of 6 tasks)** and the multi-task fixed-basis programme was stopped by decision. No experiment was run for this document. Every number below is copied from a committed report or checked against a stored result file (`stage1b_verdicts.json`, `stageM_verdict.json`, `phase0_robustness/analysis_summary.json`). **No earlier report, frozen criterion or verdict is edited or re-interpreted here**; where I disagree with a past interpretation I say so as a new statement in this document. Proposals in sections 5–6 are proposals for review, not started work.

Model and scope throughout: Qwen3-8B, bf16, one model, residual-stream hooks, templated (re-created) task data. Nothing here speaks about other models or natural data.

## 0. Preregistered verdicts, as recorded (unchanged)

| stage | recorded verdict | one-line reason |
|---|---|---|
| Phase 0 (reproduction) | no gate verdict; log in `experiment_log.md` | retention within 0.01 of intact in 3 seeds; ablation 0.500 / 0.852 / 0.500 |
| Phase 0.5 | diagnostics, no gate | retention format-bound; ablation = shift to "valid" with ranking kept |
| Phase 0.6 | no credible logic-selective target (prereg rule) | 0 of 6 candidates pass criteria 1–4, in both pilots |
| Phase 1 | MIXED (G-sparse, C-mixed) | C4 fails narrowly (Stroop +1.13 logits vs limit 1.0) |
| Phase 2 | MIXED | recompute gate FAILED; C3 side effects fail; random-head patching limit fails |
| Phase 2.5 / 2.6 | exploratory, criteria met | value-only patch 0.89; G1∪G2 positions carry 0.90 / 0.915 |
| Phase 3 | MIXED | mediation M met; generalization G not met; selectivity V failed |
| Phase 4 | MIXED | onset at hook 20 lies inside the S8 layers; A2 not met |
| Phase 5 | MIXED | none of H1–H3 supported as stated |
| Phase 6 | MIXED | O1, O2, O4 not supported; O3 literal criterion met but not read as build-up |
| Phase 7 Stage 0 | rule outcome: H-generic "supported" by family (f); "partial" by (b), (e) | family (f) contains a verdict-carrying PC (flaw disclosed at the time) |
| Phase 7 Stage M | M-A TRUE only | d_c-only equal-norm / orthogonal = 1.04 [0.84–1.04 across 4 set/format combos] |
| Phase 7 Stage 1 | L PASS, S INCONCLUSIVE, G/N/M FAIL; panel "insufficient" | frozen |
| Phase 7 Stage 1b | 0 of 6 tasks 1b-PASS; four-task gate NOT MET | stop and consolidate |

Most stages above ended MIXED or negative (Phases 0.6 to 6 and Stage 1b; Stage 0 returned the unfavourable H-generic flag). The synthesis does not hide this: the project's original question ("is the logic MVS implemented by specialised native neurons?") was answered *no or not established*, and the surviving contribution is different from the original hypothesis.

---

## 1. The strongest empirical findings (ranked by robustness, not by novelty)

Evidence labels: **[C]** interventional with random/matched controls; **[G]** geometric; **[D]** descriptive attribution; **[S]** speculative. Details in section 2.

### F1. A fitted subspace can be sufficient without being necessary or transplantable (task M; also L, S in weaker form)
* Task M (four-way QA), frozen cell hook 25 rank 4, 5 seeds, confirm contents, 1b-S **PASS**: retention KL 0.0000 [0.0000, 0.0000], 5/5 seeds, intact accuracy 1.000.
* Same subspace: ablation removes 0.05 [0.01, 0.06] of the margin gap (2.8 logits, 0/5 seeds reach 0.5), transplantation 0.093 [0.03, 0.11] (0/5 seeds). Random equal-rank: transplant 0.001, ablation about 0. 1b-N **FAIL**, 1b-G **FAIL**.
* Arm B (M, 3 seeds): orthogonal complements U₂ of rank 2, 4, 8, each fitted and retained alone, also give KL 0.000–0.0002 and accuracy 1.00, so *independent sufficiency of several orthogonal subspaces* is established. Union ablation [U, U₂] removes only 0.02–0.15.
* Dose-response of the ablation, h − α·UUᵀ(h−μ): α = 0.5, 1, 2, 4 give about 0.0, 0.02–0.06, 0.12–0.24, **0.83–0.96** (r4; r2 gives 0.54–0.57 at α = 4). Ablation at natural strength is ineffective but strongly dose-dependent.
* Weaker versions on the other tasks: L (h23, r4) necessity 0.44 [0.36, 0.58], 2/5 seeds; S/`numeric` formats 0.45 and 0.38, 0/5 seeds. Seed-to-seed spread at a fixed cell is 0.37–0.58 for L. L and S union ablation (rank 12) removes 0.65 and 0.86 vs 0.014 and 0.002 for random, **single seed**.
* Limits: one task carries the extreme case (M); Arm B contrasts for L and S are one seed; redundancy as an *active* pathway is not shown (preregistered A11); see the optimizer-carrier caveat in section 3.

### F2. The logic subspace is an accept/reject verdict channel bound to the answer mapping, not a logic variable (Phases 0, 0.5, 0.6)
* Three Phase-0 seeds retain at KL 0.0003–0.0088 and accuracy within 0.01 of intact. Under a reversed answer mapping retained accuracy falls to **0.04** (AUC 0.01, systematically inverted); under A/B or Yes/No the retained model still emits "1"/"2" (100 % off-format).
* Ablation of seeds 0 and 2 gives exactly 0.500: every example answered "1" (valid), P("1") ≈ 1. Yet the margin still ranks classes (AUC 0.95–0.99, best-threshold accuracy 0.90–0.96). Seed 1 reaches only 0.852 / 0.883 / 0.770. So ablation destroys *calibration of the verdict*, not discrimination.
* Same signature elsewhere: language localizer 1.0 → 0.562 in all three seeds, PrOntoQA 0.758 → 0.46 / 0.65 / 0.45; verdict-free tasks with the same label tokens (MMLU-binary, Stroop, word recall) are unaffected and receive transfer ≈ 0.00–0.01.
* Across seeds one direction is shared (principal-angle cosines 0.77–0.83), the second is not (0.06–0.19). Ablating only the cross-seed consensus rank-1 direction reproduces the collapse (logic 0.500, localizer 0.562, PrOntoQA 0.438). Matched-pair transplantation of the MVS: 0.49–0.64 vs ≤ 0.002 for 8 random rank-2 subspaces.
* Validity is linearly decodable (0.94–0.98) after projecting out d_c, the MVS span, or 12 INLP directions (still ≈ 0.95). A pre-registered causal search for a logic-selective subspace orthogonal to d_c found none (0 of 6 candidates in pilot 1; 1 of 6 on criterion 1 in pilot 2, 0 on 2 and 4).
* Limits: validity is a function of argument form in the re-created benchmark; the "constant 1" statement depends on first-token scoring; 3 seeds; the Phase 0.6 search is pilot-scale (300 steps, ranks ≤ 4, 2 seeds, one layer) and is *not* evidence that no logic-specific subspace exists.

### F3. Retention, ablation and transplantation dissociate across answer formats (Phase 0.5 and Stage 1b Arm C, 5 seeds)
* L: a subspace fitted on `num_orig` and evaluated on `num_rev` retains with KL 11.4 (9.6 the other way): retention is **format-bound**. Transplantation across formats still transfers 0.39–0.63 and follows the semantic class (token flip ≈ 0.01, random ≤ 0.002).
* S: cross-format retention KL 0.09–39.6; cross-format transplant has semantic transfer −0.05 to −0.12 but token-flip +0.16 to +0.27, i.e. the causal effect follows the **answer token**, not polarity.
* The two tasks therefore differ in kind, which a single sufficiency metric cannot show. Stated plainly: the same procedure yields a semantic-class handle in L and a token handle in S.
* Limits: L's "semantic" class is the verdict class, which F2 shows is not logic-specific; S has `words` vs numeric geometry almost orthogonal (overlap 0.03–0.05); Arm C cross-format cells are 5-seed but on 14 / 14 confirm contents (L / S); the L joint-format fit stays at the ln 2 plateau (KL 0.64–0.67), so no single subspace serves both mappings.

### F4. A one-line negative on geometry: only one direction of a rank-r subspace reproduces across seeds, and cosine/overlap does not predict function
* Within-format cross-seed overlap: L 0.22, S 0.27 (Haar null 0.001); top cosine ≈ 0.8, mean sorted cosines L 0.80/0.44/0.15/0.03, S 0.79/0.57/0.24/0.05. The O ≥ 0.5 benchmark is met by no pair (best 0.28; reported only, not a PASS condition by amendment A4).
* Phase 0: seed 1 has the lowest KL (0.0003) and the weakest ablation (0.852); retention KL does not predict ablation damage.
* Yet transplantation (L 5/5 seeds, S 5/5 seeds across formats) is reproducible while geometry is not. Functionally similar, geometrically different, as the plan said not to count as failure.
* Limits: principal-angle statistics on r = 4 over 5 seeds; overlap is an average of cos² and sensitive to which directions carry the function.

### F5. Layer-19 mechanism (Phases 2–6, Stage M): value-copy by three heads, with magnitude as the confound that matters
* H* = {L19H10, L19H31, L19H17} reach 0.35 / 0.39 transfer vs 0.36 / 0.39 for all 32 heads (97 % / 100 %); random 3-head sets mean 0.006 (max 0.05), magnitude-equalised null mean 0.06 (max 0.20). Attention-weight-only patch 0.00; value patch 0.34 / 0.38 vs native 0.35 / 0.39. Source of the value content: conclusion body ≈ 0.48–0.51 of the effect, conclusion-final token 0.14, fact line 0.03–0.06.
* Stage M (the one place a mis-interpretation was caught): the Phase 6 statement that the d_c-aligned component is "nearly inert" (T = 0.024) was a **magnitude effect**: rescaled to equal norm d_c transfers 0.278 vs 0.267 for the orthogonal remainder, random equal-norm directions 0.007. The dose-response is strongly convex (×2, ×4, ×8: 0.054, 0.143, 0.369).
* Convexity in dose is also seen in Phase 4 (0.07 → 0.76 for α = 0.25 → 1.25), Phase 5 (g(0.5)/g(1) = 0.32) and F1 (M ablation). This is the one theme that recurs in every phase: **effect sizes at natural magnitude are threshold-like, so "component X is inert/unnecessary" is not a dose-independent statement.**
* Limits: logic only; set A has 8 contents / 217 pairs per format (content-clustered intervals rest on 8 clusters); sets A and B are repeatedly examined; descriptive d_c accounting did not match causal pinning effects in Phase 6; the three heads are shown sufficient for the transfer, not necessary.

Findings I considered and **did not** promote: the Phase 1 and Phase 2 native-component results (concentration and ablation by ~500 units or 8 heads). They are real effects relative to their preregistered controls, but each ended MIXED on specificity, one direction at one hook on logic only, and Stage 0 showed the concentration statistic has a generic floor. They are used in narrative 3 below.

---

## 2. What is demonstrated, what is correlational, what is speculative

| tier | statement | evidence | status |
|---|---|---|---|
| **Demonstrated causal** (interventional, with random or matched controls, within Qwen3-8B on these templated tasks) | Fitted subspaces for L, S, M reproduce the output distribution when everything else is replaced by μ (KL ≤ 0.007 L numeric; ≤ 0.003 S numeric; 0.000 M) | Stage 1b, 5 seeds, confirm contents | S `words` FAIL on confirm (KL 0.133 vs 0.003 on dev) |
| | Swapping the L and S subspaces between matched items moves the answer toward the source (L 0.63 / 0.67; S 0.62–0.71) while 20 random equal-rank subspaces stay ≤ 0.003 | Stage 1b, 5/5 seeds | M 0.09 and A 0.05 do not |
| | For M, ablating the subspace at natural strength removes ≤ 6 % of the gap, at 4× 83–96 % | Arm B, 3 seeds, random ≈ 0 | |
| | Ablating the d_c direction collapses the logic, localizer and PrOntoQA verdict while leaving verdict-free tasks intact; patching moves the verdict | Phase 0/0.5, 3 seeds | accuracy collapse is calibration, AUC kept |
| | Three layer-19 heads copy value content (not attention routing) into the summary rows; random heads, norm-matched noise and routing-only patches do not | Phases 5–6, sets A–C | patching sufficiency |
| | Equal-norm d_c and its orthogonal remainder are comparably effective in the layer-19 write; random equal-norm is not | Stage M, four set/format combos | |
| **Geometric / correlational** | Top-1 % share of the verdict write in the native MLP basis 0.490 vs Haar 0.126; N_eff 60 vs 11,700 | Phase 1 | generic floor: 0.19–0.30 for non-noise contrasts; shuffled labels 0.253 |
| | Top-8 heads hold 78 % of |Δ| mass | Phase 2 | only 1.31× the head-shuffled-alignment null |
| | Principal-angle overlap across seeds 0.22–0.27 and across formats 0.03–0.25 | Arm C | descriptive of fitted bases |
| | span(U) holds 9 % of final-position variance of h−μ for M (Haar 0.3 %) | Arm B | single hook |
| | Ranking by d_c alignment vs causal single-head ranking: Spearman 0.29 / 0.30; by write norm 0.53 / 0.52 | Stage M | top-3 agree, bulk does not |
| **Descriptive attribution** | Class-difference of the d_c coordinate: +19.2 = attention +8.5 + MLP +10.7 | Phase 1 | accounting identity, not causation |
| | Routing / content accounting ≈ 4 % / 96 % for L22H5 | Phase 2.5 | eager-attention accounting; recompute gate failed |
| | Validity linearly decodable after removing 12 INLP directions | Phase 0.6 | decodable ≠ used |
| **Speculative** | The intact network actively uses redundant pathways to compute the M answer | not tested (A11) | hypothesis |
| | The fitted M subspace is an optimizer-built carrier rather than a naturally used variable | not tested; supported weakly by transplant 0.09 | hypothesis |
| | A "verdict channel" is general across task types | only logic, localizer, PrOntoQA (all binary accept/reject) | hypothesis |
| | Layer 19 (or any layer) computes logical validity | explicitly not shown (Phases 5, 6) | hypothesis |
| | No low-rank subspace exists for G or N | not shown: G and N were never OPT-limited or CAPACITY-limited by the frozen labels; N's failure is mainly unseen-name shift | open |

---

## 3. Evaluation of the three candidate narratives

### Narrative A — "Low-dimensional sufficiency does not imply causal necessity"
* **For.** Directly observed on M (KL 0.000 / ablation 0.05 / orthogonal sufficient complements); partly on L (0.44, 2/5 seeds) and S (0.45, 0.38); Phase 0 seed 1 (best KL, weakest ablation); the dose-response (α = 4) shows the subspace is causally active at larger magnitude; the union ablations for L and S show information outside the fitted span. Quantified with 5 seeds and a preregistered gate on the confirm contents. The statement is testable and was tested against random controls.
* **Against / risks.**
  1. *Not new in principle.* Redundancy and backup mechanisms are well known; the contribution is a quantified, controlled demonstration in subspace discovery, not the idea.
  2. *Sufficiency here is partly an optimization artefact.* The retained network sees a residual in which everything outside the subspace is μ, a distribution shift. A KL-trained U can *build* a carrier the downstream layers decode. M's transplant of 0.09 (a swap of U between items barely changes the answer) is consistent with this, and with narrative B. This is untested (section 5, E3).
  3. *Evidence is thinnest where the claim is strongest:* the extreme dissociation is one task; L/S necessity contrasts (union ablation) are one seed; the dev-to-confirm drop in L necessity (0.59 / 0.61 on dev → 0.44) suggests selection noise.
  4. *Correction to a statement in the Stage 1b report summary.* "Necessity falls as rank rises" is **weaker than stated**. It compares Stage 1 rank 2 (test 0.75, num_orig) with Stage 1b rank 4 (confirm 0.44), which differ in contents, partition, seed set and steps. The Arm A dev table shows rank 8 with *equal or higher* relative reduction than rank 4 in most L and S cells (e.g. L h23: 0.59/0.61 at r4 vs 0.71/0.57 at r8; S h25: 0.55/0.57 vs 0.75/0.74). I do not edit that report; this sentence should not be carried into a paper. It is an untested hypothesis.
* **Verdict.** Best supported on quantitative grounds and reproducible across tasks (M, partly L, S); needs the strengthening in section 5. Strong as the *organising frame*, weaker as a stand-alone headline because of risk 1.

### Narrative B — "Behavioural subspaces often capture answer readouts rather than the underlying computation"
* **For.** The most consistent cross-phase pattern: retention bound to the 1→valid mapping (accuracy 0.04 under reversal); ablation = shift to the affirmative verdict with ranking kept (AUC ≥ 0.95); identical signature on the language localizer and PrOntoQA; validity decodable after removing the MVS span and 12 INLP directions; the search for a validity-specific subspace orthogonal to d_c found nothing transplantable; S follows the answer token; N and G fail when a pointer/continuation must live in the subspace; M's subspace is sufficient but not swappable.
* **Against / risks.**
  1. *"Readout vs computation" is not operationally defined in the data.* What was measured is verdict-correlated, format-bound, non-logic-specific causal control. Calling it "readout" is an interpretation; Phase 0.6 reports it as such.
  2. *The negative search is pilot-scale* and does not exclude a computation subspace at other layers, ranks or objectives.
  3. *Validity is form-determined in the benchmark,* so a verdict-correlated feature and "logic" cannot be separated without a different task family. The positive result is "generalises across the tested forms and contents", not "abstract logic".
  4. *Most evidence is on one task family* (logic) plus localizer and PrOntoQA, all binary accept/reject.
* **Verdict.** Strongest *explanatory* story and the best fit to the whole evidence base, but the broad word "often" is not earned by one model and essentially one task family. Defensible as a case-study claim ("for logic validity in Qwen3-8B, the causally effective subspace is a verdict/answer channel") and as a measured contrast between L (semantic class) and S (answer token).

### Narrative C — "Native-component concentration does not necessarily imply causal specialization"
* **For.** Phase 1: concentration passes G1–G3 (4 of 4 directions) and the units are causally effective vs random/activation-/alignment-matched/rotated controls (E_gap 0.41 / 0.59, controls ≈ 0), but C4 fails. Phase 2: S8 E_gap 0.86 / 1.14, C3 fails (multiplication 1.00 → 0.78). Phase 3: U22 mediates (R = 0.61) but the same pathway carries a word-recall effect (gate V failed). Stage 0: the top-1 % share has a generic floor (shuffled labels 0.253, irrelevant partitions 0.235, verdict-irrelevant PCs ≈ 0.29, top-5 PCs 0.358 = 0.73 of native), a basis rotation fitted to one direction attains sparsity trivially (PCA basis 0.907), whereas N_eff (61 vs 175–4,760) separates native from generic.
* **Against / risks.**
  1. *Specialization was tested only against a few verdict-free tasks* (MMLU-binary, Stroop, word recall) with absolute margin limits (1.0 logit). Random 8-head sets produced verdict-free shifts up to 4.1, larger than the S8 shifts (−2.49), so "selectivity fails" is partly sensitivity to any perturbation of that size.
  2. *The decisive test (one fixed basis serving many tasks) was designed for Stages 1–4 and could not be run:* the panel is 1-and-a-half tasks.
  3. *Stage 0 family (f) was mis-specified* (it contains a verdict-carrying PC); the recorded "H-generic supported" is mechanical. It is logic-only.
  4. *A MIXED chain of verdicts* (Phases 1, 2, 3) means the claim is built from several partially failing criteria, one of which (Phase 3 V) uses a reading added after seeing the ablation numbers (disclosed).
* **Verdict.** Interesting and partly demonstrated, but the least self-contained: it needs either the stopped multi-task design or a cleaner single-task specificity test. Best kept as a secondary section or a separate follow-up, not the thesis.

### Comparison

| criterion | A: sufficiency ≠ necessity | B: answer readout, not computation | C: concentration ≠ specialization |
|---|---|---|---|
| quantified with preregistered gate on held-out data | yes (Stage 1b, 5 seeds) | partly (Phase 0.6 rule; Phase 0.5 is diagnostic) | partly (Phases 1–3 gates; Stage 0 rule) |
| tasks supporting it | M strong; L, S weak | L, localizer, PrOntoQA; S (token) | logic only |
| robust to the stopped multi-task design | yes | yes | no |
| main threat | known idea; optimizer artefact | undefined "computation" | absolute side-effect thresholds, logic only |
| fit to what the project actually found | good | best | fair |

---

## 4. Recommended research question and thesis

**Question.** *When a KL-trained low-rank residual subspace reproduces a model's behaviour, which causal claims does that license, and which require separate evidence?*

**Thesis (evidence-based, replaces the original native-neuron hypothesis).**
> In Qwen3-8B, behavioural sufficiency of a fitted low-rank residual subspace dissociates from causal necessity at natural strength and from item-level transplantability, and what such a subspace captures can be a format-bound verdict or answer channel instead of the task's computation. Reliable claims therefore need a battery (retention, dose-matched ablation, random-controlled transplantation, an answer-format factorial, and seed-level reproducibility) rather than retention alone.

Structure: narrative A is the organising frame (the *dissociation*), narrative B supplies the interpretation for the one deep case (logic, with S and L as a contrasting pair), and narrative C (native-basis) moves to a short secondary section or to future work.

Why not the original question: the multi-task test of native-basis privilege cannot be run (0 of 6 tasks pass; Stage 1 panel insufficient), and the single-direction native results are MIXED. Why not narrative B alone: "often" overreaches; the evidence does not distinguish a verdict channel from computation outside one task family. Why not narrative A alone: the sufficiency/necessity gap is only extreme in one task, and the logic case gives it content.

**Claim ceiling for a paper.** One model, six templated tasks (two usable), mixed or negative results labelled as such. The honest framing is a *methods-and-negative-results* paper with a detailed case study, not a general mechanism paper.

---

## 5. Minimum additional experiments (none broad; none a sweep)

Each would be preregistered (thresholds committed before running) and uses existing checkpoints unless noted. Combined estimate **≤ 2 GPU-h**; no new tasks, models, ranks or hooks.

* **E1. Seed coverage for the Arm B contrasts on L and S** (union and complement ablation, currently 1 seed, complement rank 8 only). Re-fit the complement U₂ on the same 5 frozen U's and report union ablation with random equal-rank controls and seed-level intervals. Needed because F1 is otherwise carried by one task and a single-seed contrast. Cost ≈ 0.7 GPU-h (fits dominate).
* **E2. Dose-response of ablation on the existing U's for L and S** (α = 0.5, 1, 2, 4), inference only, plus the matching Phase-0 rank-2 MVS. Tests whether the M threshold effect is general or specific. Success criterion fixed in advance (e.g. natural-strength vs α = 4 reduction, with random controls ≈ 0). Cost minutes.
* **E3. A non-optimized comparator for the "optimizer-built carrier" worry** (M and L): evaluate a class-mean-difference / LDA subspace of the same rank (no KL fit) for ablation and transplantation, inference plus a closed-form fit. If a naturally defined subspace transplants where the KL-fitted M subspace does not, risk 2 of narrative A is real; if neither transplants, M's dissociation is about the task and not about the optimizer. Cost minutes.
* **E4. Fresh-contents evaluation of the headline numbers.** Evaluate the frozen U's of L, S and M on newly drawn contents never in any earlier lexicon (as was done for set C: seed and manifest frozen before use), reporting the dev→confirm→fresh drift. Inference only (≈ 0.2 GPU-h). Directly addresses the reuse flagged in section 7.
* Not proposed: rank-2 grid cells for L/S, new hooks, a content-shared N, more tasks, a second model. If a reviewer asks for generality, a Qwen3-4B replication of the M dissociation alone (one task, frozen recipe) is the smallest credible extension, but it is a new stage and I do not recommend starting it now.

Narrative C would need a different single-task specificity test (matched verdict-free control tasks per unit/head set, effect sizes referred to the random-set range) or the stopped multi-task design; neither is in scope.

---

## 6. Paper outline and figures (existing data first)

**Working title:** *What does a sufficient subspace license? Dissociating sufficiency, necessity and transplantation in Qwen3-8B.*

1. **Introduction.** Subspace-discovery claims rest on retention; our preregistered programme started with a native-neuron question and ended by showing what retention does and does not license.
2. **Setup.** MVS fit (QR, KL, 1600 steps, on-manifold init), interventions (retain / ablate / dose-α / transplant), controls (random Haar and on-manifold, equal rank), data (re-created benchmark; Stage 1b repartition), gates and statistics (hierarchical bootstrap, 5 seeds).
3. **Case study: the logic subspace is a verdict channel.** Phase 0–0.6: reproduction, constant-"valid" collapse with intact ranking, reversed mapping, localizer and PrOntoQA, shared direction, no orthogonal logic target.
4. **The dissociation.** Stage 1/1b: per-task gates; M extreme; dose-response; union and complement subspaces; L/S necessity and seed spread.
5. **Format and geometry.** Arm C: retention format-bound, transplant semantic (L) vs token (S); one reproducible direction; overlap vs function.
6. **Mechanism at layer 19 (shortened).** Phases 4–6 and Stage M: value-copy heads and the magnitude confound.
7. **Native components (short or appendix).** Phases 1–3 and Stage 0: causal effects vs controls, concentration generic floor; why the multi-task test was stopped.
8. **Limitations and negative results.** Section 7 of this document; failed discovery of G and N with the three-way diagnostic labels.

**Figures (4–6 main; plotting only, none exist in the repo yet):**
1. *Dissociation panel.* Per task (L, S, M, A; G/N marked "no sufficient cell"): retention KL, necessity (relative and absolute), transplant, with 5-seed points, hierarchical intervals and random controls. Source: `stage1b_verdicts.json`, `final/`.
2. *M dose-response and complements.* Gap reduction vs α with random controls; U alone, U₂ alone, union. Source: `armB/M_*.json`, `L_sel.json`, `S_sel.json`.
3. *The verdict channel.* Ablation outcome (accuracy 0.500 vs AUC), reversed-mapping retention 0.04, localizer and PrOntoQA, per-seed spread. Source: `phase0_robustness/`, `phase05/`.
4. *Format factorial.* Transfer matrices for L and S (retention KL; semantic vs token transplant). Source: `armC/`.
5. *Geometry vs function.* Principal-angle spectra within and across formats vs Haar, next to per-seed necessity and transplant. Source: `armC/`, `final/`.
6. *(Secondary)* Concentration vs controls: top-1 % share and N_eff against Haar, shuffled-label, verdict-irrelevant PC and random-direction families; E_gap of the native sets against controls. Source: `stage0_falsification.json`, `phase1/`, `phase2/`.

---

## 7. Conclusions weakened by sample size, variance, format, multiple testing or reuse

### Small samples and few clusters
* Stage 1b confirm contents: L 14, S 14 polarity words per class and 10 nouns, M 20 + 12 entities, N 14 names, A 17 sums (55 in total). Hierarchical intervals over so few clusters are likely too narrow; the Stage 1 report already says this for its 8-content L test.
* Phase 4–6 set A: 8 contents, 217 pairs per format; content-clustered intervals rest on 8 clusters.
* Phase 0.5 controls: 128 items per control; Phase 1 control draws n = 20 (p_min = 1/21); Phase 2 n = 30 (p_min = 1/31; Holm-corrected z-tests rely on near-normal controls).
* Arm B L and S contrasts: 1 seed; `M_h25_r2`: 2 seeds.
* Task A is a weak test (55 sums, 2/5 seeds sufficient).

### Seed variance
* Phase 0 ablation: 0.500, 0.852, 0.500 (3 seeds). Stage 1b L necessity per seed 0.37–0.58; Stage 1 S per-seed 0.27–0.51; L num_orig Stage 1 0.75 / 0.77 / 0.37. With seed support defined as ≥ 3/5, a PASS or FAIL can hinge on one seed.
* Only one direction of rank 2–4 is reproducible across seeds (top cosine ≈ 0.8, rest ≈ 0.0–0.4); any claim about "the subspace" at rank > 1 is a claim about a fit, not about a unique object.
* Screening used 2 seeds and a dev rule; the dev→confirm change (L necessity 0.59 / 0.61 → 0.44; S `words` KL 0.003 → 0.133) shows selection effects and content shift. Per the preregistered rule, final PASS needed 5 seeds, which was done.

### Answer-format dependence
* Retention is format-bound for L and S; L num_rev necessity (0.36) is below num_orig (0.44); S words vs numeric geometry nearly orthogonal; G and N were scored candidate-restricted in raw formats (greedy accuracy 0.47 / 0.81) because the model puts mass on "'s", "of", "the". Conclusions about "the task" are conclusions about one answer format and scoring rule.
* Phase 0 "ablation to chance" depends on first-token scoring and the "1 = valid" mapping.

### Multiple testing
* There are dozens of comparisons across phases (gates, hooks × ranks × tasks × formats, control families, Phase 3 post-hoc class-gap reading). Only Phase 2 applied a family correction (Holm over 8 primary comparisons). Everywhere else claims rest on preregistered effect-size thresholds, not p-values, and on "exceeds all n draws", which has a floor of 1/(n+1). I did not count total tests and do not claim family-wise error control. Paper claims should be limited to effects that are large relative to controls and replicated across seeds.
* Selection of cells (hook, rank) from a 8-cell grid with 2 seeds on dev data is a form of selection; Stage 1's single passing cell per task (L, S) from a 12-cell sweep was fragile (neighbouring cells fail).

### Reused or contaminated evaluation data
* The re-created logic benchmark is reused through Phases 0–7; set A has been examined repeatedly (Phases 2.5–7); set B has documented prior exposure; only set C was frozen before any intervention (4 of 192 words overlap earlier lexicons).
* Stage 1b confirm contents come from finite lexicons and are **not independent** of Stage 1 knowledge (disclosed as A13/A14). L and S cell choice also was informed by Stage 1. L's 1b cell is rank 4, not Stage 1's rank 2, because the grid had no rank 2.
* Thresholds scaled before any ablation result (Stage 1 section 12.2, 10-logit floor); Phase 5 magnitude-null rule revised before the held-out evaluation; Phase 5 cross-content control and held-out-family breakdown were added after the freeze (exploratory); Phase 3 V reading added after seeing ablation numbers; Phase 1 hook choice came from a 300-step KL sweep in which ablation accuracy was visible.
* Phase 0 layer convention (index 23) was chosen from a sweep in which only one candidate trained.

### Other cautions
* Rank, hook, objective, 1600 steps and the neutral-corpus μ are fixed; G and N conclusions are limited to ≤ 8 dimensions, four hooks (21, 23, 25, 27) and the full-vocabulary KL objective.
* The Stage 1b budget overran (≈ 7.6 GPU-h vs 4–6 h soft) without reducing seed coverage.
* Everything is one model and templated data; "often" and any cross-model generality are unsupported.

---

## 8. Decision requested

1. Is the recommended thesis (section 4: the dissociation as organising frame, the verdict-channel interpretation for the logic case, native-basis material secondary) the direction you want?
2. Do you approve E1–E4 (≈ 2 GPU-h, preregistered, no new tasks or hooks) as the only further experiments, or do you prefer to write up the existing evidence without them (accepting the single-seed Arm B contrasts and the optimizer-artefact caveat)?
3. Should the Stage 1b summary sentence "necessity falls as rank rises" be annotated in a short errata note (a new document, not an edit of the report)?

Nothing beyond this document has been started: no experiments, no figures, no manuscript text.
