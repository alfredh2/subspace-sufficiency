# Experiment log (append-only; include failures)

## 2026-10-21 — baseline (intact Qwen3-8B, our re-created benchmark)
`results/baseline`: Items 0.945, Families 0.977, Nonwords 0.953 (first-char scoring; invalid 0%; restricted-argmax identical).
Paper (2-model mean): 0.99/0.97/0.97. Weakest cells: MT (Items 0.78, Families 0.88), reversed_if / reversed_whenever / guarantee / semi_formal
families (~0.79–0.88). Differences likely due to our unstated-wording approximations; not tuned. Controls: word recall/localizer/MSIT/multiplication 1.0,
PrOntoQA 0.758 (paper 0.736), MMLU-binary 0.8125 (paper 0.805), Stroop 0.898.

## 2026-10-21 — layer-convention sweep (pre-planned; selection criterion = validation KL only)
300 steps each, seed 0, otherwise paper config. `residual_index = i` = HF hidden_states[i]. (val_kl / val_retain_acc / val_ablate_acc at step 300)
| idx | meaning | val_kl | retain | ablate |
|---|---|---|---|---|
| 21 | output of 0-based layer 20 | 0.679 | 0.53 | 0.99 |
| 22 | output of 0-based layer 21 | 0.672 | 0.60 | 0.96 |
| **23** | **output of 0-based layer 22** | **0.017** | 0.98 | 0.52 |
| 24 | output of 0-based layer 23 | 0.654 | 0.78 | 0.98 |
Others stay near ln2 (0.693) within 300 steps (they may escape with more steps; not tested). Chosen: **residual_index = 23**, i.e. the paper's
"layer 22" read as 0-based, hook after block 22. Caveat: ablation accuracy was visible in the logs, but selection used KL; the choice is the only candidate that trained at all.
Configs/docs updated accordingly. Sweep runs are kept in results/sweep_idx*.

## 2026-10-21 — mvs_rank2_seed0 (full run: residual_index 23, rank 2, 1600 steps, seed 0, paper hyperparameters)
Train: median KL (last 5 steps) 0.0088; 348 s; peak VRAM 16.9 GB (micro-batch 8, no activation checkpointing). Single seed only.
Eval (`results/mvs_rank2_seed0/eval_metrics.json`, first-char accuracy, invalid 0% in all logic cells): 
| cond | Items | Families | Nonwords | paper |
|---|---|---|---|---|
| intact | .945 | .977 | .953 | .99/.97/.97 |
| retain | .949 | .977 | .949 | .99/.96/.96 |
| ablate | .500 | .500 | .500 | .51/.50/.51 |
Controls intact / retain / ablate: word recall 8: 1/.52/1; word recall 48: 1/.54/1; language localizer 1/1/**.56**; PrOntoQA-style .76/.49/**.46**; MMLU-binary .81/.47/.81;
Stroop .90/.51/1.0; MSIT 1/0/1; 1-digit mult 1/0/1; WikiText KL (clean‖intervened) 0 / 7.99 / 0.005.
Observations: ablation impairs logic to chance and also impairs PrOntoQA (deductive, logic-adjacent) and the language localizer (.56, not selective for that task);
other controls unaffected. Retention preserves logic but collapses controls (expected: subspace alone is not sufficient for them).
Caveats: single seed; re-created benchmark; layer convention chosen by sweep; control items are our own (prompts verbatim).

## 2026-10-21 — Phase-0 robustness extension (no tuning; Phase-0 methodology unchanged)
Seeds 1, 2: identical config/data (`data_seed=0` keeps splits and control items fixed; only init + batch sampling vary), same cached reference mu. Dataset hash identical (96b2e84595f89e08).
Full tables: `results/phase0_robustness/analysis_tables.md` (+ `analysis_summary.json`, `localizer_per_example.csv`, `random_ablation.json`). Regenerate with
`scripts/analyze_robustness.py` and `scripts/random_subspace_control.py --n 10`.

| seed | final KL | retain I/F/N | ablate I/F/N |
|---|---|---|---|
| 0 | .0088 | .949/.977/.949 | .500/.500/.500 |
| 1 | .0003 | .941/.980/.949 | **.852/.883/.770** |
| 2 | .0027 | .941/.977/.949 | .500/.500/.500 |
| mean±std | .004±.004 | .944±.005/.978±.002/.949±.000 | .617±.203/.628±.221/.590±.156 |
Intact (.945/.977/.953) is seed-independent.

* **Retention is stable** across seeds (within 0.01 of intact). **Ablation is not**: seeds 0 and 2 give exactly 0.500, seed 1 only drops to ~0.8.
* **Exact 50% = constant prediction.** In seeds 0 and 2, every example in every split gets answer "1" (valid) with P("1")≈1.0 (per-example entropy ≈0 bits). Balanced 128/128 gold ⇒ 0.500.
  Per form: all valid forms (MP, MT; ± negated) 100%, all invalid forms (AC, DA; ±) 0%. So ablation does not scramble inference; it yields an unconditional "valid" verdict.
  Seed 1 is not constant (92 vs 164 predictions of "2"/"1" on Items): it keeps most valid-form accuracy but errs mostly by calling invalid arguments valid (gold-2 recall .71/.77/.54).
* **Subspaces differ across seeds** (principal-angle cosines: s0/s1 .77,.19; s0/s2 .83,.10; s1/s2 .77,.06; random ≈.02). One direction is shared (~0.8), the second is not. Seed 1 has the lowest KL yet the weakest ablation effect, so retention KL does not predict ablation damage.
* **Random rank-2 ablations** (10 Gaussian + 10 top-variance, unoptimised, same location/mu): logic accuracy changes by ≤0.008 (max |Δ|); no control moves either. Max overlap with learned MVS 0.055. The learned MVS effect is specific to the learned subspace.
* **Language localizer**: ablated accuracy .562 in all three seeds, identical predictions: all 72 sentences "1" correct, all 56 pseudoword strings wrongly "1"; intact is 128/128, random ablations ~1.0. Not seed-dependent, not an artifact of one seed's logic damage (seed 1's logic partly survives, localizer does not).
* **Answer-channel pattern** (post-hoc interpretation, not tested): ablation produces constant "1" on the three tasks whose "1" is the affirmative verdict (valid / true / English sentence) and whose gold=2 requires a rejecting verdict (logic seeds 0,2; PrOntoQA seeds 0,2 (.46,.45; seed 1 .65 non-constant); localizer all seeds).
  Tasks without a verdict structure (word recall, Stroop, MMLU-binary, MSIT, multiplication) are unaffected. PrOntoQA is reported separately as logic-adjacent; localizer is the non-logical case.
* Caveats: 3 seeds; control items re-created; random control uses 10+10 subspaces at one location; "constant-1" is measured under teacher-free first-token scoring only.

## 2026-10-21 — Phase 0.5 (verdict/answer-token confound; no retraining)
Scripts: `phase05_formats_bias.py` (Exp 1+4), `phase05_directions.py` (Exp 2), `phase05_patching.py` (Exp 3), `phase05_patching_nonverdict.py` (Exp 3b), `phase05_report.py` (tables).
Narrative + numbers: `docs/reports/phase05_report.md`; tables: `results/phase05/report_tables.md`. Headline: retention is bound to the 1→valid mapping; ablation = shift toward the
affirmative verdict with ranking (AUC≥0.95) preserved; same signature on the language localizer and PrOntoQA; the shared rank-1 direction alone reproduces the collapse across seeds;
MVS patching transfers a verdict-like signal on matched pairs (random subspaces: nothing) but also on localizer/PrOntoQA, and not on verdict-free tasks.

## 2026-10-21 — Phase 0.6 (validity beyond the verdict direction)
Plan with pre-registered decision rule: `docs/plans/phase06_plan.md` (+ Amendment 1 for pilot 2). Report: `docs/reports/phase06_report.md`. Outputs: `results/phase06/` (decodability.json, search/ and search2/ candidates, evaluation*.json, report_tables*.md).
Result: validity decodable (~0.97) outside d_c, the MVS span and after 12 INLP directions; 12 searched candidates (2 pilots) give format-specific AUC drops with no transplantable patching effect; none passes criteria 1–4 → no credible logic-selective target found.

## 2026-10-21 — Phase 1 (native-unit implementation of the verdict direction)
Frozen plan `docs/plans/phase1_plan.md`; report `docs/reports/phase1_report.md`; outputs `results/phase1/`. Stage 1: MLP share 0.557, selected layers 22/21/20/19; Stage 2–3: top-1 % |Δ| share 0.49 vs Haar 0.126, N_eff 60 vs 11.7k, top-512 reconstructs 98 % of the signed class difference;
Stage 4: top-512 ablation reproduces 41 % / 59 % of full d_c-ablation gap loss (logic num_orig / num_rev), controls ≈0–0.18; C4 fails narrowly (Stroop +1.13 logits). Frozen rule → mixed/inconclusive (G-sparse, C-mixed). ~1.2 GPU-hours used.
Operational notes: the first Stage-4 run crashed in the last cross-task step (key-name mismatch `localizer` vs `language_localizer`); fixed and rerun with resume (phase D only), no other conditions were affected. For the full d_c-ablation condition the captured residual precedes the intervention hook (hook order), so its stored `c`/`dh` are not used; its effect enters only through margins.

## 2026-10-21/22 — Phase 2 (attention heads and the verdict direction)
Frozen plan `docs/plans/phase2_plan.md`; report `docs/reports/phase2_report.md`; outputs `results/phase2/`. Recompute gate FAILED (worst-case d_c-contribution error 2.7 %/2.4 % vs 2 %) → routing/content, source-position and A/V patching not run.
Stage 1: S8 holds 78 % of |Δ| mass, reconstructs 87 % of the held-out net; Stage 3: S8 ablation E_gap 0.86/1.14 (controls ≤ 0.29), side effects fail C3; patching transfer 0.90 (random max 0.126 → H-patching fails the < 0.02 rule); Stage 4: attention+MLP combined 1.12 (sub-additive on logic, super-additive on localizer).
Mechanical verdict: mixed / inconclusive. Bugs found and fixed during the phase: non-deterministic control seeds (k=8 controls rerun), silent skipping of all-position patches (aligned-only rerun). ≈3.0 of 4 GPU-hours.

## 2026-10-22 — Phase 2.5 (recompute diagnosis, source positions, exploratory A/V patching)
Plan `docs/plans/phase25_plan.md` (Amendments 1, 2); report `docs/reports/phase25_report.md`; outputs `results/phase25/`. The Phase 2 verdict (mixed) and the failed gate are preserved. Diagnosis: the recompute-vs-SDPA discrepancy (rel. L2 1.8e-3) is at the bf16 output-quantisation floor (1.65e-3); masks/GQA/RoPE exact; HF eager is 7× further from SDPA; S8 contribution errors ≤ 1.5 % of own std.
Descriptive (model's own eager attention): positive S8 heads read their contribution from the conclusion's final token and the generation-prompt tail; routing/content accounting ≈ 4 % / 96 % for L22H5. Exploratory A/V patching (criterion V1–V5 fixed beforehand, all passed): value-only patch transfers 0.89, attention-weight-only 0.01, A+V = full-z (0.904).

## 2026-10-22 — Phase 2.6 (position-restricted value patching of S8; exploratory)
Plan `docs/plans/phase26_plan.md`; report `docs/reports/phase26_report.md`; outputs `results/phase26/`. Primary (last query row): F(G1∪G2) = 0.900 / 0.915, F(Remaining) = −0.005 / −0.018, random count-matched ≤ 0.006 → localization criteria met in both formats. G1 and G2 alone ≈ 0.26–0.32 (super-additive jointly); all-query-row setting: F(Remaining) ≈ 0.4. ≈ 0.41 of 0.5 GPU-hours (one bug-induced rerun).
