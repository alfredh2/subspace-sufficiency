# Phase 8, Stage S1 checkpoint report: carrier discovery and validation

Status: S1 complete. **Stopped for review.** No union ablation, conditional co-ablation, transplant, dose, layerwise-recovery, compensation or pinning test, and no logic-arm fit, has been run. Nothing was pushed.

Commits: plan `1e0e0f5`; data freeze and Gate 0 `6cbffdf`; discovery freeze `d2fd7bc` (committed before any confirm-set look). Confirm outputs and this report are committed afterwards.

## 1. Gate 0 (numerical validation)
Passed (`results/phase8/stage1/gate0.json`). U1 orthonormality error <= 7.2e-7; projector algebra ~1e-17; no-op hook difference 0.0; bf16 cast relative error 1.9e-3; batch-size noise floor on the answer-letter gap 0.132 logits; U1 seed-0 dev retention KL 4.9e-6, accuracy 1.0.

## 2. Data
Fresh entities: 44 capitals + 25 elements, disjoint from the old lexicon; 50 assigned to confirm, 19 to fresh-dev (>= 36 required). The intact-accuracy screen dropped none (accuracy 1.000). The reused Stage 1b confirm items are reported separately and flagged "reused".

## 3. Discovery (dev only; the fit script never loads confirm items)
| hook | chains | cap | K*_dev per chain | retries | double failures |
|---|---|---|---|---|---|
| 25 | 5 | 8 | 8, 8, 8, 8, 8 | 0 | 0 |
| 21 | 2 | 4 | 4, 4 | 0 | 0 |

Every fit at every k was dev-sufficient on the first attempt (dev KL <= 4e-4, accuracy 1.0 = intact). **All chains stopped at the cap, not at a failure.** K* is therefore a search-dependent lower bound set by the cap (>= 8 at hook 25, >= 4 at hook 21), not an estimate of the number of carriers. A larger cap was not tried; it was not preregistered.

Orthogonality (numerical, not assumed): max pairwise error 1.3e-7 at hook 25 and 3.5e-8 at hook 21; the 1e-6 abort threshold never fired.

Determinism refit (chain 0, k=2): principal-angle cosines 1.0000 (all four), train KL 3.1e-6 vs original, dev KL 3.0e-6, so the fits are reproducible.

## 4. Confirm sufficiency (rules frozen in `discovery_freeze.json`)
- **Hook 25: K*_conf = 8 in 5/5 chains**, on fresh confirm and on reused Stage 1b confirm items. All confirm KL are <= 1.5e-4, with 95% entity-cluster bootstrap upper bounds far below 0.05, and accuracy 1.0. **MSR (frozen rule: median K*_conf >= 4, >= 4/5 chains): PASS**, as a statement about sufficiency under the KL-trained retain intervention. It says nothing yet about necessity, redundancy or compensation.
- **Hook 21 (2 chains, contrast only):** on the reused items all 4 carriers per chain pass with bounds. On fresh confirm items all 8 carriers pass on the point estimate (KL 1e-4 to 0.039, accuracy 0.995 to 1.0), but the bootstrap upper bound exceeds 0.05 for carriers 1 and 2 in both chains (about 0.10 to 0.11). The heavy tail comes from a few entities. By the point-estimate definition K*_conf = 4, 4; with the upper-bound condition required, 0, 0 on fresh items. I report both and do not choose between them post hoc. Hook 21 is a two-chain contrast and has no frozen verdict beyond feasibility, which is met on the point estimate.

## 5. Descriptive semantics, geometry, readout-likeness (hook 25 unless noted; no causal claim)
- **Carriers share information.** Median ridge R² of one carrier's coordinates predicted from another's is 0.98, and median top canonical correlation is 0.999. At hook 21 these are 0.24 and 0.87. The hook-25 carriers are close to linearly redundant as *information*, which is exactly what the S2 tests must separate from causal redundancy.
- **Letter probes do not discriminate.** A random 4-d subspace of the hook-25 residual decodes the answer letter at 0.996 (union of 4 carriers 1.0, full residual 1.0). The "decodes the letter >= 0.9" half of the frozen SEMANTIC-DUPLICATE flag is therefore satisfied by everything at hook 25. Under the frozen flag, 69 of 140 carrier pairs are flagged, driven by the logit-deviation cosine (median 0.49, range -0.30 to 0.90). The flag is descriptive only and I do not read it as evidence of duplication. At hook 21 a random 4-d subspace gives 0.38, the letter is not yet linearly available in 4-d (union 0.65), and 0/12 pairs are flagged although logit cosines are high (median 0.77).
- **Relation probes** (capital vs element): median 0.75 across hook-25 carriers, range 0.60 to 1.0, so carriers differ in how much task-relation information they carry. Hook 21 median 0.99.
- **Readout-likeness:** logit-lens accuracy of the retained residual >= 0.9 in only 3 of 40 hook-25 carriers (range 0.03 to 1.0), and 0 of 8 at hook 21 (0.25). Energy in the span of the four letter unembedding rows has median 0.0097 against a random reference of 0.00098, about 10x chance but small in absolute terms. Most carriers are not directly readable by the unembedding. The intact residual lens accuracy is 1.0 at hook 25 and 0.25 at hook 21.
- **Parallel fit (U2' orthogonal to U1, fitted independently of the sequential U2):** subspace overlap with the sequential U2 is 0.05 to 0.17 across chains against a complement-Haar null of 0.001, with top cosine 0.41 to 0.66. Independently fitted second carriers partly share directions but are far from identical, so the specific carriers are optimisation-path dependent.

## 6. Anomaly gates
None triggered: numerics pass; orthogonality pass; median K*_dev 8 (>= 2); determinism cosine 1.0; 50 confirm entities (>= 36); hook-21 chains feasible.

## 7. Caveats
1. The cap, not the data, ends the search. K*_conf = 8 is a lower bound. The preregistered 8 may be far from the limit given zero failures.
2. Each carrier is rank 4 and trained by KL against the intact output. Sufficiency of a retained subspace means the model's behaviour is reproduced when everything else is projected out at this hook, in all positions. That already shows redundancy of *information* in the residual stream at hook 25 (the whole task is recoverable from many orthogonal 4-d slices). It does not show that the intact model uses more than one of them.
3. A random 4-d subspace of the full residual decodes the letter, so decodability is uninformative at hook 25. Retention under the KL objective is the only discriminating sufficiency criterion.
4. The ridge and canonical-correlation numbers come from the script's train/confirm subsampling as configured. I did not audit them for in-sample inflation, and the 0.999 canonical correlation in particular should be read as an upper bound.
5. Hook 21 has two chains. Its bootstrap-bound failure on fresh items for carriers 1 and 2 is heavy-tailed and untested beyond that.
6. The old Stage 1b confirm items were reused as a second view only. They are not independent of Phase 7.

## 8. What S2 would test (not started, awaiting approval)
Individual and conditional necessity, joint and cumulative ablation with equal-rank, energy-matched and effect-matched controls, union transplants, dose, layerwise traces with LayerNorm controls, and the compensation test (B8), at natural and controlled strength. The preregistered categories (MSR, JCD, SCR, ACT, ASR, MIXED, INCONCLUSIVE) stand unchanged. Budget used so far is about 2.2 GPU-h of the 5 to 6 h soft budget (35 fits at roughly 162 s plus 8 at roughly 176 s, plus the extra parallel and determinism fits and evaluation).

## 9. Artifacts
`scripts/phase8_s1_{fit,freeze,confirm}.py`; `results/phase8/stage1/{discovery_freeze,confirm_descriptive,determinism,gate0,data_manifest}.json` and `carriers/*.json`. Checkpoints (`.pt`) and `fresh_items.pkl` stay out of git.
