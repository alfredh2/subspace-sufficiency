# Phase 3 report: do the layer-22 MLP units (U22) mediate the late attention heads' effect on the verdict direction?

Plan: `docs/phase3_plan.md`. Auto-generated tables: `results/phase3/report_tables.md`; gate verdicts: `results/phase3/criteria_verdict.json`.
Compute: about 0.91 GPU-hours of the 1.5 h cap (`results/phase3/budget.json`). No upstream tracing was started.

## Design in one paragraph
Interventional 2x2 on the last query row: Y00 clean, Y11 head intervention (S8_L22 or L22H5; full-z patching from the source member of an aligned valid/invalid pair, or mean ablation), Y10 rescue (head intervention, U22 units restored to clean activations), Y01 transplant (clean run, U22 overwritten with intervened activations). R, B, S, J are ratios of aggregated signed effects with paired bootstrap CIs. They are interventional rescue and transplant measures, not causal fractions of the natural pathway. U22 is the 278 layer-22 units in the Phase 1 top-512. The test-content aligned pairs are held out from the MLP ranking but were examined in Phase 2.5/2.6 attention experiments. Qwen hook order (o_proj, attn out, mediator capture/restore, down_proj, mlp out) was verified numerically before any result was used.

## Results
**1. Does U22 mediate the heads' effect on the verdict coordinate and on final answers? Yes, in the patching design, both formats (gate M passed).**
| S8_L22, patching | num_orig | num_rev |
|---|---|---|
| TE coordinate / margin | 10.71 / 9.35 | 10.64 / 7.10 |
| U22 coordinate rescue R | 0.606 | 0.613 |
| U22 margin rescue R | 0.822 | 0.796 |
| transplant S (coord / margin) | 0.606 / 0.574 | 0.613 / 0.580 |
| full MLP22 rescue (upper bound) coord / margin | 0.616 / 0.825 | 0.623 / 0.805 |
| share of induced layer-22 verdict-directed write change carried by U22 | 0.982 | 0.983 |

U22 is essentially the whole MLP22 contribution (0.606 of 0.616). Random, activation-matched, alignment-matched and complement populations rescue about 0 (control means 0.000 to 0.027; alignment-matched is highest on margin). U22 exceeds all 20 draws per family in every comparison. The empirical p floor is 1/21, so the significance comes from the bootstrap and z-scores, which are large but rest on a normal approximation over 20 draws. L22H5 alone and the joint S8 set (U22 rescue 0.51 / 0.64 coord/margin) give the same picture, with a smaller share for the joint set because earlier-layer heads act through other layers. Oracle (response-selected) rescue is no higher than U22's: the Phase 1 ranking found the units that matter. The delta-matched control (same activation-change magnitude, not native) rescues about 0, so the effect is not just the size of the activation change.

**2. How much bypasses U22?** For the verdict coordinate, 39% of the head effect bypasses U22 (B = 0.39). That is almost exactly the direct attention write (about 4.1 of 10.7 coordinate units). For the final margin, 18–20% bypasses. Margin rescue exceeds coordinate rescue and the interaction term J is +0.22 to +0.25, so the margin effects of the direct write and the unit-mediated write are not additive. The bypass is therefore not dominated by other routes (the 0.70 threshold was not reached). A coordinate clamp (reset d_c to its clean value at the hook) removes the coordinate effect and leaves 13% (num_orig) to 34% (num_rev) of the margin effect, i.e. some margin effect does not pass through the verdict coordinate.

**3. Across formats?** The patching result is stable across numeric formats (table above) and in the all-query-rows setting (coordinate R 0.606, margin R 0.815). Ablation-based mediation (S8_L22): logic num_orig/num_rev coordinate R 0.608/0.615 and margin R 0.717/0.621. The localizer coordinate R is 0.653, but the localizer margin effect of S8_L22 ablation (2.87) is below the 5-logit validity floor, so that ratio is undefined. Under the preregistered gate G this counts as not met. For the joint S8 set, U22 rescue on the localizer margin is -0.32 (TE 7.1), i.e. earlier-layer routes dominate there. Earlier-layer heads (L21H19, L20H8, L19H10) are largely below the validity floor on their own; only L21H19's coordinate effect is evaluable (R 0.32).

**4. Is the pathway verdict-selective? No evidence of selectivity; gate V failed.** Head ablation shifts the verdict-free word-recall class gap by 6.14 logits (above the floor) and U22 restoration rescues 0.51 of it (random controls about 0.00, complement 0.27). So the same head-to-unit pathway also carries a non-verdict effect. MMLU-binary, Stroop and PrOntoQA effects are below the floor, so selectivity there is untestable, not confirmed. Disclosure: I added the class-gap reading of the V check to `phase3_report.py` after seeing the ablation numbers, because the plan's wording ("head-ablation shifts") admits it; the mean-shift reading alone has all verdict-free tasks below the floor (V then not evaluable). Under either reading V is not established, and under the class-gap reading it fails.

## Outcome (preregistered, mechanical): MIXED
Mediation (M) established in both formats; generalization (G) not met because the localizer margin is under the floor; selectivity (V) failed on word recall. In line with the plan, mediation and specificity are kept separate: U22 mediates the S8_L22 effect on the verdict coordinate and the answer margin, but this is not a verdict-specific pathway.

## Caveats
- Patching pairs are numeric-format logic items only; ablation covers other tasks at class level.
- Delta-matched and oracle controls are response-selected (diagnostic, derived from validation pairs), not pre-intervention.
- U22 is selected by Phase 1 on logic data; its overlap with word-recall behavior was not designed for.
- Control family matching: alignment-matched controls match alignment well (SMD 0.17) but random controls do not (SMD 1.5); see tables.
- Supplementary stage is exploratory.
