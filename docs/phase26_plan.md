# Phase 2.6 plan — position-restricted value patching of the frozen S8 heads (new preregistered EXPLORATORY experiment)

**Standing.** Phase 2.6 is a new exploratory experiment. It does not alter or retroactively correct any earlier criterion: the Phase 2 verdict (**mixed / inconclusive**), the recorded failed original recompute gate, the frozen S8 set (L22H5, L21H19, L20H8, L22H8, L22H15, L19H10, L22H6, L22H4), all checkpoints and all earlier results are unchanged. The recompute path is the one validated in Phase 2.5 (Amendment 2, V1–V5). Outputs: `results/phase26/`. Not pushed; no co-author trailers. Budget: **≤ 0.5 GPU-hour** (tracked in `results/phase26/budget.json`; the run script aborts at 27 minutes). Out of scope: head → MLP mediation; localizer and PrOntoQA (unpaired opposite-class patching adds confounds; a matched cross-task experiment needs its own plan).

## Question
For the S8 heads, which **source positions** supply the value content whose replacement transfers the verdict effect to the answer-position head outputs? Phase 2.5 showed that replacing values at *all* positions transfers ≈ 0.89 of the margin gap (attention-weight-only: ≈ 0.01); the descriptive analysis suggests the conclusion's final token and the generation-prompt tail carry it. This experiment tests that causally.

## Position groups (assigned from the aligned token spans; verified per prompt)
For each prompt, "eligible source positions" = all non-padding positions (all are causal for the last query row). Disjoint groups:
* **G1** — the final token of the conclusion line (the sentence-final token).
* **G2** — the generation-prompt tail tokens (from `<|im_end|>` to the token before the query), **excluding the query token**.
* **Self** — the query position (the last token).
* **Remaining** = complement of G1 ∪ G2 within the eligible positions, **including Self** (everything else: first token/sink, chat prefix, instruction, rule, fact, rest of the conclusion, Self). Descriptive extra: **Remaining excluding Self**.
* **All** = every eligible position (positive control).
Assertions (per prompt, reported in aggregate): G1, G2, Self pairwise disjoint; G1 ∪ G2 ∪ Remaining = all eligible positions and G1 ∪ G2 disjoint from Remaining; Self ⊂ Remaining; no padding positions in any group; source and target prompts of an aligned pair have **identical** group masks (otherwise the pair is rejected, never realigned).

## Intervention (V-only, using the validated recompute path)
For head h and group P, the value vectors of the head's KV group at key positions in P are taken from the **source** example, all other key positions keep the **target's own** values; attention weights stay the target's. The only addition to `attention_patch` is an optional boolean per-example position mask `v_positions` (B,T).
* **Primary:** only the **last query row** of the head output is replaced (isolates what supplies the answer-position head output). **Secondary:** all query rows.
* Heads: **S8 jointly** (primary), and **each S8 head individually** (all eight kept in the detailed results to expose heterogeneity and positive vs negative contributors; individually: G1∪G2, Remaining, All for the six minor heads; G1, G2, G1∪G2, Remaining, All for L22H5 and L21H19).
* Pairs/formats: the 217 alignment-validated pairs (434 trials per format) in `num_orig` and `num_rev`, both directions.
* Conditions (S8 joint): G1, G2, G1∪G2, Remaining, Self, Remaining-excluding-Self, All.

## Controls
* **Position-count-matched random controls (deterministic, recorded):** for each trial, random source positions are drawn **only from the eligible pool = Remaining excluding Self** (non-padding, causal), with counts matched to |G1|, |G2| and |G1 ∪ G2| of that trial; 10 draws per matching (seeds from `zlib.crc32` of a fixed tag, trial and draw index; drawn positions saved). If a trial's pool is smaller than the required count, all pool positions are used and the trial is flagged and counted (expected: none).
* **Same-label lexical controls:** aligned pairs of the same family/form/negation/label with different content (the Phase 2 construction, alignment-validated); V-only patching of S8 at G1 ∪ G2, Remaining and All must not change the verdict (reported as verdict-change rate and |Δ margin|).
* **Self-patch noise floor** (source = target) measured first for S8, L22H5, L21H19; effects below 3× the floor are not interpreted.

## Implementation tests (required before the experiment)
* Full mask reproduces the existing V-only patch (tiny model exactly; on Qwen S8 the full-mask transfer reproduces the stored Phase 2.5 V-only values 0.888 / 0.886 (all query rows) and 0.711 / 0.666 (last row) within ±0.01); empty mask = no intervention (logits equal to the unpatched model within recompute tolerance).
* Group masks: disjointness, coverage and `Self ⊂ Remaining`, G1 = token before `<|im_end|>` of the user turn, G2 excludes the query; padding excluded; identical masks for aligned pairs; mismatch raises; random sampler: deterministic, pool-restricted, count-matched, small-pool handling.
* Validation V1–V3 from the stored Phase 2.5 diagnostics re-checked; V4 re-measured on this setup.

## Metrics (reported separately; effects are not assumed additive after downstream recomputation)
Per set, condition, format and query setting: (a) *direct* head-output change of the verdict contribution at the last position (Σ_h gain_h·(z_patched − z_target)), (b) downstream verdict-coordinate transfer after recomputation, (c) final answer-margin transfer `T` (signed shift toward the source ÷ source-target margin gap), (d) flip-to-source rate on clean trials, (e) interactions: `T(G1∪G2) − T(G1) − T(G2)` and `T(All) − T(G1∪G2) − T(Remaining)` with bootstrap CIs. Bootstrap (500 resamples over trials) 95 % CIs for every `T` and `F`.
**F = T(group) / T(All)** in the same format and query setting. F is **undefined** (and the condition reported as "not evaluable") if the point estimate `T(All) < 0.20` or the bootstrap 2.5th percentile of `T(All)` is `< 0.10`; bootstrap replicates with unstable denominators (`T(All) < 0.05`) are discarded and counted.

## Preregistered localization criteria (effect-size criteria on S8, primary query setting, **both** formats)
* **Localization supported:** `F(G1∪G2) ≥ 0.70` **and** `F(Remaining) ≤ 0.20` **and** the count-matched random controls (matched to G1∪G2) have mean `F ≤ 0.15` and maximum `≤ 0.30`.
* **Partially localized:** `0.40 ≤ F(G1∪G2) < 0.70`, or `F(G1∪G2) ≥ 0.70` but `F(Remaining) > 0.20`, or controls above the limits.
* **Not localized:** otherwise (including `F(G1∪G2) < 0.40`).
Mixed outcomes across formats are reported as such. All outcomes, including failures, are reported. Secondary query setting, individual heads, Self and Remaining-excluding-Self are descriptive.

## Deliverables and stop
`docs/phase26_report.md` (findings, validation details, limitations, recommendations); commit without co-author trailers; no push. **Stop after the report; the head → MLP mediation study is not started.**
