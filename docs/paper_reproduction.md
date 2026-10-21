# Paper reproduction record — Kean & Boix-Adserà, "Logical Subspace in LLMs" (arXiv 2609.32907, v1)

Source read: arXiv HTML v1 (main text, Tables 1–2, App. A–D). **No code/data release is linked.** The HTML
conversion doubles some numerals; values below were de-doubled and should be spot-checked against the PDF.
Legend: **F** = faithful to a stated paper detail, **A** = approximation/our choice (paper silent), **X** = known deviation.

## Intervention (Sec. 3, Eq. 1)
| Item | Paper | Ours |
|---|---|---|
| Retain | h ↦ μ + UUᵀ(h−μ) at **every token position**, rest of net runs normally | F |
| Ablate | replace UUᵀ with I−UUᵀ | F |
| μ | mean of h over all token positions of a neutral corpus disjoint from tasks (App. C); corpus/size unstated | A: WikiText-2 train, 1M tokens, 256-token windows, no special tokens/chat template; streaming fp64 accumulation |
| **Layer index / hook point** | "layer 22"; 0- vs 1-based and before/after block **not stated** | A: **`residual_index = 23`** ≡ HF `hidden_states[23]` ≡ output of 0-based decoder module 22 ("0-based layer 22, after the block"), chosen by a validation-KL sweep over 21–24 (see experiment_log.md; 21/22/24 did not train within 300 steps). |

## Objective & optimisation (Sec. 3 Eq. 2, Sec. 4, App. C)
| Item | Paper | Ours |
|---|---|---|
| Loss | E KL(p_orig ‖ p_retained), full next-token distribution at the **final prompt position** | F (fp32 log-softmax, full vocab) |
| Parametrisation | V∈ℝ^{d×k}; U = QR(V) at every step | F; gradient through QR is not stated → A: differentiated through (sign-fixed so diag(R)≥0) |
| Optimiser | Adam, 1600 steps, 16 prompts/step, lr 1e-2 cosine → lr/20 | F; exact cosine formula A; weight decay 0, no clipping (unstated) A |
| Init | V₀ = Q_m ξ, Q_m = top m=2048 eigvecs of μ-centred covariance, ξ~N(0,1/m) (App. C) | F; covariance from the same neutral-corpus stream; corpus A |
| Seeds / stat | 3 seeds; median KL of last 5 logged steps | stat F; seeds: we run seed 0 first |
| Massive-coordinate pinning (App. D) | needed for Qwen layers 6–14, not claimed at 22 | not implemented (not needed at layer 22 per paper); revisit if KL stalls at ln 2 |

## Data (Sec. 4.1, App. B.1, Tables 1–2) — **our dataset is a re-creation (X)**
* F: instruction text verbatim; labels 1=valid / 2=invalid; 4 forms × {affirmative, negated} consequent = 8; 14 family names and their
  **rule** wording (Table 2); negation wording rules; 6 conclusion markers; held-out families `suppose_observe`, `whenever`, `subset`;
  2 lexicons × 48 contents; three 256-item eval splits (Items / Families / Nonwords).
* A: fact and conclusion sentences (only one worked example in the paper); the 48 contents and nonword lexicon; partition of
  contents into optimisation / validation / Items-test (paper does not give optimisation set size); premise order for all families except `fact_first`
  (rule first); "which term the rule mentions first" factor not implemented.
* X: 10,752 items (paper 10,720: they removed 32 stem-sharing event items; we avoid stem sharing by construction).
* Prompt layout (chat template, system prompt, non-thinking mechanism) is **not stated**: A = Qwen3 chat template with
  `enable_thinking=False`, user message = instruction + blank line + three-line syllogism.
* Each example's `fidelity` field records the above; reports must keep faithful and approximate parts separate.

## Scoring (Sec. 4.2)
Primary: **first generated character** vs label, outside-label responses incorrect (F; greedy argmax decoded). Diagnostic only:
restricted {1,2} argmax and probability mass on {1,2}. Invalid-output rate is recorded for every condition.
Recovery: chance-normalised (a−c)/(a₀−c) (App. C Eq. 3).

## Reported numbers to compare against (Fig. 4 caption; mean of Qwen3-8B and Gemma-4-12B — the per-model values are not tabulated)
intact 0.99/0.97/0.97, retain 0.99/0.96/0.96, ablate 0.51/0.50/0.51 (Items/Families/Nonwords; column mapping is our reading of the caption).
Controls: word recall, localizer ≈ ceiling; PrOntoQA intact 0.736 (Qwen); MMLU binary 0.805 (1/2); Raven 0.540 (uninterpretable).
Because our benchmark differs, comparison is "comparable conditions", **not** exact reproduction, and 0.97–0.99 intact is not a hard gate.

## Controls (App. B.2/B.3)
Prompts quoted verbatim from the paper; items/distractors/counts are ours (unstated). Implemented: word recall (8, 48), language localizer,
PrOntoQA-style, MMLU-binary (4 subjects, 1/2 format), Stroop (congruent+incongruent), MSIT, 1-digit multiplication (leading number of an
8-token greedy generation), WikiText-2 KL (teacher-forced approximation). Skipped: Raven (paper deems it uninterpretable for Qwen).

## Layer convention
See `docs/experiment_log.md` for the pre-planned sweep over `residual_index ∈ {21,22,23,24}` (selection by validation KL only,
never by eval-split ablation accuracy) and its outcome.

## Unresolved
HF revision; the 15 depths; μ corpus; optimisation set size; QR gradient; Fig. 6 vs text layer inconsistency for Gemma (not our model).
