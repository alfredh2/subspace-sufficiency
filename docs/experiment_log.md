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
