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
