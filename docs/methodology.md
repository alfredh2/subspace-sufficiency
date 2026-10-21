# Methodology notes
* Frozen bf16 Qwen3-8B; only V (fp32, d×k) is trained; projection algebra, QR, log-softmax and KL in fp32.
* One hook implementation (`models/hooks.py`) is used for training, evaluation, and generation; out-of-place; all token positions.
* Residual convention: `residual_index=i` = HF `hidden_states[i]`; i≥1 hooks the output of `layers[i-1]`.
* Left padding with explicit `position_ids` (cumsum of mask) so padded batches equal unpadded results (tested).
* Loss uses only the last prompt position's logits (`lm_head` applied to that position only); gradients flow through all downstream layers.
* Teacher log-probs (unmodified model, full vocab) are cached once on CPU (fp32).
* Micro-batching + gradient accumulation (`hardware.micro_batch_size`); optional activation checkpointing (`model.gradient_checkpointing`).
* Model-selection discipline: validation split (disjoint contents) is used for monitoring/convention sweeps; eval splits are only used for final reports.
* Scoring: primary first-generated-character; restricted-argmax is diagnostic only; invalid rate tracked for each condition.
* Streaming reference statistics: shifted fp32 batch second moments summed in fp64 (no hidden-state cache); unit-tested against `torch.cov`.
