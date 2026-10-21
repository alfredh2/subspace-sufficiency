"""Scoring.

PRIMARY (paper, Sec. 4.2): "we score the first generated character against the correct response label;
responses outside the permitted label set are counted as incorrect." We take the greedy (argmax over the full
vocabulary) first generated token, decode it, and compare its first character with the label. Greedy
decoding makes this deterministic; it equals the first character of `generate(do_sample=False)`.
DIAGNOSTIC (not the paper's metric): argmax restricted to the label tokens, plus probability mass on them.
Multi-token labels: `sequence_logprob` scores whole candidate strings (sum and per-token mean are both
returned; which normalisation to use is a documented choice per task, never implicit).
"""
import torch

from causal_subspaces.models.hooks import residual_intervention
from causal_subspaces.models.loading import encode, format_prompt, last_logits


def label_token_ids(tok, labels: list[str]) -> list[int]:
    """Single-token ids of the labels (no leading space). Raises if a label is not exactly one token."""
    ids = []
    for lab in labels:
        t = tok.encode(lab, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"label {lab!r} tokenises to {len(t)} tokens {t}; use sequence_logprob")
        ids.append(t[0])
    return ids


def score_first_token(logits: torch.Tensor, tok, label_ids: list[int], labels: list[str], gold: list[str]):
    """logits: (B, V) fp32 at final prompt position. Returns list of per-example dicts."""
    logp = torch.log_softmax(logits, -1)
    greedy = logits.argmax(-1)
    lab_logp = logp[:, label_ids]  # (B, n_labels)
    restricted = lab_logp.argmax(-1)
    rows = []
    for i in range(logits.shape[0]):
        text = tok.decode([int(greedy[i])])
        first_char = text[:1]
        rows.append(dict(
            greedy_token=text, first_char=first_char,
            invalid=first_char not in labels,
            correct=first_char == gold[i],                           # PRIMARY
            restricted_pred=labels[int(restricted[i])],
            restricted_correct=labels[int(restricted[i])] == gold[i],  # diagnostic
            mass_on_labels=float(lab_logp[i].exp().sum()),
            **{f"p_{lab}": float(lab_logp[i, j].exp()) for j, lab in enumerate(labels)},
        ))
    return rows


@torch.no_grad()
def evaluate_prompts(model, tok, user_texts: list[str], gold: list[str], labels: list[str], *, device="cuda",
                     batch_size=32, residual_index=None, mode="noop", U=None, mu=None, enable_thinking=False):
    """Teacher-forced-free evaluation of single-token-label tasks under an optional MVS intervention."""
    label_ids = label_token_ids(tok, labels)
    prompts = [format_prompt(tok, t, enable_thinking) for t in user_texts]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # length-sorted batches, order restored below
    out = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask, pos = encode(tok, [prompts[i] for i in idx], device)
        if mode == "noop":
            logits = last_logits(model, ids, mask, pos)
        else:
            with residual_intervention(model, residual_index, mode, U, mu):
                logits = last_logits(model, ids, mask, pos)
        for i, r in zip(idx, score_first_token(logits, tok, label_ids, labels, [gold[j] for j in idx])):
            out[i] = r
    return out


@torch.no_grad()
def sequence_logprob(model, tok, user_texts: list[str], candidates: list[str], *, device="cuda",
                     enable_thinking=False) -> list[dict]:
    """For every prompt and every candidate string: log P(candidate tokens | prompt), teacher-forced.
    Returns [{cand: {"sum": ..., "mean": ..., "n_tokens": ...}}, ...]. Handles multi-token candidates."""
    res = []
    for text in user_texts:
        prompt_ids = tok(format_prompt(tok, text, enable_thinking), add_special_tokens=False, return_tensors="pt").input_ids[0]
        entry = {}
        for cand in candidates:
            c_ids = tok(cand, add_special_tokens=False, return_tensors="pt").input_ids[0]
            ids = torch.cat([prompt_ids, c_ids]).unsqueeze(0).to(device)
            logits = model(input_ids=ids, use_cache=False).logits[0].float()
            lp = torch.log_softmax(logits[len(prompt_ids) - 1:-1], -1)
            tok_lp = lp.gather(1, c_ids.to(device).unsqueeze(1)).squeeze(1)
            entry[cand] = dict(sum=float(tok_lp.sum()), mean=float(tok_lp.mean()), n_tokens=len(c_ids))
        res.append(entry)
    return res
