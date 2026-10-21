"""Shared experiment plumbing for the scripts: setup, evaluation of one condition, reference loading."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from causal_subspaces.data.controls import build_controls
from causal_subspaces.data.logic import build_splits, dataset_hash
from causal_subspaces.evaluation.metrics import summarize
from causal_subspaces.evaluation.scoring import evaluate_prompts
from causal_subspaces.models.hooks import residual_intervention
from causal_subspaces.models.loading import encode, format_prompt
from causal_subspaces.mvs.reference import cached_reference, load_neutral_corpus_ids
from causal_subspaces.utils.config import ROOT


def get_splits(cfg):
    d = cfg["data"]
    return build_splits(cfg.get("data_seed", cfg["seed"]), d["n_eval"], d["n_val_contents"], d["n_test_contents"], d["nonwords_in_optimization"])


def get_reference(model, tok, cfg, revision, device="cuda"):
    r, idx = cfg["reference"], cfg["mvs"]["residual_index"]
    key = f"ref_{cfg['model']['name'].replace('/', '_')}_{revision}_{idx}_{r['dataset'].replace('/', '_')}_{r['max_tokens']}_{r['window']}_{cfg['mvs']['init_m']}.pt"
    path = ROOT / cfg["hardware"]["cache_dir"] / key
    if path.exists():
        return torch.load(path)
    win = load_neutral_corpus_ids(tok, r["dataset"], r["config"], r["split"], r["max_tokens"], r["window"])
    return cached_reference(path, model=model, windows=win, residual_index=idx, m=cfg["mvs"]["init_m"],
                            batch_size=r["batch_size"], device=device)


@torch.no_grad()
def generate_number(model, tok, texts, golds, *, device, batch_size, think, residual_index=None, mode="noop", U=None, mu=None):
    """Multiplication: leading number of an 8-token greedy generation, exact match (paper App. B.2)."""
    rows = []
    for s in range(0, len(texts), batch_size):
        ids, mask, _ = encode(tok, [format_prompt(tok, t, think) for t in texts[s:s + batch_size]], device)
        kw = dict(input_ids=ids, attention_mask=mask, max_new_tokens=8, do_sample=False)
        if mode == "noop":
            out = model.generate(**kw)
        else:
            with residual_intervention(model, residual_index, mode, U, mu):
                out = model.generate(**kw)
        for g, o in zip(golds[s:s + batch_size], tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)):
            m = re.match(r"\s*(\d+)", o)
            rows.append(dict(first_char=o[:1], greedy_token=o, invalid=m is None, correct=bool(m and m.group(1) == g),
                             restricted_correct=False, mass_on_labels=float("nan")))
    return rows


@torch.no_grad()
def lm_kl(model, tok, passages, *, device, residual_index, mode, U, mu, skip=32, bs=4):
    """Mean per-token KL(clean || intervened) over positions >= skip of WikiText passages (teacher-forced).
    Approximation of the paper's 'continuation' control (their exact protocol is not stated)."""
    kls = []
    for s in range(0, len(passages), bs):
        ids = passages[s:s + bs].to(device)
        clean = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[:, skip:].float(), -1)
        with residual_intervention(model, residual_index, mode, U, mu):
            lg = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[:, skip:].float(), -1)
        kls.append((clean.exp() * (clean - lg)).sum(-1).mean().item())
    return float(np.mean(kls))


def evaluate_condition(model, tok, cfg, splits, controls, mode, U=None, mu=None, passages=None, device="cuda"):
    """Returns (metrics dict, prediction rows) for one condition (noop / retain / ablate)."""
    hw, think, idx = cfg["hardware"], cfg["model"]["enable_thinking"], cfg["mvs"]["residual_index"]
    kw = dict(device=device, batch_size=hw["eval_batch_size"], residual_index=idx, mode=mode, U=U, mu=mu, enable_thinking=think)
    metrics, preds = {"logic": {}, "controls": {}}, []
    for name in ("items", "families", "nonwords"):
        ex = splits[name]
        rows = evaluate_prompts(model, tok, [e.prompt_text for e in ex], [e.label for e in ex], ["1", "2"], **kw)
        df = pd.DataFrame(rows)
        for col in ("family", "form", "valid", "lexicon", "marker", "content_id"):
            df[col] = [getattr(e, col) for e in ex]
        df["id"] = [e.id for e in ex]
        metrics["logic"][name] = summarize(df, ["form", "valid", "family"])
        preds += [dict(condition=mode, task=f"logic/{name}", **r) for r in df.to_dict("records")]
    for name, t in controls.items():
        texts, golds = [x[0] for x in t["items"]], [x[1] for x in t["items"]]
        if t["labels"] is None:
            rows = generate_number(model, tok, texts, golds, device=device, batch_size=hw["eval_batch_size"], think=think,
                                   residual_index=idx, mode=mode, U=U, mu=mu)
        else:
            rows = evaluate_prompts(model, tok, texts, golds, t["labels"], **kw)
        df = pd.DataFrame(rows)
        metrics["controls"][name] = summarize(df)
        preds += [dict(condition=mode, task=f"control/{name}", gold=g, **r) for g, r in zip(golds, df.to_dict("records"))]
    if passages is not None:
        metrics["controls"]["lm_kl"] = (0.0 if mode == "noop" else
                                        lm_kl(model, tok, passages, device=device, residual_index=idx, mode=mode, U=U, mu=mu))
    return metrics, preds


def get_controls(cfg):
    c = cfg["controls"]
    return build_controls(cfg.get("data_seed", cfg["seed"]), c["n"], tuple(c["mmlu_subjects"]), c["mmlu_per_subject"])


def get_lm_passages(tok, cfg):
    from datasets import load_dataset
    ds = load_dataset(cfg["reference"]["dataset"], cfg["reference"]["config"], split="validation")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    n = cfg["controls"]["lm_kl_passages"]
    return ids[: n * 128].view(n, 128)
