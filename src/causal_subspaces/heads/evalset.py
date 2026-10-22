"""Evaluation sets and per-example evaluation for Phase 2. Exploratory = the Phase-1 subset (regression only); confirmatory = the complementary,
previously unused 384 held-out logic examples + fresh seed-5 controls + the unused WikiText passages. Criteria are evaluated on the confirmatory set."""
import numpy as np
import torch

from causal_subspaces import phase05 as P5
from causal_subspaces.data.controls import build_controls
from causal_subspaces.experiment import get_controls, get_lm_passages, get_splits
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits


class EvalSet:
    def __init__(self, model, tok, cfg, which: str, device="cuda"):
        splits = get_splits(cfg)
        rng = np.random.default_rng(0)

        def strat(sp, n):
            ex = splits[sp]
            v, i = [e for e in ex if e.valid], [e for e in ex if not e.valid]
            return [v[j] for j in rng.choice(len(v), n // 2, replace=False)] + [i[j] for j in rng.choice(len(i), n // 2, replace=False)]
        used = strat("items", 128) + strat("families", 128) + strat("nonwords", 128)       # Phase-1 exploratory logic subset (identical RNG stream)
        if which == "exploratory":
            ev = used
            controls = get_controls(cfg)
            self.passages = get_lm_passages(tok, cfg)[:32]
            sizes = {"localizer": 128, "prontoqa": 128, "mmlu_binary": 96, "stroop": 96, "word_recall_48": 96}
        else:
            ids = {e.id for e in used}
            ev = [e for sp in ("items", "families", "nonwords") for e in splits[sp] if e.id not in ids]
            c = cfg["controls"]
            controls = build_controls(5, 128, tuple(c["mmlu_subjects"]), c["mmlu_per_subject"])
            self.passages = get_lm_passages(tok, cfg)[32:64]
            sizes = {"localizer": 128, "prontoqa": 128, "mmlu_binary": 96, "stroop": 96, "word_recall_48": 96}
        self.which, self.logic_examples, self.controls = which, ev, controls
        rows = []
        for fmt in ("num_orig", "num_rev"):
            f = P5.FORMATS[fmt]
            ta, tb = P5.single_token_id(tok, f["valid"]), P5.single_token_id(tok, f["invalid"])
            rows += [(P5.build_prompt(fmt, e.syllogism), ta, tb, bool(e.valid), f"logic_{fmt}") for e in ev]
        t1, t2 = P5.single_token_id(tok, "1"), P5.single_token_id(tok, "2")
        key = {"localizer": "language_localizer", "prontoqa": "prontoqa", "mmlu_binary": "mmlu_binary", "stroop": "stroop", "word_recall_48": "word_recall_48"}
        for name, n in sizes.items():
            rows += [(x[0], t1, t2, x[1] == "1", name) for x in controls[key[name]]["items"][:n]]
        self.groups = np.array([r[4] for r in rows]); self.accept = np.array([r[3] for r in rows])
        self.prompts = [format_prompt(tok, r[0], False) for r in rows]
        self.ta = torch.tensor([r[1] for r in rows], device=device); self.tb = torch.tensor([r[2] for r in rows], device=device)
        self.order = sorted(range(len(self.prompts)), key=lambda i: len(self.prompts[i]))
        self.mult = controls["multiplication_1digit"]["items"][:64]
        self.n = len(rows)


@torch.no_grad()
def evaluate(model, tok, ev: EvalSet, idx_hook: int, d_c: torch.Tensor, ctx_factory=None, base_h=None, device="cuda", kl=True, bs=32):
    """Per-example arrays for one condition. ctx_factory(record) -> context manager (applied OUTSIDE the residual capture, so the captured verdict
    coordinate is the post-intervention one). base_h: intact residuals at the hook for the perturbation norm."""
    N = ev.n
    out = dict(margin=np.zeros(N, np.float32), c=np.zeros(N, np.float32), direct=np.full(N, np.nan, np.float32), dh=np.zeros(N, np.float32),
               lpm=np.zeros(N, np.float32), greedy=np.zeros(N, np.int64))
    hs = np.zeros((N, d_c.numel()), np.float32) if base_h is None else None
    for s in range(0, N, bs):
        idx = ev.order[s:s + bs]
        ids, mask, pos = encode(tok, [ev.prompts[i] for i in idx], device)
        sink, rec = [], _new_record()
        ctx = ctx_factory(rec) if ctx_factory else None
        from contextlib import nullcontext
        with (ctx if ctx is not None else nullcontext()):
            with capture_residual(model, idx_hook, sink):
                lg = last_logits(model, ids, mask, pos)
        ta, tb = ev.ta[idx], ev.tb[idx]
        la, lb = lg.gather(1, ta[:, None])[:, 0], lg.gather(1, tb[:, None])[:, 0]
        h = sink[0][:, -1].float()
        out["margin"][idx] = (la - lb).cpu().numpy()
        out["lpm"][idx] = (torch.logsumexp(torch.stack([la, lb], 1), 1) - torch.logsumexp(lg, -1)).cpu().numpy()
        out["c"][idx] = (h @ d_c).cpu().numpy()
        out["greedy"][idx] = lg.argmax(-1).cpu().numpy()
        if base_h is not None:
            out["dh"][idx] = (h - torch.tensor(base_h[idx], device=device)).norm(dim=-1).cpu().numpy()
        else:
            hs[idx] = h.cpu().numpy()
        td = rec.total_direct()
        if td is not None:
            out["direct"][idx] = td.cpu().numpy()
    if base_h is None:
        out["_h"] = hs
    klv = 0.0
    if kl and ctx_factory is not None:
        P = ev.passages
        for s in range(0, len(P), 4):
            ids = P[s:s + 4].to(device)
            clean = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[:, 32:].float(), -1)
            with ctx_factory(_new_record()):
                lg2 = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[:, 32:].float(), -1)
            klv += float((clean.exp() * (clean - lg2)).sum(-1).mean()) / (len(P) // 4)
    out["kl"] = np.float32(klv)
    return out


def _new_record():
    from causal_subspaces.heads.ablation import HeadRecord
    return HeadRecord()


@torch.no_grad()
def generation_check(model, tok, ev: EvalSet, ctx_factory=None, device="cuda", n_per_group=64, max_new=8):
    """8-token greedy continuations on n prompts per group (+ 1-digit multiplication); returns decoded strings and mult. correctness."""
    from contextlib import nullcontext
    res = {}
    for g in np.unique(ev.groups):
        idx = np.where(ev.groups == g)[0][:n_per_group]
        outs = []
        for s in range(0, len(idx), 16):
            ids, mask, _ = encode(tok, [ev.prompts[i] for i in idx[s:s + 16]], device)
            ctx = ctx_factory(_new_record()) if ctx_factory else None
            with (ctx if ctx is not None else nullcontext()):
                o = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False)
            outs += tok.batch_decode(o[:, ids.shape[1]:], skip_special_tokens=True)
        res[g] = outs
    import re
    outs = []
    texts = [x[0] for x in ev.mult]
    for s in range(0, len(texts), 16):
        ids, mask, _ = encode(tok, [format_prompt(tok, t, False) for t in texts[s:s + 16]], device)
        ctx = ctx_factory(_new_record()) if ctx_factory else None
        with (ctx if ctx is not None else nullcontext()):
            o = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8, do_sample=False)
        outs += tok.batch_decode(o[:, ids.shape[1]:], skip_special_tokens=True)
    ok = [bool((m := re.match(r"\s*(\d+)", o)) and m.group(1) == g) for o, (_, g) in zip(outs, ev.mult)]
    res["multiplication_acc"] = float(np.mean(ok)); res["multiplication_outputs"] = outs[:8]
    return res
