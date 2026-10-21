"""Phase 0.5 helpers: label/format variants, answer-logit statistics, aligned directions, matched pairs.

Everything here reuses the Phase-0 model loading, hook conventions (`residual_index`), checkpoints and benchmark;
nothing is retrained. Semantic quantities (valid vs invalid) are always kept separate from literal token quantities
(which token string was produced) so that label-remapping experiments can tell them apart.
"""
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import torch

from causal_subspaces.data.logic import FORMS, INSTRUCTION, MARKERS, build_syllogism
from causal_subspaces.models.hooks import capture_residual, residual_intervention
from causal_subspaces.models.loading import encode, format_prompt, last_logits

BASE = ("Please tell me whether the following reasoning is valid or invalid (i.e., does the conclusion necessarily "
        "follow from the premises?).")
# `valid` / `invalid` = the label string that CORRECTLY answers a valid / invalid argument in that format.
# `lit_first` = the label mentioned first in the instruction (literal-position preference diagnostic).
FORMATS = {
    "num_orig": dict(valid="1", invalid="2", lit_first="1", tail=" Answer 1 if it is valid, or 2 if it is invalid. Answer with just 1 or 2 and nothing else."),
    "num_rev": dict(valid="2", invalid="1", lit_first="1", tail=" Answer 1 if it is invalid, or 2 if it is valid. Answer with just 1 or 2 and nothing else."),
    "ab_orig": dict(valid="A", invalid="B", lit_first="A", tail=" Answer A if it is valid, or B if it is invalid. Answer with just A or B and nothing else."),
    "ab_rev": dict(valid="B", invalid="A", lit_first="A", tail=" Answer A if it is invalid, or B if it is valid. Answer with just A or B and nothing else."),
    # Reversed Yes/No (Yes = invalid) is not linguistically natural for "does it follow?", so only one mapping.
    "yn": dict(valid="Yes", invalid="No", lit_first="Yes", tail=" Answer Yes if it is valid, or No if it is invalid. Answer with just Yes or No and nothing else."),
}


def build_prompt(fmt: str, syllogism: str) -> str:
    return BASE + FORMATS[fmt]["tail"] + "\n\n" + syllogism


assert build_prompt("num_orig", "X") == INSTRUCTION + "\n\nX", "num_orig must be exactly the Phase-0 prompt"


def single_token_id(tok, label: str) -> int:
    t = tok.encode(label, add_special_tokens=False)
    if len(t) != 1:
        raise ValueError(f"{label!r} is {len(t)} tokens")
    return t[0]


@torch.no_grad()
def collect(model, tok, texts, aff: str, rej: str, *, device="cuda", batch_size=32, residual_index=None, mode="noop",
            U=None, mu=None) -> dict:
    """Last-position answer statistics per example: raw logits of the two answer tokens, log-sum-exp, greedy token."""
    ia, ir = single_token_id(tok, aff), single_token_id(tok, rej)
    prompts = [format_prompt(tok, t, False) for t in texts]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    la, lr, lse = (np.zeros(len(prompts), np.float32) for _ in range(3))
    gr = np.zeros(len(prompts), np.int64)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask, pos = encode(tok, [prompts[i] for i in idx], device)
        if mode == "noop":
            lg = last_logits(model, ids, mask, pos)
        else:
            with residual_intervention(model, residual_index, mode, U, mu):
                lg = last_logits(model, ids, mask, pos)
        la[idx], lr[idx] = lg[:, ia].cpu().numpy(), lg[:, ir].cpu().numpy()
        lse[idx], gr[idx] = torch.logsumexp(lg, -1).cpu().numpy(), lg.argmax(-1).cpu().numpy()
    return dict(logit_aff=la, logit_rej=lr, lse=lse, greedy=gr)


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties 0.5 (Mann-Whitney)."""
    r = pd.Series(score).rank().values
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold_acc(score: np.ndarray, pos: np.ndarray) -> float:
    """Accuracy of the best single margin threshold (an oracle that removes any global bias shift)."""
    cand = np.concatenate([[-np.inf], np.sort(np.unique(score)), [np.inf]])
    return float(max(((score > c) == pos).mean() for c in cand))


def answer_stats(tok, rec: dict, gold_aff: np.ndarray, aff: str, rej: str, lit_first: str | None = None,
                 split: np.ndarray | None = None) -> dict:
    """`aff` = label that is correct when gold_aff is True (valid verdict / '1' on localizer & PrOntoQA)."""
    first = np.array([tok.decode([g])[:1] for g in rec["greedy"]])
    pred = np.where(first == aff[0], "aff", np.where(first == rej[0], "rej", "other"))
    gold = np.where(gold_aff, "aff", "rej")
    m = rec["logit_aff"] - rec["logit_rej"]  # semantic margin when aff is the valid/affirmative answer
    la, lr = rec["logit_aff"] - rec["lse"], rec["logit_rej"] - rec["lse"]
    sd = lambda x: float(np.std(x))
    within = float(np.sqrt((np.var(m[gold_aff]) + np.var(m[~gold_aff])) / 2))
    out = dict(
        n=len(m), acc=float((pred == gold).mean()), invalid_rate=float((pred == "other").mean()),
        pred_aff=float((pred == "aff").mean()), pred_rej=float((pred == "rej").mean()),
        confusion={g: {p: int(((gold == g) & (pred == p)).sum()) for p in ("aff", "rej", "other")} for g in ("aff", "rej")},
        token_counts=dict(Counter(tok.decode([g]) for g in rec["greedy"]).most_common(6)),
        margin_mean=float(m.mean()), margin_std=sd(m), margin_mean_gold_aff=float(m[gold_aff].mean()),
        margin_mean_gold_rej=float(m[~gold_aff].mean()), margin_within_std=within,
        dprime=float((m[gold_aff].mean() - m[~gold_aff].mean()) / within) if within > 0 else float("nan"),
        auc=auc(m, gold_aff), best_threshold_acc=best_threshold_acc(m, gold_aff),
        mean_logit_aff=float(rec["logit_aff"].mean()), mean_logit_rej=float(rec["logit_rej"].mean()),
        mean_logprob_aff=float(la.mean()), mean_logprob_rej=float(lr.mean()),
        mass_on_labels=float(np.exp(la).mean() + np.exp(lr).mean()),
    )
    if lit_first is not None:  # literal token preference: how often the literally-first-listed label is produced
        out["pred_literal_first"] = float((first == lit_first[0]).mean())
    if split is not None:
        out["acc_by_split"] = {str(s): float((pred == gold)[split == s].mean()) for s in np.unique(split)}
    return out


# ---- directions ---------------------------------------------------------------------------------------------
def pairwise_principal_angles(Ua: torch.Tensor, Ub: torch.Tensor):
    """Cosines of principal angles and the principal vectors (columns) in each basis."""
    W, S, Zh = torch.linalg.svd(Ua.double().T @ Ub.double())
    return S, Ua.double() @ W, Ub.double() @ Zh.T


def consensus_directions(Us: dict) -> dict:
    """For each seed j find, INSIDE its own rank-2 span, the unit direction most contained in the other seeds' spans
    (top eigenvector of M_j = sum_k U_j^T U_k U_k^T U_j) and its orthogonal complement (the seed-specific direction).
    Eigenvalues are summed squared cosines to the other subspaces. Signs are oriented to seed order for readability."""
    out, ref = {}, None
    for j, Uj in Us.items():
        Uj = Uj.double()
        M = sum((Uj.T @ Uk.double()) @ (Uk.double().T @ Uj) for k, Uk in Us.items() if k != j)
        ev, vec = torch.linalg.eigh(M)  # ascending
        shared, spec = Uj @ vec[:, -1], Uj @ vec[:, 0]
        if ref is None:
            ref = (shared, spec)
        shared = shared if shared @ ref[0] >= 0 else -shared
        spec = spec if spec @ ref[1] >= 0 else -spec
        out[j] = dict(shared=shared.float(), specific=spec.float(), eig_shared=float(ev[-1]), eig_specific=float(ev[0]))
    return out


# ---- matched counterfactual pairs ---------------------------------------------------------------------------
PAIR_TYPES = [("MP", "AC"), ("MT", "DA")]  # (valid form, invalid form): same three sentences, fact/conclusion roles swapped


def _pair_marker(content_id, fam, neg, ptype) -> str:
    return MARKERS[(content_id * 7 + len(fam) * 3 + int(neg) + ptype) % len(MARKERS)]


def build_pairs(contents: list, content_ids, families, tok, fmt="num_orig") -> list[dict]:
    """Valid/invalid pairs built from the SAME rule, antecedent sentence and consequent sentence and the SAME marker;
    only their roles (which one is stated as fact, which as conclusion) are swapped, e.g. MP <-> AC, MT <-> DA.
    Kept only if both full prompts have equal token length (positions align from the left padding side)."""
    pairs = []
    for cid in content_ids:
        for fam in families:
            for pt, (vf, inv) in enumerate(PAIR_TYPES):
                for neg in (False, True):
                    mk = _pair_marker(cid, fam, neg, pt)
                    sv, si = (build_syllogism(contents[cid], fam, f, neg, mk) for f in (vf, inv))
                    pv, pi = (format_prompt(tok, build_prompt(fmt, s), False) for s in (sv, si))
                    tv, ti = (tok(p, add_special_tokens=False).input_ids for p in (pv, pi))
                    if len(tv) != len(ti):
                        continue
                    pairs.append(dict(content_id=cid, family=fam, ptype=f"{vf}/{inv}", neg=neg, valid_syl=sv, invalid_syl=si,
                                      n_diff_tokens=sum(a != b for a, b in zip(tv, ti)), n_tokens=len(tv)))
    return pairs


def lexical_controls(contents, pairs, tok, fmt="num_orig", seed=0) -> list[dict]:
    """For each pair, a source with the SAME validity/form/family/neg/marker as the target but different lexical
    content and equal token length ('same label, different words') -> patching it must not carry a verdict."""
    rng = np.random.default_rng(seed)
    by = {}
    for p in pairs:
        by.setdefault((p["family"], p["ptype"], p["neg"]), []).append(p)
    out = []
    for p in pairs:
        cands = [q for q in by[(p["family"], p["ptype"], p["neg"])]
                 if q["content_id"] != p["content_id"] and q["n_tokens"] == p["n_tokens"]]
        if not cands:
            continue
        q = cands[int(rng.integers(len(cands)))]
        out.append(dict(base=p, other=q))
    return out


# ---- activation patching trials -----------------------------------------------------------------------------
@torch.no_grad()
def run_trials(model, tok, trials: list[dict], aff: str, rej: str, subspaces: dict, residual_index: int, *,
               positions=("all", "last"), batch_size=16, device="cuda") -> dict:
    """trials: dicts with tgt/src prompt texts. subspaces: name -> (U, complement_flag).
    Returns arrays: margin (aff-rej logit) and greedy-first-char verdict for unpatched target/source and every
    (subspace, position) patched target. Source residuals are cached once per batch and reused across subspaces."""
    from causal_subspaces.models.hooks import residual_patch

    ia, ir = single_token_id(tok, aff), single_token_id(tok, rej)
    n = len(trials)
    res = dict(m_t=np.zeros(n), m_s=np.zeros(n), v_t=np.empty(n, "<U5"), v_s=np.empty(n, "<U5"))
    for name in subspaces:
        for pos in positions:
            res[f"m|{name}|{pos}"] = np.full(n, np.nan)
            res[f"v|{name}|{pos}"] = np.full(n, "", "<U5")

    def verdict(lg):
        f = [tok.decode([g])[:1] for g in lg.argmax(-1).tolist()]
        return ["aff" if c == aff[0] else "rej" if c == rej[0] else "other" for c in f]

    for s0 in range(0, n, batch_size):
        b = trials[s0:s0 + batch_size]
        sl = slice(s0, s0 + len(b))
        ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], device)
        ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], device)
        sink = []
        with capture_residual(model, residual_index, sink):
            lg_s = last_logits(model, ids_s, m_s, p_s)
        src_h = sink[0]
        lg_t = last_logits(model, ids_t, m_t, p_t)
        res["m_t"][sl] = (lg_t[:, ia] - lg_t[:, ir]).cpu().numpy(); res["v_t"][sl] = verdict(lg_t)
        res["m_s"][sl] = (lg_s[:, ia] - lg_s[:, ir]).cpu().numpy(); res["v_s"][sl] = verdict(lg_s)
        for name, (U, comp) in subspaces.items():
            for pos in positions:
                if pos == "all" and src_h.shape != (ids_t.shape[0], ids_t.shape[1], src_h.shape[-1]):
                    continue  # lengths differ: positions cannot be aligned
                with residual_patch(model, residual_index, U, src_h, pos, comp):
                    lg = last_logits(model, ids_t, m_t, p_t)
                res[f"m|{name}|{pos}"][sl] = (lg[:, ia] - lg[:, ir]).cpu().numpy()
                res[f"v|{name}|{pos}"][sl] = verdict(lg)
    return res


def summarize_patching(res: dict, tgt_aff: np.ndarray, src_aff: np.ndarray, name: str, pos: str) -> dict:
    """Patching effect for one (subspace, position). Direction-of-source quantities use only trials whose labels differ."""
    mp, vp = res[f"m|{name}|{pos}"], res[f"v|{name}|{pos}"]
    ok = ~np.isnan(mp)
    diff = (tgt_aff != src_aff) & ok
    same = (tgt_aff == src_aff) & ok
    out = dict(n_ok=int(ok.sum()))
    if diff.any():
        sgn = np.where(src_aff[diff], 1.0, -1.0)
        shift = sgn * (mp[diff] - res["m_t"][diff])
        gap = sgn * (res["m_s"][diff] - res["m_t"][diff])
        gold_t = np.where(tgt_aff[diff], "aff", "rej"); gold_s = np.where(src_aff[diff], "aff", "rej")
        both_ok = (res["v_t"][diff] == gold_t) & (res["v_s"][diff] == gold_s)  # clean trials: target and source answered correctly
        out.update(n_diff=int(diff.sum()), shift_mean=float(shift.mean()), shift_sem=float(shift.std() / np.sqrt(len(shift))),
                   gap_mean=float(gap.mean()), transfer=float(shift.mean() / gap.mean()),
                   n_clean=int(both_ok.sum()),
                   flip_to_source=float((vp[diff][both_ok] == gold_s[both_ok]).mean()) if both_ok.any() else float("nan"),
                   still_target=float((vp[diff][both_ok] == gold_t[both_ok]).mean()) if both_ok.any() else float("nan"))
    if same.any():
        dm = mp[same] - res["m_t"][same]
        out.update(n_same=int(same.sum()), abs_shift_mean=float(np.abs(dm).mean()),
                   verdict_change_rate=float((vp[same] != res["v_t"][same]).mean()))
    return out
