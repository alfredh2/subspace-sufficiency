"""Phase 7 Stage 1 library: task-generic evaluation, MVS fitting, gates, transplantation.

Margins are class-based: for an item with candidate tokens c_0..c_{K-1} (class order), m_ab = logit(c_a) - logit(c_b).
The margin class gap of a task is the mean over class pairs (a,b) of mean(m_ab | gold=a) - mean(m_ab | gold=b).
"""
import itertools
import math
import time

import numpy as np
import torch

from causal_subspaces.evaluation.metrics import summarize  # noqa: F401  (kept for parity with earlier phases)
from causal_subspaces.models.hooks import capture_residual, residual_intervention, residual_patch
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.mvs.optimization import kl_teacher_student, lr_at
from causal_subspaces.mvs.projection import orthonormalize


def texts_of(tok, items):
    return [it.prompt if it.raw else format_prompt(tok, it.prompt, False) for it in items]


def cand_array(items):
    return np.array([it.cand_ids for it in items])


def gold_array(items):
    return np.array([it.gold for it in items])


@torch.no_grad()
def run_stats(model, tok, items, idx=None, mode="noop", U=None, mu=None, bs=32, with_kl=False):
    """Candidate-token logits, greedy token, logsumexp and (optionally) KL(clean || mode) per item."""
    texts = texts_of(tok, items)
    cids = torch.tensor(cand_array(items))
    N, K = cids.shape
    order = sorted(range(N), key=lambda i: len(texts[i]))
    out = dict(cand_logits=np.zeros((N, K), np.float32), greedy=np.zeros(N, np.int64), lse=np.zeros(N, np.float32),
               kl=np.zeros(N, np.float32))
    for s in range(0, N, bs):
        ii = order[s:s + bs]
        ids, mask, pos = encode(tok, [texts[i] for i in ii], "cuda")
        clean = last_logits(model, ids, mask, pos) if (mode == "noop" or with_kl) else None
        if mode == "noop":
            lg = clean
        else:
            with residual_intervention(model, idx, mode, U, mu):
                lg = last_logits(model, ids, mask, pos)
        c = cids[ii].to(lg.device)
        out["cand_logits"][ii] = lg.gather(1, c).cpu().numpy()
        out["greedy"][ii] = lg.argmax(-1).cpu().numpy()
        out["lse"][ii] = torch.logsumexp(lg, -1).cpu().numpy()
        if with_kl and clean is not None:
            lc = torch.log_softmax(clean, -1)
            out["kl"][ii] = kl_teacher_student(lc, lg).cpu().numpy()
    return out


def accuracy_vec(stats, items):
    cids = cand_array(items)
    g = gold_array(items)
    return (stats["greedy"] == cids[np.arange(len(items)), g]).astype(float)


def restricted_accuracy_vec(stats, items):
    return (stats["cand_logits"].argmax(1) == gold_array(items)).astype(float)


def primary_accuracy_vec(stats, items):
    """Instructed (chat) formats: greedy first token must be the gold candidate. Raw continuation formats (no
    instruction; G, N-cont): candidate-restricted argmax (standard agreement-benchmark scoring); see plan section 12."""
    return restricted_accuracy_vec(stats, items) if items[0].raw else accuracy_vec(stats, items)


def class_gap(cl, gold):
    """Mean over class pairs of mean(m_ab | a) - mean(m_ab | b); NaN-safe for classes absent."""
    K = cl.shape[1]
    gaps = []
    for a, b in itertools.combinations(range(K), 2):
        m = cl[:, a] - cl[:, b]
        ma, mb = m[gold == a], m[gold == b]
        if len(ma) and len(mb):
            gaps.append(ma.mean() - mb.mean())
    return float(np.mean(gaps))


def cluster_boot(fn, groups, nb=1000, seed=0):
    """Percentile 95% interval of fn(index array) under resampling of whole groups."""
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nb):
        pick = rng.integers(0, len(ug), len(ug))
        vals.append(fn(np.concatenate([members[i] for i in pick])))
    vals = np.array(vals, float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def retention_metrics(model, tok, items, idx, U, mu, intact, bs=32):
    """Retained/ablated stats on a split vs the intact stats; returns summary dict + per-item arrays."""
    ret = run_stats(model, tok, items, idx, "retain", U, mu, bs, with_kl=True)
    abl = run_stats(model, tok, items, idx, "ablate", U, mu, bs, with_kl=True)
    g = gold_array(items)
    return dict(
        kl_retain=float(ret["kl"].mean()), kl_ablate=float(abl["kl"].mean()),
        acc_intact=float(primary_accuracy_vec(intact, items).mean()), acc_retain=float(primary_accuracy_vec(ret, items).mean()),
        acc_ablate=float(primary_accuracy_vec(abl, items).mean()),
        gacc_intact=float(accuracy_vec(intact, items).mean()), gacc_retain=float(accuracy_vec(ret, items).mean()),
        mass_intact=float(np.exp(intact["cand_logits"] - intact["lse"][:, None]).sum(1).mean()),
        mass_retain=float(np.exp(ret["cand_logits"] - ret["lse"][:, None]).sum(1).mean()),
        racc_intact=float(restricted_accuracy_vec(intact, items).mean()), racc_retain=float(restricted_accuracy_vec(ret, items).mean()),
        racc_ablate=float(restricted_accuracy_vec(abl, items).mean()),
        gap_intact=class_gap(intact["cand_logits"], g), gap_retain=class_gap(ret["cand_logits"], g), gap_ablate=class_gap(abl["cand_logits"], g),
    ), ret, abl


# ------------------------------------------------------------------ MVS fitting
def fit_mvs(model, tok, train_items, mu, V0, hook, steps=1600, batch=16, lr=0.01, seed=0, micro=16, log_every=100):
    """Phase-0 recipe (Adam, QR reparameterisation, KL to the unmodified model at the final position, cosine lr)."""
    dev = "cuda"
    mu = mu.to(dev)
    V = torch.nn.Parameter(V0.clone().to(dev))
    opt = torch.optim.Adam([V], lr=lr)
    texts = texts_of(tok, train_items)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    teacher = torch.empty(len(texts), model.config.vocab_size)
    with torch.no_grad():
        for s in range(0, len(order), 32):
            ii = order[s:s + 32]
            ids, mask, pos = encode(tok, [texts[i] for i in ii], dev)
            teacher[ii] = torch.log_softmax(last_logits(model, ids, mask, pos), -1).cpu()
    rng = np.random.default_rng(seed)
    kls, t0 = [], time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, 20.0)
        bi = rng.choice(len(texts), batch, replace=False)
        opt.zero_grad(set_to_none=True)
        U = orthonormalize(V)
        Ud = U.detach().requires_grad_(True)
        tot = 0.0
        for s in range(0, batch, micro):
            b = bi[s:s + micro]
            ids, mask, pos = encode(tok, [texts[i] for i in b], dev)
            with residual_intervention(model, hook, "retain", Ud, mu):
                lg = last_logits(model, ids, mask, pos)
            kl = kl_teacher_student(teacher[b].to(dev), lg)
            (kl.sum() / batch).backward()
            tot += kl.sum().item() / batch
        U.backward(Ud.grad)
        opt.step()
        kls.append(tot)
    U = orthonormalize(V.detach())
    return U, dict(kl_last5_median=float(np.median(kls[-5:])), kl_curve=[float(np.mean(kls[i:i + log_every])) for i in range(0, steps, log_every)],
                   seconds=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2**30)


# ------------------------------------------------------------------ transplantation
def pair_trials(pairs):
    """Both directions of every pair: dict(tgt, src) items."""
    tr = []
    for a, b in pairs:
        tr.append(dict(tgt=a, src=b))
        tr.append(dict(tgt=b, src=a))
    return tr


@torch.no_grad()
def transplant(model, tok, trials, hook, subspaces: dict, bs=16, positions=("all", "last")):
    """h' = h + UU^T(s - h) at the hook, positions 'all' or 'last'. Trials are grouped by exact token length so the
    target and source residuals align without padding. m = logit(target-candidate of the SOURCE class) - logit(target
    candidate of the TARGET class) evaluated in the target's own candidate tokens; m_src is the same quantity in the
    source's own tokens on the clean source run; m_tgt on the clean target run."""
    n = len(trials)
    texts_t = texts_of(tok, [t["tgt"] for t in trials])
    texts_s = texts_of(tok, [t["src"] for t in trials])
    lens = np.array([len(tok(x, add_special_tokens=False).input_ids) for x in texts_t])
    lens_s = np.array([len(tok(x, add_special_tokens=False).input_ids) for x in texts_s])
    assert (lens == lens_s).all(), "pair members must have equal token length"
    res = dict(m_tgt=np.zeros(n), m_src=np.zeros(n), group=np.array([t["tgt"].group for t in trials]))
    for nm in subspaces:
        for p in positions:
            res[f"m|{nm}|{p}"] = np.zeros(n)
    sc = np.array([t["src"].gold for t in trials])
    tc = np.array([t["tgt"].gold for t in trials])
    tcand = cand_array([t["tgt"] for t in trials])
    scand = cand_array([t["src"] for t in trials])

    def marg(lg, cands, a, b, ii):
        c = torch.tensor(cands[ii]).to(lg.device)
        v = lg.gather(1, c)
        r = torch.arange(len(ii), device=lg.device)
        return (v[r, torch.tensor(a[ii]).to(lg.device)] - v[r, torch.tensor(b[ii]).to(lg.device)]).cpu().numpy()
    for L in np.unique(lens):
        idxs = np.where(lens == L)[0]
        for s in range(0, len(idxs), bs):
            ii = idxs[s:s + bs]
            ids_t, m_t, p_t = encode(tok, [texts_t[i] for i in ii], "cuda")
            ids_s, m_s, p_s = encode(tok, [texts_s[i] for i in ii], "cuda")
            sink = []
            with capture_residual(model, hook, sink):
                lg_s = last_logits(model, ids_s, m_s, p_s)
            src_h = sink[0]
            lg_t = last_logits(model, ids_t, m_t, p_t)
            res["m_src"][ii] = marg(lg_s, scand, sc, tc, ii)
            res["m_tgt"][ii] = marg(lg_t, tcand, sc, tc, ii)
            for nm, U in subspaces.items():
                for p in positions:
                    with residual_patch(model, hook, U, src_h, p, False):
                        lg = last_logits(model, ids_t, m_t, p_t)
                    res[f"m|{nm}|{p}"][ii] = marg(lg, tcand, sc, tc, ii)
    return res


def transfer_ratio(res, nm, pos):
    num = res[f"m|{nm}|{pos}"] - res["m_tgt"]
    den = res["m_src"] - res["m_tgt"]
    return num, den


def ratio_stat(num, den, groups, nb=1000, seed=0):
    pt = float(num.sum() / den.sum())
    lo, hi = cluster_boot(lambda ix: num[ix].sum() / den[ix].sum(), groups, nb, seed)
    return pt, lo, hi


def haar_subspace(d, r, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, r, generator=g, dtype=torch.float64))
    return q.float().cuda()
