"""Phase 7 Stage 1b library (plan: docs/phase7_stage1b_plan.md incl. amendments A1-A13).

Data: the 1b content repartition (partition seed 1001; 'test' = confirm), A with a second format, N-fixedpool.
Fitting: Phase-0 recipe with dense logging, optional LDA initialisation, candidate-restricted objective and an
orthogonality constraint (complement fits). Statistics: hierarchical (seed -> content cluster) bootstrap, principal
angles. Stage 1 code is imported, never modified.
"""
import itertools
import random
import time

import numpy as np
import torch

from causal_subspaces.models.hooks import capture_residual, residual_intervention
from causal_subspaces.models.loading import encode, last_logits
from causal_subspaces.mvs.optimization import kl_teacher_student, lr_at
from causal_subspaces.mvs.projection import orthonormalize
from causal_subspaces.phase7 import tasks as T
from causal_subspaces.phase7.stage1 import cand_array, class_gap, gold_array, run_stats, texts_of

PART_SEED = 1001
FORMATS_1B = dict(T.FORMATS, A=["digits", "plus"])
HOOKS = [21, 23, 25, 27]
RANKS = [4, 8]
TASKS_1B = ["L", "S", "G", "N", "M", "A"]
CELLS = {t: [(h, r) for h in HOOKS for r in RANKS] for t in TASKS_1B}
CELLS["M"] = [(h, r) for h in (21, 25, 27) for r in RANKS]


# ------------------------------------------------------------------ data
def build_A_1b(tok, seed=PART_SEED):
    rng = random.Random(seed + 55)
    cand = [str(i) for i in range(10)]
    cids = T._ids(tok, cand)
    sums = [(a, b) for a in range(10) for b in range(10) if a + b <= 9]
    rng.shuffle(sums)
    cut = {"train": sums[:30], "val": sums[30:38], "test": sums[38:]}
    prompt = {"digits": lambda a, b: f"What is {a}+{b}? Answer with just the number and nothing else.",
              "plus": lambda a, b: f"What is the sum of {a} and {b}? Answer with just the number and nothing else."}
    out = {}
    for fmt, f in prompt.items():
        d = {}
        for sp, lst in cut.items():
            reps = 20 if sp == "train" else 1      # train: each sum repeated (batches of 16 are drawn without replacement)
            d[sp] = [T.Item(f"A|{fmt}|{sp}{i}_{k}", "A", fmt, sp, f(a, b), False, cand, cids, a + b, f"s:{a}+{b}")
                     for k in range(reps) for i, (a, b) in enumerate(lst)]
        # matched pairs inside the confirm sums: differ in exactly one operand (equal token length), different sums
        prs = []
        for (a1, b1), (a2, b2) in itertools.combinations(cut["test"], 2):
            if (a1 == a2) != (b1 == b2) and a1 + b1 != a2 + b2:
                prs.append(((a1, b1), (a2, b2)))
        rng2 = random.Random(seed + 7)
        rng2.shuffle(prs)
        d["pairs"] = [(T.Item(f"A|{fmt}|p{i}a", "A", fmt, "pairs", f(*x), False, cand, cids, sum(x), f"s:{x[0]}+{x[1]}", f"p{i}"),
                       T.Item(f"A|{fmt}|p{i}b", "A", fmt, "pairs", f(*y), False, cand, cids, sum(y), f"s:{y[0]}+{y[1]}", f"p{i}"))
                      for i, (x, y) in enumerate(prs)]
        out[fmt] = d
    return out


def build_N_fixedpool(tok, seed=PART_SEED):
    """Diagnostic (tier-2): ALL splits share the name pool; splits differ in (template, place, object) contexts."""
    rng = random.Random(seed + 77)
    names = [n for n in T.NAMES if len(tok.encode(" " + n, add_special_tokens=False)) == 1 and len(tok.encode(n, add_special_tokens=False)) == 1]
    combos = [(t, p, o) for t in range(len(T.N_TEMPL)) for p in T.N_PLACE for o in T.N_OBJ]
    rng.shuffle(combos)
    cut = {"test": combos[:80], "val": combos[80:120], "train": combos[120:]}
    out = {}
    for fmt in T.FORMATS["N"]:
        d = {}
        for sp, cs in cut.items():
            n_items = 1200 if sp == "train" else (256 if sp == "val" else 384)
            items = []
            for i in range(n_items):
                x, y = rng.sample(names, 2)
                t, p, o = rng.choice(cs)
                fia = rng.random() < 0.5
                f, s = (x, y) if fia else (y, x)
                base = T.N_TEMPL[t].format(f=f, s=s, p=p, g=y, o=o)
                text = base if fmt == "cont" else T.N_QA.format(pre=base.rsplit(" to", 1)[0] + " to", o=o)
                cand = [" " + n for n in (f, s)] if fmt == "cont" else [f, s]
                items.append(T.Item(f"Nf|{fmt}|{sp}{i}", "N", fmt, sp, text, fmt == "cont", cand, T._ids(tok, cand), 0 if fia else 1,
                                    f"ctx:{t}|{p}|{o}"))
            d[sp] = items
        d["pairs"] = []
        out[fmt] = d
    return out


def build_1b(task, tok):
    if task == "L":
        return T.build_L(tok, seed=PART_SEED, n_eval=384, n_val_c=4, n_test_c=14)
    if task == "S":
        return T.build_S(tok, seed=PART_SEED, w_val=6, w_test=14, nn_val=5, nn_test=10)
    if task == "G":
        return T.build_G(tok, seed=PART_SEED, p_val=10, p_test=30)
    if task == "N":
        return T.build_N(tok, seed=PART_SEED, p_val=6, p_test=14)
    if task == "M":
        return T.build_M(tok, seed=PART_SEED, cap=(8, 20), elem=(5, 12))
    if task == "A":
        return build_A_1b(tok)
    raise ValueError(task)


def cap_items(items, n, seed=0):
    """Fixed random subset (dev evaluation cap)."""
    if len(items) <= n:
        return list(items)
    idx = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in idx]


def leakage_report(data):
    """Content-group and prompt disjointness between train / val / test for every format of one task."""
    rep = {}
    for fmt, d in data.items():
        g = {sp: {i.group for i in d[sp]} for sp in ("train", "val", "test")}
        pr = {sp: {i.prompt for i in d[sp]} for sp in ("train", "val", "test")}
        rep[fmt] = dict(n={sp: len(d[sp]) for sp in g}, n_groups={sp: len(g[sp]) for sp in g},
                        group_overlap={f"{a}&{b}": len(g[a] & g[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
                        prompt_overlap={f"{a}&{b}": len(pr[a] & pr[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
                        n_pairs=len(d["pairs"]))
    return rep


# ------------------------------------------------------------------ statistics
def cand_kl(intact_cl, other_cl):
    """KL(softmax(intact candidate logits) || softmax(other candidate logits)) per item (tier-2 / diagnostic metric)."""
    p = torch.softmax(torch.tensor(intact_cl, dtype=torch.float64), -1)
    q = torch.log_softmax(torch.tensor(other_cl, dtype=torch.float64), -1)
    return (p * (p.clamp_min(1e-30).log() - q)).sum(-1).numpy()


def hier_ci(stat, seeds, groups, nb=400, seed=0):
    """95% interval of the seed-median of stat(seed, idx) under resampling seeds, then content clusters (independently per drawn seed)."""
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nb):
        ps = [seeds[i] for i in rng.integers(0, len(seeds), len(seeds))]
        v = []
        for s in ps:
            pick = rng.integers(0, len(ug), len(ug))
            v.append(stat(s, np.concatenate([members[i] for i in pick])))
        vals.append(np.median(v))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def principal_cosines(Ua, Ub):
    Ua, Ub = Ua.double().cpu(), Ub.double().cpu()
    return torch.linalg.svdvals(Ua.T @ Ub).clamp(max=1.0).numpy()


def overlap(Ua, Ub):
    c = principal_cosines(Ua, Ub)
    return float((c ** 2).mean())


# ------------------------------------------------------------------ LDA reference / init
@torch.no_grad()
def lda_directions(model, tok, items, hook, Qm, ridge=1e-2, bs=32, max_items=600):
    """Whitened class-mean (LDA) directions of the final-position residual at `hook`, computed inside the on-manifold span Qm
    (train items only). Returns (d, K-1) orthonormal columns. Uses development data only."""
    items = cap_items(items, max_items, 0)
    texts = texts_of(tok, items)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    H = torch.zeros(len(texts), model.config.hidden_size)
    for s in range(0, len(order), bs):
        ii = order[s:s + bs]
        ids, mask, pos = encode(tok, [texts[i] for i in ii], "cuda")
        sink = []
        with capture_residual(model, hook, sink):
            last_logits(model, ids, mask, pos)
        H[ii] = sink[0][:, -1].float().cpu()
    y = gold_array(items)
    Z = H @ Qm.cpu().float()                                   # (N, m) on-manifold coordinates
    Z = Z - Z.mean(0)
    classes = np.unique(y)
    means = torch.stack([Z[y == c].mean(0) for c in classes])
    Sb = sum(((y == c).sum()) * torch.outer(m, m) for c, m in zip(classes, means)) / len(y)
    Sw = sum(((Z[y == c] - m).T @ (Z[y == c] - m)) for c, m in zip(classes, means)) / len(y)
    Sw = Sw + ridge * Sw.diagonal().mean() * torch.eye(Sw.shape[0])
    L = torch.linalg.cholesky(Sw.double())
    Linv = torch.linalg.inv(L)
    M = Linv @ Sb.double() @ Linv.T
    ev, vec = torch.linalg.eigh(M)
    w = (Linv.T @ vec[:, -(len(classes) - 1):]).float()          # generalised eigenvectors
    d = Qm.cpu().float() @ w
    q, _ = torch.linalg.qr(d)
    return q


# ------------------------------------------------------------------ fitting
def fit_b(model, tok, train_items, mu, V0, hook, steps=1600, batch=16, lr=0.01, seed=0, micro=16, log_every=25,
          objective="full", perp=None):
    """Phase-0 recipe (Adam, QR reparameterisation, cosine lr to lr/20), plus:
    objective 'full' = KL over the full vocabulary (the frozen definition) or 'cand' = KL over the renormalised candidate
    distribution (tier-2 diagnostic); `perp` (d, r0) constrains span(U) to be orthogonal to span(perp) (complement fits).
    Returns U and a log with the dense train-KL curve."""
    dev = "cuda"
    mu = mu.to(dev)
    P = None if perp is None else perp.to(dev).float()
    V = torch.nn.Parameter(V0.clone().to(dev))
    opt = torch.optim.Adam([V], lr=lr)
    texts = texts_of(tok, train_items)
    cids = torch.tensor(cand_array(train_items))
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    full = objective == "full"
    teacher = torch.empty(len(texts), model.config.vocab_size if full else cids.shape[1])
    with torch.no_grad():
        for s in range(0, len(order), 32):
            ii = order[s:s + 32]
            ids, mask, pos = encode(tok, [texts[i] for i in ii], dev)
            lg = last_logits(model, ids, mask, pos)
            teacher[ii] = (torch.log_softmax(lg, -1) if full else torch.log_softmax(lg.gather(1, cids[ii].to(dev)), -1)).cpu()

    def make_U():
        Vp = V if P is None else V - P @ (P.T @ V)
        return orthonormalize(Vp)
    rng = np.random.default_rng(seed)
    kls, t0 = [], time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, 20.0)
        bi = rng.choice(len(texts), batch, replace=False)
        opt.zero_grad(set_to_none=True)
        U = make_U()
        Ud = U.detach().requires_grad_(True)
        tot = 0.0
        for s in range(0, batch, micro):
            b = bi[s:s + micro]
            ids, mask, pos = encode(tok, [texts[i] for i in b], dev)
            with residual_intervention(model, hook, "retain", Ud, mu):
                lg = last_logits(model, ids, mask, pos)
            if not full:
                lg = lg.gather(1, cids[b].to(dev))
            kl = kl_teacher_student(teacher[b].to(dev), lg)
            (kl.sum() / batch).backward()
            tot += kl.sum().item() / batch
        U.backward(Ud.grad)
        opt.step()
        kls.append(tot)
    U = make_U().detach()
    kls = np.array(kls)
    return U, dict(kl_last5_median=float(np.median(kls[-5:])), steps=steps,
                   kl_curve=[float(kls[i:i + log_every].mean()) for i in range(0, steps, log_every)], log_every=log_every,
                   seconds=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2**30)


def curve_slope(curve, frac=0.25):
    """Relative decrease of the train-KL curve over the last `frac` of the run: (mean of window before - mean of last window) / mean before."""
    n = max(2, int(len(curve) * frac))
    a, b = np.mean(curve[-2 * n:-n]), np.mean(curve[-n:])
    return float((a - b) / max(a, 1e-9))
