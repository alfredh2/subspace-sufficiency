"""Stage 1b Arm B: why does the retained rank-r M subspace give KL~0 while its ablation does nothing?  Interventions on U (retain, ablate at all/last
positions, dose alpha), size/energy-matched controls, activation-magnitude diagnostics, complement-subspace fits (U2 orthogonal to U; ranks 2/4/8) and
union ablation [U,U2]; L and S as necessity contrasts (complement/union/energy only). Evaluated on CONFIRM contents for the frozen cells; U2 are fitted
on train items. Reports operational sufficiency of an orthogonal subspace, NOT proof that the intact network uses redundant pathways (amendment A11).
Usage: python scripts/phase7_stage1b_armB.py"""
import json, time
from contextlib import contextmanager
import numpy as np, torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.hooks import _target_module, capture_residual
from causal_subspaces.models.loading import encode, last_logits, load_model
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.phase7.stage1 import run_stats, haar_subspace, class_gap, gold_array, texts_of, pair_trials, transplant, primary_accuracy_vec
from causal_subspaces.phase7.stage1b import build_1b, FORMATS_1B, fit_b
from causal_subspaces.utils.config import ROOT, load_config

R = ROOT / "results/phase7/stage1b"
(R / "armB").mkdir(exist_ok=True)
sel = json.load(open(R / "selection.json"))["tasks"]
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")


@contextmanager
def dose(hook, U, mu, alpha, last_only, mode="ablate"):
    module = _target_module(model, hook)
    Uf, muf = U.float(), mu.float()

    def f(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        hf = h.float()
        c = ((hf - muf) @ Uf) @ Uf.T
        if last_only:
            new = hf.clone(); new[:, -1] = hf[:, -1] - alpha * c[:, -1]
        else:
            new = hf - alpha * c
        new = new.to(h.dtype)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
    hd = module.register_forward_hook(f)
    try:
        yield
    finally:
        hd.remove()


@torch.no_grad()
def stats_ctx(items, ctx_factory, bs=32):
    texts = texts_of(tok, items); N = len(items)
    cids = torch.tensor([it.cand_ids for it in items]); out = np.zeros((N, cids.shape[1]), np.float32)
    order = sorted(range(N), key=lambda i: len(texts[i]))
    for s in range(0, N, bs):
        ii = order[s:s + bs]
        ids, mask, pos = encode(tok, [texts[i] for i in ii], "cuda")
        with ctx_factory():
            lg = last_logits(model, ids, mask, pos)
        out[ii] = lg.gather(1, cids[ii].cuda()).cpu().numpy()
    return out


@torch.no_grad()
def energy(items, hook, mu, subspaces, bs=32):
    """Mean fraction of ||h-mu||^2 captured by each subspace (final position and all non-pad positions), plus per-item final-position norm of
    the ablation perturbation and the top massive coordinates."""
    texts = texts_of(tok, items); order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    num_last = {k: 0.0 for k in subspaces}; num_all = {k: 0.0 for k in subspaces}; den_last = den_all = 0.0
    pert = {k: [] for k in subspaces}; absmean = torch.zeros(mu.shape[0], device="cuda"); npos = 0
    mu = mu.cuda().float()
    for s in range(0, len(order), bs):
        ii = order[s:s + bs]
        ids, mask, pos = encode(tok, [texts[i] for i in ii], "cuda")
        sink = []
        with capture_residual(model, hook, sink):
            last_logits(model, ids, mask, pos)
        c = sink[0].float() - mu
        m = mask.bool()
        den_all += (c[m] ** 2).sum().item(); den_last += (c[:, -1] ** 2).sum().item()
        absmean += c[m].abs().sum(0); npos += int(m.sum())
        for k, U in subspaces.items():
            pr = (c @ U.cuda().float())
            num_all[k] += (pr[m] ** 2).sum().item(); num_last[k] += (pr[:, -1] ** 2).sum().item()
            pert[k] += pr[:, -1].norm(dim=-1).cpu().tolist()
    am = (absmean / npos).cpu()
    massive = am.topk(5).indices.tolist()
    return dict(frac_last={k: num_last[k] / den_last for k in subspaces}, frac_all={k: num_all[k] / den_all for k in subspaces},
                pert_norm_last={k: dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90))) for k, v in pert.items()},
                massive_coords=massive, mean_abs_massive=[float(am[j]) for j in massive], mean_abs_median_coord=float(am.median()))


def onman(Qm, r, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(Qm.cpu().float() @ torch.randn(Qm.shape[1], r, generator=g))
    return q.cuda()


def reduction(cl, intact_cl, gold):
    gi, ga = class_gap(intact_cl, gold), class_gap(cl, gold)
    return dict(rel=(gi - ga) / gi, abs=gi - ga, gap=ga)


def analyse(name, t, fmt, h, r, seeds, data, ref, full, r2s=(2, 4, 8)):
    mu = ref["mu"].cuda()
    d = data[fmt]; test = d["test"]; gold = gold_array(test)
    intact = run_stats(model, tok, test); icl = intact["cand_logits"]
    res = dict(name=name, task=t, fmt=fmt, hook=h, rank=r, seeds=seeds, intact_gap=class_gap(icl, gold), seed_results={})
    Us = {}
    for s in seeds:
        f = R / "armB" / f"{name}_U_s{s}.pt"
        if f.exists():
            Us[s] = torch.load(f).cuda(); continue
        src = R / "final" / f"{t}_{fmt}_h{h}_r{r}_s{s}.pt"
        a_ = R / f"armA/{t}_h{h}_r{r}_s{s}.pt"
        if src.exists():
            Us[s] = torch.load(src).cuda()
        elif a_.exists() and fmt == FORMATS_1B[t][0]:
            Us[s] = torch.load(a_).cuda()
        else:
            torch.manual_seed(s)
            U, summ = fit_b(model, tok, d["train"], ref["mu"], init_V(ref["Qm"], r, s, "onmanifold"), h, seed=s)
            Us[s] = U; torch.save(U.cpu(), f); print(name, "fit", s, summ["kl_last5_median"], flush=True)
    # controls shared across seeds
    haar = [haar_subspace(4096, r, 3000 + k) for k in range(10)]
    onm = [onman(ref["Qm"], r, 4000 + k) for k in range(10)]
    pca = ref["Qm"][:, :r].cuda().float()
    ctrl = {}
    if full:
        for nm, lst in (("haar", haar), ("onmanifold", onm), ("pca_top", [pca])):
            rr = [reduction(stats_ctx(test, lambda U=U: dose(h, U, mu, 1.0, False)), icl, gold) for U in lst]
            kk = [float(np.mean(__import__("causal_subspaces.phase7.stage1b", fromlist=["cand_kl"]).cand_kl(icl, stats_ctx(test, lambda U=U: __import__("causal_subspaces.models.hooks", fromlist=["residual_intervention"]).residual_intervention(model, h, "retain", U, mu))))) for U in lst[:3]]
            ctrl[nm] = dict(ablate_rel_mean=float(np.mean([x["rel"] for x in rr])), ablate_rel_max=float(np.max([x["rel"] for x in rr])), retain_candKL_mean=float(np.mean(kk)))
        res["controls"] = ctrl
    subs = {f"U{s}": Us[s] for s in seeds}
    if full:
        subs.update({"haar0": haar[0], "onmanifold0": onm[0], "pca_top": pca})
    res["energy"] = energy(test, h, mu, subs)
    print(name, "energy last", {k: round(v, 4) for k, v in res["energy"]["frac_last"].items()}, flush=True)
    # per-seed interventions
    for s in seeds:
        U = Us[s]; sr = {}
        if full:
            for al in (0.5, 1.0, 2.0, 4.0):
                sr[f"ablate_all_a{al}"] = reduction(stats_ctx(test, lambda: dose(h, U, mu, al, False)), icl, gold)
            sr["ablate_last_a1"] = reduction(stats_ctx(test, lambda: dose(h, U, mu, 1.0, True)), icl, gold)
        # complement fits
        for r2 in r2s:
            f2 = R / "armB" / f"{name}_U2_r{r2}_s{s}.pt"
            if f2.exists():
                U2 = torch.load(f2).cuda()
            else:
                torch.manual_seed(100 + s)
                V0 = init_V(ref["Qm"], r2, 100 + s, "onmanifold")
                V0 = V0 - U.cpu() @ (U.cpu().T @ V0)
                U2, summ = fit_b(model, tok, d["train"], ref["mu"], V0, h, seed=100 + s, perp=U)
                torch.save(U2.cpu(), f2)
                sr[f"U2_r{r2}_train_kl"] = summ["kl_last5_median"]
            from causal_subspaces.models.hooks import residual_intervention
            ret = run_stats(model, tok, test, h, "retain", U2, mu, with_kl=True)
            sr[f"U2_r{r2}_retain_kl"] = float(ret["kl"].mean())
            sr[f"U2_r{r2}_retain_acc"] = float(primary_accuracy_vec(ret, test).mean())
            sr[f"U2_r{r2}_orth_err"] = float((U.T @ U2).abs().max())
            Uu = torch.linalg.qr(torch.cat([U, U2], 1))[0]
            sr[f"union_r{r + r2}_ablate"] = reduction(stats_ctx(test, lambda: dose(h, Uu, mu, 1.0, False)), icl, gold)
            rnd = [reduction(stats_ctx(test, lambda Ur=haar_subspace(4096, r + r2, 5000 + k): dose(h, Ur, mu, 1.0, False)), icl, gold)["rel"] for k in range(3)]
            sr[f"rand_r{r + r2}_ablate_rel_mean"] = float(np.mean(rnd))
        res["seed_results"][s] = sr
        print(name, "seed", s, {k: (round(v["rel"], 3) if isinstance(v, dict) else round(v, 4)) for k, v in sr.items()}, flush=True)
    if full and d["pairs"]:
        trials = pair_trials(d["pairs"])
        sub = {f"s{s}": Us[s] for s in seeds}
        for k in range(10):
            sub[f"rand{k}"] = haar_subspace(4096, r, 1000 + k)
        tr = transplant(model, tok, trials, h, sub, bs=16)
        den = tr["m_src"] - tr["m_tgt"]
        res["transplant"] = {f"{nm}|{p}": float((tr[f"m|{nm}|{p}"] - tr["m_tgt"]).sum() / den.sum()) for nm in sub for p in ("all", "last")}
    json.dump(res, open(R / "armB" / f"{name}.json", "w"), indent=1, default=float)
    return res


# budget-driven reductions (declared in the report): contrast tasks one seed and complement rank 8 only; M_h25_r2 two seeds, complement ranks 2 and 8
for name, t, fmt, cellspec, full, seeds_, r2s_ in [("M_sel", "M", "letters", None, True, [0, 1, 2], (2, 4, 8)), ("M_h25_r2", "M", "letters", (25, 2), True, [0, 1], (2, 8)),
                                                  ("L_sel", "L", "num_orig", None, False, [0], (8,)), ("S_sel", "S", "words", None, False, [0], (8,))]:
    ch = sel[t]["choice"]
    if cellspec is None:
        if ch is None:
            print(name, "no selected cell; skipped", flush=True); continue
        h, r = ch["hook"], ch["rank"]
    else:
        h, r = cellspec
    if (R / "armB" / f"{name}.json").exists():
        continue
    cfg["mvs"]["residual_index"] = h
    ref = get_reference(model, tok, cfg, rev, "cuda")
    data = build_1b(t, tok)
    analyse(name, t, fmt, h, r, seeds_, data, ref, full, r2s_)
print("ARM B DONE", flush=True)
