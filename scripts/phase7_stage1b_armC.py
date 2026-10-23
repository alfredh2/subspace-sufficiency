"""Stage 1b Arm C: cross-format stability for L and S at their frozen 1b cell. (1) transfer matrix: subspaces fitted in format F (5 seeds) evaluated on the
confirm contents of every format E (retention, ablation, transplantation, with the cell gates); (2) principal angles / overlap (within-format cross-seed,
cross-format, Haar null); (3) jointly fitted subspace (union of formats, same rank) evaluated per format; (4) cross-format transplantation (opposite-class
source -> semantic coding; same-item source in another format -> output-token coding) where prompts have equal token length.
Reported separately from same-format generalisation (amendment A12). Usage: python scripts/phase7_stage1b_armC.py --tasks L S"""
import argparse, itertools, json
import numpy as np, torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.hooks import capture_residual, residual_patch
from causal_subspaces.models.loading import encode, last_logits, load_model
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.phase7.stage1 import run_stats, haar_subspace, texts_of
from causal_subspaces.phase7.stage1b import build_1b, FORMATS_1B, fit_b, eval_confirm, cell_gates, principal_cosines, overlap
from causal_subspaces.utils.config import ROOT, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="*", default=["L", "S"])
a = ap.parse_args()
R = ROOT / "results/phase7/stage1b"
(R / "armC").mkdir(exist_ok=True)
sel = json.load(open(R / "selection.json"))["tasks"]
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
SEEDS = [0, 1, 2, 3, 4]


def load_U(t, fmt, h, r, s):
    f = R / "final" / f"{t}_{fmt}_h{h}_r{r}_s{s}.pt"
    if f.exists():
        return torch.load(f).cuda()
    return torch.load(R / f"armA/{t}_h{h}_r{r}_s{s}.pt").cuda()


def body(it):
    return it.prompt.split("\n\n", 1)[1]


@torch.no_grad()
def cross_transplant(trials, hook, subs, bs=16):
    """trials: dicts(tgt, src, src_E, kind). Groups by exact token length; patches span(U) at all positions."""
    tt, ts = texts_of(tok, [x["tgt"] for x in trials]), texts_of(tok, [x["src"] for x in trials])
    se = texts_of(tok, [x["src_E"] for x in trials])
    L = np.array([len(tok(x, add_special_tokens=False).input_ids) for x in tt]); L2 = np.array([len(tok(x, add_special_tokens=False).input_ids) for x in ts])
    keep = np.where(L == L2)[0]
    res = {k: np.zeros(len(trials)) for k in ("den", "sem_clean") }
    for nm in subs:
        res[f"sem|{nm}"] = np.zeros(len(trials)); res[f"tok|{nm}"] = np.zeros(len(trials))
    res["tok_clean"] = np.zeros(len(trials))
    for ln in np.unique(L[keep]):
        idxs = [i for i in keep if L[i] == ln]
        for s in range(0, len(idxs), bs):
            ii = idxs[s:s + bs]
            ids_t, m_t, p_t = encode(tok, [tt[i] for i in ii], "cuda"); ids_s, m_s, p_s = encode(tok, [ts[i] for i in ii], "cuda")
            ids_e, m_e, p_e = encode(tok, [se[i] for i in ii], "cuda")
            sink = []
            with capture_residual(model, hook, sink):
                last_logits(model, ids_s, m_s, p_s)
            src_h = sink[0]
            lg_t = last_logits(model, ids_t, m_t, p_t); lg_e = last_logits(model, ids_e, m_e, p_e)
            def pick(lg, toks_a, toks_b):
                r_ = torch.arange(len(ii), device=lg.device)
                return (lg[r_, torch.tensor(toks_a, device=lg.device)] - lg[r_, torch.tensor(toks_b, device=lg.device)]).cpu().numpy()
            cE_s = [trials[i]["tgt"].cand_ids[trials[i]["src"].gold] for i in ii]   # E-token of the SOURCE class
            cE_t = [trials[i]["tgt"].cand_ids[trials[i]["tgt"].gold] for i in ii]
            t_s = [trials[i]["src"].cand_ids[trials[i]["src"].gold] for i in ii]    # token the source format answers with
            res["den"][ii] = pick(lg_e, cE_s, cE_t) - pick(lg_t, cE_s, cE_t)
            res["sem_clean"][ii] = pick(lg_t, cE_s, cE_t); res["tok_clean"][ii] = pick(lg_t, t_s, cE_t)
            for nm, U in subs.items():
                with residual_patch(model, hook, U, src_h, "all", False):
                    lg = last_logits(model, ids_t, m_t, p_t)
                res[f"sem|{nm}"][ii] = pick(lg, cE_s, cE_t); res[f"tok|{nm}"][ii] = pick(lg, t_s, cE_t)
    res["keep"] = keep
    return res


for t in a.tasks:
    h, r = sel[t]["choice"]["hook"], sel[t]["choice"]["rank"]
    cfg["mvs"]["residual_index"] = h
    ref = get_reference(model, tok, cfg, rev, "cuda")
    mu = ref["mu"].cuda()
    data = build_1b(t, tok)
    fmts = FORMATS_1B[t]
    out = dict(task=t, hook=h, rank=r, formats=fmts)
    Us = {F: {s: load_U(t, F, h, r, s) for s in SEEDS} for F in fmts}
    # (3) joint fits first (needed in matrix)
    for s in SEEDS:
        f = R / "armC" / f"{t}_joint_h{h}_r{r}_s{s}.pt"
        if not f.exists():
            torch.manual_seed(s)
            tr = [it for F in fmts for it in data[F]["train"]]
            np.random.default_rng(s).shuffle(tr)
            U, summ = fit_b(model, tok, tr, ref["mu"], init_V(ref["Qm"], r, s, "onmanifold"), h, seed=s)
            torch.save(U.cpu(), f); print(t, "joint fit", s, "trainKL %.4f" % summ["kl_last5_median"], flush=True)
    Us["joint"] = {s: torch.load(R / "armC" / f"{t}_joint_h{h}_r{r}_s{s}.pt").cuda() for s in SEEDS}
    # (1) transfer matrix
    matrix = {}
    for F in fmts + ["joint"]:
        for E in fmts:
            fj = R / "armC" / f"{t}_{F}_to_{E}.json"
            if fj.exists():
                matrix[f"{F}->{E}"] = json.load(open(fj)); continue
            if F == E:
                g = json.load(open(R / "final" / f"{t}_{E}.json"))["gates"]
            else:
                dE = data[E]
                intact = run_stats(model, tok, dE["test"])
                z = eval_confirm(model, tok, dE["test"], dE["pairs"], h, Us[F], mu, intact, r)
                g = cell_gates(z, SEEDS, nb=150)
            json.dump(g, open(fj, "w"), indent=1, default=float)
            matrix[f"{F}->{E}"] = g
            print(t, F, "->", E, "KL %.4f rel %.2f abs %.1f transfer %.2f | S %s N %s T %s" % (g["median"]["kl"], g["median"]["rel"], g["median"]["abs"], g["median"].get("transfer", float("nan")),
                  g["S"]["verdict"], g["N"]["verdict"], g.get("T", {}).get("verdict")), flush=True)
    # (2) principal angles
    allU = [(F, s, Us[F][s]) for F in fmts for s in SEEDS]
    within = [principal_cosines(u1, u2) for F in fmts for (F1, s1, u1), (F2, s2, u2) in itertools.combinations([x for x in allU if x[0] == F], 2)]
    cross = [principal_cosines(u1, u2) for (F1, s1, u1), (F2, s2, u2) in itertools.combinations(allU, 2) if F1 != F2]
    haar = [principal_cosines(haar_subspace(4096, r, 7000 + 2 * k), haar_subspace(4096, r, 7001 + 2 * k)) for k in range(200)]
    summ_ = lambda L: dict(n=len(L), overlap_mean=float(np.mean([(c ** 2).mean() for c in L])), top_cos_mean=float(np.mean([c.max() for c in L])),
                          n_cos_above_0p7_mean=float(np.mean([(c > 0.7).sum() for c in L])), frac_pairs_with_any_above_0p7=float(np.mean([(c > 0.7).any() for c in L])),
                          mean_cos_sorted=np.mean(np.sort(np.stack(L), 1)[:, ::-1], 0).tolist())
    out["angles"] = dict(within_format_cross_seed=summ_(within), cross_format=summ_(cross), haar_null=summ_(haar))
    out["angles"]["per_format_pair"] = {f"{F1}|{F2}": summ_([principal_cosines(Us[F1][s1], Us[F2][s2]) for s1 in SEEDS for s2 in SEEDS if (F1 != F2 or s1 < s2)])
                                        for F1, F2 in itertools.combinations_with_replacement(fmts, 2)}
    print(t, "angles within", out["angles"]["within_format_cross_seed"]["overlap_mean"], "cross", out["angles"]["cross_format"]["overlap_mean"], "haar", out["angles"]["haar_null"]["overlap_mean"], flush=True)
    # (4) cross-format transplantation (numeric formats with equal prompt length)
    num = [f for f in fmts if f.startswith("num")]
    ct = {}
    for F, E in itertools.permutations(num, 2):
        pe, pf = data[E]["pairs"], data[F]["pairs"]
        bodyF = {body(x): x for pr in pf for x in pr}
        trials = []
        for ia, ib in pe:
            for tg, so in ((ia, ib), (ib, ia)):
                srcF_opp, srcF_same = bodyF.get(body(so)), bodyF.get(body(tg))
                if srcF_opp is not None and srcF_same is not None and srcF_opp.gold == so.gold:
                    trials.append(dict(tgt=tg, src=srcF_opp, src_E=so, kind="opp")); trials.append(dict(tgt=tg, src=srcF_same, src_E=tg, kind="same"))
        subs = {f"s{s}": Us[F][s] for s in SEEDS}
        for k in range(10):
            subs[f"rand{k}"] = haar_subspace(4096, r, 1000 + k)
        res = cross_transplant(trials, h, subs)
        kinds = np.array([x["kind"] for x in trials]); keep = np.zeros(len(trials), bool); keep[res["keep"]] = True
        mo, ms = keep & (kinds == "opp"), keep & (kinds == "same")
        den_o = res["den"][mo].sum()
        cell = dict(n_trials_total=len(trials), n_opp=int(mo.sum()), n_same=int(ms.sum()), mean_den_opp=float(res["den"][mo].mean()) if mo.any() else None)
        if mo.any():
            cell["semantic_transfer"] = {nm: float((res[f"sem|{nm}"][mo] - res["sem_clean"][mo]).sum() / den_o) for nm in subs}
        if ms.any() and mo.any():
            cell["token_flip"] = {nm: float((res[f"tok|{nm}"][ms] - res["tok_clean"][ms]).sum() / den_o * (mo.sum() / ms.sum())) for nm in subs}
        ct[f"{F}->{E}"] = cell
        print(t, "cross-transplant", F, "->", E, {k: ({n: round(v, 3) for n, v in x.items() if n.startswith("s")} if isinstance(x, dict) else x) for k, x in cell.items()}, flush=True)
    out["cross_transplant"] = ct
    json.dump(out, open(R / "armC" / f"{t}.json", "w"), indent=1, default=float)
print("ARM C DONE", flush=True)
