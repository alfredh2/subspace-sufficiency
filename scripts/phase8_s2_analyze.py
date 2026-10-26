"""Phase 8 S2 step 3: mechanical application of the frozen criteria (docs/phase8_s2_protocol.md) to the saved confirm arrays. No model needed."""
import argparse, json, pickle
import numpy as np
from causal_subspaces.phase7.stage1 import class_gap
from causal_subspaces.phase8.s2lib import R2, KS, N_DRAW, NOISE_FLOOR

ap = argparse.ArgumentParser(); ap.add_argument("--hook", type=int, default=25); ap.add_argument("--nb", type=int, default=400)
a = ap.parse_args(); h = a.hook
S = pickle.load(open(R2 / f"conf_h{h}_shared.pkl", "rb")); M = json.load(open(R2 / f"match_h{h}.json"))
chains = {}
for p in sorted(R2.glob(f"conf_h{h}_s[0-9].pkl")):
    chains[int(p.stem.split("_s")[1])] = pickle.load(open(p, "rb"))
sids = sorted(chains)
meta = S["meta"]; gold, groups, tgroups, cl_int = meta["gold"], meta["groups"], meta["tgroups"], meta["cl_int"]
ks = meta["ks"]; KC = max(len(M["targets"][str(s)]["singles"]) for s in sids)
ug = np.unique(groups); mem_i = [np.where(groups == g)[0] for g in ug]; mem_t = [np.where(tgroups == g)[0] for g in ug]
trial_names = lambda C: C["tr"]


def te(cl, ix):
    G = class_gap(cl_int[ix], gold[ix]); g = class_gap(cl[ix], gold[ix]); return (G - g) / G, G - g


def Gint(ix):
    return class_gap(cl_int[ix], gold[ix])


def T(res, nm, it):
    return float((res[f"m|{nm}|all"][it] - res["m_tgt"][it]).sum() / (res["m_src"][it] - res["m_tgt"][it]).sum())


def ctrl_max(C, k, ix, kind="te"):
    vals = {}
    for fam in ("haar", "onm"):
        vals[fam] = [te(S["abl"][(fam, k, j)], ix)[0] for j in range(N_DRAW[k])]
    for fam in ("energy", "disp"):
        vals[fam] = [te(C["abl"][f"{fam}|{k}|{j}"], ix)[0] for j in range(N_DRAW[k])]
    return vals


def chain_stats(s, ix, it):
    C = chains[s]; K = len(M["targets"][str(s)]["singles"]); out = {}
    TEk = {k: te(C["abl"][key_of_single(k)], ix) for k in range(1, K + 1)}
    out["R1_value"] = float(np.median([TEk[k][0] for k in TEk]))
    G = Gint(ix); best_jcd, best_r2 = -9, -9
    cum = {}
    for k in range(2, K + 1):
        cum[k] = te(C["abl"]["+".join(str(i) for i in range(1, k + 1))], ix)
    cum[1] = TEk[1]
    for k in (2, 3, 4, 6, 8):
        if k > K:
            continue
        rel, ab = cum[k]
        cv = ctrl_max(C, k, ix); mx = max(max(v) for v in cv.values())
        m1, m2, m3 = rel - 0.30, (ab - 10) / G, rel - mx - 0.15
        A = rel - sum(TEk[i][0] for i in range(1, k + 1))
        best_r2 = max(best_r2, min(m1, m2, m3)); best_jcd = max(best_jcd, min(m1, m2, m3, A - 0.15))
    out["R2"], out["JCD"] = best_r2, best_jcd
    out["NJ_K"] = cum[K][0]
    # R4
    tr = C["tr"]; best = -9
    for k in (2, 4, 8):
        if k > K:
            continue
        Tu = T(tr, f"u{k}", it); Ts = max(T(tr, f"c{i}", it) for i in range(1, k + 1))
        mc = -9
        cands = []
        for fam in ("haar", "onm"):
            cands += [T(S["tr"], f"{fam}|{k}|{j}", it) for j in range(6)]
        for fam in ("energy", "disp"):
            cands += [T(tr, f"{fam}|{k}|{j}", it) for j in range(6)]
        best = max(best, min(Tu - 0.25, 0.15 - Ts, 0.05 - max(cands)))
    out["R4"] = best; out["TJ_K"] = T(tr, f"u{K}", it) if K in (2, 4, 8) else T(tr, "u8", it)
    # R5
    best = -9
    for A in range(1, min(4, K) + 1):
        for B in range(A + 1, min(4, K) + 1):
            tab = te(C["abl"][f"{A}+{B}"], ix)[0]
            exc = tab - TEk[A][0] - TEk[B][0]
            cgBA, cgAB = tab - TEk[A][0], tab - TEk[B][0]
            mBA = max(te(C["cond"][(A, B, fam, j)], ix)[0] - TEk[A][0] for fam in ("haar", "energy", "disp") for j in range(6))
            mAB = max(te(C["cond"][(B, A, fam, j)], ix)[0] - TEk[B][0] for fam in ("haar", "energy", "disp") for j in range(6))
            best = max(best, min(exc - 0.10, cgBA - mBA, cgAB - mAB))
    out["R5"] = best
    return out


def key_of_single(k):
    return str(k)


def point(s):
    return chain_stats(s, np.arange(len(gold)), np.arange(len(tgroups)))


def boot(nb, seed=0):
    rng = np.random.default_rng(seed); vals = {k: [] for k in ("R1_value", "R2", "JCD", "R4", "R5", "NJ_K", "TJ_K")}
    for _ in range(nb):
        ps = [sids[i] for i in rng.integers(0, len(sids), len(sids))]; per = []
        for s in ps:
            pick = rng.integers(0, len(ug), len(ug))
            ix = np.concatenate([mem_i[i] for i in pick]); it = np.concatenate([mem_t[i] for i in pick])
            per.append(chain_stats(s, ix, it))
        for k in vals:
            vals[k].append(np.median([p[k] for p in per]))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


pt = {s: point(s) for s in sids}
bo = boot(a.nb)
med = lambda k: float(np.median([pt[s][k] for s in sids]))


def verdict(supp, med_ok, bound_ok):
    return "PASS" if supp and bound_ok else ("FAIL" if (not supp) and (not med_ok) else "INCONCLUSIVE")


res = dict(hook=h, chains=sids, per_chain={s: pt[s] for s in sids}, ci=bo, criteria={})
for name, need, key in (("R2", 3, "R2"), ("JCD", 3, "JCD"), ("R4", 4, "R4"), ("R5", 3, "R5")):
    n_ok = sum(pt[s][key] >= 0 for s in sids)
    res["criteria"][name] = dict(n_chains_ok=n_ok, median=med(key), ci=bo[key], verdict=verdict(n_ok >= need, med(key) >= 0, bo[key][0] >= 0))
n_ok = sum(pt[s]["R1_value"] <= 0.10 for s in sids)
res["criteria"]["R1"] = dict(n_chains_ok=n_ok, median=med("R1_value"), ci=bo["R1_value"], verdict=verdict(n_ok >= 3, med("R1_value") <= 0.10, bo["R1_value"][1] <= 0.15))
haar1 = np.mean([te(S["abl"][("haar", 1, j)], np.arange(len(gold)))[0] for j in range(N_DRAW[1])])
res["criteria"]["R1"]["haar_rank4_TE_mean"] = float(haar1)
if haar1 > 0.05:
    res["criteria"]["R1"]["verdict"] = "INCONCLUSIVE"
asr_n = sum(pt[s]["NJ_K"] < 0.10 and pt[s]["TJ_K"] < 0.15 for s in sids)
res["ASR"] = dict(n_chains=asr_n, holds=bool(asr_n >= 3))
V = {k: v["verdict"] for k, v in res["criteria"].items()}
if V["JCD"] == "PASS" and V["R1"] == "PASS" and V["R4"] == "PASS" and V["R5"] == "PASS":
    res["category"] = "SCR"
elif res["ASR"]["holds"]:
    res["category"] = "ASR"
elif V["JCD"] == "PASS":
    res["category"] = "JCD only (" + ", ".join(f"{k}={V[k]}" for k in ("R1", "R4", "R5")) + ")"
elif any(v == "INCONCLUSIVE" for v in V.values()):
    res["category"] = "INCONCLUSIVE (" + ", ".join(f"{k}={v}" for k, v in V.items()) + ")"
else:
    res["category"] = "MIXED (" + ", ".join(f"{k}={v}" for k, v in V.items()) + ")"
print(json.dumps(dict(criteria=res["criteria"], ASR=res["ASR"], category=res["category"]), indent=1, default=float))

# ---------------------------------------------------------------- tables (descriptive, hierarchical bootstrap bands)
def curve_ci(fn, nb=a.nb, seed=1):
    rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        ps = [sids[i] for i in rng.integers(0, len(sids), len(sids))]; v = []
        for s in ps:
            pick = rng.integers(0, len(ug), len(ug)); v.append(fn(s, np.concatenate([mem_i[i] for i in pick]), np.concatenate([mem_t[i] for i in pick])))
        out.append(np.mean(v, 0))
    out = np.array(out); return np.percentile(out, 2.5, 0).tolist(), np.percentile(out, 97.5, 0).tolist()


allix, allit = np.arange(len(gold)), np.arange(len(tgroups))
K8 = KC
cum_fn = lambda s, ix, it: np.array([te(chains[s]["abl"]["1" if k == 1 else "+".join(str(i) for i in range(1, k + 1))], ix)[0] for k in range(1, K8 + 1)])
cum_pt = np.array([cum_fn(s, allix, allit) for s in sids])
lo, hi = curve_ci(cum_fn)
res["cum_curve"] = dict(k=list(range(1, K8 + 1)), mean_over_chains=cum_pt.mean(0).tolist(), per_chain=cum_pt.tolist(), ci_lo=lo, ci_hi=hi)
cum_abs = np.array([[te(chains[s]["abl"]["1" if k == 1 else "+".join(str(i) for i in range(1, k + 1))], allix)[1] for k in range(1, K8 + 1)] for s in sids])
res["cum_curve"]["abs_logits_mean"] = cum_abs.mean(0).tolist()
single_pt = np.array([[te(chains[s]["abl"][str(k)], allix)[0] for k in range(1, K8 + 1)] for s in sids])
res["singles"] = dict(per_chain=single_pt.tolist(), mean_over_chains=single_pt.mean(0).tolist(), ci=curve_ci(lambda s, ix, it: np.array([te(chains[s]["abl"][str(k)], ix)[0] for k in range(1, K8 + 1)])))
# controls by family and k: mean and max TE over draws, and (from dev matching) achieved energy / displacement / perturbation ratios
ctab = {}
for k in ks:
    row = {}
    for fam in ("haar", "onm", "energy", "disp"):
        v = []
        for s in sids:
            C = chains[s]
            arr = [te(S["abl"][(fam, k, j)] if fam in ("haar", "onm") else C["abl"][f"{fam}|{k}|{j}"], allix)[0] for j in range(N_DRAW[k])]
            v.append(arr)
        v = np.array(v)
        row[fam] = dict(mean=float(v.mean()), max=float(v.max()), min=float(v.min()))
        if fam in ("energy", "disp"):
            row[fam]["ratio_E_mean"] = float(np.mean([np.mean(M["sel"][f"{fam}|{k}|{s}"]["ratio_E"]) for s in sids]))
            row[fam]["ratio_D_mean"] = float(np.mean([np.mean(M["sel"][f"{fam}|{k}|{s}"]["ratio_D"]) for s in sids]))
            row[fam]["ratio_P_mean"] = float(np.mean([np.mean(M["sel"][f"{fam}|{k}|{s}"]["ratio_P"]) for s in sids]))
            row[fam]["n_unmatched_chains"] = int(sum(M["sel"][f"{fam}|{k}|{s}"]["flag"] == "unmatched" for s in sids))
    row["carrier_E"] = float(np.mean([M["targets"][str(s)][str(k)]["E"] for s in sids])); row["carrier_D"] = float(np.mean([M["targets"][str(s)][str(k)]["D"] for s in sids]))
    row["carrier_P"] = float(np.mean([M["targets"][str(s)][str(k)]["P"] for s in sids]))
    ctab[k] = row
res["controls_by_k"] = ctab
# transplants
trt = {}
for s in sids:
    tr = chains[s]["tr"]; K = len(M["targets"][str(s)]["singles"])
    trt[s] = dict(single=[T(tr, f"c{k}", allit) for k in range(1, K + 1)], union={k: T(tr, f"u{k}", allit) for k in (2, 4, 8) if k <= K},
                  ctrl={f"{fam}|{k}": [T(S["tr"] if fam in ("haar", "onm") else tr, f"{fam}|{k}|{j}", allit) for j in range(6)] for fam in ("haar", "onm", "energy", "disp") for k in (1, 2, 4, 8) if k in ks})
res["transplant"] = trt
# conditional ablation table (pairs among 1..4)
cond = {}
for s in sids:
    C = chains[s]; rows = {}
    for A in range(1, 5):
        for B in range(A + 1, 5):
            tA, tB, tAB = (te(C["abl"][str(A)], allix)[0], te(C["abl"][str(B)], allix)[0], te(C["abl"][f"{A}+{B}"], allix)[0])
            cm = {fam: max(te(C["cond"][(A, B, fam, j)], allix)[0] - tA for j in range(6)) for fam in ("haar", "energy", "disp")}
            cm2 = {fam: max(te(C["cond"][(B, A, fam, j)], allix)[0] - tB for j in range(6)) for fam in ("haar", "energy", "disp")}
            rows[f"{A},{B}"] = dict(TE_A=tA, TE_B=tB, TE_AB=tAB, excess=tAB - tA - tB, CG_B_given_A=tAB - tA, ctrl_CG_B_given_A_max=cm, CG_A_given_B=tAB - tB, ctrl_CG_A_given_B_max=cm2)
    cond[s] = rows
res["conditional"] = cond
# conditional transplant
ctr = {}
for s in sids:
    r = chains[s]["ctr"]; tr = chains[s]["tr"]; den = (tr["m_src"] - tr["m_tgt"]); rows = {}
    for key, v in r.items():
        if key[0] == "AB":
            rows[f"{key[1]}->{key[2]}"] = float((v - r[("A", key[1])]).sum() / den.sum())
    ctr[s] = rows
res["conditional_transplant"] = ctr
# effect vector cosines between carriers (descriptive)
ev = {}
for s in sids:
    K = len(M["targets"][str(s)]["singles"]); E = {k: chains[s]["abl"][str(k)] - cl_int for k in range(1, K + 1)}; cs = {}
    for i in range(1, K + 1):
        for j in range(i + 1, K + 1):
            m = (np.linalg.norm(E[i], axis=1) > 2 * NOISE_FLOOR) & (np.linalg.norm(E[j], axis=1) > 2 * NOISE_FLOOR)
            cs[f"{i}-{j}"] = float(np.mean((E[i][m] * E[j][m]).sum(1) / (np.linalg.norm(E[i][m], axis=1) * np.linalg.norm(E[j][m], axis=1)))) if m.sum() > 10 else None
    ev[s] = cs
res["effect_vector_cosines"] = ev
json.dump(res, open(R2 / f"s2_results_h{h}.json", "w"), indent=1, default=float)
print("cum curve mean", np.round(cum_pt.mean(0), 3).tolist(), "ci_lo", np.round(lo, 3).tolist(), "ci_hi", np.round(hi, 3).tolist())
print("singles mean", np.round(single_pt.mean(0), 3).tolist())
for k in ks:
    r = ctab[k]; print("k", k, "carrier E %.3f D %.0f |" % (r["carrier_E"], r["carrier_D"]), {f: (round(r[f]["mean"], 3), round(r[f]["max"], 3)) for f in ("haar", "onm", "energy", "disp")})
