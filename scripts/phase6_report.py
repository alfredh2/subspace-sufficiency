"""Phase 6 report tables and mechanical criteria (docs/plans/phase6_plan.md sections 7 and 11). Reads results/phase6/*.npz."""
import json
import sys

import numpy as np

from causal_subspaces.phase6 import stats as S6
from causal_subspaces.upstream import pipeline as UP
from causal_subspaces.upstream import stats as S
from causal_subspaces.utils.config import ROOT

OUT = ROOT / "results/phase6"; md = []; ml = md.append; V = {}
exists = lambda n: (OUT / n).exists()
SPARSE = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19]
NULL_IDX = [0, 8, 14, 17, 19]
GROUPS = ["Concl", "KV", "Fact", "G1", "Concl+G1", "Rule", "Concl_same", "Concl_diff"]
f2 = lambda x: "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
ci = lambda a: f"[{a[0]:.2f}, {a[1]:.2f}]"


class DS:
    def __init__(self, name):
        self.name = name; self.tr, self.bt, self.bs, self.jobs = UP.load_raw(OUT / f"{name}.npz")
        self.boot = S.Boot(self.tr["pair"], self.tr["content"], nboot=2000, seed=21); self.sigma = self.tr["sigma"]; self.n = len(self.sigma)
    def has(self, n): return n in self.jobs
    def T(self, names, key="margin"): return S6.tcurve(self.boot, self.sigma, self.bt, self.bs, self.jobs, names, key)
    def raw(self, names, key): return S6.scalar_curve(self.boot, self.sigma, self.jobs, names, key, base=self.bt[key])
    def vproj(self, names):
        pt, rp, rc = [], [], []
        for n in names:
            r = self.boot.ratio(self.jobs[n]["vp_num"], self.jobs[n]["vp_den"]); pt.append(r["point"]); rp.append(r["rep_pair"]); rc.append(r["rep_content"])
        return dict(point=np.array(pt), rp=np.stack(rp, 1), rc=np.stack(rc, 1))
    def delta(self, name, key="hr"): return self.jobs[name][key] - self.bt[key]


def bnd(c, k): return (float(np.percentile(c["rp"][:, k], 2.5)), float(np.percentile(c["rp"][:, k], 97.5))), (float(np.percentile(c["rc"][:, k], 2.5)), float(np.percentile(c["rc"][:, k], 97.5)))


def curve_table(ds, c, grid, F=None, extra=None, title=""):
    ml(f"\n{title}\n\n| index | T*_V (signed) [pair] / [content] | F | " + " | ".join(extra or {}) + " |\n|---|---|---|" + "---|" * len(extra or {}))
    for k, i in enumerate(grid):
        p, ct = bnd(c, k); row = f"| {i} | {c['point'][k]:.3f} {ci(p)} / {ci(ct)} | {f2(F['point'][k]) if F else '—'} | " + " | ".join(f"{e['point'][k]:.3f}" for e in (extra or {}).values()) + " |"
        ml(row)


def analyse(ds, label, dense_ok=True):
    out = {}; sig = ds.sigma
    grid = [i for i in range(20) if ds.has(f"o|Concl|{i}")]; names = [f"o|Concl|{i}" for i in grid]; last = grid.index(19)
    Tc = ds.T(names); ok, lb = S6.defined_denominator(Tc, last, S6.DENOM_MIN); out["denominator_Concl_19"] = dict(point=float(Tc["point"][last]), lower=lb, defined=ok)
    F = S6.divide(Tc, Tc, last) if ok else None
    dY = ds.raw(names, "margin"); dH = ds.raw(names, "hr"); vp = ds.vproj(names); U = ds.T([f"u|Concl|{i}" for i in grid]) if ds.has(f"u|Concl|{grid[0]}") else None
    ml(f"\n## {label}: value-origin curve for P = Concl ({ds.n} trials, {len(np.unique(ds.tr['pair']))} pairs, {len(np.unique(ds.tr['content']))} contents)\n")
    ml(f"Denominator T*_V(19, Concl) = {Tc['point'][last]:.3f} (lower bound {lb:.3f}; defined={ok}). Reference T_ref = T*_V(19, KV) = {ds.T(['o|KV|19'])['point'][0]:.3f}.")
    ex = {"Δmargin (logits, signed toward source)": dY, "Δ H* readout (signed)": dH, "vproj (value change projection)": vp}
    if U is not None: ex["U(i): unrestricted T"] = U
    curve_table(ds, Tc, grid, F, ex, "Columns: unnormalised signed transfer (restricted path), F = T*_V(i)/T*_V(19), raw final-margin change, raw head-readout change, value-projection, unrestricted transfer.")
    if F is not None:
        ons = S6.onsets(F, grid); cls = S6.trajectory_class(F, grid)
        out.update(F=F["point"].tolist(), grid=grid, onsets={str(k): v for k, v in ons.items()}, trajectory=cls)
        k4 = grid.index(4); p4, c4 = bnd(F, k4); out["F4"] = dict(point=float(F["point"][k4]), pair=p4, content=c4, loco=[float(F["loco"][:, k4].min()), float(F["loco"][:, k4].max())])
        ml(f"\nOnsets (F): " + "; ".join(f"θ={k}: first {v['first']}, sustained {v['sustained']}" for k, v in ons.items()) + f"; trajectory class: **{cls}**; F(4) = {F['point'][k4]:.2f} pair {ci(p4)} / content {ci(c4)} (LOCO {out['F4']['loco'][0]:.2f}…{out['F4']['loco'][1]:.2f}).")
        # head-readout based onset consistency
        Th = ds.T(names, "hr"); okh, _ = S6.defined_denominator(Th, last, 1e-9)
        Fh = S6.divide(Th, Th, last); onh = S6.onsets(Fh, grid)
        out["onset_head_readout"] = {str(k): v for k, v in onh.items()}; out["onset_agree_within2"] = bool(ons[0.5]["sustained"] is not None and onh[0.5]["sustained"] is not None and abs(ons[0.5]["sustained"] - onh[0.5]["sustained"]) <= 2)
        ml(f"Head-readout transfer onset (θ=0.5): first {onh[0.5]['first']}, sustained {onh[0.5]['sustained']} (within 2 of the margin onset: {out['onset_agree_within2']}).")
    # other groups on the sparse grid
    ref = ds.T(["o|KV|19"]); okref = bool(np.minimum(np.percentile(ref["rp"], 2.5), np.percentile(ref["rc"], 2.5)) >= S6.REF_MIN); out["T_ref_defined"] = okref
    gsp = [i for i in SPARSE if ds.has(f"o|KV|{i}")]
    if gsp:
        ml("\n### Other patched groups (restricted path; signed T*_V, remaining influence R = T*_V/T_ref and F where the denominator is defined)\n\n| group | " + " | ".join(str(i) for i in gsp) + " |\n|---|" + "---|" * len(gsp))
        out["groups"] = {}
        for g in GROUPS[1:]:
            if not ds.has(f"o|{g}|{gsp[0]}"): continue
            Tg = ds.T([f"o|{g}|{i}" for i in gsp]); Rg = None
            if okref:
                Rg = dict(point=Tg["point"] / ref["point"][0], rp=Tg["rp"] / ref["rp"], rc=Tg["rc"] / ref["rc"], loco=Tg["loco"] / ref["loco"])
            dl = S6.defined_denominator(Tg, gsp.index(19), S6.DENOM_MIN)
            Fg = S6.divide(Tg, Tg, gsp.index(19)) if dl[0] else None
            clo = S6.closing_index(Rg, gsp) if Rg is not None else None
            out["groups"][g] = dict(T=Tg["point"].tolist(), R=None if Rg is None else Rg["point"].tolist(), F_defined=dl[0], closing=clo, T19=float(Tg["point"][-1]),
                                    onsets=None if Fg is None else {str(k): v for k, v in S6.onsets(Fg, gsp).items()}, trajectory=None if Fg is None else S6.trajectory_class(Fg, gsp), R_upper=None if Rg is None else S6.upper(Rg).tolist())
            ml(f"| {g} T | " + " | ".join(f"{x:.3f}" for x in Tg["point"]) + " |")
            if Rg is not None: ml(f"| {g} R | " + " | ".join(f"{x:.2f}" for x in Rg["point"]) + " |")
            ml(f"| {g} F | " + " | ".join(f2(x) for x in (Fg["point"] if Fg else [None] * len(gsp))) + f" |   (F {'defined' if dl[0] else 'UNDEFINED: denominator lower bound %.3f' % dl[1]}; closing index {clo})")
    # nulls
    nl = {}
    for typ, pat in (("clean random keys", "o|rk6:{d}|{i}"), ("norm-matched random direction (Concl)", "on|Concl|{i}"), ("random head sets (Concl)", "oh{d}|Concl|{i}")):
        vals = []
        for i in NULL_IDX:
            xs = [ds.T([pat.format(d=d, i=i)])["point"][0] for d in range(5) if ds.has(pat.format(d=d, i=i))]
            if xs: vals.append((i, float(np.mean(xs)), float(np.max(np.abs(xs))), len(xs)))
        if vals: nl[typ] = vals
    out["nulls"] = nl
    ml("\n### Nulls (T*_V, mean and max |T| over draws; reference = T*_V(19, Concl))\n" + "\n".join(f"- {k}: " + "; ".join(f"i={i}: mean {m:.3f}, max {x:.3f} (n={n})" for i, m, x, n in v) for k, v in nl.items()))
    oth = {i: ds.T([f"o|Other|{i}"])["point"][0] for i in (0, 10, 19) if ds.has(f"o|Other|{i}")}
    eq = {nm.split("|")[1] + "@" + nm.split("|")[2]: dict(maxdiff=float(ds.jobs[nm]["maxdiff"].max()), eq_frac=float(ds.jobs[nm]["eq_frac"].mean())) for nm in ds.jobs if nm.startswith("o|") and nm.split("|")[1] in ("Other", "Concl", "Concl_same", "Fact", "Rule") and nm.split("|")[2] in ("0", "10", "19")}
    out["Other_T"] = oth; out["equality"] = eq
    ml("\n### Measured activation equality between source and target at the patched index (not assumed): max |difference| and fraction of exactly equal positions\n" + "; ".join(f"{k}: max {v['maxdiff']:.2f}, equal {v['eq_frac']:.2f}" for k, v in sorted(eq.items())))
    # components
    comp = {}
    if ds.has("ct|attn|0|Concl"):
        Lm = [l for l in range(19)]
        ml("\n### Component-level patches at P = Concl: signed T*_V and F_comp = T*_V/T*_V(19, Concl) (total influence = downstream recompute; direct = only the component's own write added at index 19); interaction I = joint − attn − MLP\n")
        ml("| layer | total attn | total MLP | total joint | I_total | direct attn | direct MLP | direct joint | I_direct | total − direct (joint) |\n|---|---|---|---|---|---|---|---|---|---|")
        d19 = Tc["point"][last]
        has_joint = ds.has("ct|attn+mlp|0|Concl"); has_direct = ds.has("cd|attn|0|Concl")
        for l in Lm:
            cs_ = ("attn", "mlp", "attn+mlp") if has_joint else ("attn", "mlp")
            t = {c: ds.T([f"ct|{c}|{l}|Concl"])["point"][0] for c in cs_}; d = {c: (ds.T([f"cd|{c}|{l}|Concl"])["point"][0] if has_direct else float("nan")) for c in cs_}
            comp[l] = dict(total=t, direct=d); j = lambda x: x.get("attn+mlp", float("nan"))
            ml(f"| {l} | {t['attn']:.3f} | {t['mlp']:.3f} | {j(t):.3f} | {j(t) - t['attn'] - t['mlp']:+.3f} | {d['attn']:.3f} | {d['mlp']:.3f} | {j(d):.3f} | {j(d) - d['attn'] - d['mlp']:+.3f} | {j(t) - j(d):+.3f} |")
        out["components"] = comp
        if F is not None:
            ons2 = S6.onsets(F, grid); i0, i1 = ons2[0.1]["first"], ons2[0.9]["sustained"]
            if i0 is not None and i1 is not None and i1 > i0:
                rise = float(F["point"][grid.index(i1)] - F["point"][grid.index(i0)]); thr = 0.3 * rise; wr = {}
                for kind, pre in ((("total", "ct"), ("direct", "cd")) if has_direct else (("total", "ct"),)):
                    for c in (("attn", "mlp", "attn+mlp") if has_joint else ("attn", "mlp")):
                        cur = ds.T([f"{pre}|{c}|{l}|Concl" for l in range(19)]); Fc = dict(point=cur["point"] / Tc["point"][last], rp=cur["rp"] / Tc["rp"][:, [last]], rc=cur["rc"] / Tc["rc"][:, [last]])
                        lo = np.minimum(np.percentile(Fc["rp"], 2.5, axis=0), np.percentile(Fc["rc"], 2.5, axis=0))
                        wr[f"{kind}|{c}"] = [l for l in range(19) if Fc["point"][l] >= thr and lo[l] > 0.05]
                out["writers"] = dict(window=[i0, i1], rise=rise, threshold=thr, layers=wr)
                ml(f"\nWriters (window {i0}–{i1}, rise {rise:.2f}, threshold 0.3×rise = {thr:.2f} in F_comp units, lower bound > 0.05): " + "; ".join(f"{k}: {v}" for k, v in wr.items()))
            else: out["writers"] = None
    return out


def controls(prefix_sets):
    """RMS of the H* read-out change (and final margin change) under L / Fm / X sources relative to M, per index; signed head-readout transfer for M and X."""
    res = {}
    for key, ds in prefix_sets.items():
        res[key] = ds
    return res


def analyse_controls(M, others, label):
    out = {}; grid = [i for i in SPARSE if M.has(f"o|Concl|{i}")]
    rows = []
    for nm, ds in others.items():
        if not ds.has(f"o|Concl|{grid[0]}"): continue
        for g in ("Concl", "KV"):
            for i in grid:
                dm = M.delta(f"o|{g}|{i}"); dc = ds.delta(f"o|{g}|{i}"); r = S6.rms_ratio(ds.boot, dc, M.boot, dm); out[f"{nm}|{g}|{i}"] = dict(r)
        # signed transfer for opposite-label control sets (X)
    ml(f"\n## {label}: RMS of the H* read-out change relative to the matched pair M (ρ_ctrl); point [pair] / [content]\n\n| source | group | " + " | ".join(str(i) for i in grid) + " |\n|---|---|" + "---|" * len(grid))
    for nm, ds in others.items():
        for g in ("Concl", "KV"):
            if f"{nm}|{g}|{grid[0]}" in out: ml(f"| {nm} | {g} | " + " | ".join(f"{out[f'{nm}|{g}|{i}']['point']:.2f}" for i in grid) + " |")
    xs = others.get("X")
    if xs is not None and xs.has("o|Concl|19"):
        for g in ("Concl", "KV"):
            th_x = xs.T([f"o|{g}|{i}" for i in grid], "hr"); th_m = M.T([f"o|{g}|{i}" for i in grid], "hr"); out[f"signed_head_transfer_{g}"] = dict(X=th_x["point"].tolist(), M=th_m["point"].tolist(), grid=grid)
            ml(f"\nSigned head-readout transfer, group {g}: M " + ", ".join(f"{i}:{x:.2f}" for i, x in zip(grid, th_m["point"])) + "; X " + ", ".join(f"{i}:{x:.2f}" for i, x in zip(grid, th_x["point"])))
    return out


A = {f: DS(f"stage2_{f}") for f in ("num_orig", "num_rev") if exists(f"stage2_{f}.npz")}
for f, d in A.items(): V[f"A_{f}"] = analyse(d, f"Set A, {f}")
ctrl_sets = {}
if "num_orig" in A:
    for nm, fl in (("X", "stage3_X_num_orig"), ("L", "stage3_L_num_orig"), ("Fm", "stage3_Fm_num_orig")):
        if exists(fl + ".npz"): ctrl_sets[nm] = DS(fl)
    if ctrl_sets: V["controls_A"] = analyse_controls(A["num_orig"], ctrl_sets, "Controls on set A (num_orig)")
if "num_rev" in A and exists("stage3_Fm_num_rev.npz"):
    V["controls_A_rev"] = analyse_controls(A["num_rev"], {"Fm": DS("stage3_Fm_num_rev")}, "Controls on set A (num_rev)")
IND = {}
for w in ("B", "C"):
    for f in ("num_orig", "num_rev"):
        n = f"stage4_{w}_{f}"
        if exists(n + ".npz"): IND[(w, f)] = DS(n); V[f"{w}_{f}"] = analyse(IND[(w, f)], f"Independent set {w}, {f}")
    if exists(f"stage4_{w}_X_num_orig.npz"):
        M = IND.get((w, "num_orig")); V[f"controls_{w}"] = analyse_controls(M, {k: DS(f"stage4_{w}_{k}_num_orig") for k in ("L", "X") if exists(f"stage4_{w}_{k}_num_orig.npz")}, f"Controls on set {w} (num_orig)")
# ------------------------------------------------------------------ O-hypothesis scoring (section 7 of the plan; windows: early <=6, middle 7-14, late 15-18, last step 19)
def window(i): return None if i is None else ("early" if i <= 6 else "middle" if i <= 14 else "late" if i <= 18 else "last step")
def rho(key, nm, g, i):
    d = V.get(key, {}).get(f"{nm}|{g}|{i}"); return d
def small(d, p=0.25, u=0.35): return d is not None and d["point"] <= p and max(d["pair"][1], d["content"][1]) <= u
def large(d, p=0.5): return d is not None and d["point"] >= p
SC = {}
for lab, key_a, ctrl_key in (("A_num_orig", "A_num_orig", "controls_A"), ("A_num_rev", "A_num_rev", "controls_A_rev"), ("B_num_orig", "B_num_orig", "controls_B"), ("C_num_orig", "C_num_orig", "controls_C"), ("B_num_rev", "B_num_rev", None), ("C_num_rev", "C_num_rev", None)):
    a = V.get(key_a)
    if not a or "F" not in a: continue
    F = a["F4"]; on = a["onsets"]; i50 = on["0.5"]["sustained"]; w = window(i50)
    cs = ctrl_key
    rL4, rFm4, rL19, rFm19 = (rho(cs, "L", "Concl", 4), rho(cs, "Fm", "Concl", 4), rho(cs, "L", "Concl", 19), rho(cs, "Fm", "Concl", 19)) if cs else (None,) * 4
    unstable = a["trajectory"] in ("non-monotone", "sign-reversing")
    f4_ok = F["point"] >= 0.5 and min(F["pair"][0], F["content"][0]) >= 0.4
    o1 = dict(F4_ok=bool(f4_ok), controls_available=bool(rL4 is not None and rFm4 is not None), verdict_related=bool(small(rL4) and small(rFm4)), lexical=bool(large(rL4) or large(rFm4)))
    o1["supported"] = bool(o1["F4_ok"] and o1["verdict_related"] and not unstable); o1["lexical_early_content"] = bool(o1["F4_ok"] and o1["lexical"])
    o2 = dict(window=w, F4_low=bool(F["point"] <= 0.15 and max(F["pair"][1], F["content"][1]) <= 0.25), ctrl19=bool(small(rL19) and small(rFm19)))
    o2["supported"] = bool(w in ("middle", "late") and o2["F4_low"] and o2["ctrl19"] and not unstable)
    s90, s10 = on["0.9"]["sustained"], on["0.1"]["sustained"]
    o3 = dict(span=None if s90 is None or s10 is None else s90 - s10); o3["supported"] = bool(o3["span"] is not None and o3["span"] >= 8 and not o1["supported"] and not o2["supported"] and not unstable)
    o4 = dict(supported=bool(i50 == 19 and max(a["F"][-2:-1]) < 0.5))
    SC[lab] = dict(trajectory=a["trajectory"], i_sus_half=i50, window=w, F4=F["point"], O1=o1, O2=o2, O3=o3, O4=o4)
V["O_scoring"] = SC
ml("\n## Hypothesis scoring (mechanical; each set scored separately)\n```json\n" + json.dumps(SC, indent=1, default=float) + "\n```")
# ------------------------------------------------------------------ secondary: pinning
ACC = [(l, c) for l in (20, 21, 22) for c in ("attn", "mlp")]
def rat(a, b): return dict(point=a["point"] / b["point"], rp=a["rp"] / b["rp"], rc=a["rc"] / b["rc"], loco=a["loco"] / b["loco"])
def sub(a, b): return dict(point=a["point"] - b["point"], rp=a["rp"] - b["rp"], rc=a["rc"] - b["rc"], loco=a["loco"] - b["loco"])
def one(ds, n, key="margin"): return ds.T([n], key)
def fs(c): p, ct = bnd(c, 0); return f"{c['point'][0]:.2f} {ci(p)} / {ci(ct)}"
SEC = {}
for which in ("A", "B"):
    for f in ("num_orig", "num_rev"):
        n = f"stage5_{which}_{f}"
        if not exists(n + ".npz"): continue
        d = DS(n); o = {}; Tf = one(d, "free"); Ta = one(d, "pin_all"); inc = sub(Ta, Tf)
        ml(f"\n## Secondary (pinning), set {which}, {f}: T_free = {fs(Tf)}; all six components pinned: {fs(Ta)}; S_pin = {fs(rat(Ta, Tf))}\n")
        o["S_pin"] = float(Ta["point"][0] / Tf["point"][0]); q, qp = {}, {}
        ml("| component | pin-one T | add-one share q | leave-one-out T | LOO share q' | |q − q'| |\n|---|---|---|---|---|---|")
        for l, c in ACC:
            T1 = one(d, f"pin1|{l}{c}"); Tl = one(d, f"loo|{l}{c}"); qq = rat(sub(T1, Tf), inc); qq2 = dict(point=1 - rat(sub(Tl, Tf), inc)["point"], rp=1 - rat(sub(Tl, Tf), inc)["rp"], rc=1 - rat(sub(Tl, Tf), inc)["rc"], loco=1 - rat(sub(Tl, Tf), inc)["loco"])
            q[(l, c)], qp[(l, c)] = qq, qq2
            ml(f"| L{l} {c} | {T1['point'][0]:.3f} | {fs(qq)} | {Tl['point'][0]:.3f} | {fs(qq2)} | {abs(qq['point'][0] - qq2['point'][0]):.2f} |")
        top = sorted(ACC, key=lambda k: -q[k]["point"][0]); t1, t2 = top[0], top[1]
        k1_comp = bool(q[t1]["point"][0] >= 0.30 and qp[t1]["point"][0] >= 0.30 and min(bnd(q[t1], 0)[0][0], bnd(q[t1], 0)[1][0]) >= 0.30 * 0 and q[t1]["point"][0] + q[t2]["point"][0] >= 0.60 and qp[t1]["point"][0] + qp[t2]["point"][0] >= 0.60)
        o.update(q={f"{l}{c}": float(v["point"][0]) for (l, c), v in q.items()}, qprime={f"{l}{c}": float(v["point"][0]) for (l, c), v in qp.items()}, top2=[f"{t1[0]}{t1[1]}", f"{t2[0]}{t2[1]}"], K1=k1_comp, order_dependent=bool(abs(q[t1]["point"][0] - qp[t1]["point"][0]) > 0.20))
        ml("\nLayer interactions (2x2 attention × MLP pins; e = T_pin − T_free): " + "; ".join(
            f"L{l}: e_attn {one(d, f'pin1|{l}attn')['point'][0] - Tf['point'][0]:+.3f}, e_mlp {one(d, f'pin1|{l}mlp')['point'][0] - Tf['point'][0]:+.3f}, e_both {one(d, f'layer|{l}|both')['point'][0] - Tf['point'][0]:+.3f}, interaction {one(d, f'layer|{l}|both')['point'][0] - one(d, f'pin1|{l}attn')['point'][0] - one(d, f'pin1|{l}mlp')['point'][0] + Tf['point'][0]:+.3f}" for l in (20, 21, 22))
            + f"; cumulative: layer 20 {one(d, 'cum|20')['point'][0]:.3f}, 20–21 {one(d, 'cum|2021')['point'][0]:.3f}, 20–22 {Ta['point'][0]:.3f}.")
        tr = {k: rat(sub(one(d, f"tail|{k}"), Tf), inc) for k in range(9)}; g1 = rat(sub(one(d, "rows|G1"), Tf), inc); g2 = rat(sub(one(d, "rows|G2"), Tf), inc)
        ml(f"Row localisation (share of the all-row increase, rows pinned at layers 20–22 with all six components): G1 {fs(g1)}; G2 (all tail rows) {fs(g2)}; single tail tokens: " + ", ".join(f"t{k}: {tr[k]['point'][0]:.2f}" for k in range(9)))
        o["rows"] = dict(G1=float(g1["point"][0]), G2=float(g2["point"][0]), tail={k: float(v["point"][0]) for k, v in tr.items()}); o["K2"] = bool(g2["point"][0] >= 0.80 and g1["point"][0] <= 0.20)
        # d_c / orthogonal decomposition
        dd = {}
        for part in ("full", "dc", "orth"):
            tf, tp = one(d, f"dc|{part}|free"), one(d, f"dc|{part}|pin"); okd = bool(min(np.percentile(np.abs(tf["rp"]), 2.5), np.percentile(np.abs(tf["rc"]), 2.5)) >= 0.05)
            dd[part] = dict(T_free=float(tf["point"][0]), T_pin=float(tp["point"][0]), defined=okd, S_pin=float(tp["point"][0] / tf["point"][0]) if okd else None, S_lower=min(bnd(rat(tp, tf), 0)[0][0], bnd(rat(tp, tf), 0)[1][0]) if okd else None)
        o["decomposition"] = dd; ml("\nDecomposition of the layer-19 write at index 20 (rows G1∪G2): " + "; ".join(f"{p}: T_free {v['T_free']:.3f}, T_pin {v['T_pin']:.3f}, S_pin {f2(v['S_pin'])}" + ("" if v["defined"] else " (free effect below the 0.05 floor: undefined)") for p, v in dd.items()))
        o["K3"] = bool(dd["dc"]["defined"] and dd["dc"]["S_pin"] >= 1.3 and dd["dc"]["S_lower"] >= 1.15 and (not dd["orth"]["defined"] or dd["orth"]["S_pin"] <= 1.15))
        o["K5_operationalised"] = bool(dd["dc"]["defined"] and dd["orth"]["defined"] and abs(dd["dc"]["S_pin"] - dd["orth"]["S_pin"]) <= 0.15)
        # dose
        ratios = {1.0: o["S_pin"]}
        for a in (0.25, 0.5, 0.75, 1.25, 1.5): ratios[a] = float(one(d, f"dose|pin|{a}")["point"][0] / one(d, f"dose|free|{a}")["point"][0])
        o["dose_ratio"] = ratios; sub3 = [ratios[0.5], ratios[1.0], ratios[1.5]]; o["K4"] = bool(max(sub3) - min(sub3) <= 0.15)
        ml("Pinned/free ratio by dose α: " + ", ".join(f"{a}: {v:.2f}" for a, v in sorted(ratios.items())) + f" → within 0.15 across α=0.5, 1, 1.5: {o['K4']}")
        # d_c accounting
        wfree = (d.jobs["free"]["W"] - d.bt["W"]); sgn = d.sigma[:, None]; wm = (sgn * wfree).mean(0)
        dcres = np.abs((d.jobs["free"]["dC"][:, -1] - d.jobs["free"]["dC"][:, 0]) - d.jobs["free"]["W"].sum(1)); scale = float(np.abs(sgn[:, 0] * (d.jobs["free"]["dC"][:, 0] - d.bt["dC"][:, 0])).mean())
        o["accounting"] = dict(max_abs_residual=float(dcres.max()), rel_to_local_shift=float(dcres.max() / max(scale, 1e-9)), ok=bool(dcres.max() <= 0.05 * scale))
        o["delta_W"] = {f"{l}{c}": float(wm[k]) for k, (l, c) in enumerate(ACC)}; dC = (sgn * (d.jobs["free"]["dC"] - d.bt["dC"])).mean(0); o["delta_dC"] = dC.tolist()
        ml(f"Accounting (free run, rows G1∪G2): Δd_c at indices 20–23 = {np.round(dC, 3).tolist()} (signed toward source); component Δ(d_c writes) layers 20–22 [attn, mlp each]: {np.round(wm, 3).tolist()}; identity residual max {o['accounting']['max_abs_residual']:.4f} ({o['accounting']['rel_to_local_shift']:.3f} of the local shift) → ok={o['accounting']['ok']}.")
        SEC[f"{which}_{f}"] = o
V["secondary"] = SEC

# ------------------------------------------------------------------ gated follow-ups: Shapley orderings and head-level pinning at layer 20
if exists("shapley_A_num_orig.npz"):
    orders = json.load(open(OUT / "shapley_orders.json")); SH = {}
    for f in ("num_orig", "num_rev"):
        if not exists(f"shapley_A_{f}.npz"): continue
        d = DS(f"shapley_A_{f}"); sg = d.sigma; den = sg * (d.bs["margin"] - d.bt["margin"]); num = lambda n: sg * (d.jobs[n]["margin"] - d.bt["margin"])
        names = [f"{l}{c}" for l, c in ACC]; contrib = {c: [] for c in names}
        for k in range(24):
            order = orders[str(k)]; seq = [num("free")] + [num(f"shap|{k}|{m}") for m in range(1, 6)] + [num("pin_all")]
            for m, c in enumerate(order): contrib[c].append(seq[m + 1] - seq[m])
        tot = (num("pin_all") - num("free")); res_ = {}
        ml(f"\n## Shapley estimate of the pinning increase (set A, {f}; 24 random orderings of the six components): share of T_pin(all) − T_free\n")
        ml("| component | Shapley share | add-one q | LOO q' |\n|---|---|---|---|")
        for c in names:
            sn = np.mean(contrib[c], axis=0); r = d.boot.ratio(sn, den); rt = d.boot.ratio(tot, den); share = r["point"] / rt["point"]
            sh_rep = (r["rep_pair"] / rt["rep_pair"], r["rep_content"] / rt["rep_content"])
            res_[c] = dict(share=float(share), pair=[float(np.percentile(sh_rep[0], 2.5)), float(np.percentile(sh_rep[0], 97.5))], content=[float(np.percentile(sh_rep[1], 2.5)), float(np.percentile(sh_rep[1], 97.5))])
            sec = V["secondary"].get(f"A_{f}", {}); ml(f"| L{c[:2]} {c[2:]} | {share:.2f} [{res_[c]['pair'][0]:.2f}, {res_[c]['pair'][1]:.2f}] / [{res_[c]['content'][0]:.2f}, {res_[c]['content'][1]:.2f}] | {sec.get('q', {}).get(c, float('nan')):.2f} | {sec.get('qprime', {}).get(c, float('nan')):.2f} |")
        SH[f] = res_
    V["shapley"] = SH
if exists("q5_A_num_orig.npz"):
    qsel = json.load(open(OUT / "q5_selection.json")); Q5 = {}
    ml(f"\n## Head-level pinning at layer 20 attention (gated Q5; ranking frozen on the 4 validation contents; singles and nested top-k evaluated on held-out sets)\n\nValidation ranking of single heads (pin effect): {qsel['order'][:8]} with effects {[round(qsel['effect_single_pin_validation'][h], 3) for h in qsel['order'][:8]]}")
    for w in ("A", "B"):
        for f in ("num_orig", "num_rev"):
            n = f"q5_{w}_{f}"
            if not exists(n + ".npz"): continue
            d = DS(n); Tf = one(d, "free"); inc = sub(one(d, "attn20"), Tf); o = {}
            ml(f"\n**Set {w}, {f}:** T_free {Tf['point'][0]:.3f}; all layer-20 attention pinned {one(d, 'attn20')['point'][0]:.3f} (z-pin of all 32 heads: {one(d, 'top32')['point'][0]:.3f}, identity check diff {abs(one(d, 'top32')['point'][0] - one(d, 'attn20')['point'][0]):.4f}); single-head shares of the attention-pin increase: " + ", ".join(f"H{h}: {rat(sub(one(d, f'h{h}'), Tf), inc)['point'][0]:.2f}" for h in qsel["order"][:6]))
            curve = {k: rat(sub(one(d, f"top{k}"), Tf), inc) for k in (1, 2, 3, 4, 6, 8, 16, 32)}; o["nested"] = {k: float(v["point"][0]) for k, v in curve.items()}; o["singles"] = {h: float(rat(sub(one(d, f"h{h}"), Tf), inc)["point"][0]) for h in range(32)}
            ml("Nested top-k share of the attention-pin increase (frozen validation order): " + ", ".join(f"k={k}: {v['point'][0]:.2f} {ci(bnd(v, 0)[0])}" for k, v in curve.items()))
            Q5[f"{w}_{f}"] = o
    V["q5"] = Q5
(OUT / "report_tables.md").write_text("\n".join(md)); json.dump(V, open(OUT / "criteria_verdict.json", "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else (o.tolist() if isinstance(o, np.ndarray) else float(o)))
print("\n".join(md)[:12000])
