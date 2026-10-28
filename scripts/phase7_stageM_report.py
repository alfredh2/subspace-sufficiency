"""Stage M report: tables and the preregistered M-A / M-B / M-C rules (docs/plans/phase7_plan.md section 11) plus the geometric-versus-causal head ranking comparison."""
import json

import numpy as np
from scipy.stats import spearmanr

from causal_subspaces.upstream import pipeline as UP
from causal_subspaces.upstream import stats as S
from causal_subspaces.utils.config import ROOT

OUT = ROOT / "results/phase7"; md = []; ml = md.append; V = {}
sel = json.load(open(OUT / "stageM_selection.json")); sel5 = json.load(open(ROOT / "results/phase5/selection.json"))
CONDS = ["full", "dc", "orth", "geff", "rest", "dc_plus_geff", "pc1", "pc3", "dc_eqorth", "dc_eqfull", "geff_eqfull", "dc_x2", "dc_x4", "dc_x8"]


class DS:
    def __init__(self, which, fmt):
        z = np.load(OUT / f"stageM_{which}_{fmt}.npz"); self.z = z; self.sig = z["sigma"]; self.boot = S.Boot(z["pair"], z["content"], nboot=2000, seed=31); self.cache = {}
        self.names = {k[:-len("|margin")] for k in z.files if k.endswith("|margin")}
    def T(self, name, key="margin"):
        k = (name, key)
        if k not in self.cache:
            b = "margin" if key == "margin" else "c"
            num, den = S.transfer_terms(self.sig, self.z[f"{name}|{key}"], self.z[f"base|tgt_{b}"], self.z[f"base|src_{b}"]); self.cache[k] = self.boot.ratio(num, den)
        return self.cache[k]
    def Tmean(self, names):
        rs = [self.T(n) for n in names]; return dict(point=float(np.mean([r["point"] for r in rs])), rep_pair=np.mean([r["rep_pair"] for r in rs], 0), rep_content=np.mean([r["rep_content"] for r in rs], 0), loco=np.mean([r["loco"] for r in rs], 0))


def rat(a, b): return dict(point=a["point"] / b["point"], rep_pair=a["rep_pair"] / b["rep_pair"], rep_content=a["rep_content"] / b["rep_content"], loco=a["loco"] / b["loco"])
def bd(r): return (np.percentile(r["rep_pair"], 2.5), np.percentile(r["rep_pair"], 97.5)), (np.percentile(r["rep_content"], 2.5), np.percentile(r["rep_content"], 97.5))
def fs(r):
    p, c = bd(r); return f"{r['point']:.3f} [{p[0]:.2f}, {p[1]:.2f}] / [{c[0]:.2f}, {c[1]:.2f}]"
def lo(r): p, c = bd(r); return min(p[0], c[0])
def up(r): p, c = bd(r); return max(p[1], c[1])


R = {}
for which, fmt in (("A", "num_orig"), ("A", "num_rev"), ("B", "num_orig"), ("B", "num_rev")):
    d = DS(which, fmt); key = f"{which}_{fmt}"; out = {}
    ml(f"\n## Stage M, set {which} {fmt} ({len(d.sig)} trials)\n")
    ml("Norm statistics (mean over trials of the row-mean norm): " + ", ".join(f"{k.replace('stats|', '')}: {d.z['stats|' + k].mean():.3f}" for k in ("norm_full", "norm_dc", "norm_orth", "norm_geff", "cos_full_dc", "scale_dc_eqorth", "scale_dc_eqfull")) + f"; max rescale factor dc→eqorth {d.z['stats|scale_dc_eqorth'].max():.1f}, dc→eqfull {d.z['stats|scale_dc_eqfull'].max():.1f}\n")
    ml("| condition | T_margin [pair] / [content] | T_coordinate | pinned (layers 20–22 attn+MLP) T | S_pin |\n|---|---|---|---|---|")
    Tf = d.T("full")
    for c in CONDS:
        pin = f"{c}|pin" in d.names
        ml(f"| {c} | {fs(d.T(c))} | {d.T(c, 'c')['point']:.3f} | " + (f"{d.T(c + '|pin')['point']:.3f} | {d.T(c + '|pin')['point'] / d.T(c)['point']:.2f}" if pin and abs(d.T(c)['point']) > 0.05 else "— | —") + " |")
    ro = d.Tmean([f"randorth_{k}" for k in range(20)]); rf = d.Tmean([f"randfull_eqfull_{k}" for k in range(5)]); rd = d.Tmean([f"randfull_eqdc_{k}" for k in range(5)])
    ml(f"| random ⟂ d_c, norm of the orthogonal part (mean of 20) | {fs(ro)} | | | |\n| random, norm of Δ (mean of 5) | {fs(rf)} | | | |\n| random, norm of the d_c part (mean of 5) | {fs(rd)} | | | |")
    maxro = max(abs(d.T(f"randorth_{k}")["point"]) for k in range(20))
    MA = rat(d.T("dc_eqorth"), d.T("orth")); MBo = rat(d.T("orth"), Tf); MBr = rat(ro, d.T("orth")); MBd = MA
    cos_raw = [sel["cos_g1_raw_dc"], sel["cos_g2_raw_dc"]]; MCg = rat(d.T("geff"), Tf)
    out["rules"] = dict(
        M_A=bool(MA["point"] >= 0.5 and lo(MA) >= 0.35), ratio_dc_eqorth_over_orth=MA["point"], MA_bounds=[lo(MA), up(MA)],
        M_B=bool(MBo["point"] >= 0.5 and lo(MBo) >= 0.4 and MBr["point"] <= 0.1 and up(MBr) <= 0.2 and MA["point"] <= 0.1 and up(MA) <= 0.2), orth_over_full=MBo["point"], orth_over_full_lower=lo(MBo), randorth_over_orth=MBr["point"], randorth_upper=up(MBr), max_abs_single_randorth=float(maxro),
        M_C=bool(MCg["point"] >= 0.7 and lo(MCg) >= 0.6 and max(abs(c) for c in cos_raw) <= 0.3), geff_over_full=MCg["point"], geff_lower=lo(MCg), cos_raw=cos_raw,
        loco_ok_MB=bool((MBo["loco"] >= 0.5).all()), T_full=Tf["point"], T_dc=d.T("dc")["point"], T_orth=d.T("orth")["point"], T_geff=d.T("geff")["point"], T_rest=d.T("rest")["point"])
    ml(f"\nRules ({key}): M-A (magnitude artefact): ratio T(dc equal-norm)/T(orth) = {MA['point']:.3f} {fs(MA)} → {out['rules']['M_A']}; M-B (orthogonal part special): T(orth)/T(full) = {MBo['point']:.3f}, random-orth/T(orth) = {MBr['point']:.3f} (upper {up(MBr):.3f}), dc-eqnorm/orth = {MA['point']:.3f} → {out['rules']['M_B']}; M-C (position-specific direction): T(g_eff)/T(full) = {MCg['point']:.3f} (lower {lo(MCg):.3f}), |cos(g_eff, d_c)| = {max(abs(c) for c in cos_raw):.3f} → {out['rules']['M_C']}.")
    # head ranking comparison
    sg = d.sig[:, None]
    geo = {"|write on d_c|": np.abs((sg * d.z["head|proj_dc"]).mean(0)), "|write on g_eff|": np.abs((sg * d.z["head|proj_g"]).mean(0)), "write norm": d.z["head|wnorm"].mean(0)}
    if which == "A":
        tr5, bt5, bs5, jobs5 = UP.load_raw(ROOT / f"results/phase5/stage1_{fmt}.npz"); b5 = S.Boot(tr5["pair"], tr5["content"], nboot=200, seed=2)
        causal = np.array([b5.ratio(*S.transfer_terms(tr5["sigma"], jobs5[f"h{h}"]["margin"], bt5["margin"], bs5["margin"]))["point"] for h in range(32)])
        rk = {}
        for nm, sc in geo.items():
            rho = spearmanr(sc, causal).correlation; top3 = set(np.argsort(-sc)[:3]); ctop3 = set(np.argsort(-causal)[:3]); rk[nm] = dict(spearman_with_causal=float(rho), top3=[int(h) for h in np.argsort(-sc)[:3]], overlap_with_causal_top3=len(top3 & ctop3))
        rk["causal_top3"] = [int(h) for h in np.argsort(-causal)[:3]]; rk["validation_R1_vs_test_causal_spearman"] = float(spearmanr(sel5["R1_single_head_T"], causal).correlation); out["ranking"] = rk
        ml("\nHead ranking (32 layer-19 heads): " + json.dumps(rk, default=float))
    R[key] = out
V["stageM"] = R
(OUT / "stageM_report_tables.md").write_text("\n".join(md)); json.dump(V, open(OUT / "stageM_verdict.json", "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
print("\n".join(md)[:14000])
