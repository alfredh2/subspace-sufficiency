"""Phase 2.6 (EXPLORATORY): position-restricted V-only patching of the frozen S8 heads. Implements docs/phase26_plan.md.
Order: validation V1-V3 (stored diagnostics) -> mask checks -> self-patch noise floor (V4) -> full-mask reproduction of the Phase 2.5 V-only transfer ->
restricted conditions -> same-label lexical controls -> statistics. Aborts at 27 minutes."""
import json
import time
import zlib

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5
from causal_subspaces.data.logic import HELDOUT_FAMILIES, TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_splits
from causal_subspaces.heads import attention as A
from causal_subspaces.heads import positions as POS
from causal_subspaces.heads.ablation import capture_z
from causal_subspaces.heads.decompose import head_dims, head_gain
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits, load_model
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

T0 = time.time(); CAP = 27 * 60
OUT = ROOT / "results/phase26"
# ---- V1-V3 re-check from the stored Phase 2.5 diagnostics -------------------------------------------------------------------------------------
diag = json.load(open(ROOT / "results/phase25/recompute_diagnosis.json")); floor = json.load(open(ROOT / "results/phase25/rounding_floor.json"))
S8 = [(s["layer"], s["head"]) for s in json.load(open(ROOT / "results/phase2/selected_heads.json"))["S8"]]
LAY = [19, 20, 21, 22]
v1 = all(diag["pairs"][f"L{l}|MODEL_sdpa_captured|vs|R3a_sdpa_float_mask"]["z_rel_L2"] <= 1e-3 for l in LAY)
v2 = all((lambda v: v["all_rms_over_std"] <= 0.01 and v["all_max_over_std"] <= 0.02 and v["last_rms_over_std"] <= 0.01 and v["last_max_over_std"] <= 0.02)(diag["s8"][f"L{l}H{h}|MODEL_sdpa_captured|vs|R0_recompute_fp32_attention"]) for l, h in S8)
v3 = all(floor[str(l)]["z_rel_L2"]["observed"] <= 1.25 * floor[str(l)]["z_rel_L2"]["quantisation_floor"] and diag["pairs"][f"L{l}|R0_recompute_fp32_attention|vs|FP32_reference"]["z_rel_L2"] / diag["pairs"][f"L{l}|MODEL_sdpa_captured|vs|FP32_reference"]["z_rel_L2"] <= 1.05 for l in LAY)
print("V1-V3 re-check:", v1, v2, v3, flush=True)
if not (v1 and v2 and v3):
    raise SystemExit("validation V1-V3 not satisfied")

cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
H, HD, KV = head_dims(model)
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
GAIN = {l: head_gain(model, l, d_c).view(H, HD) for l in LAY}
splits = get_splits(cfg); contents = build_contents("ordinary"); test_ids = sorted({e.content_id for e in splits["items"]})
pairs = P5.build_pairs(contents, test_ids, TRAIN_FAMILIES + HELDOUT_FAMILIES, tok)
prev = json.load(open(ROOT / "results/phase25/av_patching.json"))

def spec_of(hs):
    s = {}
    for l, h in hs: s.setdefault(l, []).append(h)
    return s
SETS = {"S8": S8}
for l, h in S8: SETS[f"L{l}H{h}"] = [(l, h)]
MAJOR = {"L22H5", "L21H19"}
COND_S8 = ["G1", "G2", "G1G2", "Rem", "Self", "RemExSelf", "All"]
RANDOM_MATCH, N_DRAW = ["G1", "G2", "G1G2"], 10
def conds_for(setname):
    if setname == "S8": return COND_S8 + [f"rand_{m}_{d}" for m in RANDOM_MATCH for d in range(N_DRAW)]
    return ["G1", "G2", "G1G2", "Rem", "All"] if setname in MAJOR else ["G1G2", "Rem", "All"]

def trial_masks(tok_, user, syl_lines, family):
    prompt = format_prompt(tok_, user, False)
    cats = POS.categorize(tok_, prompt, user, POS.spans_logic(user, syl_lines, family))
    m = POS.group_masks(cats); return prompt, m

# ---- trials with group masks; reject pairs whose masks are invalid or differ between source and target -------------------------------------------
stats = {"group_checks": {}, "random": {}}
def make_trials(fmt):
    f = P5.FORMATS[fmt]; trials, rejected = [], {"alignment": 0, "mask_invalid": 0, "mask_mismatch": 0}
    for p in pairs:
        pv, pi = P5.build_prompt(fmt, p["valid_syl"]), P5.build_prompt(fmt, p["invalid_syl"])
        ok, _ = A.pair_alignment(tok, format_prompt(tok, pv), p["valid_syl"].split("\n"), format_prompt(tok, pi), p["invalid_syl"].split("\n"))
        if not ok: rejected["alignment"] += 1; continue
        _, mv = trial_masks(tok, pv, p["valid_syl"].split("\n"), p["family"]); _, mi = trial_masks(tok, pi, p["invalid_syl"].split("\n"), p["family"])
        if POS.check_groups(mv) or POS.check_groups(mi): rejected["mask_invalid"] += 1; continue
        if any(not np.array_equal(mv[k], mi[k]) for k in mv): rejected["mask_mismatch"] += 1; continue
        trials += [dict(tgt=pi, src=pv, tgt_aff=False, src_aff=True, masks=mi), dict(tgt=pv, src=pi, tgt_aff=True, src_aff=False, masks=mv)]
    return trials, rejected
def cond_mask(cond, masks, trial_idx, fmt):
    if cond in masks: return masks[cond]
    _, match, d = cond.split("_"); pool = np.where(masks["RemExSelf"])[0]; cnt = int(masks[match].sum())
    pos, flag = POS.random_positions(pool, cnt, f"phase26|{fmt}|{match}", trial_idx, int(d))
    stats["random"].setdefault(f"{fmt}|{match}", dict(flagged=0, draws=0)); stats["random"][f"{fmt}|{match}"]["draws"] += 1; stats["random"][f"{fmt}|{match}"]["flagged"] += int(flag)
    m = np.zeros(len(masks["All"]), bool); m[pos] = True; return m

@torch.no_grad()
def run_pass(trials, aff, rej, setconds, qsettings, fmt, bs=16, selfpatch=False):
    ia, ir = P5.single_token_id(tok, aff), P5.single_token_id(tok, rej)
    n = len(trials); res = {k: np.zeros(n) for k in ("m_t", "m_s", "c_t", "c_s")}; res["v_t"] = np.empty(n, "<U5"); res["v_s"] = np.empty(n, "<U5")
    verdict = lambda lg: ["aff" if tok.decode([g])[:1] == aff[0] else "rej" if tok.decode([g])[:1] == rej[0] else "other" for g in lg.argmax(-1).tolist()]
    keys = [(sn, c, qs) for sn, cs in setconds.items() for c in cs for qs in qsettings]
    for k in keys:
        nm = "|".join(k); res[f"m|{nm}"] = np.full(n, np.nan); res[f"c|{nm}"] = np.full(n, np.nan); res[f"d|{nm}"] = np.full(n, np.nan); res[f"v|{nm}"] = np.full(n, "", "<U5")
    for s0 in range(0, n, bs):
        if time.time() - T0 > CAP: raise TimeoutError("27-minute cap reached")
        b = trials[s0:s0 + bs]; sl = slice(s0, s0 + len(b))
        ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], dev)
        ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], dev)
        assert ids_t.shape == ids_s.shape and torch.equal(m_t, m_s)
        B, T = ids_t.shape
        st = A.AttnStash(model, LAY); cs_ = []
        with capture_residual(model, idx_hook, cs_):
            lg_s = last_logits(model, ids_s, m_s, p_s)
        src = {l: A.recompute(model, l, *st.data[l], m_s) for l in LAY}
        ct, zt = [], {}
        with capture_residual(model, idx_hook, ct), capture_z(model, LAY, zt):
            lg_t = last_logits(model, ids_t, m_t, p_t)
        tgt = {l: A.recompute(model, l, *st.data[l], m_t) for l in LAY}
        st.close()
        res["m_s"][sl] = (lg_s[:, ia] - lg_s[:, ir]).cpu().numpy(); res["m_t"][sl] = (lg_t[:, ia] - lg_t[:, ir]).cpu().numpy(); res["v_s"][sl] = verdict(lg_s); res["v_t"][sl] = verdict(lg_t)
        res["c_s"][sl] = (cs_[0][:, -1].float() @ d_c).cpu().numpy(); res["c_t"][sl] = (ct[0][:, -1].float() @ d_c).cpu().numpy()
        for sn, cs in setconds.items():
            hs = SETS[sn]; spec = spec_of(hs); layers = sorted(spec)
            source = tgt if selfpatch else src
            srcd = {l: dict(A=source[l]["A"], v=source[l]["v"]) for l in layers}
            for c in cs:
                vp = torch.tensor(np.stack([POS.pad_mask(cond_mask(c, t["masks"], s0 + i, fmt), T) for i, t in enumerate(b)]))
                for qs in qsettings:
                    cp, zp = [], {}
                    with A.attention_patch(model, spec, srcd, m_t, idx_hook, patch="V", position=qs, v_positions=vp), capture_residual(model, idx_hook, cp), capture_z(model, layers, zp):
                        lg = last_logits(model, ids_t, m_t, p_t)
                    nm = f"{sn}|{c}|{qs}"
                    res[f"m|{nm}"][sl] = (lg[:, ia] - lg[:, ir]).cpu().numpy(); res[f"v|{nm}"][sl] = verdict(lg); res[f"c|{nm}"][sl] = (cp[0][:, -1].float() @ d_c).cpu().numpy()
                    res[f"d|{nm}"][sl] = sum(((zp[l].float().view(B, T, H, HD)[:, -1, h] - zt[l].float().view(B, T, H, HD)[:, -1, h]) * GAIN[l][h]).sum(-1) for l, h in hs).cpu().numpy()
    return res

def transfers(res, ta, sa):
    diff = ta != sa; sgn = np.where(sa, 1.0, -1.0)
    return diff, sgn, (sgn * (res["m_s"] - res["m_t"]))[diff].mean(), (sgn * (res["c_s"] - res["c_t"]))[diff].mean()

all_out, raw = {"validation": {"V1": v1, "V2": v2, "V3": v3}, "formats": {}}, {}
for fmt in ("num_orig", "num_rev"):
    f = P5.FORMATS[fmt]; trials, rejected = make_trials(fmt)
    ta, sa = np.array([t["tgt_aff"] for t in trials]), np.array([t["src_aff"] for t in trials])
    sizes = {k: float(np.mean([t["masks"][k].sum() for t in trials])) for k in ("G1", "G2", "G1G2", "Self", "Rem", "RemExSelf", "All")}
    stats["group_checks"][fmt] = dict(n_trials=len(trials), rejected=rejected, mean_sizes=sizes,
                                      all_partition_checks_passed=bool(all(not POS.check_groups(t["masks"]) for t in trials)))
    print(fmt, "trials", len(trials), "rejected", rejected, "mean group sizes", sizes, flush=True)
    # V4: self-patch noise floor (source = target), full mask, before anything else
    r0 = run_pass(trials, f["valid"], f["invalid"], {k: ["All"] for k in ("S8", "L22H5", "L21H19")}, ("last", "all"), fmt, selfpatch=True)
    v4 = {}
    for k in ("S8", "L22H5", "L21H19"):
        for qs in ("last", "all"):
            dm = np.abs(r0[f"m|{k}|All|{qs}"] - r0["m_t"]); vc = float((r0[f"v|{k}|All|{qs}"] != r0["v_t"]).mean())
            v4[f"{k}|{qs}"] = dict(max_abs=float(dm.max()), mean_abs=float(dm.mean()), verdict_change=vc, passed=bool(dm.max() <= 1.0 and dm.mean() <= 0.3 and vc <= 0.01))
    all_out["validation"]["V4_" + fmt] = v4
    print(fmt, "V4", {k: (round(v["max_abs"], 3), round(v["mean_abs"], 3), v["verdict_change"], v["passed"]) for k, v in v4.items()}, flush=True)
    if not all(v["passed"] for v in v4.values()):
        all_out["formats"][fmt] = dict(stopped="V4 failed"); break
    noise = max(v["max_abs"] for v in v4.values())
    setconds = {sn: conds_for(sn) for sn in SETS}
    res = run_pass(trials, f["valid"], f["invalid"], setconds, ("last", "all"), fmt)
    diff, sgn, gap, cgap = transfers(res, ta, sa)
    # full-mask reproduction of the Phase 2.5 V-only result (S8, All)
    chk = {}
    for qs in ("last", "all"):
        t_all = float((sgn * (res[f"m|S8|All|{qs}"] - res["m_t"]))[diff].mean() / gap)
        stored = prev["formats"][fmt]["sets"][f"S8|{'last' if qs == 'last' else 'all'}"]["margin_transfer"]["V"]
        chk[qs] = dict(T_all_now=t_all, stored_phase25_V_only=stored, abs_diff=abs(t_all - stored), passed=bool(abs(t_all - stored) <= 0.01))
    all_out["validation"]["full_mask_reproduces_phase25_" + fmt] = chk
    print(fmt, "full-mask reproduction", chk, flush=True)
    rg = np.random.default_rng(zlib.crc32(f"boot|{fmt}".encode()))
    didx = np.where(diff)[0]; boots = [rg.choice(didx, len(didx)) for _ in range(500)]
    def Tm(key, rows=None, which="m"):
        a_ = res[f"{which}|{key}"] - (res["m_t"] if which == "m" else res["c_t"]); g_ = gap if which == "m" else cgap
        if rows is None: return float((sgn * a_)[diff].mean() / g_)
        gg = (sgn * (res["m_s"] - res["m_t"]) if which == "m" else sgn * (res["c_s"] - res["c_t"]))[rows].mean()
        return float((sgn * a_)[rows].mean() / gg)
    clean = diff & (res["v_t"] == np.where(ta, "aff", "rej")) & (res["v_s"] == np.where(sa, "aff", "rej"))
    out = {"n_trials": len(trials), "noise_floor_max_abs": noise, "gap": float(gap), "n_clean": int(clean.sum()), "sets": {}}
    for sn in SETS:
        for qs in ("last", "all"):
            rowset = {}
            Tall_pt = Tm(f"{sn}|All|{qs}"); Tall_b = np.array([Tm(f"{sn}|All|{qs}", rows) for rows in boots])
            evaluable = bool(Tall_pt >= 0.20 and np.quantile(Tall_b, .025) >= 0.10)
            for c in setconds[sn]:
                key = f"{sn}|{c}|{qs}"
                T = Tm(key); Tb = np.array([Tm(key, rows) for rows in boots])
                F = T / Tall_pt if evaluable else None
                Fb = Tb / np.where(Tall_b > 0.05, Tall_b, np.nan)
                rowset[c] = dict(T=T, T_ci=[float(np.quantile(Tb, .025)), float(np.quantile(Tb, .975))], F=F, F_ci=[float(np.nanquantile(Fb, .025)), float(np.nanquantile(Fb, .975))] if evaluable else None,
                                 coord_T=Tm(key, which="c"), direct_T=float((sgn * res[f"d|{key}"])[diff].mean() / cgap), flip=float((res[f"v|{key}"][clean] == res["v_s"][clean]).mean()),
                                 discarded_bootstrap_replicates=int(np.isnan(Fb).sum()))
            # random controls summarised per matching (mean/max over the 10 draws of F)
            rand = {}
            for m in RANDOM_MATCH:
                Fd = [rowset[f"rand_{m}_{d}"]["T"] / Tall_pt for d in range(N_DRAW)] if evaluable else None
                rand[m] = dict(F_mean=float(np.mean(Fd)) if Fd else None, F_max=float(np.max(Fd)) if Fd else None, T_mean=float(np.mean([rowset[f"rand_{m}_{d}"]["T"] for d in range(N_DRAW)])) if sn == "S8" else None) if sn == "S8" else None
            ir = None
            if "G2" in rowset and "G1" in rowset:
                i1 = [Tm(f"{sn}|G1G2|{qs}", r) - Tm(f"{sn}|G1|{qs}", r) - Tm(f"{sn}|G2|{qs}", r) for r in boots]; i2 = [Tm(f"{sn}|All|{qs}", r) - Tm(f"{sn}|G1G2|{qs}", r) - Tm(f"{sn}|Rem|{qs}", r) for r in boots]
                ir = dict(G1G2_minus_G1_minus_G2=Tm(f"{sn}|G1G2|{qs}") - Tm(f"{sn}|G1|{qs}") - Tm(f"{sn}|G2|{qs}"), ci1=[float(np.quantile(i1, .025)), float(np.quantile(i1, .975))],
                          All_minus_G1G2_minus_Rem=Tm(f"{sn}|All|{qs}") - Tm(f"{sn}|G1G2|{qs}") - Tm(f"{sn}|Rem|{qs}"), ci2=[float(np.quantile(i2, .025)), float(np.quantile(i2, .975))])
            elif "G1G2" in rowset:
                i2 = [Tm(f"{sn}|All|{qs}", r) - Tm(f"{sn}|G1G2|{qs}", r) - Tm(f"{sn}|Rem|{qs}", r) for r in boots]
                ir = dict(All_minus_G1G2_minus_Rem=Tm(f"{sn}|All|{qs}") - Tm(f"{sn}|G1G2|{qs}") - Tm(f"{sn}|Rem|{qs}"), ci2=[float(np.quantile(i2, .025)), float(np.quantile(i2, .975))])
            out["sets"][f"{sn}|{qs}"] = dict(T_all=Tall_pt, T_all_ci=[float(np.quantile(Tall_b, .025)), float(np.quantile(Tall_b, .975))], evaluable=evaluable, conditions=rowset, random=rand, interactions=ir)
    all_out["formats"][fmt] = out
    for k, v in res.items(): raw[f"{fmt}|{k}"] = v
    raw[f"{fmt}|ta"], raw[f"{fmt}|sa"] = ta, sa
    s8 = out["sets"]["S8|last"]
    print(fmt, "S8|last T_all", round(s8["T_all"], 3), {c: round(s8["conditions"][c]["F"], 3) for c in COND_S8 if s8["conditions"][c]["F"] is not None}, "rand", s8["random"], flush=True)
    json.dump(dict(**all_out, **stats), open(OUT / "position_patching.json", "w"), indent=1)

# ---- same-label lexical controls (S8; G1G2, Rem, All; both query settings), aligned pairs, identical masks -----------------------------------------------------
try:
    f = P5.FORMATS["num_orig"]; ctl = P5.lexical_controls(contents, pairs, tok); ltr = []; rej = 0
    for c in ctl:
        b_, o_ = c["base"], c["other"]
        for key, y in (("valid_syl", True), ("invalid_syl", False)):
            ub, uo = P5.build_prompt("num_orig", b_[key]), P5.build_prompt("num_orig", o_[key])
            ok, _ = A.pair_alignment(tok, format_prompt(tok, ub), b_[key].split("\n"), format_prompt(tok, uo), o_[key].split("\n"))
            if not ok: rej += 1; continue
            _, mb = trial_masks(tok, ub, b_[key].split("\n"), b_["family"]); _, mo = trial_masks(tok, uo, o_[key].split("\n"), o_["family"])
            if POS.check_groups(mb) or any(not np.array_equal(mb[k], mo[k]) for k in mb): rej += 1; continue
            ltr.append(dict(tgt=ub, src=uo, tgt_aff=y, src_aff=y, masks=mb))
    rl = run_pass(ltr, f["valid"], f["invalid"], {"S8": ["G1G2", "Rem", "All"]}, ("last", "all"), "lex")
    lex = {}
    for c in ("G1G2", "Rem", "All"):
        for qs in ("last", "all"):
            k = f"S8|{c}|{qs}"; lex[k] = dict(verdict_change=float((rl[f"v|{k}"] != rl["v_t"]).mean()), mean_abs_margin_change=float(np.abs(rl[f"m|{k}"] - rl["m_t"]).mean()), max_abs_margin_change=float(np.abs(rl[f"m|{k}"] - rl["m_t"]).max()))
    all_out["lexical_controls"] = dict(n_trials=len(ltr), rejected=rej, results=lex)
    print("lexical controls", lex, flush=True)
except TimeoutError as e:
    all_out["lexical_controls"] = dict(error=str(e))
json.dump(dict(**all_out, **stats), open(OUT / "position_patching.json", "w"), indent=1)
np.savez_compressed(OUT / "position_patching_arrays.npz", **raw)
json.dump(dict(cap_gpu_hours=0.5, wall_seconds=time.time() - T0, wall_gpu_hours=(time.time() - T0) / 3600), open(OUT / "budget.json", "w"), indent=1)
print("done", time.time() - T0)
