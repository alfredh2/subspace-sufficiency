"""Phase 2.5 step 3 (EXPLORATORY): A-only / V-only / A+V head patching on alignment-validated valid/invalid pairs, gated by the Amendment-2 criterion
(docs/plans/phase25_plan.md). V1-V3 are re-checked from the stored diagnostics; V4 (self-patch noise floor) is measured BEFORE any A/V effect is computed and stops the
script if it fails. Nothing here can change the Phase 2 verdict."""
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
from causal_subspaces.heads.ablation import capture_z, head_z_patch
from causal_subspaces.heads.decompose import head_dims
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits, load_model
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

T0 = time.time()
OUT = ROOT / "results/phase25"
# ------------------------------------------------------------------ V1-V3 from the stored diagnostics ------------------------------------------------------------
diag = json.load(open(OUT / "recompute_diagnosis.json")); floor = json.load(open(OUT / "rounding_floor.json"))
S8 = [(s["layer"], s["head"]) for s in json.load(open(ROOT / "results/phase2/selected_heads.json"))["S8"]]
LAY = [19, 20, 21, 22]
V = {"V1": {}, "V2": {}, "V3": {}}
for l in LAY:
    V["V1"][l] = diag["pairs"][f"L{l}|MODEL_sdpa_captured|vs|R3a_sdpa_float_mask"]["z_rel_L2"] <= 1e-3
    fl = floor[str(l)]["z_rel_L2"]
    V["V3"][l] = bool(fl["observed"] <= 1.25 * fl["quantisation_floor"] and diag["pairs"][f"L{l}|R0_recompute_fp32_attention|vs|FP32_reference"]["z_rel_L2"] / diag["pairs"][f"L{l}|MODEL_sdpa_captured|vs|FP32_reference"]["z_rel_L2"] <= 1.05)
for l, h in S8:
    v = diag["s8"][f"L{l}H{h}|MODEL_sdpa_captured|vs|R0_recompute_fp32_attention"]
    V["V2"][f"L{l}H{h}"] = bool(v["all_rms_over_std"] <= 0.01 and v["all_max_over_std"] <= 0.02 and v["last_rms_over_std"] <= 0.01 and v["last_max_over_std"] <= 0.02)
print("V1", V["V1"], "\nV2", V["V2"], "\nV3", V["V3"], flush=True)
v123 = all(V["V1"].values()) and all(V["V2"].values()) and all(V["V3"].values())
if not v123:
    json.dump(dict(V=V, stopped="V1-V3 failed"), open(OUT / "av_patching.json", "w"), indent=1); raise SystemExit("V1-V3 failed: no routing/content interventions")

cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
H, HD, KV = head_dims(model)
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
splits = get_splits(cfg); contents = build_contents("ordinary"); test_ids = sorted({e.content_id for e in splits["items"]})
pairs = P5.build_pairs(contents, test_ids, TRAIN_FAMILIES + HELDOUT_FAMILIES, tok)

def spec_of(hs):
    s = {}
    for l, h in hs: s.setdefault(l, []).append(h)
    return s
def random_set(hs, d):
    rng = np.random.default_rng(100_003 * len(hs) + 7919 * d + zlib.crc32(b"random") % 1000); nat = set(hs); out = []
    for l in sorted({l for l, _ in hs}):
        k = sum(1 for x, _ in hs if x == l); pool = [h for h in range(H) if (l, h) not in nat]; out += [(l, int(h)) for h in rng.choice(pool, min(k, len(pool)), replace=False)]
    return out
SETS = {"S8": S8, "L22H5": [(22, 5)], "L21H19": [(21, 19)]}
for d in range(10): SETS[f"random|d={d}"] = random_set(S8, d)

@torch.no_grad()
def run_pass(trials, aff, rej, types, sets, positions, bs=16):
    ia, ir = P5.single_token_id(tok, aff), P5.single_token_id(tok, rej)
    n = len(trials); res = {k: np.zeros(n) for k in ("m_t", "m_s", "c_t", "c_s")}
    verdict = lambda lg: ["aff" if tok.decode([g])[:1] == aff[0] else "rej" if tok.decode([g])[:1] == rej[0] else "other" for g in lg.argmax(-1).tolist()]
    res["v_t"] = np.empty(n, "<U5"); res["v_s"] = np.empty(n, "<U5")
    for ty in types:
        for nm in sets:
            for p in positions:
                res[f"m|{ty}:{nm}|{p}"] = np.full(n, np.nan); res[f"c|{ty}:{nm}|{p}"] = np.full(n, np.nan); res[f"v|{ty}:{nm}|{p}"] = np.full(n, "", "<U5")
    layers = sorted({l for hs in sets.values() for l, _ in hs})
    for s0 in range(0, n, bs):
        b = trials[s0:s0 + bs]; sl = slice(s0, s0 + len(b))
        ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], dev)
        ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], dev)
        assert ids_t.shape == ids_s.shape and torch.equal(m_t, m_s), "source/target layouts differ: pair should have been rejected"
        st = A.AttnStash(model, layers); zs, cs = {}, []
        with capture_z(model, layers, zs), capture_residual(model, idx_hook, cs):
            lg_s = last_logits(model, ids_s, m_s, p_s)
        src = {l: A.recompute(model, l, *st.data[l], m_s) for l in layers}
        ct = []
        with capture_residual(model, idx_hook, ct):
            lg_t = last_logits(model, ids_t, m_t, p_t)
        tgt = {l: A.recompute(model, l, *st.data[l], m_t) for l in layers}
        st.close()
        res["m_s"][sl] = (lg_s[:, ia] - lg_s[:, ir]).cpu().numpy(); res["m_t"][sl] = (lg_t[:, ia] - lg_t[:, ir]).cpu().numpy()
        res["v_s"][sl] = verdict(lg_s); res["v_t"][sl] = verdict(lg_t)
        res["c_s"][sl] = (cs[0][:, -1].float() @ d_c).cpu().numpy(); res["c_t"][sl] = (ct[0][:, -1].float() @ d_c).cpu().numpy()
        for nm, hs in sets.items():
            spec = spec_of(hs)
            for p in positions:
                for ty in types:
                    cp = []
                    if ty == "Z":
                        ctx = head_z_patch(model, spec, zs, idx_hook, position=p)
                    else:
                        source = tgt if ty == "self" else src
                        ctx = A.attention_patch(model, spec, {l: dict(A=source[l]["A"], v=source[l]["v"]) for l in spec}, m_t, idx_hook, patch="AV" if ty in ("self", "AV") else ty, position=p)
                    with ctx, capture_residual(model, idx_hook, cp):
                        lg = last_logits(model, ids_t, m_t, p_t)
                    res[f"m|{ty}:{nm}|{p}"][sl] = (lg[:, ia] - lg[:, ir]).cpu().numpy(); res[f"v|{ty}:{nm}|{p}"][sl] = verdict(lg); res[f"c|{ty}:{nm}|{p}"][sl] = (cp[0][:, -1].float() @ d_c).cpu().numpy()
    return res

summary = {"V": V, "formats": {}, "control_sets": {k: v for k, v in SETS.items() if k.startswith("random")}}
def fmt_prompt(fmt, syl): return format_prompt(tok, P5.build_prompt(fmt, syl), False)
for fmt in ("num_orig", "num_rev"):
    f = P5.FORMATS[fmt]; trials = []
    for p in pairs:
        ok, why = A.pair_alignment(tok, fmt_prompt(fmt, p["valid_syl"]), p["valid_syl"].split("\n"), fmt_prompt(fmt, p["invalid_syl"]), p["invalid_syl"].split("\n"))
        if not ok: continue
        v, i = P5.build_prompt(fmt, p["valid_syl"]), P5.build_prompt(fmt, p["invalid_syl"])
        trials += [dict(tgt=i, src=v, tgt_aff=False, src_aff=True), dict(tgt=v, src=i, tgt_aff=True, src_aff=False)]
    ta, sa = np.array([t["tgt_aff"] for t in trials]), np.array([t["src_aff"] for t in trials])
    # ---- V4: self-patch noise floor, measured before any A/V effect --------------------------------------------------------------------------------------
    r0 = run_pass(trials, f["valid"], f["invalid"], ["self"], {k: SETS[k] for k in ("S8", "L22H5", "L21H19")}, ("all", "last"))
    v4 = {}
    for nm in ("S8", "L22H5", "L21H19"):
        for p in ("all", "last"):
            dm = np.abs(r0[f"m|self:{nm}|{p}"] - r0["m_t"]); vc = float((r0[f"v|self:{nm}|{p}"] != r0["v_t"]).mean())
            v4[f"{nm}|{p}"] = dict(max_abs=float(dm.max()), mean_abs=float(dm.mean()), verdict_change=vc, passed=bool(dm.max() <= 1.0 and dm.mean() <= 0.3 and vc <= 0.01))
    V["V4_" + fmt] = v4
    print(fmt, "V4", {k: (round(v["max_abs"], 3), round(v["mean_abs"], 3), v["verdict_change"], v["passed"]) for k, v in v4.items()}, flush=True)
    if not all(v["passed"] for v in v4.values()):
        summary["formats"][fmt] = dict(stopped="V4 failed"); json.dump(summary, open(OUT / "av_patching.json", "w"), indent=1); print("V4 failed: stopping before computing A/V effects"); break
    floor_v4 = max(v["max_abs"] for v in v4.values())
    # ---- exploratory A / V / A+V / full-z -----------------------------------------------------------------------------------------------------------------
    res = run_pass(trials, f["valid"], f["invalid"], ["A", "V", "AV", "Z"], SETS, ("all", "last"))
    diff = ta != sa; sgn = np.where(sa, 1.0, -1.0)
    gap = (sgn * (res["m_s"] - res["m_t"]))[diff].mean(); cgap = (sgn * (res["c_s"] - res["c_t"]))[diff].mean()
    def T(key, which="m"):
        base = res["m_t"] if which == "m" else res["c_t"]; return float((sgn * (res[f"{which}|{key}"] - base))[diff].mean() / (gap if which == "m" else cgap))
    out = {"n_trials": len(trials), "source_target_margin_gap": float(gap), "self_patch_noise_floor_max_abs": floor_v4, "sets": {}}
    rg = np.random.default_rng(0)
    for nm in SETS:
        for p in ("all", "last"):
            tr = {ty: T(f"{ty}:{nm}|{p}") for ty in ("A", "V", "AV", "Z")}; trc = {ty: T(f"{ty}:{nm}|{p}", "c") for ty in ("A", "V", "AV", "Z")}
            idx = np.where(diff)[0]; bi = []
            for _ in range(500):
                rr = rg.choice(idx, len(idx)); g_ = (sgn * (res["m_s"] - res["m_t"]))[rr].mean()
                x = {ty: (sgn * (res[f"m|{ty}:{nm}|{p}"] - res["m_t"]))[rr].mean() / g_ for ty in ("A", "V", "AV")}
                bi.append(x["AV"] - x["A"] - x["V"])
            flips = {ty: float(np.mean([res[f"v|{ty}:{nm}|{p}"][i] == res["v_s"][i] for i in idx if res["v_t"][i] != res["v_s"][i] and False] or [np.nan])) for ty in ("A",)}
            clean = diff & (res["v_t"] == np.where(ta, "aff", "rej")) & (res["v_s"] == np.where(sa, "aff", "rej"))
            fl = {ty: float((res[f"v|{ty}:{nm}|{p}"][clean] == res["v_s"][clean]).mean()) for ty in ("A", "V", "AV", "Z")}
            lab = ("attention-routing-dominant" if tr["A"] >= 2 * tr["V"] and tr["A"] >= 0.10 else "value/content-dominant" if tr["V"] >= 2 * tr["A"] and tr["V"] >= 0.10 else "mixed/interactive")
            out["sets"][f"{nm}|{p}"] = dict(margin_transfer=tr, coordinate_transfer=trc, flip_to_source=fl, interaction_AV_minus_A_minus_V=float(tr["AV"] - tr["A"] - tr["V"]), interaction_ci95=[float(np.quantile(bi, .025)), float(np.quantile(bi, .975))],
                                            AV_minus_Z=float(tr["AV"] - tr["Z"]), label=lab)
    out["V5_AV_vs_fullz_S8_all_positions"] = dict(diff=abs(out["sets"]["S8|all"]["margin_transfer"]["AV"] - out["sets"]["S8|all"]["margin_transfer"]["Z"]), passed=bool(abs(out["sets"]["S8|all"]["margin_transfer"]["AV"] - out["sets"]["S8|all"]["margin_transfer"]["Z"]) <= 0.05))
    summary["formats"][fmt] = out
    print(fmt, "S8|all", out["sets"]["S8|all"]["margin_transfer"], "V5", out["V5_AV_vs_fullz_S8_all_positions"], flush=True)
    json.dump(summary, open(OUT / "av_patching.json", "w"), indent=1)
print("done", time.time() - T0)
