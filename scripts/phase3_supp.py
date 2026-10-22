"""Phase 3 Stage 4 (supplementary, exploratory): earlier-layer heads (mediators restricted to layers >= the head's layer), coordinate-clamp run, all-query-rows setting.
Test-content aligned pairs. Usage: python scripts/phase3_supp.py"""
import json
import time

import numpy as np

from causal_subspaces import phase05 as P5
from causal_subspaces.data.logic import HELDOUT_FAMILIES, TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_splits
from causal_subspaces.heads import attention as A
from causal_subspaces.mediation import analysis as AN
from causal_subspaces.mediation import controls as CT
from causal_subspaces.mediation.runner import M_UNITS, run_cells
from causal_subspaces.mediation.setup import load_phase3
from causal_subspaces.models.loading import format_prompt
from causal_subspaces.utils.config import ROOT

T0 = time.time(); CAP = 15 * 60
OUT = ROOT / "results/phase3"
P = load_phase3(); model, tok, d_c, idx_hook, cfg = P["model"], P["tok"], P["d_c"], P["idx_hook"], P["cfg"]
splits = get_splits(cfg); contents = build_contents("ordinary"); test_ids = sorted({e.content_id for e in splits["items"]})
def trials_for(fmt):
    f = P5.FORMATS[fmt]; ta_, tb_ = P5.single_token_id(tok, f["valid"]), P5.single_token_id(tok, f["invalid"]); tr = []
    for p in P5.build_pairs(contents, test_ids, TRAIN_FAMILIES + HELDOUT_FAMILIES, tok):
        ok, _ = A.pair_alignment(tok, format_prompt(tok, P5.build_prompt(fmt, p["valid_syl"])), p["valid_syl"].split("\n"), format_prompt(tok, P5.build_prompt(fmt, p["invalid_syl"])), p["invalid_syl"].split("\n"))
        if not ok: continue
        v, i = P5.build_prompt(fmt, p["valid_syl"]), P5.build_prompt(fmt, p["invalid_syl"])
        tr += [dict(tgt=i, src=v, ta=ta_, tb=tb_, src_aff=True), dict(tgt=v, src=i, ta=ta_, tb=tb_, src_aff=False)]
    return tr
def eff(r, cell, sgn, nf):
    return {o: AN.mediation_effects(r["clean"][o], r["inter"][o], cell["Y10"][o], cell["Y01"][o] if "Y01" in cell else cell["Y10"][o], sgn=sgn, floor=fl, noise_floor=nf[o], seed=3)
            for o, fl in (("c", AN.FLOORS["coordinate"]), ("margin", AN.FLOORS["margin"]))}
summary = {}
tr = trials_for("num_orig"); sgn = np.where([t["src_aff"] for t in tr], 1.0, -1.0)
# (a) earlier-layer heads: mediators = Phase-1 top units / all units at layers >= head layer (units at lower layers cannot be affected: causality)
summary["earlier_layer_heads"] = {}
for l_h, h in ((21, 19), (20, 8), (19, 10)):
    ls = [l for l in P["LS"] if l >= l_h]
    us = {"U_geL": {l: P["U_all"][l] for l in ls}, "full_geL": {l: np.arange(M_UNITS) for l in ls}}
    for d in range(5): us[f"random{d}"] = {l: CT.random_set(M_UNITS, P["U_all"][l], d) for l in ls}
    r = run_cells(model, tok, tr, heads=[(l_h, h)], intervention="patch", usets=us, idx_hook=idx_hook, d_c=d_c, full_layers=tuple(ls), self_restore=(), time_cap=CAP, t0=T0)
    nf = {"c": 0.0, "margin": 0.0}
    row = {nm: eff(r, cell, sgn, nf) for nm, cell in r["cells"].items()}
    summary["earlier_layer_heads"][f"L{l_h}H{h}"] = dict(layers=ls, sets=row, direct_attn_change={f"attn{l}": float((sgn * (r["inter"][f"attn{l}"] - r["clean"][f"attn{l}"])).mean()) for l in (19, 20, 21, 22)},
                                                      mlp_write_change={f"mlp{l}": float((sgn * (r["inter"][f"mlp{l}"] - r["clean"][f"mlp{l}"])).mean()) for l in (19, 20, 21, 22)})
    u = row["U_geL"]; print(f"L{l_h}H{h}", f"({time.time() - T0:.0f}s)", {o: {k: round(v, 3) for k, v in u[o].items() if k in ("TE", "R", "B", "S")} if not u[o].get("undefined") else u[o]["reason"] for o in ("c", "margin")}, flush=True)
    json.dump(summary, open(OUT / "supp.json", "w"), indent=1, default=float)
# (b) coordinate clamp + (c) all-query-rows, S8_L22, both formats for the clamp
summary["coordinate_clamp"] = {}
for fmt in ("num_orig", "num_rev"):
    trf = trials_for(fmt); sg = np.where([t["src_aff"] for t in trf], 1.0, -1.0)
    r = run_cells(model, tok, trf, heads=P["heads"]["S8_L22"], intervention="patch", usets={"U22": {22: P["U22"]}}, idx_hook=idx_hook, d_c=d_c, clamp=True, time_cap=CAP, t0=T0)
    te = {o: float((sg * (r["inter"][o] - r["clean"][o])).mean()) for o in ("c", "margin")}
    cl = {o: float((sg * (r["cells"]["clamp_dc"]["Y"][o] - r["clean"][o])).mean()) for o in ("c", "margin")}
    summary["coordinate_clamp"][fmt] = dict(TE=te, effect_with_coordinate_clamped=cl, fraction_margin_effect_not_through_dc_coordinate=cl["margin"] / te["margin"] if abs(te["margin"]) > 5 else None,
                                            coordinate_after_clamp_residual=cl["c"])
    print("clamp", fmt, summary["coordinate_clamp"][fmt], flush=True)
tr_all = trials_for("num_orig")
r = run_cells(model, tok, tr_all, heads=P["heads"]["S8_L22"], intervention="patch", usets={"U22": {22: P["U22"]}, "full22": {22: np.arange(M_UNITS)}, "random0": {22: CT.random_set(M_UNITS, P["U22"], 0)}}, idx_hook=idx_hook, d_c=d_c, position="all", bs=8, self_restore=(), time_cap=CAP, t0=T0)
summary["all_query_rows"] = {nm: eff(r, cell, sgn, {"c": 0.0, "margin": 0.0}) for nm, cell in r["cells"].items()}
print("all-rows U22:", {o: {k: round(v, 3) for k, v in summary["all_query_rows"]["U22"][o].items() if k in ("TE", "R", "B", "S", "J")} for o in ("c", "margin") if not summary["all_query_rows"]["U22"][o].get("undefined")}, flush=True)
json.dump(summary, open(OUT / "supp.json", "w"), indent=1, default=float)
json.dump(dict(cap_gpu_hours=1.5, supp_seconds=time.time() - T0), open(OUT / "budget_supp.json", "w"))
print("done", time.time() - T0)
