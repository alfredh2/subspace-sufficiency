"""Phase 3 Stage 2: confirmatory interventional mediation by PATCHING (docs/phase3_plan.md), test-content aligned pairs, both formats.
S8_L22 with 20 draws per pre-intervention control family; L22H5 and S8 (joint) with reduced controls. Usage: python scripts/phase3_patching.py"""
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from causal_subspaces import phase05 as P5
from causal_subspaces.data.logic import HELDOUT_FAMILIES, TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_splits
from causal_subspaces.heads import attention as A
from causal_subspaces.mediation import analysis as AN
from causal_subspaces.mediation import controls as CT
from causal_subspaces.mediation.runner import M_UNITS, run_cells
from causal_subspaces.mediation.setup import build_usets, family_of, load_phase3, response_sets
from causal_subspaces.models.loading import format_prompt
from causal_subspaces.utils.config import ROOT

T0 = time.time(); CAP = 38 * 60
OUT = ROOT / "results/phase3"
P = load_phase3(); model, tok, d_c, idx_hook, cfg = P["model"], P["tok"], P["d_c"], P["idx_hook"], P["cfg"]
splits = get_splits(cfg); contents = build_contents("ordinary")
fams = TRAIN_FAMILIES + HELDOUT_FAMILIES

def trials_for(fmt, ids_):
    f = P5.FORMATS[fmt]; ta_, tb_ = P5.single_token_id(tok, f["valid"]), P5.single_token_id(tok, f["invalid"]); tr = []
    for p in P5.build_pairs(contents, ids_, fams, tok):
        ok, _ = A.pair_alignment(tok, format_prompt(tok, P5.build_prompt(fmt, p["valid_syl"])), p["valid_syl"].split("\n"), format_prompt(tok, P5.build_prompt(fmt, p["invalid_syl"])), p["invalid_syl"].split("\n"))
        if not ok: continue
        v, i = P5.build_prompt(fmt, p["valid_syl"]), P5.build_prompt(fmt, p["invalid_syl"])
        tr += [dict(tgt=i, src=v, ta=ta_, tb=tb_, src_aff=True), dict(tgt=v, src=i, ta=ta_, tb=tb_, src_aff=False)]
    return tr

# response-selected controls (delta-matched, oracle) come from the VALIDATION pairs (never from the confirmatory pairs)
val_ids = sorted({e.content_id for e in splits["val"]}); test_ids = sorted({e.content_id for e in splits["items"]})
val_tr = trials_for("num_orig", val_ids); val_sgn = np.where([t["src_aff"] for t in val_tr], 1.0, -1.0)
RESP = {}
for hn in ("S8_L22", "L22H5"):
    r0 = run_cells(model, tok, val_tr, heads=P["heads"][hn], intervention="patch", usets={}, idx_hook=idx_hook, d_c=d_c, self_restore=())
    RESP[hn] = response_sets(r0["a_clean"][22], r0["a_inter"][22], val_sgn, P["U22"], P["g"][22])
RESP["S8"] = RESP["S8_L22"]
print("response sets from validation pairs ready", f"({time.time() - T0:.0f}s)", flush=True)

PLAN = {"S8_L22": dict(n_draw=20, mode="layer22", dose=True, full_layers=(22,)), "L22H5": dict(n_draw=5, mode="layer22", dose=True, full_layers=(22,)), "S8": dict(n_draw=5, mode="all", dose=False, full_layers=(19, 20, 21, 22))}
summary, raw = {"formats": {}}, {}
for fmt in ("num_orig", "num_rev"):
    tr = trials_for(fmt, test_ids); sgn = np.where([t["src_aff"] for t in tr], 1.0, -1.0); n = len(tr)
    summary["formats"][fmt] = {"n_trials": n, "headsets": {}}
    for hn, pl in PLAN.items():
        us = build_usets(P, hn, RESP[hn] if hn != "S8" else RESP["S8_L22"], pl["n_draw"], dose=pl["dose"], mode=pl["mode"])
        r = run_cells(model, tok, tr, heads=P["heads"][hn], intervention="patch", usets=us, idx_hook=idx_hook, d_c=d_c, full_layers=pl["full_layers"], bs=16, time_cap=CAP, t0=T0)
        nf = {o: float(max(np.abs(r["cells"]["self_clean"]["Y"][o] - r["clean"][o]).max(), np.abs(r["cells"]["self_inter"]["Y"][o] - r["inter"][o]).max())) for o in ("margin", "c")}
        sets = {}
        for name, cell in r["cells"].items():
            if name.startswith("self"): continue
            sets[name] = {o: AN.mediation_effects(r["clean"][o], r["inter"][o], cell["Y10"][o], cell["Y01"][o], sgn=sgn, floor=fl, noise_floor=nf[o], seed=1) for o, fl in (("c", AN.FLOORS["coordinate"]), ("margin", AN.FLOORS["margin"]))}
        ent = dict(noise_floor=nf, sets=sets, TE=dict(c=sets["U22"]["c"]["TE"], margin=sets["U22"]["margin"]["TE"]))
        # accounting / influence / concentration (layer 22): induced activation and verdict-directed write changes, aggregated and signed
        da = (r["a_inter"][22].astype(np.float32) - r["a_clean"][22].astype(np.float32)); ds = (sgn[:, None] * da * P["g"][22]).mean(0); tot = float(ds.sum())
        conc = {nm: float(ds[u[22]].sum() / tot) for nm, u in us.items() if 22 in u}
        fam_vals = {}
        for nm, v in conc.items():
            fm = family_of(nm)
            if fm in ("random", "act", "align"): fam_vals.setdefault(fm, []).append(v)
        ent["concentration"] = dict(shares=conc, total_induced_signed_write_change=tot, families={k: dict(mean=float(np.mean(v)), max=float(np.max(v)), n=len(v)) for k, v in fam_vals.items()},
                                    U22_vs_control_mean_ratio=float(conc["U22"] / max(np.mean([x for v in fam_vals.values() for x in v]), 1e-9)) if fam_vals else None,
                                    rank_corr_phase1_vs_induced=float(spearmanr(np.abs(P["D"][0].numpy()), np.abs(ds)).correlation))
        mi = r["mlp_in_inter"].astype(np.float32) - r["mlp_in_clean"].astype(np.float32)
        ent["strength"] = dict(mlp_input_change_norm_mean=float(np.linalg.norm(mi, axis=1).mean()), mlp_input_norm_clean=float(np.linalg.norm(r["mlp_in_clean"].astype(np.float32), axis=1).mean()),
                               sum_abs_dA={nm: float(np.abs(da[:, u[22]]).sum(1).mean()) for nm, u in us.items() if 22 in u and (nm in ("U22", "oracle", "delta_matched", "complement") or nm.endswith("0"))},
                               sum_abs_gdA={nm: float(np.abs(da[:, u[22]] * P["g"][22][u[22]]).sum(1).mean()) for nm, u in us.items() if 22 in u and (nm in ("U22", "oracle", "delta_matched", "complement") or nm.endswith("0"))})
        ent["reported_signed_aggregated"] = dict(direct_attn22_change=float((sgn * (r["inter"]["attn22"] - r["clean"]["attn22"])).mean()), mlp22_write_change=float((sgn * (r["inter"]["mlp22"] - r["clean"]["mlp22"])).mean()),
                                                 U22_write_change=float(ds[P["U22"]].sum()), U22_mean_abs_dA=float(np.abs(da[:, P["U22"]]).mean()), coordinate_change=float((sgn * (r["inter"]["c"] - r["clean"]["c"])).mean()),
                                                 margin_change=float((sgn * (r["inter"]["margin"] - r["clean"]["margin"])).mean()))
        mq = {}
        for nm, u in us.items():
            fm = family_of(nm)
            if fm in ("random", "act", "align") and 22 in u:
                mq.setdefault(fm, []).append({k: abs(v) for k, v in CT.matching_quality(dict(act_abs=P["feats"][22]["act"][:, 0], act_sq=P["feats"][22]["act"][:, 1], align=P["feats"][22]["align"][:, 0]), P["U22"], u[22]).items()})
        mq["delta_matched"] = [{"mean_abs_dA_smd": abs(CT.matching_quality(dict(f=np.log(RESP[hn]["mean_abs_da"] + 1e-9)), P["U22"], us["delta_matched"][22])["f"])}] if "delta_matched" in us else []
        ent["matching_quality_mean_abs_smd"] = {k: {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]} for k, v in mq.items() if v}
        summary["formats"][fmt]["headsets"][hn] = ent
        raw.update({f"{fmt}|{hn}|{nm}|{tag}|{o}": cell[tag][o] for nm, cell in r["cells"].items() if not nm.startswith("self") for tag in ("Y10", "Y01") for o in ("c", "margin") if tag in cell})
        raw.update({f"{fmt}|{hn}|clean|{o}": r["clean"][o] for o in ("c", "margin")}); raw.update({f"{fmt}|{hn}|inter|{o}": r["inter"][o] for o in ("c", "margin")}); raw[f"{fmt}|sgn"] = sgn
        u = sets["U22"]
        print(fmt, hn, f"({time.time() - T0:.0f}s) TE c/m {u['c']['TE']:.2f}/{u['margin']['TE']:.2f}; U22 R_c {u['c'].get('R', float('nan')):.3f} R_m {u['margin'].get('R', float('nan')):.3f} S_c {u['c'].get('S', float('nan')):.3f} S_m {u['margin'].get('S', float('nan')):.3f} | full22 R_c {sets['full22']['c'].get('R', float('nan')):.3f} R_m {sets['full22']['margin'].get('R', float('nan')):.3f} | concentration U22 {conc['U22']:.3f}", flush=True)
        json.dump(summary, open(OUT / "patching.json", "w"), indent=1, default=float)
np.savez_compressed(OUT / "patching_arrays.npz", **raw)
json.dump(dict(cap_gpu_hours=1.5, patching_seconds=time.time() - T0), open(OUT / "budget_patching.json", "w"))
print("done", time.time() - T0)
