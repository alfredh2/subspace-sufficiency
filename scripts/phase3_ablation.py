"""Phase 3 Stage 3: ablation-based interventional mediation on the Phase-2 confirmatory class-level set (logic in both formats, localizer, PrOntoQA, and the
verdict-free negative controls). Head sets S8_L22 and S8; mediator sets as in the plan. Usage: python scripts/phase3_ablation.py"""
import json
import time

import numpy as np
import torch

from causal_subspaces.heads.evalset import EvalSet
from causal_subspaces.mediation import analysis as AN
from causal_subspaces.mediation import controls as CT
from causal_subspaces.mediation.runner import M_UNITS, run_cells
from causal_subspaces.mediation.setup import build_usets, family_of, load_phase3
from causal_subspaces.utils.config import ROOT

T0 = time.time(); CAP = 30 * 60
OUT = ROOT / "results/phase3"
P = load_phase3(); model, tok, d_c, idx_hook, cfg, dev = P["model"], P["tok"], P["d_c"], P["idx_hook"], P["cfg"], P["dev"]
ev = EvalSet(model, tok, cfg, "confirmatory", dev)
NSH = torch.load(ROOT / "results/phase2/neutral_head_stats.pt"); zbar = {l: NSH["zbar"][l].to(dev) for l in range(idx_hook)}
items = [dict(tgt=p, ta=int(a), tb=int(b)) for p, a, b in zip(ev.prompts, ev.ta.cpu().numpy(), ev.tb.cpu().numpy())]
groups, accept = ev.groups, ev.accept
print("class-level eval items", len(items), {g: int((groups == g).sum()) for g in np.unique(groups)}, flush=True)
# response-selected controls from the pilot (validation pairs, patching) are not applicable to ablation; derive delta-matched/oracle from the ablation response on the
# EXPLORATORY Phase-1/2 subset instead (never from the confirmatory set)
ev_e = EvalSet(model, tok, cfg, "exploratory", dev)
items_e = [dict(tgt=p, ta=int(a), tb=int(b)) for p, a, b in zip(ev_e.prompts, ev_e.ta.cpu().numpy(), ev_e.tb.cpu().numpy())]
sg1 = np.where(ev_e.accept, 1.0, -1.0)
RESP = {}
for hn in ("S8_L22", "S8"):
    r0 = run_cells(model, tok, items_e, heads=P["heads"][hn], intervention="ablate", zbar=zbar, usets={}, idx_hook=idx_hook, d_c=d_c, self_restore=(), formatted=True, bs=32)
    da = r0["a_inter"][22].astype(np.float32) - r0["a_clean"][22].astype(np.float32); ds = (da * P["g"][22]).mean(0)
    RESP[hn] = dict(delta_matched=CT.delta_matched_set(np.abs(da).mean(0), P["U22"]), oracle=CT.oracle_set(ds, P["U22"], len(P["U22"])), mean_abs_da=np.abs(da).mean(0), ds=ds)
print("response sets ready", f"({time.time() - T0:.0f}s)", flush=True)
TASKS = {"logic_num_orig": ("c", "margin"), "logic_num_rev": ("c", "margin"), "localizer": ("c", "margin"), "prontoqa": ("margin",), "mmlu_binary": ("margin",), "stroop": ("margin",), "word_recall_48": ("margin",)}
FL = dict(c=AN.FLOORS["gap"], margin=AN.FLOORS["gap"])
summary = {"headsets": {}}
raw = {}
for hn, mode in (("S8_L22", "layer22"), ("S8", "all")):
    us = build_usets(P, hn, RESP[hn], 10, dose=True, mode=mode)
    core = [k for k in us if not family_of(k)]                                  # U22, full, complement, delta, oracle, dose..., U_all...
    ctrls = {k: us[k] for k in us if family_of(k)}
    full_layers = (22,) if mode == "layer22" else (19, 20, 21, 22)
    rA = run_cells(model, tok, items, heads=P["heads"][hn], intervention="ablate", zbar=zbar, usets={k: us[k] for k in core}, idx_hook=idx_hook, d_c=d_c, full_layers=full_layers, formatted=True, bs=32, time_cap=CAP, t0=T0)
    print(hn, "core cells done", f"({time.time() - T0:.0f}s)", flush=True)
    rB = run_cells(model, tok, items, heads=P["heads"][hn], intervention="ablate", zbar=zbar, usets=ctrls, idx_hook=idx_hook, d_c=d_c, full_layers=full_layers, formatted=True, bs=32, time_cap=CAP, t0=T0, restore_tags=("Y10",), self_restore=())
    cells = {**rA["cells"], **rB["cells"]}
    nf = {o: float(max(np.abs(rA["cells"]["self_clean"]["Y"][o] - rA["clean"][o]).max(), np.abs(rA["cells"]["self_inter"]["Y"][o] - rA["inter"][o]).max())) for o in ("margin", "c")}
    ent = {"noise_floor": nf, "tasks": {}}
    for tk, outs in TASKS.items():
        idx = np.where(groups == tk)[0]; acc = accept[idx]; tko = {}
        for o in outs:
            row = {}
            for name, cell in cells.items():
                if name.startswith("self"): continue
                y01 = cell["Y01"][o][idx] if "Y01" in cell else cell["Y10"][o][idx]
                e = AN.gap_effects(rA["clean"][o][idx], rA["inter"][o][idx], cell["Y10"][o][idx], y01, acc, floor=FL[o], noise_floor=nf[o], seed=2)
                if "Y01" not in cell: e["S"] = e["J"] = e["Phi"] = None
                row[name] = e
            tko[o] = row
        ent["tasks"][tk] = tko
        u = tko["margin"]["U22"]; print(hn, tk, "margin:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in u.items() if k in ("TE", "R", "B", "S", "J", "undefined", "reason")}, flush=True)
    # verdict-free tasks: signed mean shifts (not class gaps) -- raw, aggregated
    vf = {}
    for tk in ("mmlu_binary", "stroop", "word_recall_48"):
        idx = np.where(groups == tk)[0]; m00, m11 = rA["clean"]["margin"][idx], rA["inter"]["margin"][idx]
        vf[tk] = dict(head_ablation_mean_shift=float((m11 - m00).mean()), U22_restored_mean_shift=float((cells["U22"]["Y10"]["margin"][idx] - m00).mean()),
                      random_restored_mean_shift_range=[float(min((cells[k]["Y10"]["margin"][idx] - m00).mean() for k in ctrls)), float(max((cells[k]["Y10"]["margin"][idx] - m00).mean() for k in ctrls))] if ctrls else None)
    ent["verdict_free_mean_shifts"] = vf
    summary["headsets"][hn] = ent
    raw.update({f"{hn}|clean|{o}": rA["clean"][o] for o in ("c", "margin")}); raw.update({f"{hn}|inter|{o}": rA["inter"][o] for o in ("c", "margin")})
    raw.update({f"{hn}|{nm}|{tag}|{o}": cell[tag][o] for nm, cell in cells.items() if not nm.startswith("self") for tag in ("Y10", "Y01") for o in ("c", "margin") if tag in cell})
    json.dump(summary, open(OUT / "ablation.json", "w"), indent=1, default=float)
raw["groups"] = groups; raw["accept"] = accept
np.savez_compressed(OUT / "ablation_arrays.npz", **raw)
json.dump(dict(cap_gpu_hours=1.5, ablation_seconds=time.time() - T0), open(OUT / "budget_ablation.json", "w"))
print("done", time.time() - T0)
