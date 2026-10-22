"""Phase 3 Stage 0 preflight + Stage 1 pilot (docs/phase3_plan.md): validation-content aligned pairs, num_orig, S8_L22 and L22H5, U22 with matched
controls (5 draws per family), complement, full MLP22, dose-response, delta-matched and oracle sets (derived from the pilot response). Usage: python scripts/phase3_pilot.py"""
import json
import time

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from causal_subspaces import phase05 as P5
from causal_subspaces.data.logic import HELDOUT_FAMILIES, TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_splits
from causal_subspaces.heads import attention as A
from causal_subspaces.heads.ablation import head_z_patch, capture_z
from causal_subspaces.heads.decompose import head_gain
from causal_subspaces.mediation import analysis as AN
from causal_subspaces.mediation import controls as CT
from causal_subspaces.mediation.runner import M_UNITS, run_cells
from causal_subspaces.mediation.units import OrderLog, capture_units, log_modules, restore_units
from causal_subspaces.models.loading import encode, format_prompt, last_logits, load_model
from causal_subspaces.neurons.decompose import unit_gain
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

T0 = time.time(); OUT = ROOT / "results/phase3"; OUT.mkdir(exist_ok=True)
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
S8 = [(s["layer"], s["head"]) for s in json.load(open(ROOT / "results/phase2/selected_heads.json"))["S8"]]
HEADSETS = {"S8_L22": [(l, h) for l, h in S8 if l == 22], "L22H5": [(22, 5)]}
LS = json.load(open(ROOT / "results/phase1/stage1.json"))["selection"]["layers"]
V1 = np.load(ROOT / "results/phase1/stage1_vectors.npz")
NS = torch.load(ROOT / "results/phase1/neutral_unit_stats.pt")
g22 = unit_gain(model, 22, d_c).float().cpu().numpy()

# ---- U22 = layer-22 units of the Phase-1 top-512 (ranked by |Delta| = |class-mean difference x gain|) -------------------------------------------------
D = torch.stack([torch.tensor(V1["logic/mean_acc"][l] - V1["logic/mean_rej"][l]) * torch.tensor(V1["consensus/g"][l]) for l in LS])
pos = D.abs().reshape(-1).topk(512).indices.numpy(); lay, ix = np.array(LS)[pos // M_UNITS], pos % M_UNITS
rank_order = ix[lay == 22]                                              # layer-22 units in Phase-1 rank order
U22 = np.sort(rank_order); print("U22 size", len(U22), flush=True)
feat_act = np.stack([np.log(NS[22]["abs"].numpy() + 1e-6), np.log(NS[22]["sq"].numpy() + 1e-9)], 1); feat_act = (feat_act - feat_act.mean(0)) / feat_act.std(0)
feat_al = np.log(np.abs(g22) + 1e-9)[:, None]; feat_al = (feat_al - feat_al.mean()) / feat_al.std()

# ---- preflight: explicit hook-order verification on Qwen3 for every layer that carries an intervened head or a mediator --------------------------------------
splits = get_splits(cfg); contents = build_contents("ordinary"); val_ids = sorted({e.content_id for e in splits["val"]})
pairs = P5.build_pairs(contents, val_ids, TRAIN_FAMILIES + HELDOUT_FAMILIES, tok)
def trials_for(fmt):
    f = P5.FORMATS[fmt]; ta_, tb_ = P5.single_token_id(tok, f["valid"]), P5.single_token_id(tok, f["invalid"]); tr = []
    for p in pairs:
        ok, _ = A.pair_alignment(tok, format_prompt(tok, P5.build_prompt(fmt, p["valid_syl"])), p["valid_syl"].split("\n"), format_prompt(tok, P5.build_prompt(fmt, p["invalid_syl"])), p["invalid_syl"].split("\n"))
        if not ok: continue
        v, i = P5.build_prompt(fmt, p["valid_syl"]), P5.build_prompt(fmt, p["invalid_syl"])
        tr += [dict(tgt=i, src=v, ta=ta_, tb=tb_, src_aff=True), dict(tgt=v, src=i, ta=ta_, tb=tb_, src_aff=False)]
    return tr
trials = trials_for("num_orig"); sgn = np.where([t["src_aff"] for t in trials], 1.0, -1.0)
print("pilot trials", len(trials), flush=True)
log = OrderLog(); ids, mask, pos_ = encode(tok, [format_prompt(tok, t["tgt"], False) for t in trials[:4]], dev)
ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in trials[:4]], dev); zs = {}
with torch.no_grad(), capture_z(model, [22, 21], zs):
    last_logits(model, ids_s, m_s, p_s)
dummy = {22: torch.zeros(4, len(U22), device=dev), 21: torch.zeros(4, 3, device=dev)}
with torch.no_grad(), log_modules(model, [20, 21, 22], log), head_z_patch(model, {22: [5], 21: [19]}, zs, idx_hook), capture_units(model, {22: torch.as_tensor(U22), 21: torch.arange(3)}, {}, log=log), restore_units(model, {22: torch.as_tensor(U22), 21: torch.arange(3)}, dummy, log=log):
    last_logits(model, ids, mask, pos_)
order_ok = True
for l in (21, 22):
    p = {n: log.positions(n, l)[0] for n in ("o_proj_pre", "attn_out", "down_proj_pre", "mlp_out")}
    order_ok &= p["o_proj_pre"] < p["attn_out"] < p["down_proj_pre"] < p["mlp_out"]
    for n in ("unit_capture", "unit_restore"):
        order_ok &= log.positions("attn_out", l)[0] < log.positions(n, l)[0] < log.positions("mlp_out", l)[0]
order_ok &= log.positions("down_proj_pre", 21)[0] < log.positions("o_proj_pre", 22)[0]          # layer 21 completes before layer 22 attention
print("PREFLIGHT hook order verified on Qwen3:", order_ok, flush=True)
assert order_ok

# ---- pass 1: response of layer-22 units to the head intervention (pilot pairs) -> delta-matched and oracle sets ------------------------------------------
res_sets, pilot = {}, {}
for hn, hs in HEADSETS.items():
    r0 = run_cells(model, tok, trials, heads=hs, intervention="patch", usets={}, idx_hook=idx_hook, d_c=d_c, self_restore=())
    da = r0["a_inter"][22].astype(np.float32) - r0["a_clean"][22].astype(np.float32)
    mean_abs_da = np.abs(da).mean(0); ds = (sgn[:, None] * da * g22).mean(0)                           # signed, toward-source verdict-directed change per unit
    res_sets[hn] = dict(delta_matched=CT.delta_matched_set(mean_abs_da, U22), oracle=CT.oracle_set(ds, U22, len(U22)), mean_abs_da=mean_abs_da, ds=ds)
    pilot[hn] = dict(TE_c=float((sgn * (r0["inter"]["c"] - r0["clean"]["c"])).mean()), TE_m=float((sgn * (r0["inter"]["margin"] - r0["clean"]["margin"])).mean()))
    print(hn, "pilot TE", pilot[hn], flush=True)

# ---- pass 2: the cells ---------------------------------------------------------------------------------------------------------------------------------------
def usets_for(hn):
    us = {"U22": {22: U22}, "full22": {22: np.arange(M_UNITS)}, "complement": {22: CT.complement(M_UNITS, U22)},
          "delta_matched": {22: res_sets[hn]["delta_matched"]}, "oracle": {22: res_sets[hn]["oracle"]}}
    for k in (16, 64, 128): us[f"dose{k}"] = {22: rank_order[:k]}
    for d in range(5):
        us[f"random{d}"] = {22: CT.random_set(M_UNITS, U22, d)}; us[f"act{d}"] = {22: CT.nearest_set(feat_act, U22, "act", d)}; us[f"align{d}"] = {22: CT.nearest_set(feat_al, U22, "align", d)}
    return us
summary = {"preflight_order_ok": bool(order_ok), "n_trials": len(trials), "U22": len(U22), "pilot_TE": pilot, "headsets": {}}
raw = {}
for hn, hs in HEADSETS.items():
    us = usets_for(hn)
    r = run_cells(model, tok, trials, heads=hs, intervention="patch", usets=us, idx_hook=idx_hook, d_c=d_c)
    raw[hn] = r
    sc = r["cells"]["self_clean"]["Y"]; si = r["cells"]["self_inter"]["Y"]
    nf = {o: float(max(np.abs(sc[o] - r["clean"][o]).max(), np.abs(si[o] - r["inter"][o]).max())) for o in ("margin", "c")}
    out = {"noise_floor": nf, "sets": {}}
    for name, cell in r["cells"].items():
        if name.startswith("self"): continue
        row = {}
        for o, fl in (("c", AN.FLOORS["coordinate"]), ("margin", AN.FLOORS["margin"])):
            row[o] = AN.mediation_effects(r["clean"][o], r["inter"][o], cell["Y10"][o], cell["Y01"][o], sgn=sgn, floor=fl, noise_floor=nf[o])
        out["sets"][name] = row
    # T1/T2: concentration of the induced verdict-directed write change (layer 22): share carried by each unit set
    ds = res_sets[hn]["ds"]; tot = ds.sum()
    out["concentration"] = {nm: float(ds[u[22]].sum() / tot) for nm, u in us.items()}
    out["rank_corr_phase1_vs_induced_abs_ds"] = float(spearmanr(np.abs(D[0].numpy()), np.abs(ds)).correlation)
    out["induced_total_ds22"] = float(tot)
    # separately reported quantities (aggregated, signed): direct attention change, U22 activation/write changes, coordinate, margin
    a_c, a_i = r["clean"], r["inter"]
    out["reported"] = dict(direct_attn22_change=float((sgn * (a_i["attn22"] - a_c["attn22"])).mean()), mlp22_write_change=float((sgn * (a_i["mlp22"] - a_c["mlp22"])).mean()),
                           U22_write_change=float(ds[U22].sum()), U22_mean_abs_dA=float(res_sets[hn]["mean_abs_da"][U22].mean()), coordinate_change=float((sgn * (a_i["c"] - a_c["c"])).mean()),
                           margin_change=float((sgn * (a_i["margin"] - a_c["margin"])).mean()))
    summary["headsets"][hn] = out
    print(hn, "noise floors", nf, "| U22:", {o: {k: round(v, 3) if isinstance(v, float) else v for k, v in out["sets"]["U22"][o].items() if k in ("TE", "R", "B", "S", "J", "undefined", "reason")} for o in ("c", "margin")},
          "| full22:", {o: {k: round(v, 3) for k, v in out["sets"]["full22"][o].items() if k in ("R", "B")} for o in ("c", "margin") if not out["sets"]["full22"][o].get("undefined")}, flush=True)
json.dump(summary, open(OUT / "pilot.json", "w"), indent=1, default=float)
np.savez_compressed(OUT / "pilot_arrays.npz", sgn=sgn, **{f"{hn}|{k}": v for hn, r in raw.items() for k, v in [("clean_margin", r["clean"]["margin"]), ("inter_margin", r["inter"]["margin"]), ("clean_c", r["clean"]["c"]), ("inter_c", r["inter"]["c"])]})
json.dump(dict(cap_gpu_hours=1.5, pilot_seconds=time.time() - T0), open(OUT / "budget_pilot.json", "w"))
print("done", time.time() - T0)
