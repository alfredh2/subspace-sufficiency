"""Shared setup for the Phase-3 scripts: frozen S8, Phase-1 unit ranking (U22, U_all), matching features, unit sets per head set, bootstrap/Holm comparisons."""
import json

import numpy as np
import torch
import yaml

from causal_subspaces.mediation import analysis as AN
from causal_subspaces.mediation import controls as CT
from causal_subspaces.mediation.runner import M_UNITS
from causal_subspaces.models.loading import load_model
from causal_subspaces.neurons.decompose import unit_gain
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything


def load_phase3():
    cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
    seed_everything(0)
    dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
    model, tok, rev = load_model(cfg["model"], dev)
    d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
    S8 = [(s["layer"], s["head"]) for s in json.load(open(ROOT / "results/phase2/selected_heads.json"))["S8"]]
    LS = json.load(open(ROOT / "results/phase1/stage1.json"))["selection"]["layers"]
    V1 = np.load(ROOT / "results/phase1/stage1_vectors.npz")
    NS = torch.load(ROOT / "results/phase1/neutral_unit_stats.pt")
    D = torch.stack([torch.tensor(V1["logic/mean_acc"][l] - V1["logic/mean_rej"][l]) * torch.tensor(V1["consensus/g"][l]) for l in LS])
    pos = D.abs().reshape(-1).topk(512).indices.numpy(); lay, ix = np.array(LS)[pos // M_UNITS], pos % M_UNITS
    U_all = {int(l): np.sort(ix[lay == l]) for l in LS}                       # the Phase-1 top-512 split by layer
    rank22 = ix[lay == 22]                                                    # layer-22 units in Phase-1 rank order
    g = {l: unit_gain(model, l, d_c).float().cpu().numpy() for l in LS}
    feats = {}
    for l in LS:
        fa = np.stack([np.log(NS[l]["abs"].numpy() + 1e-6), np.log(NS[l]["sq"].numpy() + 1e-9)], 1); fa = (fa - fa.mean(0)) / fa.std(0)
        fl = np.log(np.abs(g[l]) + 1e-9)[:, None]; fl = (fl - fl.mean()) / fl.std()
        feats[l] = dict(act=fa, align=fl)
    heads = {"S8_L22": [(l, h) for l, h in S8 if l == 22], "L22H5": [(22, 5)], "S8": S8}
    return dict(cfg=cfg, dev=dev, idx_hook=idx_hook, model=model, tok=tok, d_c=d_c, S8=S8, LS=LS, U22=np.sort(rank22), rank22=rank22, U_all=U_all, g=g, feats=feats, heads=heads, D=D, NS=NS)


def build_usets(P, hn, response, n_draw, dose=True, mode="layer22"):
    """Unit populations. mode 'layer22': everything in layer 22 (U22 native); 'all': U_all across layers 19-22 (each control drawn per layer with the native count)."""
    U22, f = P["U22"], P["feats"]
    us = {"U22": {22: U22}, "full22": {22: np.arange(M_UNITS)}, "complement": {22: CT.complement(M_UNITS, U22)}}
    if response is not None:
        us["delta_matched"] = {22: response["delta_matched"]}; us["oracle"] = {22: response["oracle"]}
    if dose:
        for k in (16, 64, 128): us[f"dose{k}"] = {22: P["rank22"][:k]}
    for d in range(n_draw):
        us[f"random{d}"] = {22: CT.random_set(M_UNITS, U22, d)}
        us[f"act{d}"] = {22: CT.nearest_set(f[22]["act"], U22, "act", d)}
        us[f"align{d}"] = {22: CT.nearest_set(f[22]["align"], U22, "align", d)}
    if mode == "all":
        us["U_all"] = {l: ix for l, ix in P["U_all"].items()}
        us["full_all"] = {l: np.arange(M_UNITS) for l in P["LS"]}
        for d in range(min(n_draw, 5)):
            us[f"random_all{d}"] = {l: CT.random_set(M_UNITS, ix, d) for l, ix in P["U_all"].items()}
    return us


def family_of(name):
    for fam in ("random_all", "random", "act", "align"):
        if name.startswith(fam) and name[len(fam):].isdigit():
            return fam
    return None


def response_sets(res_clean, res_inter, sgn, U22, g22):
    da = res_inter.astype(np.float32) - res_clean.astype(np.float32)
    mean_abs_da = np.abs(da).mean(0); ds = (sgn[:, None] * da * g22).mean(0)
    return dict(delta_matched=CT.delta_matched_set(mean_abs_da, U22), oracle=CT.oracle_set(ds, U22, len(U22)), mean_abs_da=mean_abs_da, ds=ds)
