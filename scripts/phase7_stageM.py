"""Phase 7 Stage M (docs/phase7_plan.md section 11): verdict-aligned versus orthogonal contributions of the layer-19 write, with norm/dose controls and a
geometric-versus-causal ranking comparison. Usage: python scripts/phase7_stageM.py {select|eval}"""
import json
import sys
import time

import numpy as np
import torch

from causal_subspaces.heads.ablation import capture_z
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.phase5 import pipeline as PL5
from causal_subspaces.phase5.patching import head_z_set_patch
from causal_subspaces.phase6.early import exit_after_layer, run_partial
from causal_subspaces.phase6.runner import _pad
from causal_subspaces.phase7 import stage_m as SM
from causal_subspaces.upstream import pipeline as UP
from causal_subspaces.upstream.patching import capture_component, component_patch, residual_set_patch
from causal_subspaces.utils.config import ROOT

stage = sys.argv[1]; T0 = time.time(); OUT = ROOT / "results/phase7"; OUT.mkdir(exist_ok=True)
LAYER, LOCAL, PIN_LAYERS = 19, 20, (20, 21, 22)
sel5 = json.load(open(ROOT / "results/phase5/selection.json")); Hs = sel5["H_star"]
P = PL5.setup(); c5 = P["ctx5"]; model, tok, dev = c5.model, c5.tok, c5.dev; d_c = c5.d_c.float(); H, HD = c5.H, c5.HD
Wo = model.model.layers[LAYER].self_attn.o_proj.weight.float()


def enc(b, s0):
    ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], dev); ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], dev)
    assert ids_t.shape == ids_s.shape and torch.equal(m_t, m_s); T = ids_t.shape[1]
    masks = {c: _pad(b, c, T, dev, s0) for c in ("G1G2", "G1", "G2")}
    return (ids_t, m_t, p_t), (ids_s, m_s, p_s), masks


def delta_of(b, s0, bs_idx):
    """Clean target/source passes and the per-position change of the layer output caused by the H* head patch at the summary rows."""
    (ids_t, m_t, p_t), (ids_s, m_s, p_s), masks = enc(b, s0); R = masks["G1G2"]
    zs, zt, ht, cT, cS, hS = {}, {}, [], [], [], []
    with capture_z(model, [LAYER], zs), capture_residual(model, 23, cS), capture_residual(model, LOCAL, hS): lg_s = last_logits(model, ids_s, m_s, p_s)
    comp = {"attn": {}, "mlp": {}}
    with capture_z(model, [LAYER], zt), capture_residual(model, 23, cT), capture_residual(model, LOCAL, ht), capture_component(model, list(PIN_LAYERS), "attn", comp["attn"]), capture_component(model, list(PIN_LAYERS), "mlp", comp["mlp"]):
        lg_t = last_logits(model, ids_t, m_t, p_t)
    store = {}; run_partial(model, ids_t, m_t, p_t, exit_after_layer(model, LAYER, store), [head_z_set_patch(model, LAYER, Hs, R, zs[LAYER])])
    h20t = ht[0]; D = (store["out"].float() - h20t.float()) * R[..., None].float()
    return dict(enc=((ids_t, m_t, p_t), (ids_s, m_s, p_s)), masks=masks, zs=zs[LAYER], zt=zt[LAYER], h20t=h20t, D=D, comp=comp, lg_t=lg_t, lg_s=lg_s, cT=cT[0][:, -1].float() @ d_c, cS=cS[0][:, -1].float() @ d_c)


if stage == "select":
    val, cen = UP.trial_sets(P, "val", "num_orig"); print("validation", len(val), cen, flush=True)
    sig = torch.tensor([t["sigma"] for t in val], device=dev); g1, g2, cnt = torch.zeros(d_c.numel(), device=dev), torch.zeros(d_c.numel(), device=dev), 0
    Dall, sall = [], []
    for s0 in range(0, len(val), 8):
        b = val[s0:s0 + 8]; ids_t, m_t, p_t = enc(b, s0)[0]; masks = enc(b, s0)[2]
        ta = torch.tensor([t["ta"] for t in b], device=dev); tb = torch.tensor([t["tb"] for t in b], device=dev); st = {}
        with torch.enable_grad(), SM.leaf_at_layer_output(model, LAYER, st):
            out = model.model(input_ids=ids_t, attention_mask=m_t, position_ids=p_t, use_cache=False); lg = model.lm_head(out.last_hidden_state[:, -1]).float()
            Y = lg.gather(1, ta[:, None])[:, 0] - lg.gather(1, tb[:, None])[:, 0]; Y.sum().backward()
        gr = st["leaf"].grad.float() * sig[s0:s0 + 8, None, None]
        m1, m2 = masks["G1"][..., None].float(), masks["G2"][..., None].float()
        g1 += (gr * m1).sum((0, 1)); g2 += ((gr * m2).sum(1) / m2.sum(1).clamp_min(1)).sum(0); cnt += len(b)
        with torch.no_grad():
            dd = delta_of(b, s0, None)["D"]; R = masks["G1G2"]
            Dall.append((dd * sig[s0:s0 + 8, None, None])[R].cpu())
    g1, g2 = g1 / cnt, g2 / cnt
    ortho = lambda g: torch.nn.functional.normalize(g - (g @ d_c) * d_c, dim=0)
    D_all = torch.cat(Dall); _, S_, Vh = torch.linalg.svd(D_all, full_matrices=False); pcs = Vh[:3].contiguous()
    res = dict(cos_g1_raw_dc=float(torch.nn.functional.cosine_similarity(g1, d_c, dim=0)), cos_g2_raw_dc=float(torch.nn.functional.cosine_similarity(g2, d_c, dim=0)),
               norm_g1=float(g1.norm()), norm_g2=float(g2.norm()), pc_var_frac=(S_[:5] ** 2 / (S_ ** 2).sum()).tolist(), cos_pc_dc=[float(p.cpu() @ d_c.cpu()) for p in pcs], cos_pc1_g2=float(pcs[0].cpu() @ torch.nn.functional.normalize(g2, dim=0).cpu()),
               mean_delta_norm=float(D_all.norm(dim=-1).mean()), n_rows=int(D_all.shape[0]), n_trials=cnt)
    torch.save(dict(g_G1=ortho(g1).cpu(), g_G2=ortho(g2).cpu(), g1_raw=g1.cpu(), g2_raw=g2.cpu(), pcs=pcs.cpu()), OUT / "stageM_selection.pt"); json.dump(res, open(OUT / "stageM_selection.json", "w"), indent=1)
    print(json.dumps(res, indent=1), f"\nselect done {time.time() - T0:.0f}s", flush=True)

elif stage == "eval":
    selp = torch.load(OUT / "stageM_selection.pt"); b1, b2, pcs = selp["g_G1"].to(dev), selp["g_G2"].to(dev), selp["pcs"].to(dev)
    PINNED = ("full", "dc", "orth", "geff", "rest")
    for which, fmt in (("A", "num_orig"), ("A", "num_rev"), ("B", "num_orig"), ("B", "num_rev")):
        t1 = time.time(); trials, cen = UP.trial_sets(P, which, fmt); n = len(trials); sig = torch.tensor([t["sigma"] for t in trials], device=dev)
        ta_all = torch.tensor([t["ta"] for t in trials], device=dev); tb_all = torch.tensor([t["tb"] for t in trials], device=dev)
        res, base = {}, dict(tgt_margin=np.zeros(n), src_margin=np.zeros(n), tgt_c=np.zeros(n), src_c=np.zeros(n))
        stats = {k: np.zeros(n) for k in ("norm_full", "norm_dc", "norm_orth", "norm_geff", "cos_full_dc", "scale_dc_eqorth", "scale_dc_eqfull")}
        hp = {k: np.zeros((n, H)) for k in ("proj_dc", "proj_g", "wnorm")}
        for s0 in range(0, n, 32):
            b = trials[s0:s0 + 32]; sl = slice(s0, s0 + len(b)); out = delta_of(b, s0, None); (ids_t, m_t, p_t), _ = out["enc"]; masks = out["masks"]; R = masks["G1G2"]
            ta, tb = ta_all[sl], tb_all[sl]; marg = lambda lg: (lg.gather(1, ta[:, None])[:, 0] - lg.gather(1, tb[:, None])[:, 0])
            base["tgt_margin"][sl] = marg(out["lg_t"]).cpu().numpy(); base["src_margin"][sl] = marg(out["lg_s"]).cpu().numpy(); base["tgt_c"][sl] = out["cT"].cpu().numpy(); base["src_c"][sl] = out["cS"].cpu().numpy()
            U = masks["G1"][..., None].float() * b1 + masks["G2"][..., None].float() * b2
            conds = SM.conditions(out["D"], R, d_c, U, pcs, tag=f"{which}|{fmt}|{s0}")
            D = out["D"]; m = R[..., None].float(); cnt = m.sum((1, 2)).clamp_min(1)
            rm = lambda x: ((x.norm(dim=-1)) * R.float()).sum(1) / R.float().sum(1).clamp_min(1)
            stats["norm_full"][sl] = rm(conds["full"]).cpu().numpy(); stats["norm_dc"][sl] = rm(conds["dc"]).cpu().numpy(); stats["norm_orth"][sl] = rm(conds["orth"]).cpu().numpy(); stats["norm_geff"][sl] = rm(conds["geff"]).cpu().numpy()
            cosr = (D * d_c).sum(-1) / D.norm(dim=-1).clamp_min(1e-9); stats["cos_full_dc"][sl] = ((cosr * R.float()).sum(1) / R.float().sum(1).clamp_min(1)).cpu().numpy()
            stats["scale_dc_eqorth"][sl] = rm(conds["dc_eqorth"]).cpu().numpy() / np.maximum(stats["norm_dc"][sl], 1e-9); stats["scale_dc_eqfull"][sl] = rm(conds["dc_eqfull"]).cpu().numpy() / np.maximum(stats["norm_dc"][sl], 1e-9)
            dz = (out["zs"].float() - out["zt"].float()) * R[..., None].float(); w = R.float() / R.float().sum(1, keepdim=True).clamp_min(1)
            ph_dc, wn = SM.head_projections(dz, Wo, d_c.expand(*dz.shape[:2], -1), H, HD); ph_g, _ = SM.head_projections(dz, Wo, U, H, HD)
            for k, v in (("proj_dc", ph_dc), ("proj_g", ph_g), ("wnorm", wn)): hp[k][sl] = (v * w[..., None]).sum(1).cpu().numpy()
            def run(delta, pins=False):
                zs_, c23 = {}, []
                from contextlib import ExitStack
                with ExitStack() as es:
                    es.enter_context(residual_set_patch(model, LOCAL, R, out["h20t"], delta=delta))
                    if pins:
                        for l in PIN_LAYERS:
                            for c in ("attn", "mlp"): es.enter_context(component_patch(model, l, c, R, out["comp"][c][l]))
                    es.enter_context(capture_residual(model, 23, c23)); lg = last_logits(model, ids_t, m_t, p_t)
                return marg(lg).cpu().numpy(), (c23[0][:, -1].float() @ d_c).cpu().numpy()
            for name, dl in conds.items():
                mg, cc = run(dl); res.setdefault(name, dict(margin=np.zeros(n), c=np.zeros(n)))
                res[name]["margin"][sl], res[name]["c"][sl] = mg, cc
            for name in PINNED:
                mg, cc = run(conds[name], pins=True); res.setdefault(f"{name}|pin", dict(margin=np.zeros(n), c=np.zeros(n)))
                res[name + "|pin"]["margin"][sl], res[name + "|pin"]["c"][sl] = mg, cc
        np.savez_compressed(OUT / f"stageM_{which}_{fmt}.npz", sigma=UP.sigma_of(trials), pair=np.array([t["pair"] for t in trials]), content=np.array([t["content"] for t in trials]),
                            **{f"base|{k}": v for k, v in base.items()}, **{f"stats|{k}": v for k, v in stats.items()}, **{f"head|{k}": v for k, v in hp.items()}, **{f"{c}|{k}": v for c, d in res.items() for k, v in d.items()})
        print(which, fmt, n, "trials", len(res), "conditions", f"{time.time() - t1:.0f}s", flush=True)
