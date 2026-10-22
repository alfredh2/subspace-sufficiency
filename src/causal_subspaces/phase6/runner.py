"""Phase 6 job runner (docs/phase6_plan.md): two-pass restricted origin experiments, unrestricted residual/component patches, pinning with accounting,
d_c / orthogonal decomposition of the layer-19 write."""
import time
import zlib
from contextlib import ExitStack

import numpy as np
import torch

from causal_subspaces.heads import positions as POS
from causal_subspaces.heads.ablation import capture_z
from causal_subspaces.heads.attention import AttnStash, recompute
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.phase5.patching import attention_rows_patch, head_z_set_patch
from causal_subspaces.phase6.data import cond_mask6
from causal_subspaces.phase6.early import exit_after_layer, exit_before_attention, run_partial
from causal_subspaces.upstream.patching import capture_component, component_patch, forward_states, residual_set_patch

LAYER = 19            # layer whose heads H* read the values
LOCAL = LAYER + 1     # hook index of the layer output
FINAL = 23            # verdict hook
ACC_LAYERS = (20, 21, 22)
ROWS, KEYS = "G1G2", "KV"
COMPS = ("attn", "mlp")


def _pad(b, cond, T, dev, s0=0):
    return torch.tensor(np.stack([POS.pad_mask(cond_mask6(t, cond, s0 + i), T) for i, t in enumerate(b)])).to(dev)


def _rmean(x, rows):
    m = rows.float(); return (x * m).sum(1) / m.sum(1).clamp_min(1)


def _hr(ctx, z, rows, heads):
    """d_c read-out of the listed heads (sum_h gain_h . z_h) averaged over the rows; z: (B,T,H*HD)."""
    B, T, _ = z.shape; zz = z.view(B, T, ctx.H, ctx.HD).float()
    return _rmean(sum((zz[:, :, h] * ctx.gain[h]).sum(-1) for h in heads), rows)


def _fresh(n, dev, extra=()):
    d = {k: torch.zeros(n, device=dev) for k in ("margin", "c", "c20", "w20", "hr")}
    d.update({k: torch.zeros(n, device=dev) for k in extra}); return d


@torch.no_grad()
def run_jobs(ctx, heads, trials, jobs, *, bs=32, t0=None, cap=None, comp_layers=range(LAYER)):
    """jobs[i] = dict(name, kind in {'origin','u','pin','dcomp'}, ...). Per-job arrays: margin, c (hook FINAL), c20/w20/hr (rows G1G2 of the layer output:
    d_c / w coordinates and H* read-out), origin: eq_frac, maxdiff, relnorm, vp_num, vp_den; pin/dcomp: dC (n,4) d_c at indices 20..23, W (n,6) d_c writes of
    layers 20-22 (attn, mlp). Returns (base_tgt, base_src, results)."""
    model, tok, dev, n = ctx.model, ctx.tok, ctx.dev, len(trials)
    need_comp = any(j["kind"] in ("origin", "u") and j["spec"]["type"].startswith("comp") for j in jobs)
    need_av = any(j["kind"] == "origin" for j in jobs)
    nW = 2 * len(ACC_LAYERS)
    new = lambda: dict(_fresh(n, dev, ("eq_frac", "maxdiff", "relnorm", "vp_num", "vp_den")), dC=torch.zeros(n, FINAL - LOCAL + 1, device=dev), W=torch.zeros(n, nW, device=dev))
    base = dict(tgt=new(), src=new()); res = {j["name"]: new() for j in jobs}
    for s0 in range(0, n, bs):
        if cap is not None and time.time() - t0 > cap: raise TimeoutError("stage time cap")
        b = trials[s0:s0 + bs]; sl = slice(s0, s0 + len(b))
        ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], dev); ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], dev)
        assert ids_t.shape == ids_s.shape and torch.equal(m_t, m_s), "source/target not aligned"
        B, T = ids_t.shape
        ta = torch.tensor([t["ta"] for t in b], device=dev); tb = torch.tensor([t["tb"] for t in b], device=dev)
        marg = lambda lg: lg.gather(1, ta[:, None])[:, 0] - lg.gather(1, tb[:, None])[:, 0]
        R, KV = _pad(b, ROWS, T, dev, s0), _pad(b, KEYS, T, dev, s0); cache = {}
        def clean(ids, m, p):
            z, cF, h20, h21, h22, comp, st = {}, [], [], [], [], {"attn": {}, "mlp": {}}, None
            sinks = {i: [] for i in range(LOCAL, FINAL + 1)}
            with ExitStack() as es:
                es.enter_context(capture_z(model, [LAYER], z))
                for i in sinks: es.enter_context(capture_residual(model, i, sinks[i]))
                cl = sorted(set(ACC_LAYERS) | (set(comp_layers) if need_comp else set()))
                for w in COMPS: es.enter_context(capture_component(model, cl, w, comp[w]))
                st = AttnStash(model, [LAYER]) if need_av else None
                lg, hs = forward_states(model, ids, m, p, n_states=LAYER + 1)
            av = None
            if st is not None: av = recompute(model, LAYER, *st.data[LAYER], m); st.close()
            return dict(lg=lg, hs=hs, z=z[LAYER], sinks={i: sinks[i][0] for i in sinks}, comp=comp, av=av)
        S_, T_ = clean(ids_s, m_s, p_s), clean(ids_t, m_t, p_t)

        def acc(c_, rows):
            dC = torch.stack([_rmean(c_["sinks"][i].float() @ ctx.d_c, rows) for i in range(LOCAL, FINAL + 1)], 1)
            W = torch.stack([_rmean(c_["comp"][w][l].float() @ ctx.d_c, rows) for l in ACC_LAYERS for w in COMPS], 1); return dC, W
        for nm, c_ in (("src", S_), ("tgt", T_)):
            r = base[nm]; r["margin"][sl] = marg(c_["lg"]); r["c"][sl] = c_["sinks"][FINAL][:, -1].float() @ ctx.d_c
            r["c20"][sl] = _rmean(c_["sinks"][LOCAL].float() @ ctx.d_c, R); r["w20"][sl] = _rmean(c_["sinks"][LOCAL].float() @ ctx.w, R) if ctx.w is not None else 0
            r["hr"][sl] = _hr(ctx, c_["z"], R, heads); r["dC"][sl], r["W"][sl] = acc(c_, R)

        def record(name, lg, sinks, z, hh=None):
            r = res[name]; r["margin"][sl] = marg(lg); r["c"][sl] = sinks[FINAL][0][:, -1].float() @ ctx.d_c
            r["c20"][sl] = _rmean(sinks[LOCAL][0].float() @ ctx.d_c, R); r["w20"][sl] = _rmean(sinks[LOCAL][0].float() @ ctx.w, R) if ctx.w is not None else 0
            r["hr"][sl] = _hr(ctx, z[LAYER], R, hh if hh is not None else heads)
            return r

        for j in jobs:
            kind, nm = j["kind"], j["name"]
            if kind in ("origin", "u"):
                sp = j["spec"]; g = sp["group"]
                if g not in cache: cache[g] = _pad(b, g, T, dev, s0)
                P = cache[g]; typ = sp["type"]; ctxs = []; r = res[nm]
                if typ in ("resid", "noise"):
                    i = sp["index"]; diff = T_["hs"][i].float() - S_["hs"][i].float()
                    delta = None
                    if typ == "noise":
                        gen = torch.Generator(device=dev); gen.manual_seed(zlib.crc32(f"{nm}|{s0}".encode()))
                        rn = torch.randn(diff.shape, device=dev, generator=gen); delta = rn / rn.norm(dim=-1, keepdim=True).clamp_min(1e-12) * diff.norm(dim=-1, keepdim=True)
                    ctxs.append(residual_set_patch(model, i, P, S_["hs"][i], delta=delta))
                    ad = (S_["hs"][i].float() - T_["hs"][i].float()).abs().amax(-1); mk = P.float()
                    r["eq_frac"][sl] = (((ad == 0).float() * mk).sum(1) / mk.sum(1).clamp_min(1)); r["maxdiff"][sl] = (ad * mk).amax(1)
                    r["relnorm"][sl] = _rmean((S_["hs"][i].float() - T_["hs"][i].float()).norm(dim=-1) / T_["hs"][i].float().norm(dim=-1).clamp_min(1e-6), P)
                else:
                    l, comps = sp["layer"], sp["comps"]
                    dif = sum(S_["comp"][c][l].float() - T_["comp"][c][l].float() for c in comps); ad = dif.abs().amax(-1); mk = P.float()
                    r["eq_frac"][sl] = (((ad == 0).float() * mk).sum(1) / mk.sum(1).clamp_min(1)); r["maxdiff"][sl] = (ad * mk).amax(1)
                    r["relnorm"][sl] = _rmean(dif.norm(dim=-1) / T_["hs"][l + 1].float().norm(dim=-1).clamp_min(1e-6) if l + 1 <= LAYER else dif.norm(dim=-1), P)
                    if typ == "comp_total": ctxs += [component_patch(model, l, c, P, S_["comp"][c][l]) for c in comps]
                    elif typ == "comp_direct": ctxs.append(residual_set_patch(model, LAYER, P, S_["hs"][LAYER], delta=dif))
                    else: raise ValueError(typ)
                if kind == "origin":
                    store = {}; run_partial(model, ids_t, m_t, p_t, exit_before_attention(model, LAYER, store), ctxs)
                    rec = recompute(model, LAYER, store["h"], store["pe"], m_t)
                    hl = list(j.get("heads", heads)); kk = KV.float()[:, None, :, None]
                    dvA = (rec["v"] - T_["av"]["v"])[:, hl]; dvS = (S_["av"]["v"] - T_["av"]["v"])[:, hl]
                    r["vp_num"][sl] = (dvA * dvS * kk).sum((1, 2, 3)); r["vp_den"][sl] = (dvS * dvS * kk).sum((1, 2, 3))
                    zz, sk = {}, {i: [] for i in (LOCAL, FINAL)}
                    with ExitStack() as es:
                        es.enter_context(attention_rows_patch(model, LAYER, hl, dict(A=rec["A"], v=rec["v"]), m_t, R, patch="V", keys=KV))
                        es.enter_context(capture_z(model, [LAYER], zz))
                        for i in sk: es.enter_context(capture_residual(model, i, sk[i]))
                        lg = last_logits(model, ids_t, m_t, p_t)
                    record(nm, lg, sk, zz, hl)
                else:
                    zz, sk = {}, {i: [] for i in (LOCAL, FINAL)}
                    with ExitStack() as es:
                        for c in ctxs: es.enter_context(c)
                        es.enter_context(capture_z(model, [LAYER], zz))
                        for i in sk: es.enter_context(capture_residual(model, i, sk[i]))
                        lg = last_logits(model, ids_t, m_t, p_t)
                    record(nm, lg, sk, zz)
            elif kind in ("pin", "dcomp"):
                rows_c = j.get("rows", ROWS)
                if rows_c not in cache: cache[rows_c] = _pad(b, rows_c, T, dev, s0)
                rows = cache[rows_c]; pin_rows = cache.setdefault(j.get("pin_rows", ROWS), _pad(b, j.get("pin_rows", ROWS), T, dev, s0)); r = res[nm]
                ctxs = []
                if kind == "pin":
                    ctxs.append(head_z_set_patch(model, LAYER, j["heads"], rows, S_["z"], j.get("alpha", 1.0)))
                else:
                    store = {}; run_partial(model, ids_t, m_t, p_t, exit_after_layer(model, LAYER, store), [head_z_set_patch(model, LAYER, heads, R, S_["z"])])
                    d = (store["out"].float() - T_["sinks"][LOCAL].float()) * R[..., None].float(); u = ctx.d_c.float()
                    dc = (d @ u)[..., None] * u; part = {"full": d, "dc": dc, "orth": d - dc}[j["part"]]
                    ctxs.append(residual_set_patch(model, LOCAL, R, T_["sinks"][LOCAL], delta=part))
                for (l, c) in j.get("pins", []): ctxs.append(component_patch(model, l, c, pin_rows, T_["comp"][c][l]))
                zz, sk, cp = {}, {i: [] for i in range(LOCAL, FINAL + 1)}, {"attn": {}, "mlp": {}}
                with ExitStack() as es:
                    for c in ctxs: es.enter_context(c)
                    es.enter_context(capture_z(model, [LAYER], zz))
                    for i in sk: es.enter_context(capture_residual(model, i, sk[i]))
                    for w in COMPS: es.enter_context(capture_component(model, list(ACC_LAYERS), w, cp[w]))
                    lg = last_logits(model, ids_t, m_t, p_t)
                record(nm, lg, {i: sk[i] for i in sk}, zz)
                r["dC"][sl] = torch.stack([_rmean(sk[i][0].float() @ ctx.d_c, R) for i in range(LOCAL, FINAL + 1)], 1)
                r["W"][sl] = torch.stack([_rmean(cp[w][l].float() @ ctx.d_c, R) for l in ACC_LAYERS for w in COMPS], 1)
            else:
                raise ValueError(kind)
    toc = lambda d: {k: v.cpu().numpy() for k, v in d.items()}
    return toc(base["tgt"]), toc(base["src"]), {k: toc(v) for k, v in res.items()}
