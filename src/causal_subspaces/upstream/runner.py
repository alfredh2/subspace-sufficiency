"""Batched execution of Phase-4 patch / pathway-freeze / writer jobs (docs/phase4_plan.md sections 3-5)."""
import time
import zlib
from contextlib import ExitStack

import numpy as np
import torch

from causal_subspaces.heads import positions as POS
from causal_subspaces.heads.decompose import head_dims, head_gain
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.upstream.patching import (capture_component, capture_last_z, component_patch, forward_states, freeze_heads, residual_set_patch)

RL = (19, 20, 21, 22)           # layers of the S8 readout


class Ctx:
    def __init__(self, model, tok, d_c, S8, dev="cuda", idx_hook=23):
        self.model, self.tok, self.d_c, self.S8, self.dev, self.idx_hook = model, tok, d_c, [tuple(x) for x in S8], dev, idx_hook
        self.H, self.HD, _ = head_dims(model); self.NL = model.config.num_hidden_layers
        self.gain = {l: head_gain(model, l, d_c).view(self.H, self.HD).float() for l in RL}
        self.s8 = {l: sorted(h for ll, h in self.S8 if ll == l) for l in RL}
        self._spec = {}

    def freeze_spec(self, name: str, i: int) -> dict:
        """Heads whose LAST-row z is pinned to the clean target value, restricted to layers >= i (earlier layers cannot be affected by a patch at i)."""
        key = (name, i)
        if key in self._spec:
            return self._spec[key]
        allh = list(range(self.H)); ls = range(max(i, 0), self.NL)
        S8 = {l: set(self.s8.get(l, [])) for l in range(self.NL)}
        if name == "S8": sp = {l: sorted(S8[l]) for l in ls if l in RL}
        elif name == "nonS8": sp = {l: [h for h in allh if h not in S8[l]] for l in ls if l in RL}
        elif name == "late": sp = {l: allh for l in ls if l >= 23}
        elif name == "all": sp = {l: allh for l in ls}
        elif name == "onlyS8": sp = {l: [h for h in allh if h not in S8[l]] for l in ls}
        elif name == "S8late": sp = {**{l: sorted(S8[l]) for l in ls if l in RL}, **{l: allh for l in ls if l >= 23}}
        elif name.startswith("rnd"):                       # layer-count-matched random non-S8 heads in layers 19-22
            d = int(name[3:]); sp = {}
            for l in ls:
                if l in RL and S8[l]:
                    rg = np.random.default_rng(zlib.crc32(f"phase4|rnd|{d}|{l}".encode()))
                    pool = [h for h in allh if h not in S8[l]]; sp[l] = sorted(int(x) for x in rg.choice(pool, len(S8[l]), replace=False))
        else:
            raise ValueError(name)
        self._spec[key] = {l: v for l, v in sp.items() if v}
        return self._spec[key]


def _pad_masks(b, cond, T, dev):
    return torch.tensor(np.stack([POS.pad_mask(t["sets"][cond], T) for t in b])).to(dev)


def _readout(ctx, z):
    return torch.stack([sum((z[l].view(-1, ctx.H, ctx.HD)[:, h].float() * ctx.gain[l][h]).sum(-1) for h in ctx.s8[l]) if ctx.s8[l] else torch.zeros(z[l].shape[0], device=z[l].device) for l in RL], 1)


@torch.no_grad()
def run_jobs(ctx: Ctx, trials: list, jobs: list, *, bs=32, t0=None, cap=None, diag=True):
    """jobs: dicts(name, kind in {'patch','noise','comp'}, index, cond, [alpha], [freeze], [comp, layer]). Returns (base, results); arrays over trials.
    Outcomes per job: margin (valid-invalid logit), c (verdict coordinate at hook 23), r (S8 readout per layer 19..22), verify (max |frozen - clean| z)."""
    model, tok, dev, n = ctx.model, ctx.tok, ctx.dev, len(trials)
    new = lambda: dict(margin=torch.zeros(n, device=dev), c=torch.zeros(n, device=dev), r=torch.zeros(n, len(RL), device=dev))
    base = dict(tgt=new(), src=new())
    res = {}
    for j in jobs:
        res[j["name"]] = dict(**new(), verify=torch.zeros((), device=dev))
        if diag and j["kind"] != "comp":
            res[j["name"]].update(nrm=torch.full((n,), float("nan"), device=dev), dc_frac=torch.full((n,), float("nan"), device=dev), npos=torch.zeros(n, device=dev))
    comp_layers = sorted({j["layer"] for j in jobs if j["kind"] == "comp"})
    for s0 in range(0, n, bs):
        if cap is not None and time.time() - t0 > cap:
            raise TimeoutError("stage time cap")
        b = trials[s0:s0 + bs]; sl = slice(s0, s0 + len(b))
        ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], dev)
        ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], dev)
        assert ids_t.shape == ids_s.shape and torch.equal(m_t, m_s), "source/target not aligned"
        B, T = ids_t.shape
        ta = torch.tensor([t["ta"] for t in b], device=dev); tb = torch.tensor([t["tb"] for t in b], device=dev)
        marg = lambda lg: lg.gather(1, ta[:, None])[:, 0] - lg.gather(1, tb[:, None])[:, 0]
        # source pass
        zs, cs, comp_s = {}, [], {"attn": {}, "mlp": {}}
        with ExitStack() as st:
            st.enter_context(capture_residual(model, ctx.idx_hook, cs)); st.enter_context(capture_last_z(model, RL, zs))
            if comp_layers:
                st.enter_context(capture_component(model, comp_layers, "attn", comp_s["attn"])); st.enter_context(capture_component(model, comp_layers, "mlp", comp_s["mlp"]))
            lg_s, hs_s = forward_states(model, ids_s, m_s, p_s)
        base["src"]["margin"][sl] = marg(lg_s); base["src"]["c"][sl] = cs[0][:, -1].float() @ ctx.d_c; base["src"]["r"][sl] = _readout(ctx, zs)
        # target pass (clean); last-row z of ALL layers for freezing
        zt, ct = {}, []
        with capture_residual(model, ctx.idx_hook, ct), capture_last_z(model, range(ctx.NL), zt):
            lg_t, hs_t = forward_states(model, ids_t, m_t, p_t)
        base["tgt"]["margin"][sl] = marg(lg_t); base["tgt"]["c"][sl] = ct[0][:, -1].float() @ ctx.d_c; base["tgt"]["r"][sl] = _readout(ctx, zt)
        mcache = {}
        gen = torch.Generator(device=dev)
        for j in jobs:
            cond = j["cond"]
            if cond not in mcache: mcache[cond] = _pad_masks(b, cond, T, dev)
            mask = mcache[cond]
            with ExitStack() as st:
                verify = {}
                kind = j["kind"]
                if kind in ("patch", "noise"):
                    i = j["index"]; delta = None
                    if kind == "noise":                                    # random direction with the exact per-position norm of the true patch difference
                        gen.manual_seed(_seed(j["name"], s0))
                        diff = (hs_s[i].float() - hs_t[i].float()); nrm = diff.norm(dim=-1, keepdim=True)
                        rnd = torch.randn(diff.shape, device=dev, generator=gen); delta = rnd / rnd.norm(dim=-1, keepdim=True) * nrm
                    st.enter_context(residual_set_patch(model, i, mask, hs_s[i], j.get("alpha", 1.0), delta))
                else:
                    st.enter_context(component_patch(model, j["layer"], j["comp"], mask, comp_s[j["comp"]][j["layer"]]))
                if j.get("freeze"):
                    st.enter_context(freeze_heads(model, ctx.freeze_spec(j["freeze"], j["index"]), zt, verify))
                zp, cp = {}, []
                st.enter_context(capture_residual(model, ctx.idx_hook, cp)); st.enter_context(capture_last_z(model, RL, zp))
                lg = last_logits(model, ids_t, m_t, p_t)
            r = res[j["name"]]
            r["margin"][sl] = marg(lg); r["c"][sl] = cp[0][:, -1].float() @ ctx.d_c; r["r"][sl] = _readout(ctx, zp)
            if "max" in verify: r["verify"] = torch.maximum(r["verify"], verify["max"])
            if diag and kind != "comp":
                i = j["index"]; d = (hs_s[i].float() - hs_t[i].float()); mk = mask[..., None].float(); cnt = mask.float().sum(1)
                dn = d.norm(dim=-1); hn = hs_t[i].float().norm(dim=-1).clamp_min(1e-6)
                ok = cnt > 0
                r["nrm"][sl] = torch.where(ok, ((dn / hn) * mask.float()).sum(1) / cnt.clamp_min(1), torch.full_like(cnt, float("nan")))
                num = (((d @ ctx.d_c) ** 2) * mask.float()).sum(1); den = ((dn ** 2) * mask.float()).sum(1).clamp_min(1e-12)
                r["dc_frac"][sl] = torch.where(ok, num / den, torch.full_like(cnt, float("nan"))); r["npos"][sl] = cnt
    toc = lambda d: {k: v.cpu().numpy() for k, v in d.items()}
    return toc(base["tgt"]), toc(base["src"]), {k: toc(v) for k, v in res.items()}


def _seed(name, s0):
    return zlib.crc32(f"{name}|{s0}".encode())
