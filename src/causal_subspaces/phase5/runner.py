"""Batched execution of Phase-5 jobs (docs/phase5_plan.md): row-set head patches, row-restricted A/V patches, residual patches, summary-row pinning."""
import time
import zlib
from contextlib import ExitStack

import numpy as np
import torch

from causal_subspaces.heads import positions as POS
from causal_subspaces.heads.ablation import capture_z
from causal_subspaces.heads.attention import AttnStash, recompute
from causal_subspaces.heads.decompose import head_dims, head_gain
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.phase5.patching import attention_rows_patch, head_z_set_patch, pinned_rows
from causal_subspaces.upstream.patching import capture_component, forward_states, residual_set_patch

LAYER = 19
LOCAL = 20          # hook index of the layer-19 output
FINAL = 23          # verdict hook
ROWS = "G1G2"
PIN_LAYERS = {"22": (20, 21, 22), "35": tuple(range(20, 36))}


class Ctx5:
    def __init__(self, model, tok, d_c, dev="cuda", w=None):
        self.model, self.tok, self.d_c, self.dev, self.w = model, tok, d_c, dev, w
        self.H, self.HD, _ = head_dims(model)
        self.gain = head_gain(model, LAYER, d_c).view(self.H, self.HD).float()


def cond_mask(t, cond: str, trial_idx: int = 0, draw_tag: str = "") -> np.ndarray:
    """Composite position-set name: 'Fact+Concl' = union; 'rk:<group>:<d>' = random key set of the group's size drawn from the variable region outside
    the summary positions and the group; 'Other' = positions before the first token difference (exact no-op)."""
    sets = t["sets"]
    if cond.startswith("rk:"):
        _, g, d = cond.split(":"); grp = np.logical_or.reduce([sets[x] for x in g.split("+")])
        pool = np.where(sets["NonSummary"] & ~grp)[0]
        pos, _ = POS.random_positions(pool, int(grp.sum()), f"phase5|rk|{g}", trial_idx, int(d))
        m = np.zeros(len(grp), bool); m[pos] = True; return m
    cond = cond.replace("Other", "PreDiff")
    return np.logical_or.reduce([sets[x] for x in cond.split("+")])


def _pad(b, cond, T, dev, s0=0):
    return torch.tensor(np.stack([POS.pad_mask(cond_mask(t, cond, s0 + i), T) for i, t in enumerate(b)])).to(dev)


def _local(ctx, h20, rows):
    m = rows.float(); n = m.sum(1).clamp_min(1)
    c = (((h20.float() @ ctx.d_c) * m).sum(1)) / n
    w = (((h20.float() @ ctx.w) * m).sum(1)) / n if ctx.w is not None else torch.zeros_like(c)
    return c, w


@torch.no_grad()
def run_jobs(ctx: Ctx5, trials: list, jobs: list, *, bs=32, t0=None, cap=None, keep_dh=False):
    """Job kinds: 'z' (model-native head-output patch at rows), 'av' (delta-form recompute patch: patch in V/A/AV, optional key group), 'res'
    (residual patch of hidden_states[index] at positions). Optional per-job: alpha, noise, pin={'layers':'22'|'35','which':('attn','mlp')}.
    Returns base_tgt, base_src, per-job arrays margin, c (hook 23), c20, w20 (mean over rows G1G2 of the layer-19 output), dn (relative change of the
    layer-19 output at the patched rows), vz (patched-head output verification), vp (pin verification)."""
    model, tok, dev, n = ctx.model, ctx.tok, ctx.dev, len(trials)
    f = lambda: dict(margin=torch.zeros(n, device=dev), c=torch.zeros(n, device=dev), c20=torch.zeros(n, device=dev), w20=torch.zeros(n, device=dev))
    base = dict(tgt=f(), src=f()); res = {}
    for j in jobs:
        res[j["name"]] = dict(**f(), dn=torch.zeros(n, device=dev), vz=torch.zeros((), device=dev), vp=torch.zeros((), device=dev))
    if keep_dh: dh = torch.zeros(n, model.config.hidden_size, device=dev)
    need_pin = any(j.get("pin") for j in jobs); need_av = any(j["kind"] == "av" for j in jobs)
    pin_layers = sorted({l for j in jobs if j.get("pin") for l in PIN_LAYERS[j["pin"]["layers"]]})
    for s0 in range(0, n, bs):
        if cap is not None and time.time() - t0 > cap: raise TimeoutError("stage time cap")
        b = trials[s0:s0 + bs]; sl = slice(s0, s0 + len(b))
        ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], dev)
        ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], dev)
        assert ids_t.shape == ids_s.shape and torch.equal(m_t, m_s), "source/target not aligned"
        B, T = ids_t.shape
        ta = torch.tensor([t["ta"] for t in b], device=dev); tb = torch.tensor([t["tb"] for t in b], device=dev)
        marg = lambda lg: lg.gather(1, ta[:, None])[:, 0] - lg.gather(1, tb[:, None])[:, 0]
        R = _pad(b, ROWS, T, dev, s0)
        # source pass
        zs, c23s, h20s = {}, [], []; st = AttnStash(model, [LAYER]) if need_av else None
        with ExitStack() as es:
            es.enter_context(capture_z(model, [LAYER], zs)); es.enter_context(capture_residual(model, FINAL, c23s)); es.enter_context(capture_residual(model, LOCAL, h20s))
            lg_s, hs_s = forward_states(model, ids_s, m_s, p_s, n_states=FINAL + 1)
        src_av = recompute(model, LAYER, *st.data[LAYER], m_s) if need_av else None
        if st is not None: st.close()
        # target pass
        zt, c23t, h20t = {}, [], []; cl = {"attn": {}, "mlp": {}}
        with ExitStack() as es:
            es.enter_context(capture_z(model, [LAYER], zt)); es.enter_context(capture_residual(model, FINAL, c23t)); es.enter_context(capture_residual(model, LOCAL, h20t))
            if need_pin:
                es.enter_context(capture_component(model, pin_layers, "attn", cl["attn"])); es.enter_context(capture_component(model, pin_layers, "mlp", cl["mlp"]))
            lg_t, hs_t = forward_states(model, ids_t, m_t, p_t, n_states=FINAL + 1)
        for nm, lg, c23, h20 in (("src", lg_s, c23s, h20s[0]), ("tgt", lg_t, c23t, h20t[0])):
            c20, w20 = _local(ctx, h20, R)
            base[nm]["margin"][sl] = marg(lg); base[nm]["c"][sl] = c23[0][:, -1].float() @ ctx.d_c; base[nm]["c20"][sl] = c20; base[nm]["w20"][sl] = w20
        if keep_dh:
            m = R.float(); dh[sl] = (((h20s[0].float() - h20t[0].float()) * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1))
        cache = {}
        for j in jobs:
            cond = j.get("rows", j.get("cond", ROWS))
            if cond not in cache: cache[cond] = _pad(b, cond, T, dev, s0)
            rows = cache[cond]; keys = None
            if j["kind"] == "av" and j.get("keys"):
                if j["keys"] not in cache: cache[j["keys"]] = _pad(b, j["keys"], T, dev, s0)
                keys = cache[j["keys"]]
            c23p, h20p, zp, vp = [], [], {}, {}
            with ExitStack() as es:
                if j["kind"] == "z":
                    delta = None
                    if j.get("noise"):                                     # random direction with the exact per-head, per-row norm of the true patch difference
                        g = torch.Generator(device=dev); g.manual_seed(zlib.crc32(f"{j['name']}|{s0}".encode()))
                        zz = zt[LAYER].view(B, T, ctx.H, ctx.HD); sz = zs[LAYER].view(B, T, ctx.H, ctx.HD)
                        d = torch.stack([sz[:, :, h].float() - zz[:, :, h].float() for h in j["heads"]], 2); r = torch.randn(d.shape, device=dev, generator=g)
                        delta = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12) * d.norm(dim=-1, keepdim=True)
                    es.enter_context(head_z_set_patch(model, LAYER, j["heads"], rows, zs[LAYER], j.get("alpha", 1.0), delta))
                elif j["kind"] == "av":
                    es.enter_context(attention_rows_patch(model, LAYER, j["heads"], src_av, m_t, rows, patch=j["patch"], keys=keys))
                elif j["kind"] == "res":
                    es.enter_context(residual_set_patch(model, j["index"], rows, hs_s[j["index"]], j.get("alpha", 1.0)))
                if j.get("pin"):
                    pr = cache.setdefault("__R", R)
                    es.enter_context(pinned_rows(model, PIN_LAYERS[j["pin"]["layers"]], j["pin"]["which"], pr, cl))
                    for w in j["pin"]["which"]:
                        es.enter_context(capture_component(model, PIN_LAYERS[j["pin"]["layers"]], w, vp.setdefault(w, {})))
                es.enter_context(capture_z(model, [LAYER], zp)); es.enter_context(capture_residual(model, FINAL, c23p)); es.enter_context(capture_residual(model, LOCAL, h20p))
                lg = last_logits(model, ids_t, m_t, p_t)
            r = res[j["name"]]; c20, w20 = _local(ctx, h20p[0], R)
            r["margin"][sl] = marg(lg); r["c"][sl] = c23p[0][:, -1].float() @ ctx.d_c; r["c20"][sl] = c20; r["w20"][sl] = w20
            mk = rows.float(); den = h20t[0].float().norm(dim=-1).clamp_min(1e-6)
            r["dn"][sl] = ((((h20p[0].float() - h20t[0].float()).norm(dim=-1) / den) * mk).sum(1) / mk.sum(1).clamp_min(1))
            if j["kind"] == "z" and not j.get("noise") and j.get("alpha", 1.0) == 1.0:      # verify patched heads/rows hold the source z, other rows the target's
                H_, HD_ = ctx.H, ctx.HD; a = zp[LAYER].view(B, T, H_, HD_).float(); tt = zt[LAYER].view(B, T, H_, HD_).float(); ss = zs[LAYER].view(B, T, H_, HD_).float()
                hm = torch.zeros(H_, dtype=torch.bool, device=dev); hm[list(j["heads"])] = True
                exp = torch.where(rows[:, :, None, None] & hm[None, None, :, None], ss, tt)
                r["vz"] = torch.maximum(r["vz"], (a - exp).abs().max())
            if j.get("pin"):
                for w, sink in vp.items():
                    for l, o in sink.items():
                        r["vp"] = torch.maximum(r["vp"], ((o.float() - cl[w][l].float()).abs() * R[..., None].float()).max())
    toc = lambda d: {k: v.cpu().numpy() for k, v in d.items()}
    out = (toc(base["tgt"]), toc(base["src"]), {k: toc(v) for k, v in res.items()})
    return out + ((dh.cpu().numpy(),) if keep_dh else ())
