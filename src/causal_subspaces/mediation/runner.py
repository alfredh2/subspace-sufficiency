"""Batch runner for the 2x2 interventional cells (docs/phase3_plan.md section 2): clean (Y00), intervened (Y11), rescue (Y10: head intervention + units
restored to their clean activations), transplant (Y01: clean run with units overwritten by their intervened activations)."""
from contextlib import ExitStack

import numpy as np
import torch

from causal_subspaces.heads.ablation import capture_z, head_ablation, head_z_patch
from causal_subspaces.mediation.units import capture_blocks, capture_units, restore_units
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits

M_UNITS = 12288


def spec_heads(hs):
    s = {}
    for l, h in hs:
        s.setdefault(l, []).append(h)
    return s


@torch.no_grad()
def run_cells(model, tok, items, *, heads, intervention, usets, idx_hook, d_c, zbar=None, position="last", full_layers=(22,), block_layers=(19, 20, 21, 22),
              bs=16, self_restore=("22",), time_cap=None, t0=None, restore_tags=("Y10", "Y01")):
    """items: dicts with 'tgt' (user text), optional 'src', 'ta', 'tb' (answer token ids). intervention: 'patch' (full-z from src) or 'ablate' (mean z).
    usets: name -> {layer: np.ndarray idx}. Returns per-item arrays for each cell and the full last-position layer activations of the clean/intervened runs."""
    import time
    n = len(items); spec = spec_heads(heads); hl = sorted(spec)
    full = {l: torch.arange(M_UNITS) for l in full_layers}
    out = {"clean": {}, "inter": {}, "cells": {}, "a_clean": {l: np.zeros((n, M_UNITS), np.float16) for l in full_layers}, "a_inter": {l: np.zeros((n, M_UNITS), np.float16) for l in full_layers}}

    def store(dst, key, sl, v):
        if key not in dst:
            dst[key] = np.zeros((n,) + tuple(v.shape[1:]), np.float32)
        dst[key][sl] = v.float().cpu().numpy()

    def record(dst, sl, lg, ta, tb, resid, blocks):
        store(dst, "margin", sl, lg.gather(1, ta[:, None])[:, 0] - lg.gather(1, tb[:, None])[:, 0])
        store(dst, "c", sl, resid[0][:, -1].float() @ d_c)
        for l in block_layers:
            store(dst, f"attn{l}", sl, blocks["attn"][l]); store(dst, f"mlp{l}", sl, blocks["mlp"][l])

    def one(ids, mask, pos, ta, tb, head_ctx=None, restore=None, capture=False):
        sink_u, blocks, resid = {}, {}, []
        with ExitStack() as st:
            if head_ctx is not None:
                st.enter_context(head_ctx)                                    # registered first: fires first at o_proj
            if restore is not None:
                st.enter_context(restore_units(model, restore[0], restore[1], position))
            if capture:
                st.enter_context(capture_units(model, full, sink_u, position))
            st.enter_context(capture_blocks(model, block_layers, d_c, blocks))
            st.enter_context(capture_residual(model, idx_hook, resid))
            lg = last_logits(model, ids, mask, pos)
        return lg, sink_u, blocks, resid

    for s0 in range(0, n, bs):
        if time_cap is not None and time.time() - t0 > time_cap:
            raise TimeoutError("time cap")
        b = items[s0:s0 + bs]; sl = slice(s0, s0 + len(b))
        ids, mask, pos = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], "cuda")
        ta = torch.tensor([t["ta"] for t in b], device="cuda"); tb = torch.tensor([t["tb"] for t in b], device="cuda")
        zs = {}
        if intervention == "patch":
            ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], "cuda")
            assert ids_s.shape == ids.shape and torch.equal(m_s, mask)
            with capture_z(model, hl, zs):
                last_logits(model, ids_s, m_s, p_s)
        def hctx():
            if intervention == "patch":
                return head_z_patch(model, spec, zs, idx_hook, position=position)
            return head_ablation(model, spec, idx_hook, mode="mean", zbar=zbar, position=position)
        lgC, uC, bC, rC = one(ids, mask, pos, ta, tb, capture=True); record(out["clean"], sl, lgC, ta, tb, rC, bC)
        lgI, uI, bI, rI = one(ids, mask, pos, ta, tb, head_ctx=hctx(), capture=True); record(out["inter"], sl, lgI, ta, tb, rI, bI)
        for l in full_layers:
            out["a_clean"][l][sl] = (uC[l][:, -1] if position == "all" else uC[l]).float().cpu().numpy().astype(np.float16)
            out["a_inter"][l][sl] = (uI[l][:, -1] if position == "all" else uI[l]).float().cpu().numpy().astype(np.float16)
        for name, us in usets.items():
            sp = {l: torch.as_tensor(ix, dtype=torch.long) for l, ix in us.items()}
            cell = out["cells"].setdefault(name, {})
            if "Y10" in restore_tags:
                vals = {l: uC[l][..., sp[l].to(uC[l].device)] for l in sp}
                lg, _, bl, rs = one(ids, mask, pos, ta, tb, head_ctx=hctx(), restore=(sp, vals)); record(cell.setdefault("Y10", {}), sl, lg, ta, tb, rs, bl)
            if "Y01" in restore_tags:
                vals = {l: uI[l][..., sp[l].to(uI[l].device)] for l in sp}
                lg, _, bl, rs = one(ids, mask, pos, ta, tb, restore=(sp, vals)); record(cell.setdefault("Y01", {}), sl, lg, ta, tb, rs, bl)
        for tag in self_restore:                                              # self-restore noise floors: restore a unit set with its OWN run's values
            l = int(tag)
            if l in full_layers and "U22" in usets and tag == "22":
                sp = {l_: torch.as_tensor(ix, dtype=torch.long) for l_, ix in usets["U22"].items()}
                vals = {l_: uC[l_][..., sp[l_].to(uC[l_].device)] for l_ in sp}
                lg, _, bl, rs = one(ids, mask, pos, ta, tb, restore=(sp, vals)); record(out["cells"].setdefault("self_clean", {}).setdefault("Y", {}), sl, lg, ta, tb, rs, bl)
                vals = {l_: uI[l_][..., sp[l_].to(uI[l_].device)] for l_ in sp}
                lg, _, bl, rs = one(ids, mask, pos, ta, tb, head_ctx=hctx(), restore=(sp, vals)); record(out["cells"].setdefault("self_inter", {}).setdefault("Y", {}), sl, lg, ta, tb, rs, bl)
    return out
