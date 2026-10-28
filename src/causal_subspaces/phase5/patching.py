"""Phase 5 interventions at layer-19 attention (docs/plans/phase5_plan.md): row-set head-output patch (model-native), row-restricted delta-form
A/V recompute patch, summary-row pinning of later components."""
from contextlib import ExitStack, contextmanager

import torch

from causal_subspaces.heads.attention import AttnStash, recompute
from causal_subspaces.heads.decompose import head_dims
from causal_subspaces.upstream.patching import component_patch


@contextmanager
def head_z_set_patch(model, layer: int, heads, rows: torch.Tensor, src_z: torch.Tensor, alpha: float = 1.0, delta: torch.Tensor | None = None):
    """z_h[rows] <- z_h[rows] + alpha*(src_z_h[rows] - z_h[rows]) for the listed heads (alpha=1 and delta None: exact replacement). rows: (B,T) bool;
    src_z: (B,T,H*HD) captured with `capture_z` on the source forward; delta (B,T,len(heads),HD) optionally replaces (src - z) (norm-matched controls).
    Pre-hook on o_proj of `layer`, i.e. before the output projection; model-native, no recompute."""
    H, HD, _ = head_dims(model); hl = list(heads)

    def hook(mod, inp):
        z = inp[0]; B, T, _ = z.shape
        zz = z.view(B, T, H, HD); new = zz.clone(); sz = src_z.view(B, T, H, HD).to(z.device)
        for j, h in enumerate(hl):
            if delta is None and alpha == 1.0:
                new[:, :, h] = torch.where(rows[:, :, None], sz[:, :, h], zz[:, :, h])
            else:
                d = (sz[:, :, h].float() - zz[:, :, h].float()) if delta is None else delta[:, :, j].float()
                new[:, :, h] = torch.where(rows[:, :, None], (zz[:, :, h].float() + alpha * d).to(z.dtype), zz[:, :, h])
        return (new.view(B, T, H * HD),) + tuple(inp[1:])

    handle = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def attention_rows_patch(model, layer: int, heads, source: dict, mask2d: torch.Tensor, rows: torch.Tensor, *, patch: str = "V", keys: torch.Tensor | None = None):
    """Delta-form recompute patch for the listed heads at query ROWS only: z' = z_native + [rec(patched) - rec(unpatched)] on `rows`, unchanged elsewhere
    (the common-mode recompute error cancels). patch in {'V','A','AV'}: 'V' = key positions `keys` ((B,T) bool; None = all) take the SOURCE value vectors,
    attention weights stay the target's; 'A' = source attention weights on the rows (target values); source = dict(A=(B,H,T,T), v=(B,H,T,HD)) from
    `recompute` on the source forward (same batch order)."""
    if not patch or not set(patch) <= {"A", "V"}:
        raise ValueError(patch)
    H, HD, _ = head_dims(model); hl = list(heads); stash = AttnStash(model, [layer])

    def hook(mod, inp):
        z = inp[0]; B, T, _ = z.shape
        if source["A"].shape[-1] != T or source["v"].shape[0] != B:
            raise ValueError("source/target shapes differ; pair rejected")
        hidden, pe = stash.data[layer]
        base = recompute(model, layer, hidden, pe, mask2d)
        A_ov, V_ov = {}, {}
        for h in hl:
            if "A" in patch:
                A_ov[h] = torch.where(rows[:, :, None], source["A"][:, h], base["A"][:, h])
            if "V" in patch:
                V_ov[h] = source["v"][:, h] if keys is None else torch.where(keys[:, :, None], source["v"][:, h], base["v"][:, h])
        rec = recompute(model, layer, hidden, pe, mask2d, A_override=A_ov or None, V_override=V_ov or None)
        new = z.float().view(B, T, H, HD).clone()
        for h in hl:
            d = rec["z"][:, :, h].float() - base["z"][:, :, h].float()
            new[:, :, h] = new[:, :, h] + d * rows[:, :, None].float()
        return (new.view(B, T, H * HD).to(z.dtype),) + tuple(inp[1:])

    handle = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove(); stash.close()


@contextmanager
def pinned_rows(model, layers, whichs, rows: torch.Tensor, clean: dict):
    """Force the listed components' outputs ('attn' / 'mlp') of `layers` at `rows` to the target's own clean outputs clean[which][layer]."""
    with ExitStack() as st:
        for l in layers:
            for w in whichs:
                st.enter_context(component_patch(model, l, w, rows, clean[w][l]))
        yield
