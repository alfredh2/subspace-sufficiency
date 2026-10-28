"""Residual position-set patching, component (attn/mlp output) patching, last-row head freezing with direct verification (docs/plans/phase4_plan.md).
Hook index i = HF hidden_states[i] (0 = embedding output, i>=1 = output of layers[i-1]); layer l reads hidden_states[l]."""
from contextlib import contextmanager

import torch

from causal_subspaces.heads.decompose import head_dims
from causal_subspaces.models.hooks import _target_module

N_IDX = 24


def forward_states(model, ids, mask, pos, n_states=N_IDX):
    """Clean forward: last-position logits and the first n_states hidden states (hidden_states[0..n_states-1], raw residual stream)."""
    out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, output_hidden_states=True)
    lg = model.lm_head(out.last_hidden_state[:, -1]).float()
    return lg, tuple(h.detach() for h in out.hidden_states[:n_states])


@contextmanager
def residual_set_patch(model, index: int, mask: torch.Tensor, src_h: torch.Tensor, alpha: float = 1.0, delta: torch.Tensor | None = None):
    """h[mask] <- src_h[mask] (alpha=1, exact) / h + alpha*(src-h) at masked positions; `delta` (B,T,d) replaces (src-h) (norm-matched controls).
    mask: (B,T) bool on the model device; everything else keeps the target's residual."""
    module = _target_module(model, index)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if delta is None and alpha == 1.0:
            new = torch.where(mask[..., None], src_h.to(h.dtype), h)
        else:
            hf = h.float(); d = (src_h.float() - hf) if delta is None else delta.float()
            new = torch.where(mask[..., None], hf + alpha * d, hf).to(h.dtype)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _component(model, layer, which):
    return model.model.layers[layer].self_attn if which == "attn" else model.model.layers[layer].mlp


@contextmanager
def capture_component(model, layers, which, sink: dict):
    """sink[layer] = module output (B,T,d), detached, for `which` in {'attn','mlp'}."""
    hs = []
    for l in layers:
        def hook(_m, _i, out, l=l):
            sink[l] = (out[0] if isinstance(out, tuple) else out).detach()
        hs.append(_component(model, l, which).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@contextmanager
def component_patch(model, layer: int, which: str, mask: torch.Tensor, src: torch.Tensor):
    """Replace the attention-output / MLP-output of `layer` at masked positions by the source's (the write into hidden_states[layer+1])."""
    def hook(_m, _i, out):
        o = out[0] if isinstance(out, tuple) else out
        new = torch.where(mask[..., None], src.to(o.dtype), o)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    handle = _component(model, layer, which).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_last_z(model, layers, sink: dict):
    """sink[layer] = o_proj input at the LAST position, (B, H*HD), detached. Registered after a freeze it sees the frozen values."""
    hs = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, i, l=l: sink.__setitem__(l, i[0][:, -1].detach())) for l in layers]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@contextmanager
def freeze_heads(model, spec: dict, clean_last_z: dict, verify: dict):
    """Pin the LAST-row z of spec[layer] heads to clean_last_z[layer] (B,H*HD). A second pre-hook (registered after, so it sees the pinned input)
    records the max |z_actual - z_clean| over all frozen heads in verify['max'] (a device scalar tensor)."""
    H, HD, _ = head_dims(model)
    handles = []
    for l, heads in spec.items():
        if not heads:
            continue
        hm = torch.zeros(H, dtype=torch.bool); hm[list(heads)] = True

        def freeze(mod, inp, l=l, hm=hm):
            z = inp[0]; B, T, _ = z.shape; m = hm.to(z.device)
            new = z.clone().view(B, T, H, HD)
            new[:, -1, m] = clean_last_z[l].view(B, H, HD)[:, m].to(z.dtype)
            return (new.view(B, T, H * HD),) + tuple(inp[1:])

        def check(mod, inp, l=l, hm=hm):
            m = hm.to(inp[0].device); B = inp[0].shape[0]
            d = (inp[0][:, -1].view(B, H, HD)[:, m].float() - clean_last_z[l].view(B, H, HD)[:, m].float()).abs().max()
            verify["max"] = d if "max" not in verify else torch.maximum(verify["max"], d)

        o = model.model.layers[l].self_attn.o_proj
        handles += [o.register_forward_pre_hook(freeze), o.register_forward_pre_hook(check)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
