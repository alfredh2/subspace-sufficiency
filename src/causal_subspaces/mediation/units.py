"""Capture / restore of MLP unit activations (inputs of `down_proj`) and per-block verdict-direction writes, with an optional hook-order log.
Ordering guarantee (docs/plans/phase3_plan.md section 5): inside a decoder layer the module order is  o_proj(pre) -> self_attn(out) -> [residual add, norm]
-> mlp.down_proj(pre) -> mlp(out); unit capture/restore fires at down_proj(pre), i.e. AFTER the layer's attention output and BEFORE the MLP write."""
from contextlib import contextmanager

import torch


class OrderLog:
    def __init__(self):
        self.events = []

    def add(self, name, layer):
        self.events.append((name, layer))

    def positions(self, name, layer):
        return [i for i, e in enumerate(self.events) if e == (name, layer)]


@contextmanager
def log_modules(model, layers, log: OrderLog):
    """Log the firing order of the module-level hook points of the given layers."""
    hs = []
    for l in layers:
        L = model.model.layers[l]
        hs.append(L.self_attn.o_proj.register_forward_pre_hook(lambda m, i, l=l: log.add("o_proj_pre", l)))
        hs.append(L.self_attn.register_forward_hook(lambda m, i, o, l=l: log.add("attn_out", l)))
        hs.append(L.mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: log.add("down_proj_pre", l)))
        hs.append(L.mlp.register_forward_hook(lambda m, i, o, l=l: log.add("mlp_out", l)))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@contextmanager
def capture_units(model, spec: dict, sink: dict, position="last", log: OrderLog | None = None):
    """sink[layer] <- activations of units spec[layer] (LongTensor) at the answer position (B,|idx|) or all positions (B,T,|idx|) (detached copy, native dtype)."""
    hs = []
    for l, idx in spec.items():
        it = torch.as_tensor(idx, dtype=torch.long)

        def hook(mod, inp, l=l, it=it):
            if log is not None:
                log.add("unit_capture", l)
            a = inp[0]
            sink[l] = (a[:, -1] if position == "last" else a)[..., it.to(a.device)].detach().clone()
        hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@contextmanager
def restore_units(model, spec: dict, values: dict, position="last", log: OrderLog | None = None):
    """Overwrite units spec[layer] by values[layer] ((B,|idx|) for 'last', (B,T,|idx|) for 'all') in the same dtype; out of place."""
    hs = []
    for l, idx in spec.items():
        it = torch.as_tensor(idx, dtype=torch.long)

        def hook(mod, inp, l=l, it=it):
            if log is not None:
                log.add("unit_restore", l)
            a = inp[0].clone()
            v = values[l].to(a.device, a.dtype)
            ii = it.to(a.device)
            if position == "last":
                a[:, -1, ii] = v
            else:
                if v.shape[:2] != a.shape[:2]:
                    raise ValueError(f"restored values {tuple(v.shape)} do not match activations {tuple(a.shape)}")
                a[:, :, ii] = v
            return (a,) + tuple(inp[1:])
        hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@contextmanager
def capture_blocks(model, layers, d: torch.Tensor, sink: dict):
    """sink['attn'][l], sink['mlp'][l] <- d . (attention / MLP output) at the last position, fp32 (B,)."""
    sink["attn"], sink["mlp"] = {}, {}
    hs = []
    for l in layers:
        L = model.model.layers[l]
        hs.append(L.self_attn.register_forward_hook(lambda m, i, o, l=l: sink["attn"].__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -1].float() @ d.float().to(o[0].device if isinstance(o, tuple) else o.device))))
        hs.append(L.mlp.register_forward_hook(lambda m, i, o, l=l: sink["mlp"].__setitem__(l, o[:, -1].float() @ d.float().to(o.device))))
    try:
        yield
    finally:
        for h in hs:
            h.remove()
