"""Head-level interventions applied to the input of o_proj (i.e. to z_h, before the output projection).
head_ablation: replace z_h by its neutral-corpus mean / zero / a resampled value; head_z_patch: replace z_h by a cached source value."""
from contextlib import contextmanager

import torch

from causal_subspaces.heads.decompose import head_dims


class HeadRecord:
    def __init__(self):
        self.direct = {}

    def total_direct(self):
        parts = [torch.cat(v) for v in self.direct.values()]
        return sum(parts) if parts else None


@contextmanager
def head_ablation(model, spec: dict, residual_index: int, *, mode="mean", zbar=None, position="all", gains=None, record: HeadRecord | None = None):
    """spec[layer] = list of head indices. mode: 'mean' (zbar[layer]: (H*HD,)), 'zero', 'resample' (value of the next row in the batch, same
    padded position). Direct change of the verdict coordinate from the ablated heads' own writes is recorded at the last position."""
    if mode not in ("mean", "zero", "resample") or position not in ("all", "last"):
        raise ValueError((mode, position))
    H, HD, _ = head_dims(model)
    handles = []
    for l, heads in spec.items():
        if not 0 <= l < residual_index:
            raise ValueError(f"layer {l} cannot write into hidden_states[{residual_index}]")
        heads = list(heads)

        def hook(mod, inp, l=l, heads=heads):
            z = inp[0]
            B, T, _ = z.shape
            zf = z.float().view(B, T, H, HD)
            new = zf.clone()
            sl = slice(T - 1, T) if position == "last" else slice(0, T)
            dsum = 0
            for h in heads:
                if mode == "mean":
                    rep = zbar[l].to(z.device).float().view(H, HD)[h].expand(B, new[:, sl, h].shape[1], HD)
                elif mode == "zero":
                    rep = torch.zeros_like(new[:, sl, h])
                else:
                    rep = zf[torch.roll(torch.arange(B, device=z.device), -1), sl, h]
                if record is not None and gains is not None:   # direct change of c at the last position from this head's own write
                    dsum = dsum - ((zf[:, -1, h] - rep[:, -1]) * gains[l].view(H, HD)[h]).sum(-1)
                new[:, sl, h] = rep
            if record is not None and gains is not None:
                record.direct.setdefault(l, []).append(dsum)
            return (new.view(B, T, H * HD).to(z.dtype),) + tuple(inp[1:])

        handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def head_z_patch(model, spec: dict, source_z: dict, residual_index: int, position="all"):
    """Replace z_h of the listed heads by source_z[layer] ((B,T,H*HD) cached from the source forward; shapes must equal the target's for 'all')."""
    H, HD, _ = head_dims(model)
    handles = []
    for l, heads in spec.items():
        if not 0 <= l < residual_index:
            raise ValueError(f"layer {l} cannot write into hidden_states[{residual_index}]")

        def hook(mod, inp, l=l, heads=list(heads)):
            z = inp[0]
            B, T, _ = z.shape
            src = source_z[l]
            if position == "all" and src.shape != z.shape:
                raise ValueError(f"source {tuple(src.shape)} and target {tuple(z.shape)} differ")
            new = z.float().view(B, T, H, HD).clone()
            s = src.float().to(z.device).view(src.shape[0], src.shape[1], H, HD)
            for h in heads:
                if position == "all":
                    new[:, :, h] = s[:, :, h]
                else:
                    new[:, -1, h] = s[:, -1, h]
            return (new.view(B, T, H * HD).to(z.dtype),) + tuple(inp[1:])

        handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def capture_z(model, layers, sink: dict):
    """Record the full o_proj input (B,T,H*HD) of the given layers (detached)."""
    handles = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, i, l=l: sink.__setitem__(l, i[0].detach())) for l in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
