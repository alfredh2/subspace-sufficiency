"""Residual-stream intervention via a forward hook.

Residual boundary convention (the paper does NOT define it; see docs/paper_reproduction.md):
`residual_index = i` refers to HuggingFace `output_hidden_states` entry i:
  i = 0 : output of the embedding layer
  i >= 1: output of decoder module `layers[i-1]` (residual stream after that block, before the next)
Qwen3-8B has 36 modules, so i in 0..36 (hidden_states[36] additionally has the final RMSNorm applied
by HF and is not a legal hook target here).
The same hook object is used for training and evaluation so conventions cannot drift.
"""
from contextlib import contextmanager

import torch

from causal_subspaces.mvs.projection import apply_mode


def _target_module(model, residual_index: int):
    n = model.config.num_hidden_layers  # not len(layers): reference computation temporarily truncates the list
    if not 0 <= residual_index < n:
        raise ValueError(f"residual_index must be in [0, {n - 1}], got {residual_index}")
    return model.model.embed_tokens if residual_index == 0 else model.model.layers[residual_index - 1]


@contextmanager
def residual_intervention(model, residual_index: int, mode: str, U=None, mu=None):
    """Within the context, the residual stream at `residual_index` is replaced (all token positions,
    out of place) by retain/ablate/noop of the MVS. The hook is always removed on exit."""
    if mode != "noop" and (U is None or mu is None):
        raise ValueError("U and mu are required unless mode == 'noop'")
    module = _target_module(model, residual_index)

    def hook(_mod, _inp, out):
        if mode == "noop":
            return out
        if isinstance(out, tuple):
            return (apply_mode(out[0], mode, U, mu),) + tuple(out[1:])
        return apply_mode(out, mode, U, mu)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_residual(model, residual_index: int, sink: list):
    """Append the residual stream at `residual_index` (detached) to `sink` on each forward call."""
    module = _target_module(model, residual_index)

    def hook(_mod, _inp, out):
        sink.append((out[0] if isinstance(out, tuple) else out).detach())

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def residual_patch(model, residual_index: int, U: torch.Tensor, source_h: torch.Tensor, position: str = "all",
                   complement: bool = False):
    """Activation patching restricted to span(U) (Phase 0.5, Exp. 3).

    Target residual h, cached source residual s (same layer, same batch order):
        patch (default):  h' = h + U U^T (s - h)      -> coordinates in the MVS come from the source,
                                                         the orthogonal complement stays the target's
        complement=True:  h' = h + (I - U U^T)(s - h) -> everything EXCEPT the MVS comes from the source
    The mean mu cancels, so it is not needed. `position`: "all" (needs source_h of shape (B,T,d), same T as the
    target) or "last" (only the final token position; source_h may be (B,d) or (B,T,d)). Out of place, fp32 math.
    """
    if position not in ("all", "last"):
        raise ValueError(position)
    module = _target_module(model, residual_index)
    Uf = U.float()

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hf = h.float()
        if position == "last":
            s = source_h if source_h.dim() == 2 else source_h[:, -1]
            diff = s.float().to(hf.device) - hf[:, -1]
            comp = (diff @ Uf) @ Uf.T
            delta = diff - comp if complement else comp
            new = hf.clone()
            new[:, -1] = hf[:, -1] + delta
        else:
            if source_h.shape != h.shape:
                raise ValueError(f"source {tuple(source_h.shape)} and target {tuple(h.shape)} residuals must have equal shape")
            diff = source_h.float().to(hf.device) - hf
            comp = (diff @ Uf) @ Uf.T
            new = hf + (diff - comp if complement else comp)
        new = new.to(h.dtype)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
