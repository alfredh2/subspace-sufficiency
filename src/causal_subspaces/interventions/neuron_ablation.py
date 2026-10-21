"""Mean-replacement ablation of post-gating MLP units (native) or of rotated coordinates, via pre-hooks on `down_proj`.
Layers must be < residual_index (they cannot be downstream of the verdict hook). Out of place; no d x d matrices; no weight copies.
Optionally records the *direct* change of the verdict coordinate caused by the ablated units' own writes."""
from contextlib import contextmanager

import numpy as np
import torch


class AblationRecord:
    def __init__(self):
        self.direct = {}  # layer -> list of (B,) tensors (last position)

    def total_direct(self) -> torch.Tensor:
        parts = [torch.cat(v) for v in self.direct.values()]
        return sum(parts) if parts else None


@contextmanager
def unit_ablation(model, spec: dict, residual_index: int, gains: dict | None = None, position: str = "all", record: AblationRecord | None = None):
    """spec[layer] is either
         dict(idx=LongTensor(k), repl=Tensor(k))                      native units: a_idx <- repl
         dict(RS=Tensor(k, m), abar=Tensor(m))                         rotated coords (rows of R): a <- a - ((a - abar) RS^T) RS
       gains[layer] = g^l (needed only for `record`)."""
    if position not in ("all", "last"):
        raise ValueError(position)
    handles = []
    for l, sp in spec.items():
        if not 0 <= l < residual_index:
            raise ValueError(f"layer {l} cannot write into hidden_states[{residual_index}]")

        def hook(mod, inp, l=l, sp=sp):
            a = inp[0]
            af = a.float()
            if "idx" in sp:
                idx, repl = sp["idx"].to(a.device), sp["repl"].to(a.device).float()
                new = af.clone()
                part = new[:, -1:] if position == "last" else new  # view: in-place write below modifies `new` only
                part[..., idx] = repl
                if record is not None and gains is not None:
                    record.direct.setdefault(l, []).append(-((af[:, -1, idx] - repl) * gains[l][idx].to(a.device)).sum(-1))
            else:
                RS, abar = sp["RS"].to(a.device).float(), sp["abar"].to(a.device).float()
                new = af.clone()
                part = new[:, -1:] if position == "last" else new
                coef = (part - abar) @ RS.T
                part -= coef @ RS
                if record is not None and gains is not None:
                    c_last = (af[:, -1] - abar) @ RS.T
                    record.direct.setdefault(l, []).append(-(c_last @ (RS @ gains[l].to(a.device).float())))
            return (new.to(a.dtype),) + tuple(inp[1:])

        handles.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---- unit sets and matched controls --------------------------------------------------------------------------------
def to_spec(unit_layers: np.ndarray, unit_idx: np.ndarray, abar: dict, device) -> dict:
    """Group a flat set of (layer, index) into per-layer mean-replacement specs."""
    spec = {}
    for l in np.unique(unit_layers):
        idx = torch.as_tensor(unit_idx[unit_layers == l], dtype=torch.long)
        spec[int(l)] = dict(idx=idx.to(device), repl=abar[int(l)].to(device)[idx.to(device)])
    return spec


def random_units(layer_counts: dict, m: int, rng, exclude: dict | None = None):
    ls, ix = [], []
    for l, k in layer_counts.items():
        pool = np.setdiff1d(np.arange(m), (exclude or {}).get(l, np.array([], int)))
        ix.append(rng.choice(pool, k, replace=False)); ls.append(np.full(k, l))
    return np.concatenate(ls), np.concatenate(ix)


def matched_units(native_layers, native_idx, feat: dict, rng, exclude_native=True, n_cand: int = 1):
    """For every native unit pick a distinct unit of the same layer whose feature vector (rows of feat[l], shape (m, f), pre-standardised)
    is nearest among units not in the native set; sampling without replacement. Used for activation- and alignment-matched controls."""
    ls, ix = [], []
    for l in np.unique(native_layers):
        nat = native_idx[native_layers == l]
        F = feat[int(l)]
        used = np.zeros(F.shape[0], bool)
        if exclude_native:
            used[nat] = True
        for j in rng.permutation(nat):
            d = ((F - F[j]) ** 2).sum(1)
            d[used] = np.inf
            cand = np.argsort(d)[:max(1, n_cand)]
            c = int(rng.choice(cand))
            used[c] = True
            ls.append(l); ix.append(c)
    return np.array(ls), np.array(ix)
