"""Stage M (docs/phase7_plan.md section 11): decomposition of the layer-19 write at the summary rows into d_c-aligned, effect-optimal, principal and
orthogonal components, with norm- and dose-matched controls. Pure tensor helpers + a gradient estimator that hooks a layer output."""
import zlib
from contextlib import contextmanager

import torch


def proj(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """projection of x (...,d) onto the unit vector(s) u (d,) or (...,d), same leading shape as x."""
    return (x * u).sum(-1, keepdim=True) * u


def nrm(x):
    return x.norm(dim=-1, keepdim=True)


def rescale_to(x: torch.Tensor, target_norm: torch.Tensor, floor: float = 1e-3, ref_norm: torch.Tensor | None = None):
    """x scaled so that its norm equals target_norm; rows with ||x|| below floor * ref_norm are left at zero scale (flagged by the returned factor)."""
    ref = ref_norm if ref_norm is not None else target_norm
    den = nrm(x).clamp_min(floor * ref.clamp_min(1e-12))
    return x * (target_norm / den)


def random_unit(shape, seed_tag: str, device, exclude: torch.Tensor | None = None):
    g = torch.Generator(device=device).manual_seed(zlib.crc32(seed_tag.encode()))
    r = torch.randn(shape, generator=g, device=device)
    if exclude is not None:
        r = r - (r * exclude).sum(-1, keepdim=True) * exclude
    return r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def conditions(delta: torch.Tensor, rows: torch.Tensor, d_c: torch.Tensor, U_eff: torch.Tensor, pcs: torch.Tensor, tag: str, n_orth=20, n_full=5):
    """All Stage-M interventions as per-position residual deltas (B,T,d), zero outside `rows` (B,T bool).
    delta: patch-induced change Delta; d_c: (d,) unit; U_eff: (B,T,d) per-row effect-optimal unit directions (orthogonalised against d_c, zero outside rows);
    pcs: (r,d) principal directions of Delta. Returns dict name -> delta tensor."""
    m = rows[..., None].float(); D = delta * m
    dc = proj(D, d_c) * m; orth = (D - dc) * m
    gc = proj(D, U_eff) * m
    out = {"full": D, "dc": dc, "orth": orth, "geff": gc, "rest": (D - dc - gc) * m, "dc_plus_geff": (dc + gc) * m}
    out["dc_eqorth"] = rescale_to(dc, nrm(orth), ref_norm=nrm(D)) * m
    out["dc_eqfull"] = rescale_to(dc, nrm(D)) * m
    out["geff_eqfull"] = rescale_to(gc, nrm(D)) * m
    for a in (2, 4, 8): out[f"dc_x{a}"] = dc * a
    for r in (1, 3):
        P = pcs[:r]; out[f"pc{r}"] = ((D @ P.T) @ P) * m
    for k in range(n_orth):
        out[f"randorth_{k}"] = random_unit(D.shape, f"{tag}|ro|{k}", D.device, exclude=d_c) * nrm(orth) * m
    for k in range(n_full):
        out[f"randfull_eqfull_{k}"] = random_unit(D.shape, f"{tag}|rf|{k}", D.device) * nrm(D) * m
        out[f"randfull_eqdc_{k}"] = random_unit(D.shape, f"{tag}|rd|{k}", D.device) * nrm(dc) * m
    return out


@contextmanager
def leaf_at_layer_output(model, layer: int, store: dict):
    """Forward hook on layers[layer] replacing its output by a detached leaf that requires grad (stored in store['leaf'])."""
    def hook(mod, args, kwargs, out):
        o = out[0] if isinstance(out, tuple) else out
        leaf = o.detach().clone().requires_grad_(True); store["leaf"] = leaf
        return (leaf,) + tuple(out[1:]) if isinstance(out, tuple) else leaf
    h = model.model.layers[layer].register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


def head_projections(dz: torch.Tensor, W_o: torch.Tensor, U: torch.Tensor, H: int, HD: int):
    """dz: (B,T,H*HD) change of o_proj inputs; W_o: (d_out, H*HD); U: (B,T,d) unit directions. Returns per-head projection of the head's write W_h dz_h on U
    ((B,T,H)) and per-head write norms ((B,T,H))."""
    B, T, _ = dz.shape
    gains = (U @ W_o).view(B, T, H, HD)                         # W_h^T u per head
    zz = dz.view(B, T, H, HD).float()
    proj_h = (zz * gains).sum(-1)
    wh = torch.einsum("bthk,ohk->btho", zz, W_o.view(W_o.shape[0], H, HD).float())
    return proj_h, wh.norm(dim=-1)
