"""Low-rank projection algebra for the MVS intervention (Kean & Boix-Adsera, Sec. 3, Eq. 1).

retain:  h -> mu + U U^T (h - mu)
ablate:  h -> h  - U U^T (h - mu)  ( = mu + (I - U U^T)(h - mu) )

The d x d projector is never materialised: (h - mu) @ U is (..., k), then @ U^T.
Maths is done in float32 and the result is cast back to h.dtype (the model runs in bf16).
"""
import torch


def orthonormalize(V: torch.Tensor) -> torch.Tensor:
    """U = Q of the reduced QR of V, recomputed (and differentiated through) at every step.

    The paper (Sec. 4) says U is obtained from V "at every step of the optimization by taking a QR
    decomposition". Whether gradients flow through QR is NOT stated; we differentiate through it
    (torch.linalg.qr supports backward for reduced mode). Sign convention is fixed so that
    diag(R) >= 0, which makes U a continuous function of V.
    """
    Q, R = torch.linalg.qr(V.float(), mode="reduced")
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return Q * sign.unsqueeze(-2)


def _component(h: torch.Tensor, U: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """U U^T (h - mu) in fp32, shape of h."""
    c = h.float() - mu.float()
    return (c @ U.float()) @ U.float().transpose(-1, -2)


def retain(h: torch.Tensor, U: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return (mu.float() + _component(h, U, mu)).to(h.dtype)


def ablate(h: torch.Tensor, U: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return (h.float() - _component(h, U, mu)).to(h.dtype)


def apply_mode(h: torch.Tensor, mode: str, U: torch.Tensor | None, mu: torch.Tensor | None) -> torch.Tensor:
    if mode == "noop":
        return h
    if mode == "retain":
        return retain(h, U, mu)
    if mode == "ablate":
        return ablate(h, U, mu)
    raise ValueError(f"unknown mode {mode!r}")


def orthonormality_error(U: torch.Tensor) -> float:
    k = U.shape[-1]
    return (U.float().T @ U.float() - torch.eye(k, device=U.device)).abs().max().item()
