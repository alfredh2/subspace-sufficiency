"""Phase 7 Stage 0: held-out concentration statistics under a train-rank / test-evaluate protocol, for arbitrary contrasts and target directions
(docs/plans/phase7_plan.md section 11). Pure tensor code: all functions take class-contrast vectors and gain vectors, so they can be tested without a model."""
import torch


def contrast_weights(y: torch.Tensor) -> torch.Tensor:
    """(n,) bool -> (n,) weights w with w @ A = mean(A[y]) - mean(A[~y])."""
    y = y.bool()
    return y.float() / y.sum() - (~y).float() / (~y).sum()


def pooled_stats(D_tr: torch.Tensor, D_te: torch.Tensor, q: int, k_recon=(512,)) -> dict:
    """D_tr, D_te: (M,) pooled contributions Delta_j = delta_j * g_j from the training / held-out split. Ranking by |D_tr|; evaluation on D_te."""
    absr = D_tr.abs(); idx = absr.topk(q).indices
    same = absr[idx].sum() / absr.sum().clamp_min(1e-30)
    ate = D_te.abs(); share = ate[idx].sum() / ate.sum().clamp_min(1e-30)
    pte = ate / ate.sum().clamp_min(1e-30); ptr = absr / absr.sum().clamp_min(1e-30)
    order = absr.argsort(descending=True); net = D_te.sum()
    cs = D_te[order].cumsum(0)
    out = dict(share_same=float(same), share_heldout=float(share), neff_heldout=float(1 / (pte ** 2).sum()), neff_train=float(1 / (ptr ** 2).sum()),
               net_over_abs=float(net.abs() / ate.sum().clamp_min(1e-30)), net=float(net))
    for k in k_recon:
        out[f"recon_{k}"] = float(cs[min(k, len(cs)) - 1] / net) if net.abs() > 1e-12 else float("nan")
    if net.abs() > 1e-12:
        r = cs / net; hit = (r >= 0.9).nonzero()
        out["k90"] = int(hit[0, 0]) + 1 if len(hit) else None
    else:
        out["k90"] = None
    return out


def haar_orthogonal(m: int, seed: int, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(m, m, generator=g, device=device))
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def completed_basis(V: torch.Tensor, seed: int, device="cuda") -> torch.Tensor:
    """Orthonormal basis B (m x m) whose first r columns are the orthonormal columns of V (m x r) and whose remaining columns are a Haar-random
    orthonormal completion of the orthogonal complement. Rotated coordinates of a vector x are B^T x."""
    m, r = V.shape
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(m, m - r, generator=g, device=device)
    X = X - V @ (V.T @ X)
    X = X - V @ (V.T @ X)
    Q, _ = torch.linalg.qr(X)
    return torch.cat([V, Q], 1)
