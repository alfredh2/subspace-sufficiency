import torch

from causal_subspaces.phase7 import concentration as C7


def test_contrast_weights_and_pooled_stats():
    A = torch.randn(40, 7); y = torch.arange(40) % 2 == 0
    w = C7.contrast_weights(y); assert torch.allclose(w @ A, A[y].mean(0) - A[~y].mean(0), atol=1e-6)
    D = torch.tensor([5., -1., 0.5, 0.1, 3., 0.0, -0.2, 0.3]); s = C7.pooled_stats(D, D, q=2, k_recon=(2, 3))
    assert abs(s["share_same"] - (5 + 3) / D.abs().sum()) < 1e-6 and abs(s["share_heldout"] - s["share_same"]) < 1e-6
    assert abs(s["recon_2"] - 8 / D.sum()) < 1e-5 and abs(s["net"] - D.sum()) < 1e-6
    D2 = torch.zeros(8); D2[0] = 1.0; assert abs(C7.pooled_stats(D2, D2, q=1)["neff_heldout"] - 1.0) < 1e-6 and C7.pooled_stats(D2, D2, q=1)["k90"] == 1


def test_rotation_preserves_total_and_fitted_basis_makes_single_direction_trivially_sparse():
    m = 64; torch.manual_seed(0)
    delta, g = torch.randn(m), torch.randn(m)
    R = C7.haar_orthogonal(m, 1, "cpu"); assert torch.allclose(R @ R.T, torch.eye(m), atol=1e-4)
    assert abs(float(((R @ delta) * (R @ g)).sum() - (delta * g).sum())) < 1e-3                      # function-preserving: the net contribution is invariant
    # fitted rotation aligning g with the first axis: every contrast (here a different delta) concentrates on ONE coordinate
    V = (g / g.norm())[:, None]; B = C7.completed_basis(V, 3, "cpu"); assert torch.allclose(B.T @ B, torch.eye(m), atol=1e-4)
    for d2 in (delta, torch.randn(m)):
        D = (B.T @ d2) * (B.T @ g); assert D[1:].abs().max() < 1e-4 * D[0].abs() + 1e-5 and abs(float(D.sum() - (d2 * g).sum())) < 1e-3
