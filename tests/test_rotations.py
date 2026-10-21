import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp

from causal_subspaces.neurons import concentration as C
from causal_subspaces.rotations import haar as H


def test_haar_orthogonal_and_reproducible():
    R = H.haar_orthogonal(128, 0, "cpu", torch.float64)
    assert (R @ R.T - torch.eye(128, dtype=torch.float64)).abs().max() < 1e-10
    assert torch.equal(R, H.haar_orthogonal(128, 0, "cpu", torch.float64)) and not torch.equal(R, H.haar_orthogonal(128, 1, "cpu", torch.float64))


def test_rotated_class_difference_matches_per_example_average():
    """Delta_rot_j = (R delta)_j (R g)_j equals the explicit class-mean of per-example rotated contributions."""
    torch.manual_seed(0)
    m, n = 64, 40
    A = torch.randn(n, m, dtype=torch.float64); y = torch.rand(n) > .5
    g = torch.randn(m, dtype=torch.float64); R = H.haar_orthogonal(m, 1, "cpu", torch.float64)
    s = (A @ R.T) * (R @ g)
    explicit = s[y].mean(0) - s[~y].mean(0)
    delta = A[y].mean(0) - A[~y].mean(0)
    assert torch.allclose(explicit, H.rotated_class_difference(R @ delta, R @ g), atol=1e-10)
    assert float(explicit.sum()) == pytest.approx(float(delta @ g), abs=1e-10)  # rotation-invariant net difference


def test_two_frame_sampler_matches_true_haar_distribution():
    m = 256
    torch.manual_seed(0)
    delta = torch.randn(m).pow(3); g = 0.5 * delta + torch.randn(m).pow(3)   # heavy tailed, correlated
    true = []
    for s in range(300):
        R = H.haar_orthogonal(m, 1000 + s, "cpu")
        true.append(float(C.topk_abs_share(H.rotated_class_difference(R @ delta, R @ g), 3)))
    Rd, Rg = H.two_frame_pair(delta, g, 3000, seed=0)
    samp = C.topk_abs_share(Rd * Rg, 3).numpy()
    assert ks_2samp(true, samp).pvalue > 0.01
    # exact invariants of the sampler: norms and inner product are preserved for every draw
    assert torch.allclose(Rd.norm(dim=1), delta.norm().expand(3000), rtol=1e-4)
    assert torch.allclose((Rd * Rg).sum(1), (delta @ g).expand(3000), rtol=1e-3, atol=1e-3)


def test_structured_rotation_orthogonal_but_labelled_not_haar():
    S = H.StructuredRotation(256, seed=0, block=64, rounds=3, device="cpu")
    M = S.matrix()
    assert (M @ M.T - torch.eye(256)).abs().max() < 1e-4
    x = torch.randn(5, 256)
    assert torch.allclose(S.apply(x).norm(dim=1), x.norm(dim=1), rtol=1e-4)
    assert torch.allclose(S.apply(x), x @ M.T, atol=1e-4)
    S2 = H.StructuredRotation(256, seed=0, block=64, rounds=3, device="cpu")
    assert torch.equal(S.perms[0], S2.perms[0])


def test_native_concentration_exceeds_rotated_for_sparse_vectors():
    m = 512
    delta = torch.zeros(m); delta[:4] = 3.0; g = torch.zeros(m); g[:4] = 2.0
    delta += 0.01 * torch.randn(m); g += 0.01 * torch.randn(m)
    native = float(C.topk_abs_share(delta * g, 5))
    Rd, Rg = H.two_frame_pair(delta, g, 200, 0)
    assert native > 0.9 and float(C.topk_abs_share(Rd * Rg, 5).mean()) < 0.5
