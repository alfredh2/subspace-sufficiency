import torch

from causal_subspaces.mvs.optimization import kl_teacher_student, lr_at
from causal_subspaces.mvs.reference import StreamingMoments, init_V


def test_streaming_moments_match_direct_with_large_offsets():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5000, 24, generator=g, dtype=torch.float64) * torch.linspace(1, 5, 24)
    x[:, 3] += 800.0  # massive-coordinate stand-in
    m = StreamingMoments(24, "cpu")
    for c in x.split(777):
        m.update(c)
    mu, cov = m.finalize()
    assert torch.allclose(mu, x.mean(0), atol=1e-4)
    assert torch.allclose(cov, torch.cov(x.T, correction=0), atol=1e-3)


def test_init_in_top_subspace_and_reproducible():
    Q = torch.linalg.qr(torch.randn(32, 8))[0]
    V = init_V(Q, 2, seed=1)
    assert torch.allclose(V, init_V(Q, 2, seed=1)) and not torch.allclose(V, init_V(Q, 2, seed=2))
    assert torch.allclose(Q @ (Q.T @ V), V, atol=1e-5)


def test_kl_properties():
    lp = torch.log_softmax(torch.randn(3, 50), -1)
    assert kl_teacher_student(lp, lp).abs().max() < 1e-6
    assert (kl_teacher_student(lp, torch.randn(3, 50)) > 0).all()


def test_lr_schedule_endpoints():
    assert lr_at(0, 100, 1e-2) == 1e-2 and abs(lr_at(100, 100, 1e-2) - 5e-4) < 1e-12
