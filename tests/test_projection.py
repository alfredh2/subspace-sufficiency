import torch

from causal_subspaces.mvs.projection import ablate, orthonormality_error, orthonormalize, retain

D, K = 16, 3


def setup(dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 5, D, generator=g, dtype=dtype)
    mu = torch.randn(D, generator=g, dtype=dtype)
    U = orthonormalize(torch.randn(D, K, generator=g))
    return h, mu, U


def test_retain_plus_ablate_reconstructs():
    h, mu, U = setup(torch.float32)
    # retain + ablate = mu + h  (both keep (h-mu) split into in/out of subspace)
    assert torch.allclose(retain(h, U, mu) + ablate(h, U, mu), h + mu, atol=1e-5)


def test_idempotent():
    h, mu, U = setup(torch.float32)
    r = retain(h, U, mu)
    assert torch.allclose(retain(r, U, mu), r, atol=1e-5)
    a = ablate(h, U, mu)
    assert torch.allclose(ablate(a, U, mu), a, atol=1e-5)


def test_orthonormal_after_qr_and_signs_deterministic():
    V = torch.randn(D, K)
    U = orthonormalize(V)
    assert orthonormality_error(U) < 1e-5
    assert torch.allclose(U, orthonormalize(V * 1.0))


def test_full_rank_retain_identity_and_ablate_is_mu():
    h, mu, _ = setup(torch.float32)
    U = orthonormalize(torch.randn(D, D))
    assert torch.allclose(retain(h, U, mu), h, atol=1e-4)
    assert torch.allclose(ablate(h, U, mu), mu.expand_as(h), atol=1e-4)


def test_rank_zero_retain_is_mu_ablate_is_identity():
    h, mu, _ = setup(torch.float32)
    U = torch.zeros(D, 0)
    assert torch.allclose(retain(h, U, mu), mu.expand_as(h))
    assert torch.allclose(ablate(h, U, mu), h)


def test_matches_explicit_projector_formula():
    h, mu, U = setup(torch.float32)  # ops run in fp32 internally; compare with the explicit d x d projector
    P = U @ U.T
    I = torch.eye(D)
    assert torch.allclose(retain(h, U, mu), mu + (h - mu) @ P, atol=1e-5)
    assert torch.allclose(ablate(h, U, mu), mu + (h - mu) @ (I - P), atol=1e-5)


def test_mu_is_fixed_point_of_retain():
    _, mu, U = setup(torch.float32)
    assert torch.allclose(retain(mu, U, mu), mu, atol=1e-6)


def test_gradient_flows_to_V_and_not_nan():
    V = torch.randn(D, K, requires_grad=True)
    h, mu, _ = setup(torch.float32)
    loss = retain(h, orthonormalize(V), mu).pow(2).sum()
    loss.backward()
    assert V.grad is not None and V.grad.abs().sum() > 0 and torch.isfinite(V.grad).all()


def test_bf16_input_returns_bf16_without_inplace():
    h, mu, U = setup(torch.float32)
    hb = h.bfloat16()
    before = hb.clone()
    out = retain(hb, U, mu)
    assert out.dtype == torch.bfloat16 and torch.equal(hb, before)
