import numpy as np
import pytest
import torch

from causal_subspaces import phase06 as P6
from causal_subspaces.models.loading import encode, format_prompt


def test_perp_basis_orthonormal_and_orthogonal_to_d():
    torch.manual_seed(0)
    d = torch.nn.functional.normalize(torch.randn(40), dim=0)
    U = P6.perp_basis(torch.randn(40, 4), d)
    assert torch.allclose(U.T @ U, torch.eye(4), atol=1e-5) and (d @ U).abs().max() < 1e-5


def test_perp_basis_gradient_flows():
    d = torch.nn.functional.normalize(torch.randn(30), dim=0)
    W = torch.randn(30, 2, requires_grad=True)
    (P6.perp_basis(W, d).sum() ** 2).backward()
    assert W.grad is not None and torch.isfinite(W.grad).all() and W.grad.abs().sum() > 0


def test_random_orth_subspace_reproducible_and_orthogonal():
    Q = torch.linalg.qr(torch.randn(64, 16))[0]
    d = torch.nn.functional.normalize(torch.randn(64), dim=0)
    a, b = P6.random_orth_subspace(Q, d, 2, 5), P6.random_orth_subspace(Q, d, 2, 5)
    assert torch.allclose(a, b) and (d @ a).abs().max() < 1e-5
    assert not torch.allclose(a, P6.random_orth_subspace(Q, d, 2, 6))


def test_subspace_variance_matches_direct_covariance():
    torch.manual_seed(0)
    X = torch.randn(4000, 20, dtype=torch.float64) * torch.linspace(0.5, 3, 20)
    cov = torch.cov(X.T, correction=0)
    ev, vec = torch.linalg.eigh(cov)
    Qm, eig = vec[:, -20:].flip(-1), ev[-20:].flip(-1)
    U = torch.linalg.qr(torch.randn(20, 3, dtype=torch.float64))[0]
    assert P6.subspace_variance(U, Qm, eig) == pytest.approx(float(torch.trace(U.T @ cov @ U)), rel=1e-6)


def test_project_out_removes_span():
    X = torch.randn(10, 12)
    D = torch.linalg.qr(torch.randn(12, 3))[0]
    assert (P6.project_out(X, D) @ D).abs().max() < 1e-5
    assert torch.equal(P6.project_out(X, None), X)


def test_probe_learns_linear_signal_and_inlp_degrades():
    torch.manual_seed(0)
    n, d = 600, 30
    y = (torch.rand(n) > .5)
    X = torch.randn(n, d)
    X[:, 3] += 3 * (y.float() - .5)            # signal in a single coordinate
    p = P6.fit_probe(X[:400], y[:400], l2=0.01)
    r = P6.probe_eval(p, X[400:], y[400:].numpy())
    assert r["acc"] > .9 and r["auc"] > .95
    e3 = torch.zeros(d, 1); e3[3] = 1
    r2 = P6.probe_eval(P6.fit_probe(P6.project_out(X[:400], e3), y[:400], .01), P6.project_out(X[400:], e3), y[400:].numpy())
    assert r2["acc"] < .65                      # removing the only informative direction removes decodability
    curve = P6.inlp(X[:400], y[:400], {"t": (X[400:], y[400:].numpy())}, l2=.01, n_iter=3)
    assert curve[0]["t"]["acc"] > .9 and curve[-1]["t"]["acc"] < .65


def test_search_loss_prefers_removed_discrimination_but_penalises_bias_and_control_damage():
    mv0, mi0 = torch.full((8,), 16.), torch.full((8,), -12.)
    z = torch.zeros(8)
    ctrl0 = torch.randn(6) * 10
    base = lambda mv, mi, c=ctrl0: P6.search_loss(mv, mi, mv0, mi0, z, z, c, ctrl0, torch.zeros(6), torch.zeros(6), gap_scale=28.)
    intact = base(mv0, mi0)
    scrambled = base(torch.full((8,), 2.), torch.full((8,), 2.))             # gap closed, pair-mean preserved
    biased = base(torch.full((8,), 18.), torch.full((8,), 18.))              # gap closed by a global shift
    damaged = base(torch.full((8,), 2.), torch.full((8,), 2.), c=ctrl0 + 8)  # gap closed but controls shifted
    assert scrambled["total"] < intact["total"]
    assert biased["total"] > scrambled["total"] and damaged["total"] > scrambled["total"]
    assert intact["disc"] == pytest.approx(1.0, abs=1e-6) and scrambled["bias"] < 1e-6


def test_collect_residuals_matches_hook_capture_and_restores_layers(tiny, tok):
    from causal_subspaces.models.hooks import capture_residual
    texts = ["the cat sat down", "a much longer prompt about a dog"]
    n_layers = len(tiny.model.layers)
    out = P6.collect_residuals(tiny, tok, texts, 2, device="cpu", batch_size=2)
    assert len(tiny.model.layers) == n_layers
    ids, m, pos = encode(tok, [format_prompt(tok, t) for t in texts], "cpu")
    sink = []
    with capture_residual(tiny, 2, sink):
        tiny.model(input_ids=ids, attention_mask=m, position_ids=pos)
    h = sink[0]
    assert np.allclose(out["last"], h[:, -1].numpy(), atol=1e-5)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for i in range(2):
        e = int((ids[i] == im_end).nonzero()[0])
        assert np.allclose(out["concl"][i], h[i, e - 1].numpy(), atol=1e-5)
