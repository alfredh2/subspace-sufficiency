import torch

from causal_subspaces.models.hooks import capture_residual, residual_intervention
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.mvs.projection import orthonormalize, retain


def batch(tok, texts=("hello there", "a somewhat longer prompt here")):
    return encode(tok, [format_prompt(tok, t) for t in texts], "cpu")


def test_no_hook_leftovers_and_noop_matches(tiny, tok):
    ids, m, pos = batch(tok)
    base = last_logits(tiny, ids, m, pos)
    with residual_intervention(tiny, 2, "noop"):
        assert torch.equal(last_logits(tiny, ids, m, pos), base)
    assert all(len(l._forward_hooks) == 0 for l in tiny.model.layers)
    d = tiny.config.hidden_size
    U, mu = orthonormalize(torch.randn(d, 2)), torch.randn(d)
    with residual_intervention(tiny, 2, "ablate", U, mu):
        changed = last_logits(tiny, ids, m, pos)
    assert not torch.allclose(changed, base)
    assert torch.equal(last_logits(tiny, ids, m, pos), base)  # removal restores baseline


def test_hook_removed_on_exception(tiny):
    try:
        with residual_intervention(tiny, 1, "noop"):
            raise RuntimeError
    except RuntimeError:
        pass
    assert all(len(l._forward_hooks) == 0 for l in tiny.model.layers)


def test_full_rank_retain_is_noop_through_model(tiny, tok):
    ids, m, pos = batch(tok)
    d = tiny.config.hidden_size
    U = orthonormalize(torch.randn(d, d))
    base = last_logits(tiny, ids, m, pos)
    with residual_intervention(tiny, 3, "retain", U, torch.randn(d)):
        assert torch.allclose(last_logits(tiny, ids, m, pos), base, atol=1e-4)


def test_residual_index_convention_matches_hf_hidden_states(tiny, tok):
    ids, m, pos = batch(tok)
    hs = tiny.model(input_ids=ids, attention_mask=m, position_ids=pos, output_hidden_states=True).hidden_states
    for i in (1, 2, 3):  # hidden_states[i] = output of layers[i-1]; i == n_layers excluded (final norm applied by HF)
        sink = []
        with capture_residual(tiny, i, sink):
            tiny.model(input_ids=ids, attention_mask=m, position_ids=pos)
        assert torch.allclose(sink[0], hs[i], atol=1e-6), i


def test_hook_applies_to_all_positions(tiny, tok):
    ids, m, pos = batch(tok)
    d = tiny.config.hidden_size
    U, mu = orthonormalize(torch.randn(d, 1)), torch.randn(d)
    sink = []
    with residual_intervention(tiny, 1, "retain", U, mu), capture_residual(tiny, 1, sink):
        tiny.model(input_ids=ids, attention_mask=m, position_ids=pos)
    # capture hook registered after the intervention hook sees the modified output; it must lie in mu + span(U)
    r = sink[0] - mu
    assert torch.allclose(r - (r @ U) @ U.T, torch.zeros_like(r), atol=1e-5)


def test_params_frozen_and_grad_reaches_subspace_param(tiny, tok):
    assert all(not p.requires_grad for p in tiny.parameters())
    ids, m, pos = batch(tok)
    d = tiny.config.hidden_size
    V = torch.randn(d, 2, requires_grad=True)
    with residual_intervention(tiny, 2, "retain", orthonormalize(V), torch.zeros(d)):
        last_logits(tiny, ids, m, pos).logsumexp(-1).sum().backward()
    assert V.grad is not None and V.grad.abs().sum() > 0
    assert all(p.grad is None for p in tiny.parameters())


def test_padding_does_not_change_last_logits(tiny, tok):
    """Left padding + explicit position_ids must equal unpadded single-prompt result."""
    t = "hello there"
    ids1, m1, p1 = batch(tok, (t,))
    ids2, m2, p2 = batch(tok, (t, "a much much longer second prompt in the same batch"))
    assert torch.allclose(last_logits(tiny, ids1, m1, p1)[0], last_logits(tiny, ids2, m2, p2)[0], atol=1e-4)
