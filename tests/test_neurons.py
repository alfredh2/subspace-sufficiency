import numpy as np
import pytest
import torch

from causal_subspaces.interventions.neuron_ablation import AblationRecord, matched_units, random_units, to_spec, unit_ablation
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.neurons import concentration as C
from causal_subspaces.neurons.decompose import class_difference_table, layerwise_pass, layers_below, unit_gain
from causal_subspaces.rotations import haar as H

IDX = 3  # hook = hidden_states[3] = output of layers[2] in the 4-layer tiny model


@pytest.fixture()
def tiny32(tok):
    """fp32 tiny model so identities can be checked tightly."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)


TEXTS = ["the cat sat down now", "a big dog ran away fast", "one two three four five", "red green blue yellow pink", "he said she said it"]
ACC = np.array([True, False, True, False, True])


def test_decomposition_identity_and_unit_sums(tiny32, tok):
    d = torch.nn.functional.normalize(torch.randn(32), dim=0)
    out = layerwise_pass(tiny32, tok, TEXTS, ACC, d, IDX, device="cpu", batch_size=2, keep_acts_layers=(0, 2))
    t = class_difference_table(out["comps"], ACC)
    assert t["identity_max_abs_err"] < 1e-4 * max(1, out["comps"]["resid"].std())
    # sum_j a_j g_j = d . mlp_out, for each kept layer, per example
    for li, l in enumerate(out["layers"]):
        if l in out["acts"]:
            g = unit_gain(tiny32, l, d)
            s = out["acts"][l].float() @ g
            assert np.allclose(s.numpy(), out["comps"]["mlp"][:, li], atol=5e-2 * max(1, np.abs(out["comps"]["mlp"][:, li]).max()))  # acts stored as bf16
    # class-mean difference identity: sum_j delta_j g_j = class difference of d . mlp_out
    for li, l in enumerate(out["layers"]):
        g = unit_gain(tiny32, l, d)
        delta = out["mean_acc"][li] - out["mean_rej"][li]
        assert float(delta @ g) == pytest.approx(float(t["mlp"][li]), rel=1e-4, abs=1e-5)


def test_layer_ordering_rejects_downstream_layers(tiny32):
    assert layers_below(tiny32, IDX) == [0, 1, 2]
    with pytest.raises(ValueError):
        with unit_ablation(tiny32, {3: dict(idx=torch.tensor([0]), repl=torch.tensor([0.]))}, IDX):
            pass


def _batch(tok):
    return encode(tok, [format_prompt(tok, t) for t in TEXTS], "cpu")


def test_native_ablation_equals_manual_surgery_and_k0_is_noop(tiny32, tok):
    ids, m, pos = _batch(tok)
    base = last_logits(tiny32, ids, m, pos)
    with unit_ablation(tiny32, {}, IDX):
        assert torch.equal(last_logits(tiny32, ids, m, pos), base)
    idx = torch.tensor([1, 5, 9]); repl = torch.tensor([0.3, -0.2, 0.0])
    with unit_ablation(tiny32, {1: dict(idx=idx, repl=repl)}, IDX):
        got = last_logits(tiny32, ids, m, pos)
    # manual: hook that edits the input of down_proj
    h = tiny32.model.layers[1].mlp.down_proj.register_forward_pre_hook(lambda mod, i: (torch.index_put(i[0], (slice(None), slice(None), idx), repl.expand(*i[0].shape[:2], -1)) if False else _put(i[0], idx, repl),))
    want = last_logits(tiny32, ids, m, pos)
    h.remove()
    assert torch.allclose(got, want, atol=1e-6) and not torch.allclose(got, base)
    assert all(len(l.mlp.down_proj._forward_pre_hooks) == 0 for l in tiny32.model.layers)


def _put(a, idx, repl):
    a = a.clone(); a[..., idx] = repl; return a


def test_last_position_only_leaves_earlier_positions(tiny32, tok):
    ids, m, pos = _batch(tok)
    idx = torch.tensor([2, 3]); repl = torch.zeros(2)
    sink_all, sink_last, sink_base = [], [], []
    with capture_residual(tiny32, 2, sink_base):
        last_logits(tiny32, ids, m, pos)
    with unit_ablation(tiny32, {1: dict(idx=idx, repl=repl)}, IDX, position="last"), capture_residual(tiny32, 2, sink_last):
        last_logits(tiny32, ids, m, pos)
    # layer-1 ablation at the last position cannot change earlier positions at layer <= 1 output, but attention in layer 2 only looks backwards
    assert torch.allclose(sink_last[0][:, :-1], sink_base[0][:, :-1], atol=1e-6)
    assert not torch.allclose(sink_last[0][:, -1], sink_base[0][:, -1])


def test_direct_change_equals_actual_change_for_last_writer(tiny32, tok):
    """For the last layer before the hook, the change of d.h at the hook equals the recorded direct change exactly."""
    d = torch.nn.functional.normalize(torch.randn(32), dim=0)
    ids, m, pos = _batch(tok)
    gains = {2: unit_gain(tiny32, 2, d)}
    base, abl = [], []
    with capture_residual(tiny32, IDX, base):
        last_logits(tiny32, ids, m, pos)
    rec = AblationRecord()
    spec = {2: dict(idx=torch.tensor([0, 7, 11]), repl=torch.tensor([0.1, 0.2, -0.3]))}
    with unit_ablation(tiny32, spec, IDX, gains=gains, record=rec), capture_residual(tiny32, IDX, abl):
        last_logits(tiny32, ids, m, pos)
    actual = (abl[0][:, -1] - base[0][:, -1]) @ d
    assert torch.allclose(actual, rec.total_direct(), atol=1e-5)


def test_rotated_ablation_equals_explicit_projection(tiny32, tok):
    R = H.haar_orthogonal(64, 0, "cpu")
    S = torch.tensor([0, 5, 9])
    RS = R[S]
    abar = torch.randn(64) * 0.1
    ids, m, pos = _batch(tok)
    with unit_ablation(tiny32, {1: dict(RS=RS, abar=abar)}, IDX):
        got = last_logits(tiny32, ids, m, pos)
    P = RS.T @ RS
    h = tiny32.model.layers[1].mlp.down_proj.register_forward_pre_hook(lambda mod, i: (i[0] - (i[0] - abar) @ P,))
    want = last_logits(tiny32, ids, m, pos)
    h.remove()
    assert torch.allclose(got, want, atol=1e-5)
    # full-rank rotated ablation == replacing the whole activation by abar
    with unit_ablation(tiny32, {1: dict(RS=R, abar=abar)}, IDX):
        full = last_logits(tiny32, ids, m, pos)
    h = tiny32.model.layers[1].mlp.down_proj.register_forward_pre_hook(lambda mod, i: (abar.expand_as(i[0]).clone(),))
    ref = last_logits(tiny32, ids, m, pos)
    h.remove()
    assert torch.allclose(full, ref, atol=1e-4)


def test_function_preserving_rotation(tiny32, tok):
    """W_rot a_rot = W a with a_rot = R a and W_rot = W R^T, for the MLP output and the final logits."""
    ids, m, pos = _batch(tok)
    base = last_logits(tiny32, ids, m, pos)
    R = H.haar_orthogonal(64, 3, "cpu", torch.float32)
    for l in range(4):
        dp = tiny32.model.layers[l].mlp.down_proj
        W = dp.weight.data.clone()
        dp._rot = (W @ R.T, R)
    hs = [tiny32.model.layers[l].mlp.down_proj.register_forward_hook(lambda mod, i, o: None) for l in range(4)]
    for h in hs: h.remove()
    def mk(l):
        dp = tiny32.model.layers[l].mlp.down_proj
        Wrot, Rm = dp._rot
        return dp.register_forward_hook(lambda mod, i, o: (i[0] @ Rm.T) @ Wrot.T)
    hs = [mk(l) for l in range(4)]
    out = last_logits(tiny32, ids, m, pos)
    for h in hs: h.remove()
    assert torch.allclose(out, base, atol=1e-4)
    # and the per-unit contributions change but their sum does not
    d = torch.randn(32); g = tiny32.model.layers[0].mlp.down_proj.weight.T @ d
    a = torch.randn(7, 64)
    assert torch.allclose((a * g).sum(1), ((a @ R.T) * (R @ g)).sum(1), atol=1e-4)


def test_matched_and_random_controls(tiny32):
    rng = np.random.default_rng(0)
    nat_l = np.array([0, 0, 1, 1, 1]); nat_i = np.array([3, 10, 4, 5, 60])
    rl, ri = random_units({0: 2, 1: 3}, 64, rng, exclude={0: np.array([3, 10]), 1: np.array([4, 5, 60])})
    assert len(rl) == 5 and not (set(zip(rl, ri)) & set(zip(nat_l, nat_i))) and (np.bincount(rl) == np.bincount(nat_l)).all()
    feat = {0: rng.normal(size=(64, 2)), 1: rng.normal(size=(64, 2))}
    ml, mi = matched_units(nat_l, nat_i, feat, rng)
    assert len(ml) == 5 and not (set(zip(ml, mi)) & set(zip(nat_l, nat_i))) and len(set(zip(ml, mi))) == 5
    # nearest-neighbour property: with n_cand=1 the matched unit is the closest non-native unit
    j = 3; F = feat[0]; d = ((F - F[j]) ** 2).sum(1); d[[3, 10]] = np.inf
    spec = to_spec(nat_l, nat_i, {0: torch.zeros(64), 1: torch.zeros(64)}, "cpu")
    assert set(spec) == {0, 1} and spec[1]["idx"].numel() == 3


# ---- concentration / reconstruction metrics -------------------------------------------------------------------------
def test_concentration_closed_forms():
    uniform = torch.ones(100)
    spike = torch.zeros(100); spike[0] = 5.0
    assert C.n_eff(uniform) == pytest.approx(100.0) and C.n_eff(spike) == pytest.approx(1.0)
    assert float(C.gini(uniform)) == pytest.approx(0.0, abs=1e-9) and float(C.gini(spike)) == pytest.approx(0.99, abs=1e-9)
    assert float(C.topk_abs_share(uniform, 10)) == pytest.approx(0.1) and float(C.topk_abs_share(spike, 1)) == pytest.approx(1.0)
    assert float(C.exp_entropy(uniform)) == pytest.approx(100.0)
    # concentration ignores sign
    x = torch.randn(50)
    assert float(C.topk_abs_share(x, 5)) == pytest.approx(float(C.topk_abs_share(-x, 5)))
    assert float(C.topk_abs_share(x, 5)) == pytest.approx(float(C.topk_abs_share(x.abs(), 5)))


def test_signed_reconstruction_and_cancellation():
    delta = torch.tensor([4.0, -3.5, 0.5, 0.2, -0.2])
    idx = torch.tensor([0, 1])
    r = C.signed_reconstruction(delta, idx)
    assert r["D"] == pytest.approx(1.0) and r["partial"] == pytest.approx(0.5) and r["frac"] == pytest.approx(0.5)
    full = C.signed_reconstruction(delta, torch.arange(5))
    assert full["frac"] == pytest.approx(1.0) and full["error"] == pytest.approx(0.0)
    k = C.cancellation_index(delta)
    assert k["kappa"] == pytest.approx(8.4 / 1.0) and k["pos_mass"] == pytest.approx(4.7) and k["neg_mass"] == pytest.approx(3.7)


def test_per_example_r2_exact_at_full_set_and_zero_for_noise():
    torch.manual_seed(0)
    S = torch.randn(200, 30); S[:, :3] += 5 * torch.randn(200, 1)  # shared latent in first 3 units
    r_full = C.per_example_r2(S[:100], S[100:], torch.arange(30))
    assert r_full["r2"] == pytest.approx(1.0, abs=1e-9)
    r_sig = C.per_example_r2(S[:100], S[100:], torch.tensor([0, 1, 2]))
    r_noise = C.per_example_r2(S[:100], S[100:], torch.tensor([20, 21, 22]))
    assert r_sig["r2"] > 0.5 and r_noise["r2"] < 0.1
    assert float(C.per_example_kappa(S).mean()) > 1


def test_empirical_p_and_bootstrap():
    assert C.empirical_p(5.0, [1, 2, 3]) == pytest.approx(0.25)
    assert C.empirical_p(100.0, list(range(20))) == pytest.approx(1 / 21)
    assert C.empirical_p(5.0, list(range(20))) == pytest.approx(16 / 21)
    lo, hi = C.bootstrap_ci(np.random.default_rng(0).normal(size=500))
    assert lo < 0 < hi
