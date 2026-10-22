import numpy as np
import pytest
import torch

from causal_subspaces.heads.ablation import capture_z, head_ablation, head_z_patch
from causal_subspaces.heads.decompose import head_dims
from causal_subspaces.mediation import analysis as AN
from causal_subspaces.mediation import controls as CT
from causal_subspaces.mediation.units import OrderLog, capture_blocks, capture_units, log_modules, restore_units
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits

L = 2            # head-intervention layer; hook = output of layer L  -> residual_index L+1
IDX = L + 1
TEXTS = ["the cat sat down now", "a big dog ran away fast today", "one two three four five", "red green blue yellow"]


@pytest.fixture()
def m32(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)


def batch(tok):
    return encode(tok, [format_prompt(tok, t) for t in TEXTS], "cpu")


def run(m32, b, ctxs=(), units=None, blocks=None, d=None, resid=None, position="last"):
    ids, mask, pos = b
    from contextlib import ExitStack
    with ExitStack() as st:
        for c in ctxs: st.enter_context(c)
        sink_u = {}
        if units is not None: st.enter_context(capture_units(m32, units, sink_u, position))
        if blocks is not None: st.enter_context(capture_blocks(m32, range(4), d, blocks))
        if resid is not None: st.enter_context(capture_residual(m32, IDX, resid))
        lg = last_logits(m32, ids, mask, pos)
    return lg, sink_u


def test_capture_matches_manual_hook_and_self_restore_is_noop(m32, tok):
    b = batch(tok); idx = torch.tensor([1, 5, 9, 30])
    lg, su = run(m32, b, units={2: idx})
    man = {}
    h = m32.model.layers[2].mlp.down_proj.register_forward_pre_hook(lambda m, i: man.__setitem__("a", i[0][:, -1][:, idx].clone())); last_logits(m32, *b); h.remove()
    assert torch.equal(su[2], man["a"])
    with restore_units(m32, {2: idx}, {2: su[2]}):
        again = last_logits(m32, *b)
    assert torch.equal(again, lg)                                  # restoring a unit set with its own clean values is the identity
    assert all(len(l.mlp.down_proj._forward_pre_hooks) == 0 for l in m32.model.layers)


def test_full_restore_reproduces_clean_mlp_output_and_isolates_direct_attention_write(m32, tok):
    b = batch(tok); H, HD, _ = head_dims(m32)
    d = torch.nn.functional.normalize(torch.randn(32), dim=0)
    full = {L: torch.arange(64)}
    blocks_c, resid_c = {}, []
    lg_c, su_c = run(m32, b, units=full, blocks=blocks_c, d=d, resid=resid_c)
    zbar = {l: torch.randn(H * HD) * 0.3 for l in range(4)}
    blocks_i, resid_i = {}, []
    lg_i, su_i = run(m32, b, ctxs=[head_ablation(m32, {L: [0, 2]}, IDX, zbar=zbar)], units=full, blocks=blocks_i, d=d, resid=resid_i)
    # accounting identity at the hook: c = d.(h_in) + d.attn + d.mlp ; the head intervention in layer L leaves h_in unchanged
    hin = []
    h = capture_residual(m32, L, hin)
    with h: last_logits(m32, *b)
    for blocks, resid in ((blocks_c, resid_c), (blocks_i, resid_i)):
        c = resid[0][:, -1].float() @ d
        assert torch.allclose(c, hin[0][:, -1].float() @ d + blocks["attn"][L] + blocks["mlp"][L], atol=1e-4)
    dc = (resid_i[0][:, -1].float() - resid_c[0][:, -1].float()) @ d
    assert torch.allclose(dc, (blocks_i["attn"][L] - blocks_c["attn"][L]) + (blocks_i["mlp"][L] - blocks_c["mlp"][L]), atol=1e-4)
    # restore ALL MLP units of layer L to clean values after the head intervention -> MLP write equals clean; coordinate change == direct attention change
    blocks_r, resid_r = {}, []
    lg_r, _ = run(m32, b, ctxs=[head_ablation(m32, {L: [0, 2]}, IDX, zbar=zbar), restore_units(m32, full, {L: su_c[L]})], blocks=blocks_r, d=d, resid=resid_r)
    assert torch.allclose(blocks_r["mlp"][L], blocks_c["mlp"][L], atol=1e-5)
    dcr = (resid_r[0][:, -1].float() - resid_c[0][:, -1].float()) @ d
    assert torch.allclose(dcr, blocks_i["attn"][L] - blocks_c["attn"][L], atol=1e-4)
    assert not torch.allclose(lg_r, lg_c, atol=1e-5)              # the attention write still changes the logits (and is not the clean model)


def test_transplant_into_clean_run(m32, tok):
    """Y01: clean heads, units overwritten by the intervened run's activations -> the MLP write equals the intervened MLP write for those units."""
    b = batch(tok); H, HD, _ = head_dims(m32)
    d = torch.nn.functional.normalize(torch.randn(32), dim=0)
    full = {L: torch.arange(64)}; zbar = {l: torch.randn(H * HD) * 0.3 for l in range(4)}
    bl_c, bl_i, bl_t = {}, {}, {}
    run(m32, b, units=full, blocks=bl_c, d=d)
    _, su_i = run(m32, b, ctxs=[head_ablation(m32, {L: [1]}, IDX, zbar=zbar)], units=full, blocks=bl_i, d=d)
    run(m32, b, ctxs=[restore_units(m32, full, {L: su_i[L]})], blocks=bl_t, d=d)
    assert torch.allclose(bl_t["mlp"][L], bl_i["mlp"][L], atol=1e-5)
    assert torch.allclose(bl_t["attn"][L], bl_c["attn"][L], atol=1e-6)      # attention untouched in the transplant run


def test_hook_order_and_layer_causality(m32, tok):
    b = batch(tok); H, HD, _ = head_dims(m32)
    log = OrderLog(); idx = torch.tensor([0, 3])
    zbar = {l: torch.randn(H * HD) * 0.3 for l in range(4)}
    with log_modules(m32, range(4), log):
        with head_ablation(m32, {L: [0]}, IDX, zbar=zbar), capture_units(m32, {L: idx, 1: idx}, {}, log=log), restore_units(m32, {L: idx}, {L: torch.zeros(4, 2)}, log=log):
            last_logits(m32, *b)
    for l in range(4):
        p = {n: log.positions(n, l)[0] for n in ("o_proj_pre", "attn_out", "down_proj_pre", "mlp_out")}
        assert p["o_proj_pre"] < p["attn_out"] < p["down_proj_pre"] < p["mlp_out"]            # module order inside every layer
    # mediator hooks fire at down_proj_pre of THEIR layer: after the layer's attention output and before the layer's MLP output
    for n in ("unit_capture", "unit_restore"):
        pos = log.positions(n, L)[0]
        assert log.positions("attn_out", L)[0] < pos < log.positions("mlp_out", L)[0]
    # causality: a head intervention at layer L cannot change units of lower layers but changes units of layer L (same layer, after attention)
    _, su_c = run(m32, b, units={1: torch.arange(64), L: torch.arange(64), L + 1: torch.arange(64)})
    _, su_i = run(m32, b, ctxs=[head_ablation(m32, {L: [0, 1]}, IDX, zbar=zbar)], units={1: torch.arange(64), L: torch.arange(64), L + 1: torch.arange(64)})
    assert torch.equal(su_c[1], su_i[1])
    assert not torch.allclose(su_c[L].float(), su_i[L].float()) and not torch.allclose(su_c[L + 1].float(), su_i[L + 1].float())


def test_last_row_head_intervention_leaves_earlier_positions_unchanged(m32, tok):
    b = batch(tok); H, HD, _ = head_dims(m32); zbar = {l: torch.randn(H * HD) * 0.3 for l in range(4)}
    full = {L: torch.arange(64)}
    _, c_all = run(m32, b, units=full, position="all")
    _, i_all = run(m32, b, ctxs=[head_ablation(m32, {L: [0, 1]}, IDX, zbar=zbar, position="last")], units=full, position="all")
    assert torch.equal(c_all[L][:, :-1], i_all[L][:, :-1]) and not torch.equal(c_all[L][:, -1], i_all[L][:, -1])


def test_head_z_patch_then_restore_shapes_and_position_all(m32, tok):
    b = batch(tok); idx = torch.arange(10)
    _, su = run(m32, b, units={L: idx}, position="all")
    with restore_units(m32, {L: idx}, {L: su[L]}, position="all"):
        a = last_logits(m32, *b)
    assert torch.equal(a, last_logits(m32, *b))
    with pytest.raises(ValueError):
        with restore_units(m32, {L: idx}, {L: su[L][:, :-1]}, position="all"):
            last_logits(m32, *b)


# ---- analysis ------------------------------------------------------------------------------------------------------
def test_algebra_and_undefined_rules():
    rg = np.random.default_rng(0); n = 400
    y00 = rg.normal(0, 1, n); te_true = 10.0
    y11 = y00 + te_true + rg.normal(0, 1, n); y10 = y00 + 6 + rg.normal(0, 1, n); y01 = y00 + 5 + rg.normal(0, 1, n)
    r = AN.mediation_effects(y00, y11, y10, y01, floor=2.0)
    assert not r["undefined"] and r["R"] + r["B"] == pytest.approx(1.0) and r["J"] == pytest.approx(r["R"] - r["S"]) and r["Phi"] == pytest.approx(0.5 * (r["R"] + r["S"]))
    assert r["R"] == pytest.approx(0.4, abs=0.05) and r["B"] == pytest.approx(0.6, abs=0.05) and r["S"] == pytest.approx(0.5, abs=0.05)
    assert r["R_ci"][0] < r["R"] < r["R_ci"][1]
    small = AN.mediation_effects(y00, y00 + 0.5 + rg.normal(0, 1, n), y00, y00, floor=2.0)
    assert small["undefined"] and "below floor" in small["reason"]
    noisy = AN.mediation_effects(y00[:20], y00[:20] + 3.0 + rg.normal(0, 8, 20), y00[:20], y00[:20], floor=2.0, n_boot=200)   # few noisy trials: TE unstable in the bootstrap
    assert noisy["undefined"]
    flo = AN.mediation_effects(y00, y11, y10, y01, floor=2.0, noise_floor=4.0)
    assert flo["undefined"]                                                    # |TE| < 3 x noise floor


def test_aggregated_ratio_is_not_the_mean_of_per_example_ratios():
    rg = np.random.default_rng(1); n = 300
    y00 = np.zeros(n); y11 = rg.normal(4, 6, n); y10 = 0.5 * y11 + rg.normal(0, 4, n)          # true aggregated R = 0.5; per-example ratios explode where y11 ~ 0
    agg = AN.mediation_effects(y00, y11, y10, y10, floor=2.0)
    per_ex = (y11 - y10) / y11
    assert abs(agg["R"] - 0.5) < 0.15 and agg["R_ci"][0] < 0.5 < agg["R_ci"][1] + 0.2
    assert per_ex.std() > 5 * agg["R_ci"][1] - agg["R_ci"][0] or np.abs(per_ex).max() > 20            # the per-example ratio is wildly unstable


def test_direction_sign_handling_for_patching():
    sgn = np.array([1.0, -1.0] * 100); rg = np.random.default_rng(2)
    y00 = rg.normal(0, 1, 200); y11 = y00 + sgn * 8; y10 = y00 + sgn * 2; y01 = y00 + sgn * 3
    r = AN.mediation_effects(y00, y11, y10, y01, sgn=sgn, floor=2.0)
    assert r["TE"] == pytest.approx(8.0) and r["R"] == pytest.approx(0.75) and r["S"] == pytest.approx(0.375)


def test_holm_and_control_comparison():
    adj = AN.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.08) and adj["c"] == pytest.approx(0.5)
    r = AN.compare_to_controls(1.0, np.random.default_rng(0).normal(0, 0.1, 20))
    assert r["above_all"] and r["empirical_p"] == pytest.approx(1 / 21) and r["z"] > 5


# ---- controls --------------------------------------------------------------------------------------------------------
def test_control_sets_deterministic_disjoint_and_matched():
    m = 500; rng = np.random.default_rng(0); native = np.sort(rng.choice(m, 30, replace=False)); feat = rng.normal(size=(m, 2))
    a, b = CT.random_set(m, native, 3), CT.random_set(m, native, 3)
    assert np.array_equal(a, b) and len(a) == 30 and not set(a) & set(native) and not np.array_equal(a, CT.random_set(m, native, 4))
    n1, n2 = CT.nearest_set(feat, native, "act", 0), CT.nearest_set(feat, native, "act", 0)
    assert np.array_equal(n1, n2) and len(set(n1)) == 30 and not set(n1) & set(native)
    q = CT.matching_quality({"f0": feat[:, 0]}, native, n1)
    assert abs(q["f0"]) < 0.5
    rand_q = np.mean([abs(CT.matching_quality({"f0": feat[:, 0]}, native, CT.random_set(m, native, d))["f0"]) for d in range(20)])
    assert abs(q["f0"]) <= rand_q + 0.3
    comp = CT.complement(m, native)
    assert len(comp) == m - 30 and not set(comp) & set(native)
    resp = np.zeros(m); resp[[7, 8, 9]] = [3, -5, 4]
    assert set(CT.oracle_set(resp, native, 2)) == {8, 9}
    dm = CT.delta_matched_set(np.abs(rng.normal(size=m)), native)
    assert len(dm) == 30 and not set(dm) & set(native)


def test_coordinate_clamp_restores_clean_coordinate_but_not_other_directions(m32, tok):
    """The clamp hook (used in runner) resets only the d_c coordinate at the hook; the rest of the residual keeps the intervened values."""
    b = batch(tok); H, HD, _ = head_dims(m32)
    d = torch.nn.functional.normalize(torch.randn(32), dim=0); zbar = {l: torch.randn(H * HD) * 0.3 for l in range(4)}
    _, _ = run(m32, b)
    res_c, res_i = [], []
    run(m32, b, resid=res_c); run(m32, b, ctxs=[head_ablation(m32, {L: [0, 2]}, IDX, zbar=zbar)], resid=res_i)
    c_clean = res_c[0][:, -1].float() @ d

    def clamp_hook(m, i, o, tgt=c_clean):
        h = o[0] if isinstance(o, tuple) else o
        new = h.clone(); cur = h[:, -1].float() @ d
        new[:, -1] = (h[:, -1].float() + (tgt - cur)[:, None] * d[None]).to(h.dtype)
        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
    hc = m32.model.layers[IDX - 1].register_forward_hook(clamp_hook)
    res_k = []
    run(m32, b, ctxs=[head_ablation(m32, {L: [0, 2]}, IDX, zbar=zbar)], resid=res_k)
    hc.remove()
    assert torch.allclose(res_k[0][:, -1].float() @ d, c_clean, atol=1e-4)
    perp = lambda x: x[:, -1].float() - (x[:, -1].float() @ d)[:, None] * d[None]
    assert torch.allclose(perp(res_k[0]), perp(res_i[0]), atol=1e-4) and not torch.allclose(perp(res_i[0]), perp(res_c[0]), atol=1e-5)


def test_gap_effects_algebra_and_undefined():
    rg = np.random.default_rng(0); n = 400; acc = np.arange(n) % 2 == 0
    y00 = np.where(acc, 10.0, -10.0) + rg.normal(0, 1, n)                       # clean class gap 20
    y11 = np.where(acc, 2.0, -2.0) + rg.normal(0, 1, n)                         # intervened gap 4 -> TE = 16
    y10 = np.where(acc, 6.0, -6.0) + rg.normal(0, 1, n)                         # rescue: gap 12 -> effect_fixed = 8
    y01 = np.where(acc, 7.0, -7.0) + rg.normal(0, 1, n)                         # transplant: gap 14 -> effect = 6
    r = AN.gap_effects(y00, y11, y10, y01, acc)
    assert not r["undefined"] and r["TE"] == pytest.approx(16, abs=0.5) and r["R"] == pytest.approx(0.5, abs=0.05) and r["S"] == pytest.approx(0.375, abs=0.05) and r["R"] + r["B"] == pytest.approx(1)
    assert AN.gap_effects(y00, y00 + rg.normal(0, 1, n), y10, y01, acc)["undefined"]
