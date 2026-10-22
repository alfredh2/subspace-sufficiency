import numpy as np
import pytest
import torch

from causal_subspaces.heads.attention import AttnStash, recompute
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.phase5.patching import attention_rows_patch
from causal_subspaces.phase6 import runner as R6
from causal_subspaces.phase6 import stats as S6
from causal_subspaces.phase6.early import exit_before_attention, run_partial
from causal_subspaces.upstream import stats as S
from causal_subspaces.upstream.patching import capture_component, forward_states, residual_set_patch

LAYER = 3


@pytest.fixture()
def m6(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(1)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=512, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)


@pytest.fixture()
def tiny_ctx(m6, tok, monkeypatch):
    for k, v in dict(LAYER=LAYER, LOCAL=4, FINAL=5, ACC_LAYERS=(4,)).items(): monkeypatch.setattr(R6, k, v)
    from causal_subspaces.heads.decompose import head_gain
    ctx = R6.Ctx5.__new__(R6.Ctx5) if hasattr(R6, "Ctx5") else None
    from causal_subspaces.phase5.runner import Ctx5
    ctx = Ctx5.__new__(Ctx5); ctx.model, ctx.tok, ctx.dev = m6, tok, "cpu"; ctx.d_c = torch.nn.functional.normalize(torch.randn(32), dim=0); ctx.w = torch.nn.functional.normalize(torch.randn(32), dim=0)
    ctx.H, ctx.HD = 4, 8; ctx.gain = head_gain(m6, LAYER, ctx.d_c).view(4, 8).float()
    return ctx


def make_trials(tok):
    ta, tb = tok.encode("1", add_special_tokens=False)[0], tok.encode("2", add_special_tokens=False)[0]
    texts = [("the cat sat down now", "the dog sat down now"), ("a big bird ran away fast", "a big fish ran away fast"), ("one cat sat down now", "one dog sat down now")]
    trials = []
    for a, b in texts:
        ia, ib = tok(format_prompt(tok, a), add_special_tokens=False).input_ids, tok(format_prompt(tok, b), add_special_tokens=False).input_ids
        assert len(ia) == len(ib); fd = next(i for i, (x, y) in enumerate(zip(ia, ib)) if x != y); n = len(ia); idx = np.arange(n); same = np.array([x == y for x, y in zip(ia, ib)])
        sets = dict(G1G2=idx >= n - 4, G1=idx == n - 4, G2=(idx > n - 4) & (idx < n - 1), All=np.ones(n, bool), PreDiff=idx < fd, Other=idx < fd, NonSummary=(idx >= fd) & (idx < n - 4),
                    Fact=(idx >= fd) & (idx < fd + 1), Concl=(idx >= fd + 1) & (idx < n - 5), Rule=idx < 0, KV=(idx >= fd) & (idx <= n - 4), Self=idx == n - 1)
        sets["tokens_same_after_fd"] = (idx > fd) & same & (idx < n - 4); sets["Conclsame"] = sets["tokens_same_after_fd"]
        for tgt, src, sv in ((b, a, 1.0), (a, b, -1.0)): trials.append(dict(tgt=tgt, src=src, sigma=sv, ta=ta, tb=tb, sets=sets, pair=len(trials) // 2, content=len(trials) // 2))
    return trials


def test_early_exit_stash_equals_full_forward_and_hooks_removed(m6, tok):
    ids, mask, pos = encode(tok, [format_prompt(tok, "the cat sat down now"), format_prompt(tok, "a big dog ran away fast today")], "cpu")
    st = AttnStash(m6, [LAYER])
    with torch.no_grad(): lg = last_logits(m6, ids, mask, pos)
    full = (st.data[LAYER][0].clone(), st.data[LAYER][1]); st.close()
    store = {}; run_partial(m6, ids, mask, pos, exit_before_attention(m6, LAYER, store))
    assert torch.equal(store["h"], full[0]) and torch.equal(store["pe"][0], full[1][0])
    with torch.no_grad(): assert torch.equal(last_logits(m6, ids, mask, pos), lg)                       # nothing left hooked
    # a raised error inside the context still removes the hook
    with pytest.raises(RuntimeError):
        with exit_before_attention(m6, LAYER, {}): raise RuntimeError("x")
    with torch.no_grad(): assert torch.equal(last_logits(m6, ids, mask, pos), lg)


def test_origin_equals_direct_value_patch_at_layer_input_and_identity_for_same_source(m6, tok, tiny_ctx):
    trials = make_trials(tok); heads = [0, 2]
    jobs = [dict(name="o_kv", kind="origin", spec=dict(type="resid", index=LAYER, group="KV")), dict(name="o_all", kind="origin", spec=dict(type="resid", index=LAYER, group="All")),
            dict(name="u_kv", kind="u", spec=dict(type="resid", index=LAYER, group="KV"))]
    tb, sb, res = R6.run_jobs(tiny_ctx, heads, trials, jobs, bs=3, comp_layers=range(LAYER))
    # direct reference with the validated Phase-5 value patch (source values from the source forward)
    ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in trials], "cpu"); ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in trials], "cpu")
    st = AttnStash(m6, [LAYER])
    with torch.no_grad(): last_logits(m6, ids_s, m_s, p_s)
    src = recompute(m6, LAYER, *st.data[LAYER], m_s); st.close()
    T = ids_t.shape[1]; pad = lambda c: torch.tensor(np.stack([np.pad(t["sets"][c], (T - len(t["sets"][c]), 0)) for t in trials]))
    with torch.no_grad(), attention_rows_patch(m6, LAYER, heads, src, m_t, pad("G1G2"), patch="V", keys=pad("KV")):
        lg = last_logits(m6, ids_t, m_t, p_t)
    ta_, tb_ = trials[0]["ta"], trials[0]["tb"]; ref = (lg[:, ta_] - lg[:, tb_]).numpy()
    assert np.allclose(res["o_kv"]["margin"], ref, atol=1e-4)
    assert np.allclose(res["o_kv"]["vp_num"] / res["o_kv"]["vp_den"], 1.0, atol=1e-3)                    # injected values = source values at KV
    assert not np.allclose(res["o_kv"]["margin"], tb["margin"])
    # identity: source = target text
    same = [dict(t, src=t["tgt"]) for t in trials]
    tb2, sb2, res2 = R6.run_jobs(tiny_ctx, heads, same, [jobs[0]], bs=3, comp_layers=range(LAYER))
    assert np.allclose(res2["o_kv"]["margin"], tb2["margin"], atol=1e-5)


def test_equality_is_measured_not_assumed(m6, tok, tiny_ctx):
    trials = make_trials(tok)
    jobs = [dict(name=f"o{i}|{g}", kind="origin", spec=dict(type="resid", index=i, group=g)) for i in (0, LAYER) for g in ("Other", "Conclsame")]
    tb, sb, res = R6.run_jobs(tiny_ctx, [0], trials, jobs, bs=3, comp_layers=range(LAYER))
    assert (res[f"o0|Other"]["maxdiff"] == 0).all() and (res[f"o{LAYER}|Other"]["maxdiff"] == 0).all() and (res[f"o{LAYER}|Other"]["eq_frac"] == 1).all()   # before the first difference: equal by causality
    assert (res["o0|Conclsame"]["maxdiff"] == 0).all()                                                   # same tokens -> same embeddings
    assert (res[f"o{LAYER}|Conclsame"]["maxdiff"] > 0).any() and (res[f"o{LAYER}|Conclsame"]["eq_frac"] < 1).any()   # same tokens, different context-dependent activations later


def test_component_direct_write_total_and_joint(m6, tok, tiny_ctx):
    trials = make_trials(tok); l = 1
    jobs = [dict(name="d_attn", kind="origin", spec=dict(type="comp_direct", layer=l, comps=("attn",), group="KV")), dict(name="t_attn", kind="origin", spec=dict(type="comp_total", layer=l, comps=("attn",), group="KV")),
            dict(name="d_mlp", kind="origin", spec=dict(type="comp_direct", layer=l, comps=("mlp",), group="KV")), dict(name="d_joint", kind="origin", spec=dict(type="comp_direct", layer=l, comps=("attn", "mlp"), group="KV")),
            dict(name="t_joint", kind="origin", spec=dict(type="comp_total", layer=l, comps=("attn", "mlp"), group="KV"))]
    tb, sb, res = R6.run_jobs(tiny_ctx, [0, 2], trials, jobs, bs=3, comp_layers=range(LAYER))
    # direct-write identity: hidden_states[LAYER] at P = target + dW exactly
    ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in trials], "cpu"); ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in trials], "cpu")
    cs, ct, hs_t = {"attn": {}, "mlp": {}}, {"attn": {}, "mlp": {}}, None
    with torch.no_grad(), capture_component(m6, [l], "attn", cs["attn"]), capture_component(m6, [l], "mlp", cs["mlp"]): _, hs_s = forward_states(m6, ids_s, m_s, p_s, n_states=LAYER + 1)
    with torch.no_grad(), capture_component(m6, [l], "attn", ct["attn"]), capture_component(m6, [l], "mlp", ct["mlp"]): _, hs_t = forward_states(m6, ids_t, m_t, p_t, n_states=LAYER + 1)
    T = ids_t.shape[1]; P = torch.tensor(np.stack([np.pad(t["sets"]["KV"], (T - len(t["sets"]["KV"]), 0)) for t in trials]))
    dW = cs["attn"][l] - ct["attn"][l]; got = []
    with torch.no_grad(), residual_set_patch(m6, LAYER, P, hs_s[LAYER], delta=dW), capture_residual(m6, LAYER, got): last_logits(m6, ids_t, m_t, p_t)
    assert torch.allclose(got[0][P], (hs_t[LAYER] + dW)[P], atol=1e-5) and torch.allclose(got[0][~P], hs_t[LAYER][~P], atol=1e-6)
    # total influence (downstream recompute through layers l+1..LAYER-1) differs from the direct write; joint differs from the sum of the singles
    assert not np.allclose(res["d_attn"]["margin"], res["t_attn"]["margin"], atol=1e-6)
    s_ = lambda n: res[n]["margin"] - tb["margin"]
    assert not np.allclose(s_("d_joint"), s_("d_attn") + s_("d_mlp"), atol=1e-7)                       # interaction exists in general; reported, never assumed zero


def test_pin_dcomp_accounting_and_consistency(m6, tok, tiny_ctx):
    trials = make_trials(tok); heads = [0, 2]
    jobs = [dict(name="free", kind="pin", heads=heads), dict(name="pinall", kind="pin", heads=heads, pins=[(4, "attn"), (4, "mlp")]), dict(name="pinattn", kind="pin", heads=heads, pins=[(4, "attn")]),
            dict(name="full", kind="dcomp", part="full"), dict(name="dc", kind="dcomp", part="dc"), dict(name="orth", kind="dcomp", part="orth"), dict(name="dc_pin", kind="dcomp", part="dc", pins=[(4, "attn"), (4, "mlp")])]
    tb, sb, res = R6.run_jobs(tiny_ctx, heads, trials, jobs, bs=3, comp_layers=range(LAYER))
    # d_c-write accounting: change of d_c between index 4 and 5 equals the d_c writes of layer 4 (attn + mlp) at the rows
    for nm in ("free", "pinall"):
        assert np.allclose(res[nm]["dC"][:, 1] - res[nm]["dC"][:, 0], res[nm]["W"].sum(1), atol=1e-4)
    assert np.allclose(tb["dC"][:, 1] - tb["dC"][:, 0], tb["W"].sum(1), atol=1e-4)
    assert np.allclose(res["pinall"]["W"], tb["W"], atol=1e-5)                                           # pinned components keep the target's writes
    assert not np.allclose(res["free"]["W"], tb["W"])
    assert np.allclose(res["full"]["margin"], res["free"]["margin"], atol=1e-4)                           # delta-residual patch at the layer output = head patch
    assert not np.allclose(res["dc"]["margin"], res["orth"]["margin"])
    assert np.allclose(res["dc_pin"]["W"], tb["W"], atol=1e-5)


def test_stats_onsets_closing_classes_and_rms():
    rng = np.random.default_rng(0); grid = list(range(0, 20, 2)) + [19]; nb = 400
    def mk(f, sd=0.02):
        f = np.asarray(f, float); return dict(point=f, rp=f + rng.normal(0, sd, (nb, len(f))), rc=f + rng.normal(0, sd, (nb, len(f))), loco=f[None] + rng.normal(0, sd / 2, (8, len(f))))
    f = np.array([0, 0, 0, 0.02, 0.05, 0.1, 0.3, 0.6, 0.9, 0.97, 1.0])
    F = mk(f); o = S6.onsets(F, grid)
    assert o[0.5]["sustained"] == 14 and o[0.5]["first"] == 14 and o[0.9]["sustained"] in (16, 18) and o[0.1]["sustained"] == 12
    assert S6.trajectory_class(F, grid) == "monotone rise"
    dip = f.copy(); dip[8] = 0.0                                                                          # a dip after a rise: first crossing earlier than sustained, non-monotone
    Fd = mk(dip + 0); dip2 = np.array([0, 0, 0, 0, 0, 0.6, 0.7, 0.2, 0.9, 0.97, 1.0]); Fn = mk(dip2)
    assert S6.trajectory_class(Fn, grid) == "non-monotone" and S6.onsets(Fn, grid)[0.5]["first"] < S6.onsets(Fn, grid)[0.5]["sustained"]
    assert S6.trajectory_class(mk([0, 0, -0.3, -0.3, 0, 0, 0, 0, 0, 0.5, 1.0]), grid) == "sign-reversing"
    assert S6.trajectory_class(mk(np.zeros(11) + np.r_[np.zeros(10), 1.0]), grid) in ("monotone rise",)
    R = mk(np.array([0.02, 0.5, 0.4, 0.3, 0.15, 0.08, 0.05, 0.04, 0.03, 0.03, 0.03])); assert S6.closing_index(R, grid) == 12
    assert S6.closing_index(mk(np.full(11, 0.5)), grid) is None and S6.closing_index(mk(np.full(11, 0.0)), grid) == 0
    ref = mk(np.array([0.34])); assert S6.defined_denominator(ref, 0, 0.1)[0] and not S6.defined_denominator(mk(np.array([0.02])), 0, 0.08)[0]
    pair = np.repeat(np.arange(50), 2); cont = np.repeat(np.arange(50) % 5, 2); b = S.Boot(pair, cont, nboot=300, seed=0)
    x, y = rng.normal(0, 1, 100), rng.normal(0, 2, 100); r = S6.rms_ratio(b, x, b, y); assert abs(r["point"] - 0.5) < 0.15 and r["pair"][0] < r["point"] < r["pair"][1]


def test_token_identity_partition_and_fm_controls(tok):
    from causal_subspaces.data.logic import build_contents
    from causal_subspaces.phase6.data import augment, build_fm_trials, cond_mask6
    from causal_subspaces.upstream.data import build_trials
    cs = build_contents("ordinary")
    tr, _ = build_trials(tok, cs, [0, 1], ["if_then", "all_are"], "num_orig"); augment(tok, tr)
    for t in tr:
        s = t["sets"]
        assert not (s["Concl_same"] & s["Concl_diff"]).any() and ((s["Concl_same"] | s["Concl_diff"]) == s["Concl"]).all()
        assert ((s["KV_same"] | s["KV_diff"]) == s["KV"]).all() and (s["KV"] == (s["Fact"] | s["Concl"] | s["G1"])).all()
        assert sum(s[f"tail{k}"].sum() for k in range(9)) == s["G2"].sum()
        rk = cond_mask6(t, "rk6:0"); assert not (rk & s["KV"]).any() and not (rk & s["G1G2"]).any() and (rk & s["Self"]).sum() == 0
    fm, cen = build_fm_trials(tok, cs, [0, 1, 2], ["if_then", "all_are", "fact_first"], "num_orig"); assert len(fm) > 0
    for t in fm:
        assert t["form_pair"] in ("MP/MT", "AC/DA")
        assert len(tok(format_prompt(tok, t["tgt"], False), add_special_tokens=False).input_ids) == len(tok(format_prompt(tok, t["src"], False), add_special_tokens=False).input_ids)
