import numpy as np
import pytest
import torch

from causal_subspaces.heads import positions as POS
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.upstream import runner as R
from causal_subspaces.upstream.patching import (capture_component, capture_last_z, component_patch, forward_states, freeze_heads, residual_set_patch)
from causal_subspaces.upstream import stats as S


@pytest.fixture()
def m32(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)


def pair(vocab, T=9, B=3, fd=4, pad=(0, 2, 1), seed=0):
    """Random id batches (src/tgt equal up to position fd, equal left padding), explicit position ids."""
    g = torch.Generator().manual_seed(seed)
    tgt = torch.randint(5, vocab, (B, T), generator=g); src = tgt.clone(); src[:, fd:] = torch.randint(5, vocab, (B, T - fd), generator=g)
    mask = torch.ones(B, T, dtype=torch.long)
    for i, p in enumerate(pad): mask[i, :p] = 0
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return tgt, src, mask, pos


def pm(mask, sel):  # (B,T) bool mask over positions `sel` of the padded sequence
    m = torch.zeros_like(mask, dtype=torch.bool); m[:, sel] = True; return m & mask.bool()


def states(m, ids, mask, pos): 
    with torch.no_grad(): return forward_states(m, ids, mask, pos, n_states=4)


def test_identity_noop_and_all_exact_at_every_index(m32, tok):
    tgt, src, mask, pos = pair(len(tok))
    lg_t, hs_t = states(m32, tgt, mask, pos); lg_s, hs_s = states(m32, src, mask, pos)
    allp = mask.bool()
    for i in range(4):
        with torch.no_grad(), residual_set_patch(m32, i, torch.zeros_like(allp), hs_s[i]):
            assert torch.equal(last_logits(m32, tgt, mask, pos), lg_t)                    # empty set
        with torch.no_grad(), residual_set_patch(m32, i, allp, hs_t[i]):
            assert torch.equal(last_logits(m32, tgt, mask, pos), lg_t)                    # source = target
        with torch.no_grad(), residual_set_patch(m32, i, allp, hs_s[i], alpha=0.0):
            assert torch.allclose(last_logits(m32, tgt, mask, pos), lg_t, atol=1e-6)      # alpha 0
        with torch.no_grad(), residual_set_patch(m32, i, allp, hs_s[i]):
            assert torch.allclose(last_logits(m32, tgt, mask, pos), lg_s, atol=1e-5)      # all-position patch = source run (padding positions are inert)
    # padding-only set is inert
    padm = ~mask.bool()
    with torch.no_grad(), residual_set_patch(m32, 2, padm, hs_s[2]):
        assert torch.equal(last_logits(m32, tgt, mask, pos), lg_t)


def test_causality_prediff_noop_lower_layers_unchanged_and_earlier_positions_unchanged(m32, tok):
    tgt, src, mask, pos = pair(len(tok), fd=5)
    lg_t, hs_t = states(m32, tgt, mask, pos); lg_s, hs_s = states(m32, src, mask, pos)
    for i in range(4):                                                                   # positions before the first difference are identical at every index
        pre = (torch.arange(tgt.shape[1]) < 5)[None].expand_as(mask) & mask.bool()
        assert torch.allclose(hs_s[i][pre], hs_t[i][pre], atol=1e-6)
        with torch.no_grad(), residual_set_patch(m32, i, pre, hs_s[i]):
            assert torch.allclose(last_logits(m32, tgt, mask, pos), lg_t, atol=1e-6)     # pre-difference patch = exact no-op
    sel = pm(mask, [6, 7]); seen = {}
    for i in (2, 3):
        with torch.no_grad(), residual_set_patch(m32, i, sel, hs_s[i]):
            _, hs_p = forward_states(m32, tgt, mask, pos, n_states=4)
        for l in range(i): assert torch.equal(hs_p[l], hs_t[l])                          # lower hidden states untouched
        for l in range(i, 4):
            changed = (hs_p[l] - hs_t[l]).abs().amax(-1) > 1e-7
            assert not changed[:, :6].any()                                               # positions before the patched ones unchanged at later layers


def test_hook_order_patch_precedes_layer_i_attention_input(m32, tok):
    tgt, src, mask, pos = pair(len(tok))
    _, hs_s = states(m32, src, mask, pos); sel = pm(mask, [6, 8]); seen = {}
    h = m32.model.layers[2].register_forward_pre_hook(lambda m, a, kw: seen.__setitem__("in", a[0].detach().clone() if a else kw["hidden_states"].detach().clone()), with_kwargs=True)
    with torch.no_grad(), residual_set_patch(m32, 2, sel, hs_s[2]):
        last_logits(m32, tgt, mask, pos)
    h.remove()
    assert torch.allclose(seen["in"][sel], hs_s[2][sel]) and not torch.allclose(seen["in"][~sel & mask.bool()], hs_s[2][~sel & mask.bool()])


def test_interpolation_linear(m32, tok):
    tgt, src, mask, pos = pair(len(tok))
    _, hs_t = states(m32, tgt, mask, pos); _, hs_s = states(m32, src, mask, pos); sel = pm(mask, [6, 7, 8]); seen = {}
    for a in (0.25, 0.5, 1.25):
        h = m32.model.layers[2].register_forward_pre_hook(lambda m, aa, kw: seen.__setitem__("in", (aa[0] if aa else kw["hidden_states"]).detach().clone()), with_kwargs=True)
        with torch.no_grad(), residual_set_patch(m32, 2, sel, hs_s[2], alpha=a):
            last_logits(m32, tgt, mask, pos)
        h.remove()
        assert torch.allclose(seen["in"][sel], (hs_t[2] + a * (hs_s[2] - hs_t[2]))[sel], atol=1e-5)


def test_norm_matched_random_delta_has_exact_norm(m32, tok):
    tgt, src, mask, pos = pair(len(tok))
    _, hs_t = states(m32, tgt, mask, pos); _, hs_s = states(m32, src, mask, pos); sel = pm(mask, [6, 7]); seen = {}
    diff = hs_s[2] - hs_t[2]; rnd = torch.randn(diff.shape); delta = rnd / rnd.norm(dim=-1, keepdim=True) * diff.norm(dim=-1, keepdim=True)
    h = m32.model.layers[2].register_forward_pre_hook(lambda m, aa, kw: seen.__setitem__("in", (aa[0] if aa else kw["hidden_states"]).detach().clone()), with_kwargs=True)
    with torch.no_grad(), residual_set_patch(m32, 2, sel, hs_s[2], delta=delta):
        last_logits(m32, tgt, mask, pos)
    h.remove()
    got = (seen["in"] - hs_t[2])[sel]
    assert torch.allclose(got.norm(dim=-1), diff[sel].norm(dim=-1), rtol=1e-4)


def test_component_accounting(m32, tok):
    tgt, src, mask, pos = pair(len(tok))
    lg_t, hs_t = states(m32, tgt, mask, pos); l = 1; allp = mask.bool(); sa, sm, ta, tm = {}, {}, {}, {}
    with torch.no_grad():
        with capture_component(m32, [l], "attn", ta), capture_component(m32, [l], "mlp", tm): forward_states(m32, tgt, mask, pos)
        with capture_component(m32, [l], "attn", sa), capture_component(m32, [l], "mlp", sm): forward_states(m32, src, mask, pos)
    assert torch.allclose(hs_t[l + 1], hs_t[l] + ta[l] + tm[l], atol=1e-5)               # h_out = h_in + attn_out + mlp_out
    with torch.no_grad(), component_patch(m32, l, "attn", allp, ta[l]), component_patch(m32, l, "mlp", allp, tm[l]):
        assert torch.allclose(last_logits(m32, tgt, mask, pos), lg_t, atol=1e-6)         # patching a component with its own value is the identity
    with torch.no_grad(), component_patch(m32, l, "attn", allp, sa[l]):
        _, hp = forward_states(m32, tgt, mask, pos, n_states=4)
    for k in range(l + 1): assert torch.equal(hp[k], hs_t[k])                            # nothing before the write changes
    assert not torch.allclose(hp[l + 1], hs_t[l + 1], atol=1e-6)
    with torch.no_grad(), component_patch(m32, l, "mlp", allp, sm[l]):
        _, hq = forward_states(m32, tgt, mask, pos, n_states=4)
    assert not torch.allclose(hp[l + 1], hq[l + 1], atol=1e-6)                           # attn-only and mlp-only patches are different interventions


def test_freeze_identity_verification_and_lower_layers(m32, tok):
    tgt, src, mask, pos = pair(len(tok))
    lg_t, hs_t = states(m32, tgt, mask, pos); _, hs_s = states(m32, src, mask, pos); zt = {}
    with torch.no_grad(), capture_last_z(m32, range(4), zt): forward_states(m32, tgt, mask, pos)
    spec = {1: [0, 2], 2: [1, 3], 3: [0, 1, 2, 3]}
    v = {}
    with torch.no_grad(), freeze_heads(m32, spec, zt, v):                                # identity in an unpatched run
        assert torch.allclose(last_logits(m32, tgt, mask, pos), lg_t, atol=1e-6)
    assert float(v["max"]) <= 1e-6
    sel = pm(mask, [6, 7, 8]); v = {}; zp = {}
    with torch.no_grad(), residual_set_patch(m32, 2, sel, hs_s[2]):
        patched = last_logits(m32, tgt, mask, pos)
    with torch.no_grad(), residual_set_patch(m32, 2, sel, hs_s[2]), freeze_heads(m32, {1: [0, 1, 2, 3]}, zt, v):   # layer 1 < index 2: no effect on a patch at 2
        assert torch.allclose(last_logits(m32, tgt, mask, pos), patched, atol=1e-6)
    v = {}
    with torch.no_grad(), residual_set_patch(m32, 2, sel, hs_s[2]), freeze_heads(m32, spec, zt, v), capture_last_z(m32, [2, 3], zp):
        frozen = last_logits(m32, tgt, mask, pos)
    assert float(v["max"]) <= 1e-6                                                       # frozen heads' actual inputs equal the clean cached outputs
    H, HD = 4, 8
    assert torch.allclose(zp[2].view(-1, H, HD)[:, [1, 3]], zt[2].view(-1, H, HD)[:, [1, 3]], atol=1e-6)
    assert not torch.allclose(frozen, patched, atol=1e-5)                                # freezing changes the patched effect (the check is not vacuous)
    # all-heads freeze: no requirement on the effect, only the verification
    v = {}
    with torch.no_grad(), residual_set_patch(m32, 2, sel, hs_s[2]), freeze_heads(m32, {l: [0, 1, 2, 3] for l in (2, 3)}, zt, v):
        last_logits(m32, tgt, mask, pos)
    assert float(v["max"]) <= 1e-6


def test_batch_invariance(m32, tok):
    tgt, src, mask, pos = pair(len(tok))
    with torch.no_grad():
        full = last_logits(m32, tgt, mask, pos)
        single = torch.cat([last_logits(m32, tgt[i:i + 1, mask[i].argmax():], mask[i:i + 1, mask[i].argmax():], pos[i:i + 1, mask[i].argmax():]) for i in range(3)])
    assert torch.allclose(full, single, atol=1e-4)


def test_position_sets_and_random_controls(tok):
    cats = ["first_token"] + ["chat_prefix"] * 3 + ["instruction"] * 4 + ["rule"] * 3 + ["fact"] * 2 + ["conclusion"] * 2 + ["conclusion_final_token"] + ["template_tail"] * 4 + ["self"]
    from causal_subspaces.upstream.data import position_sets
    s = position_sets(cats, 9, "t", 0); s2 = position_sets(cats, 9, "t", 0)
    n = len(cats)
    assert all(np.array_equal(s[k], s2[k]) for k in s if k != "_rand_flagged")                                # deterministic
    assert s["G1"].sum() == 1 and s["Self"].sum() == 1 and s["G2"].sum() == 4
    assert not (s["NonSummary"] & (s["G1G2Self"])).any() and not (s["NonSummary"][:9]).any()                 # variable region only, no summary overlap
    for d in range(5):
        r = s[f"rand{d}"]; assert r.sum() == s["G1G2"].sum() and (r <= s["NonSummary"]).all()
    assert np.array_equal(s["PreDiff"], np.arange(n) < 9) and not (s["PreDiff"] & s["NonSummary"]).any()
    assert not (s["rand0"] == s["rand1"]).all()


def test_stats_algebra_bootstrap_and_onsets():
    rg = np.random.default_rng(0)
    pair = np.repeat(np.arange(60), 2); cont = np.repeat(np.arange(60) % 6, 2)
    sig = np.tile([1.0, -1.0], 60); yt = rg.normal(0, 1, 120); ys = yt + sig * 20
    yp = yt + sig * 10 + rg.normal(0, 1, 120)
    num, den = S.transfer_terms(sig, yp, yt, ys)
    assert num.sum() / den.sum() == pytest.approx(0.5, abs=0.03)
    b = S.Boot(pair, cont, nboot=500, seed=1); r = b.ratio(num, den)
    assert r["point"] == pytest.approx(num.sum() / den.sum()) and S.Boot.ci(r["rep_pair"])[0] < 0.5 < S.Boot.ci(r["rep_pair"])[1] and len(r["loco"]) == 6
    # reversed format: swapping label tokens flips margin sign and sigma stays tied to the VALID member -> ratio unchanged under sign flip of all margins
    n2, d2 = S.transfer_terms(-sig, yp, yt, ys)
    assert n2.sum() / d2.sum() == pytest.approx(0.5, abs=0.03)       # the ratio is invariant to flipping the sign convention of both terms
    # content-clustered CI is wider than pair-level when contents differ
    num_c = num * (1 + 0.6 * (cont % 2 * 2 - 1)); rc = b.ratio(num_c, den)
    assert np.ptp(rc["rep_content"]) > np.ptp(rc["rep_pair"])
    # onsets: monotone rise, noisy dip (first crossing earlier than sustained), and never-sustained
    i = np.arange(24); pt = np.clip((i - 8) / 10, 0, 1.0)
    o = S.onsets(pt, pt - 0.05, pt - 0.1)
    assert o["first_crossing"] <= o["sustained_pair"] <= o["sustained_content"] and o["sustained_both"] == o["sustained_content"]
    dip = pt.copy(); dip[16] = 0.0; lo = dip - 0.05
    o2 = S.onsets(dip, lo, lo)
    assert o2["first_crossing"] < o2["sustained_pair"] and o2["sustained_pair"] == 17
    flat = np.full(24, 0.3); lo3 = flat - 0.4
    assert S.onsets(flat, lo3, lo3)["sustained_both"] is None and S.onsets(flat, lo3, lo3)["first_crossing"] == 0


def test_run_jobs_integration(m32, tok, monkeypatch):
    """End-to-end on a tiny model: all-position patch -> source margin; PreDiff patch -> target; freeze verification; deterministic noise job."""
    monkeypatch.setattr(R, "RL", (1, 2)); monkeypatch.setattr(R, "forward_states", lambda m, i, mk, p: forward_states(m, i, mk, p, n_states=4))
    m32.config.num_hidden_layers = 4
    ctx = R.Ctx.__new__(R.Ctx); ctx.model, ctx.tok, ctx.dev, ctx.idx_hook = m32, tok, "cpu", 3
    ctx.d_c = torch.nn.functional.normalize(torch.randn(32), dim=0); ctx.S8 = [(2, 0), (2, 1), (1, 3)]; ctx.H, ctx.HD, ctx.NL = 4, 8, 4; ctx._spec = {}
    from causal_subspaces.heads.decompose import head_gain
    ctx.gain = {l: head_gain(m32, l, ctx.d_c).view(4, 8).float() for l in (1, 2)}; ctx.s8 = {1: [3], 2: [0, 1]}
    ta, tb = tok.encode("1", add_special_tokens=False)[0], tok.encode("2", add_special_tokens=False)[0]
    texts = [("the cat sat down now", "the dog sat down now"), ("a big bird ran away fast", "a big fish ran away fast")]
    trials = []
    for k, (a, b) in enumerate(texts):
        ia, ib = tok(format_prompt(tok, a), add_special_tokens=False).input_ids, tok(format_prompt(tok, b), add_special_tokens=False).input_ids
        assert len(ia) == len(ib); fd = next(i for i, (x, y) in enumerate(zip(ia, ib)) if x != y); n = len(ia); idx = np.arange(n)
        sets = dict(All=np.ones(n, bool), PreDiff=idx < fd, Tail=idx >= n - 3, NS=(idx >= fd) & (idx < n - 3))
        for tgt, src, sv in ((b, a, 1.0), (a, b, -1.0)):
            trials.append(dict(tgt=tgt, src=src, sigma=sv, ta=ta, tb=tb, sets=sets))
    # format_prompt is applied inside run_jobs: pass the raw user text
    jobs = [dict(name="all", kind="patch", index=2, cond="All"), dict(name="pre", kind="patch", index=2, cond="PreDiff"), dict(name="tail", kind="patch", index=2, cond="Tail"),
            dict(name="tailS8", kind="patch", index=2, cond="Tail", freeze="S8"), dict(name="tailall", kind="patch", index=2, cond="Tail", freeze="all"),
            dict(name="noise", kind="noise", index=2, cond="Tail"), dict(name="comp", kind="comp", index=2, cond="Tail", layer=1, comp="mlp")]
    tgt_b, src_b, res = R.run_jobs(ctx, trials, jobs, bs=4)
    assert np.allclose(res["all"]["margin"], src_b["margin"], atol=1e-4)
    assert np.allclose(res["all"]["r"][:, 1], src_b["r"][:, 1], atol=1e-4) and np.allclose(res["all"]["r"][:, 0], tgt_b["r"][:, 0], atol=1e-6)   # layer 2 reads the patch, layer 1 precedes it
    assert np.allclose(res["pre"]["margin"], tgt_b["margin"], atol=1e-5)
    assert res["tailS8"]["verify"] < 1e-6 and res["tailall"]["verify"] < 1e-6
    assert not np.allclose(res["tail"]["margin"], tgt_b["margin"])
    _, _, res2 = R.run_jobs(ctx, trials, jobs, bs=4)
    assert np.array_equal(res["noise"]["margin"], res2["noise"]["margin"])                            # deterministic noise control
    assert np.allclose(res["noise"]["nrm"], res["tail"]["nrm"], rtol=1e-4, equal_nan=True)            # same per-position patch norm
