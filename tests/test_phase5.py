import numpy as np
import pytest
import torch

from causal_subspaces.heads.ablation import capture_z
from causal_subspaces.heads.attention import AttnStash, recompute
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.phase5 import patching as PP
from causal_subspaces.phase5 import runner as R5
from causal_subspaces.upstream.patching import capture_component, component_patch, forward_states, residual_set_patch

L = 1


@pytest.fixture()
def m32(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)


def pair(vocab, T=10, B=3, fd=4, pad=(0, 2, 1), seed=0):
    g = torch.Generator().manual_seed(seed)
    tgt = torch.randint(5, vocab, (B, T), generator=g); src = tgt.clone(); src[:, fd:] = torch.randint(5, vocab, (B, T - fd), generator=g)
    mask = torch.ones(B, T, dtype=torch.long)
    for i, p in enumerate(pad): mask[i, :p] = 0
    return tgt, src, mask, (mask.cumsum(-1) - 1).clamp(min=0)


def rows(mask, sel):
    m = torch.zeros_like(mask, dtype=torch.bool); m[:, sel] = True; return m & mask.bool()


def zcap(m, ids, mask, pos, layer=L):
    z = {}
    with torch.no_grad(), capture_z(m, [layer], z): last_logits(m, ids, mask, pos)
    return z[layer]


def test_z_set_patch_identity_exactness_and_equivalence_to_component_patch(m32, tok):
    tgt, src, mask, pos = pair(len(tok)); r = rows(mask, [6, 7, 9])
    with torch.no_grad(): lg_t = last_logits(m32, tgt, mask, pos)
    zs, zt = zcap(m32, src, mask, pos), zcap(m32, tgt, mask, pos)
    with torch.no_grad(), PP.head_z_set_patch(m32, L, [], r, zs): assert torch.equal(last_logits(m32, tgt, mask, pos), lg_t)           # no heads
    with torch.no_grad(), PP.head_z_set_patch(m32, L, [0, 1, 2, 3], torch.zeros_like(r), zs): assert torch.equal(last_logits(m32, tgt, mask, pos), lg_t)   # no rows
    with torch.no_grad(), PP.head_z_set_patch(m32, L, [0, 1, 2, 3], r, zt): assert torch.allclose(last_logits(m32, tgt, mask, pos), lg_t, atol=1e-6)    # source = target
    seen = {}
    with torch.no_grad(), PP.head_z_set_patch(m32, L, [1, 3], r, zs), capture_z(m32, [L], seen): last_logits(m32, tgt, mask, pos)
    H, HD = 4, 8; a = seen[L].view(3, -1, H, HD); s_ = zs.view(3, -1, H, HD); t_ = zt.view(3, -1, H, HD)
    assert torch.equal(a[:, :, [1, 3]][r], s_[:, :, [1, 3]][r]) and torch.equal(a[:, :, [0, 2]], t_[:, :, [0, 2]]) and torch.equal(a[~r], t_[~r])
    # all heads at rows == attention-output component patch (o_proj is linear, bias free)
    sa = {}
    with torch.no_grad(), capture_component(m32, [L], "attn", sa): last_logits(m32, src, mask, pos)
    with torch.no_grad(), PP.head_z_set_patch(m32, L, [0, 1, 2, 3], r, zs): za = last_logits(m32, tgt, mask, pos)
    with torch.no_grad(), component_patch(m32, L, "attn", r, sa[L]): zb = last_logits(m32, tgt, mask, pos)
    assert torch.allclose(za, zb, atol=1e-5)


def test_z_set_patch_alpha_and_norm_matched_delta(m32, tok):
    tgt, src, mask, pos = pair(len(tok)); r = rows(mask, [6, 7])
    zs, zt = zcap(m32, src, mask, pos), zcap(m32, tgt, mask, pos); seen = {}
    with torch.no_grad(), PP.head_z_set_patch(m32, L, [0, 2], r, zs, alpha=0.25), capture_z(m32, [L], seen): last_logits(m32, tgt, mask, pos)
    H, HD = 4, 8; a = seen[L].view(3, -1, H, HD); s_ = zs.view(3, -1, H, HD); t_ = zt.view(3, -1, H, HD)
    assert torch.allclose(a[:, :, [0, 2]][r], (t_ + 0.25 * (s_ - t_))[:, :, [0, 2]][r], atol=1e-6)
    d = (s_[:, :, [0, 2]] - t_[:, :, [0, 2]]); rnd = torch.randn(d.shape); delta = rnd / rnd.norm(dim=-1, keepdim=True) * d.norm(dim=-1, keepdim=True)
    with torch.no_grad(), PP.head_z_set_patch(m32, L, [0, 2], r, zs, delta=delta), capture_z(m32, [L], seen): last_logits(m32, tgt, mask, pos)
    got = (seen[L].view(3, -1, H, HD) - t_)[:, :, [0, 2]][r]
    assert torch.allclose(got.norm(dim=-1), d[r].norm(dim=-1), rtol=1e-4)


def _src_av(m, src, mask, pos):
    st = AttnStash(m, [L])
    with torch.no_grad(): last_logits(m, src, mask, pos)
    out = recompute(m, L, *st.data[L], mask); st.close(); return out


def test_attention_rows_patch_null_additivity_causality_and_av_equals_z(m32, tok):
    tgt, src, mask, pos = pair(len(tok)); r = rows(mask, [7, 8, 9]); B, T = tgt.shape; heads = [0, 1, 2, 3]
    sav, tav = _src_av(m32, src, mask, pos), _src_av(m32, tgt, mask, pos)
    zt, zs = zcap(m32, tgt, mask, pos), zcap(m32, src, mask, pos)

    def z_after(patch, keys=None, source=sav):
        seen = {}
        with torch.no_grad(), PP.attention_rows_patch(m32, L, heads, source, mask, r, patch=patch, keys=keys), capture_z(m32, [L], seen): last_logits(m32, tgt, mask, pos)
        return seen[L]
    assert torch.allclose(z_after("V", source=tav), zt, atol=1e-6) and torch.allclose(z_after("AV", source=tav), zt, atol=1e-6)      # null patch = native z
    k1, k2 = rows(mask, [4, 5]), rows(mask, [6, 8]); H, HD = 4, 8
    d = lambda z: (z - zt).view(B, T, H, HD)
    d1, d2, d12 = d(z_after("V", k1)), d(z_after("V", k2)), d(z_after("V", k1 | k2))
    assert torch.allclose(d1 + d2, d12, atol=1e-5)                                         # Delta z is additive over disjoint key groups (attention weights fixed)
    assert (d(z_after("V")).abs().amax((2, 3))[~r] < 1e-6).all()                            # rows outside R unchanged
    # causality: a key later than the row cannot affect it
    kl = rows(mask, [9]); dl = d(z_after("V", kl)); assert (dl[:, :9].abs().amax() < 1e-6)
    # AV with every key/head at rows R reproduces the model-native z-patch of the same rows
    assert torch.allclose(z_after("AV")[r.view(B, T, 1).expand(-1, -1, H * HD)], zs[r.view(B, T, 1).expand(-1, -1, H * HD)], atol=1e-4)


def test_pinning_mechanics(m32, tok):
    tgt, src, mask, pos = pair(len(tok)); r = rows(mask, [7, 8, 9])
    with torch.no_grad(): lg_t = last_logits(m32, tgt, mask, pos)
    clean = {"attn": {}, "mlp": {}}
    with torch.no_grad(), capture_component(m32, [2, 3], "attn", clean["attn"]), capture_component(m32, [2, 3], "mlp", clean["mlp"]): last_logits(m32, tgt, mask, pos)
    with torch.no_grad(), PP.pinned_rows(m32, [2, 3], ("attn", "mlp"), r, clean): assert torch.allclose(last_logits(m32, tgt, mask, pos), lg_t, atol=1e-6)   # pin w/o patch = identity
    zs = zcap(m32, src, mask, pos); free, pinned_out = {}, {}
    with torch.no_grad(), PP.head_z_set_patch(m32, L, [0, 1, 2, 3], r, zs): lf = last_logits(m32, tgt, mask, pos)
    sink = {}
    with torch.no_grad(), PP.head_z_set_patch(m32, L, [0, 1, 2, 3], r, zs), PP.pinned_rows(m32, [2, 3], ("attn", "mlp"), r, clean), capture_component(m32, [2, 3], "attn", sink):
        lp = last_logits(m32, tgt, mask, pos)
    for l in (2, 3): assert torch.allclose(sink[l][r], clean["attn"][l][r], atol=1e-6)        # pinned outputs equal the clean target's
    assert not torch.allclose(lf, lp, atol=1e-5)                                               # pinning changes the patched effect (non-vacuous)


def test_run_jobs_integration(m32, tok, monkeypatch):
    monkeypatch.setattr(R5, "LAYER", 1); monkeypatch.setattr(R5, "LOCAL", 2); monkeypatch.setattr(R5, "FINAL", 3); monkeypatch.setattr(R5, "PIN_LAYERS", {"22": (2, 3), "35": (2, 3)})
    ctx = R5.Ctx5.__new__(R5.Ctx5); ctx.model, ctx.tok, ctx.dev = m32, tok, "cpu"; ctx.d_c = torch.nn.functional.normalize(torch.randn(32), dim=0); ctx.w = torch.nn.functional.normalize(torch.randn(32), dim=0)
    ctx.H, ctx.HD = 4, 8; ctx.gain = torch.randn(4, 8)
    ta, tb = tok.encode("1", add_special_tokens=False)[0], tok.encode("2", add_special_tokens=False)[0]
    texts = [("the cat sat down now", "the dog sat down now"), ("a big bird ran away fast", "a big fish ran away fast")]
    trials = []
    for a, b in texts:
        ia, ib = tok(format_prompt(tok, a), add_special_tokens=False).input_ids, tok(format_prompt(tok, b), add_special_tokens=False).input_ids
        fd = next(i for i, (x, y) in enumerate(zip(ia, ib)) if x != y); n = len(ia); idx = np.arange(n)
        sets = dict(G1G2=idx >= n - 4, G1=idx == n - 4, G2=(idx > n - 4) & (idx < n - 1), All=np.ones(n, bool), PreDiff=idx < fd, NonSummary=(idx >= fd) & (idx < n - 4), Fact=(idx >= fd) & (idx < fd + 2), Concl=(idx >= fd + 2) & (idx < n - 4))
        for tgt, src, sv in ((b, a, 1.0), (a, b, -1.0)): trials.append(dict(tgt=tgt, src=src, sigma=sv, ta=ta, tb=tb, sets=sets))
    heads = [0, 1, 2, 3]
    jobs = [dict(name="zall", kind="z", heads=heads, rows="G1G2"), dict(name="zall_pin", kind="z", heads=heads, rows="G1G2", pin=dict(layers="22", which=("attn", "mlp"))),
            dict(name="avall", kind="av", heads=heads, patch="AV", rows="G1G2"), dict(name="vnoop", kind="av", heads=heads, patch="V", rows="G1G2", keys="Other"),
            dict(name="z_half", kind="z", heads=[0, 1], rows="G1G2", alpha=0.5), dict(name="noise", kind="z", heads=[0, 1], rows="G1G2", noise=True), dict(name="res", kind="res", index=2, cond="G1G2")]
    tb_, sb_, res = R5.run_jobs(ctx, trials, jobs, bs=4)
    assert float(res["zall"]["vz"]) == 0.0 and float(res["zall_pin"]["vp"]) <= 1e-6
    assert np.allclose(res["avall"]["margin"], res["zall"]["margin"], atol=1e-4) and np.allclose(res["avall"]["w20"], res["zall"]["w20"], atol=1e-4)
    assert np.allclose(res["vnoop"]["margin"], tb_["margin"], atol=1e-5)                             # 'Other' keys are identical -> exact no-op
    _, _, res2 = R5.run_jobs(ctx, trials, jobs, bs=4); assert np.array_equal(res["noise"]["margin"], res2["noise"]["margin"])   # deterministic given the batch layout
    assert not np.allclose(res["z_half"]["margin"], tb_["margin"])
    # dh storage
    out = R5.run_jobs(ctx, trials, jobs[:1], bs=4, keep_dh=True); assert out[3].shape == (4, 32)
