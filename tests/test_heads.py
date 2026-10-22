import numpy as np
import pytest
import torch

from causal_subspaces.heads import attention as A
from causal_subspaces.heads import nulls as N
from causal_subspaces.heads.ablation import HeadRecord, capture_z, head_ablation, head_z_patch
from causal_subspaces.heads.decompose import head_dims, head_gain, head_pass, kv_head_of
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.neurons import concentration as C

IDX = 3
TEXTS = ["the cat sat down now", "a big dog ran away fast today", "one two three four five", "red green blue yellow", "he said she said it twice"]
ACC = np.array([True, False, True, False, True])


@pytest.fixture()
def m32(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)


def batch(tok, texts=TEXTS):
    return encode(tok, [format_prompt(tok, t) for t in texts], "cpu")


def test_gqa_mapping_and_dims(m32):
    assert head_dims(m32) == (4, 8, 2)
    assert [kv_head_of(h, m32) for h in range(4)] == [0, 0, 1, 1]
    assert m32.model.layers[0].self_attn.o_proj.bias is None


def test_head_contributions_sum_to_attention_output(m32, tok):
    d = torch.nn.functional.normalize(torch.randn(32), dim=0)
    out = head_pass(m32, tok, TEXTS, ACC, d, IDX, device="cpu", batch_size=len(TEXTS), keep_z=True)
    assert np.allclose(out["S"].sum(-1), out["attn_dot"], atol=1e-4)
    # vector level: sum_h W_O,h z_h == o_proj(z) (no bias)
    H, HD, _ = head_dims(m32)
    W = m32.model.layers[1].self_attn.o_proj.weight
    z = out["z"][:, 1].float()
    full = z @ W.T
    parts = sum(z[:, h * HD:(h + 1) * HD] @ W[:, h * HD:(h + 1) * HD].T for h in range(H))
    assert torch.allclose(full, parts, atol=1e-4)


def test_recompute_matches_original_with_padding(m32, tok):
    ids, mask, pos = batch(tok)
    stash = A.AttnStash(m32, [0, 2])
    z_orig = {}
    with capture_z(m32, [0, 2], z_orig):
        m32.model(input_ids=ids, attention_mask=mask, position_ids=pos)
    for l in (0, 2):
        h, pe = stash.data[l]
        rec = A.recompute(m32, l, h, pe, mask)
        B, T, H, HD = rec["z"].shape
        want = z_orig[l].view(B, T, H, HD)
        real = mask.bool()  # compare real (non-pad) query positions only
        assert torch.allclose(rec["z"][real], want[real], atol=1e-5)
        assert torch.allclose(rec["A"].sum(-1), torch.ones(B, H, T), atol=1e-5)
        assert (torch.triu(rec["A"][0, 0], diagonal=1).abs().max() < 1e-6)
    stash.close()


def test_patch_identities_and_interaction(m32, tok):
    ids_t, m_t, p_t = batch(tok, ["the cat sat down now", "a big dog ran away"])
    ids_s, m_s, p_s = batch(tok, ["red dog sat there now", "one big cat ran home"])
    assert ids_t.shape == ids_s.shape
    st = A.AttnStash(m32, [1]); m32.model(input_ids=ids_t, attention_mask=m_t, position_ids=p_t); ht = st.data[1]
    m32.model(input_ids=ids_s, attention_mask=m_s, position_ids=p_s); hs = st.data[1]; st.close()
    T_ = A.recompute(m32, 1, ht[0], ht[1], m_t)
    S_ = A.recompute(m32, 1, hs[0], hs[1], m_s)
    H = 4
    for h in range(H):
        ASs, VSs = {h: S_["A"][:, h]}, {h: S_["v"][:, h]}
        ATs, VTs = {h: T_["A"][:, h]}, {h: T_["v"][:, h]}
        z_ss = A.recompute(m32, 1, ht[0], ht[1], m_t, A_override=ASs, V_override=VSs)["z"][:, :, h]
        # A+V from the source reproduces the SOURCE head output exactly (given the source's own mask/positions the same as the target's here)
        assert torch.allclose(z_ss, S_["z"][:, :, h], atol=1e-5) or True  # source z also depends on source q only via A, which is overridden
        z_tt = A.recompute(m32, 1, ht[0], ht[1], m_t, A_override=ATs, V_override=VTs)["z"][:, :, h]
        assert torch.allclose(z_tt, T_["z"][:, :, h], atol=1e-5)
        z_st = A.recompute(m32, 1, ht[0], ht[1], m_t, A_override=ASs, V_override=VTs)["z"][:, :, h]
        z_ts = A.recompute(m32, 1, ht[0], ht[1], m_t, A_override=ATs, V_override=VSs)["z"][:, :, h]
        inter = z_ss - z_st - z_ts + z_tt
        # exact accounting: full effect = A-only + V-only + interaction, and the interaction is NOT zero in general
        assert torch.allclose(z_ss - z_tt, (z_st - z_tt) + (z_ts - z_tt) + inter, atol=1e-5)
        # the AV patch reproduces the source head output when the source z is computed with the same mask (same lengths here)
        assert torch.allclose(z_ss, S_["z"][:, :, h], atol=1e-5)
    assert inter.abs().max() > 1e-6


def test_attention_patch_hook_matches_head_z_patch_for_AV_and_self_patch(m32, tok):
    ids_t, m_t, p_t = batch(tok, ["the cat sat down now", "a big dog ran away"])
    ids_s, m_s, p_s = batch(tok, ["red dog sat there now", "one big cat ran home"])
    spec = {1: [0, 3]}
    # cache the source A,v,z
    st = A.AttnStash(m32, [1]); zs = {}
    with capture_z(m32, [1], zs):
        m32.model(input_ids=ids_s, attention_mask=m_s, position_ids=p_s)
    hs = st.data[1]; src = {1: A.recompute(m32, 1, hs[0], hs[1], m_s)}; st.close()
    base = last_logits(m32, ids_t, m_t, p_t)
    with head_z_patch(m32, spec, zs, IDX):
        want = last_logits(m32, ids_t, m_t, p_t)
    with A.attention_patch(m32, spec, {1: dict(A=src[1]["A"], v=src[1]["v"])}, m_t, IDX, patch="AV"):
        got = last_logits(m32, ids_t, m_t, p_t)
    assert torch.allclose(got, want, atol=1e-4) and not torch.allclose(got, base, atol=1e-4)
    # self patch (noise floor): source := target
    stt = A.AttnStash(m32, [1]); m32.model(input_ids=ids_t, attention_mask=m_t, position_ids=p_t); ht = stt.data[1]; stt.close()
    tgt = {1: A.recompute(m32, 1, ht[0], ht[1], m_t)}
    with A.attention_patch(m32, spec, tgt, m_t, IDX, patch="AV"):
        selfp = last_logits(m32, ids_t, m_t, p_t)
    assert torch.allclose(selfp, base, atol=1e-4)
    assert all(len(l.self_attn.o_proj._forward_pre_hooks) == 0 and len(l.self_attn._forward_pre_hooks) == 0 for l in m32.model.layers)
    with pytest.raises(ValueError):
        with A.attention_patch(m32, {3: [0]}, src, m_t, IDX):
            pass


def test_routing_content_identity():
    torch.manual_seed(0)
    Aa, Ab = torch.softmax(torch.randn(6, 20), -1), torch.softmax(torch.randn(6, 20), -1)
    ua, ub = torch.randn(6, 20), torch.randn(6, 20)
    r, c, d = A.routing_content(Aa, Ab, ua, ub)
    assert torch.allclose(r + c, d, atol=1e-5)
    # identical attention => zero routing; identical values => zero content
    r0, _, _ = A.routing_content(Aa, Aa, ua, ub); assert r0.abs().max() < 1e-6
    _, c0, _ = A.routing_content(Aa, Ab, ua, ua); assert c0.abs().max() < 1e-6


def test_head_ablation_equals_manual_surgery_and_direct_change(m32, tok):
    ids, m, pos = batch(tok)
    H, HD, _ = head_dims(m32)
    zbar = {l: torch.randn(H * HD) * 0.1 for l in range(4)}
    base = last_logits(m32, ids, m, pos)
    with head_ablation(m32, {}, IDX, zbar=zbar):
        assert torch.equal(last_logits(m32, ids, m, pos), base)
    with head_ablation(m32, {1: [0, 2]}, IDX, zbar=zbar):
        got = last_logits(m32, ids, m, pos)

    def manual(mod, inp):
        z = inp[0].clone().view(*inp[0].shape[:2], H, HD)
        for h in (0, 2):
            z[:, :, h] = zbar[1].view(H, HD)[h]
        return (z.view(*inp[0].shape),)
    hh = m32.model.layers[1].self_attn.o_proj.register_forward_pre_hook(manual)
    want = last_logits(m32, ids, m, pos); hh.remove()
    assert torch.allclose(got, want, atol=1e-6) and not torch.allclose(got, base)
    # the recorded direct change equals the change of d . (attention output) of that layer (the layer's MLP then reacts, so the hook-level change differs)
    d = torch.nn.functional.normalize(torch.randn(32), dim=0)
    gains = {2: head_gain(m32, 2, d)}
    b_, a_ = {}, {}
    hb = m32.model.layers[2].self_attn.register_forward_hook(lambda mod, i, o: b_.__setitem__("o", o[0][:, -1].clone()))
    last_logits(m32, ids, m, pos); hb.remove()
    rec = HeadRecord()
    ha = m32.model.layers[2].self_attn.register_forward_hook(lambda mod, i, o: a_.__setitem__("o", o[0][:, -1].clone()))
    with head_ablation(m32, {2: [1, 3]}, IDX, zbar=zbar, gains=gains, record=rec):
        last_logits(m32, ids, m, pos)
    ha.remove()
    assert torch.allclose((a_["o"] - b_["o"]) @ d, rec.total_direct(), atol=1e-5)
    with pytest.raises(ValueError):
        with head_ablation(m32, {3: [0]}, IDX, zbar=zbar):
            pass
    assert all(len(l.self_attn.o_proj._forward_pre_hooks) == 0 for l in m32.model.layers)


def test_resample_and_zero_modes_and_last_position(m32, tok):
    ids, m, pos = batch(tok)
    H, HD, _ = head_dims(m32)
    base = last_logits(m32, ids, m, pos)
    for mode in ("zero", "resample"):
        with head_ablation(m32, {1: [0]}, IDX, mode=mode):
            assert not torch.allclose(last_logits(m32, ids, m, pos), base)
    sink_b, sink_l = [], []
    with capture_residual(m32, 2, sink_b):
        last_logits(m32, ids, m, pos)
    with head_ablation(m32, {1: [0, 1]}, IDX, mode="zero", position="last"), capture_residual(m32, 2, sink_l):
        last_logits(m32, ids, m, pos)
    assert torch.allclose(sink_l[0][:, :-1], sink_b[0][:, :-1], atol=1e-6) and not torch.allclose(sink_l[0][:, -1], sink_b[0][:, -1])


def test_z_patch_full_rank_semantics_and_shape_guard(m32, tok):
    ids, m, pos = batch(tok, ["the cat sat down now", "a big dog ran away"])
    ids2, m2, p2 = batch(tok, ["red dog sat there now", "one big cat ran home"])
    zs = {}
    with capture_z(m32, [1], zs):
        m32.model(input_ids=ids2, attention_mask=m2, position_ids=p2)
    sink = []
    with head_z_patch(m32, {1: [0, 1, 2, 3]}, zs, IDX), capture_z(m32, [1], sink_ := {}):
        pass
    # patching ALL heads of a layer reproduces the source o_proj input at that layer
    got = {}
    with head_z_patch(m32, {1: [0, 1, 2, 3]}, zs, IDX), capture_z(m32, [1], got):
        m32.model(input_ids=ids, attention_mask=m, position_ids=pos)
    assert torch.allclose(got[1], zs[1], atol=1e-6)
    with pytest.raises(ValueError):
        with head_z_patch(m32, {1: [0]}, {1: zs[1][:, :-1]}, IDX):
            m32.model(input_ids=ids, attention_mask=m, position_ids=pos)


def test_pair_alignment_accepts_equal_spans_and_rejects_mismatch(tok):
    pa = "<|im_start|>user\nInstr.\n\nThe cat is red.\nIf the cat is big, then it is red.\nHence, the cat is big.<|im_end|>\n<|im_start|>assistant\n"
    la = ["The cat is red.", "If the cat is big, then it is red.", "Hence, the cat is big."]
    pb = pa.replace("The cat is red.\nIf", "The cat is big.\nIf").replace("Hence, the cat is big.", "Hence, the cat is red.")
    lb = ["The cat is big.", "If the cat is big, then it is red.", "Hence, the cat is red."]
    ok, why = A.pair_alignment(tok, pa, la, pb, lb)
    assert ok, why
    pc = pa.replace("The cat is red.", "The extremely large cat is red.")
    ok2, why2 = A.pair_alignment(tok, pa, la, pc, ["The extremely large cat is red."] + la[1:])
    assert not ok2 and why2 in ("length", "line spans differ")
    pd = pa.replace("Instr.", "Other.")
    ok3, why3 = A.pair_alignment(tok, pa, la, pd, la)
    assert not ok3 and "outside" in why3


def test_null_functions_function_preserving_and_layer_matched():
    torch.manual_seed(0)
    n, hd = 256, 32
    delta, gain = torch.randn(n), torch.randn(n)
    nat = N.native_head_delta(delta, gain, hd)
    assert nat.numel() == n // hd and float(nat.sum()) == pytest.approx(float(delta @ gain), abs=1e-4)
    for f in (lambda s: N.haar_block_delta(delta, gain, hd, s, "cpu"), lambda s: N.permuted_block_delta(delta, gain, hd, s), lambda s: N.head_shuffled_delta(delta, gain, hd, s)):
        x = f(1)
        assert x.numel() == n // hd and torch.equal(x, f(1)) and not torch.equal(x, f(2))
    assert float(N.haar_block_delta(delta, gain, hd, 1, "cpu").sum()) == pytest.approx(float(delta @ gain), abs=1e-3)   # net is rotation-invariant
    assert float(N.permuted_block_delta(delta, gain, hd, 1).sum()) == pytest.approx(float(delta @ gain), abs=1e-4)
    z = N.match_layer_mass(N.haar_block_delta(delta, gain, hd, 1, "cpu"), 7.5)
    assert float(z.abs().sum()) == pytest.approx(7.5, rel=1e-5)
    s = N.haar_block_delta_sampler(delta, gain, hd, 50, 0)
    assert s.shape == (50, n // hd)
