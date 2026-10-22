import numpy as np
import pytest
import torch

from causal_subspaces import phase05 as P5
from causal_subspaces.data.logic import build_contents
from causal_subspaces.heads import attention as A
from causal_subspaces.heads import positions as POS
from causal_subspaces.heads.ablation import capture_z
from causal_subspaces.models.loading import encode, format_prompt, last_logits

IDX = 3
SRC = ["red dog sat there now", "one big cat ran home"]
TGT = ["the cat sat down now", "a big dog ran away"]


@pytest.fixture()
def m32(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)


def _sources(m32, tok, layers=(1,)):
    ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t) for t in TGT], "cpu")
    ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t) for t in SRC], "cpu")
    assert ids_t.shape == ids_s.shape
    st = A.AttnStash(m32, list(layers)); m32.model(input_ids=ids_s, attention_mask=m_s, position_ids=p_s)
    src = {l: A.recompute(m32, l, *st.data[l], m_s) for l in layers}; st.close()
    return (ids_t, m_t, p_t), {l: dict(A=v["A"], v=v["v"]) for l, v in src.items()}


@pytest.mark.parametrize("position", ["last", "all"])
def test_full_mask_equals_unrestricted_and_empty_mask_is_no_intervention(m32, tok, position):
    (ids, m, pos), src = _sources(m32, tok)
    spec = {1: [0, 2]}
    B, T = ids.shape
    base = last_logits(m32, ids, m, pos)
    with A.attention_patch(m32, spec, src, m, IDX, patch="V", position=position):
        unrestricted = last_logits(m32, ids, m, pos)
    with A.attention_patch(m32, spec, src, m, IDX, patch="V", position=position, v_positions=torch.ones(B, T, dtype=torch.bool)):
        full = last_logits(m32, ids, m, pos)
    with A.attention_patch(m32, spec, src, m, IDX, patch="V", position=position, v_positions=torch.zeros(B, T, dtype=torch.bool)):
        empty = last_logits(m32, ids, m, pos)
    assert torch.allclose(full, unrestricted, atol=1e-6)
    assert torch.allclose(empty, base, atol=1e-4) and not torch.allclose(unrestricted, base, atol=1e-4)
    half = torch.zeros(B, T, dtype=torch.bool); half[:, T // 2:] = True
    with A.attention_patch(m32, spec, src, m, IDX, patch="V", position=position, v_positions=half):
        part = last_logits(m32, ids, m, pos)
    assert not torch.allclose(part, base, atol=1e-5) and not torch.allclose(part, unrestricted, atol=1e-5)
    with pytest.raises(ValueError):
        with A.attention_patch(m32, spec, src, m, IDX, patch="V", position=position, v_positions=torch.ones(B, T + 1, dtype=torch.bool)):
            last_logits(m32, ids, m, pos)
    assert all(len(l.self_attn.o_proj._forward_pre_hooks) == 0 for l in m32.model.layers)


def test_padding_positions_are_inert_and_causality_respected(m32, tok):
    """Marking only padding positions changes nothing (keys at padding are masked); for the last query row every real position is causal."""
    texts_t = ["the cat sat down now", "a big dog ran away today again"]
    texts_s = ["red dog sat there now", "one big cat ran home today again"]
    ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t) for t in texts_t], "cpu")
    ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t) for t in texts_s], "cpu")
    if ids_t.shape != ids_s.shape or not torch.equal(m_t, m_s):
        pytest.skip("lengths differ in this tokenisation")
    st = A.AttnStash(m32, [1]); m32.model(input_ids=ids_s, attention_mask=m_s, position_ids=p_s)
    src = {1: A.recompute(m32, 1, *st.data[1], m_s)}; st.close()
    src = {l: dict(A=v["A"], v=v["v"]) for l, v in src.items()}
    base = last_logits(m32, ids_t, m_t, p_t)
    pad_only = ~m_t.bool()
    with A.attention_patch(m32, {1: [0]}, src, m_t, IDX, patch="V", position="last", v_positions=pad_only):
        out = last_logits(m32, ids_t, m_t, p_t)
    assert torch.allclose(out, base, atol=1e-4)


def _example_pair(tok):
    contents = build_contents("ordinary")
    pairs = P5.build_pairs(contents, [0, 1, 2], ["if_then", "fact_first", "guarantee"], tok)
    p = next(q for q in pairs if A.pair_alignment(tok, format_prompt(tok, P5.build_prompt("num_orig", q["valid_syl"])), q["valid_syl"].split("\n"),
                                                  format_prompt(tok, P5.build_prompt("num_orig", q["invalid_syl"])), q["invalid_syl"].split("\n"))[0])
    return p


def test_group_assignment_partition_and_alignment(tok):
    p = _example_pair(tok)
    masks = []
    for syl in (p["valid_syl"], p["invalid_syl"]):
        user = P5.build_prompt("num_orig", syl); prompt = format_prompt(tok, user)
        cats = POS.categorize(tok, prompt, user, POS.spans_logic(user, syl.split("\n"), p["family"]))
        m = POS.group_masks(cats)
        assert POS.check_groups(m) == []
        ids = tok(prompt, add_special_tokens=False).input_ids
        g1 = int(np.where(m["G1"])[0][0])
        assert tok.decode([ids[g1]]).strip() == "." and tok.decode([ids[g1 + 1]]) == "<|im_end|>"           # G1 = the final token of the conclusion, right before the end of the user turn
        g2 = np.where(m["G2"])[0]
        assert g2[0] == g1 + 1 and g2[-1] == len(ids) - 2 and m["Self"][-1] and len(ids) - 1 not in g2          # G2 = tail excluding the query
        assert m["Rem"].sum() + m["G1G2"].sum() == len(ids) and m["RemExSelf"].sum() == m["Rem"].sum() - 1
        assert m["Rem"][0] and not m["G1"][0]                                                                  # first (sink) token and chat prefix are in Remaining
        masks.append(m)
    for k in masks[0]:                                                                                          # aligned pair: identical group masks
        assert np.array_equal(masks[0][k], masks[1][k]), k


def test_pad_mask_never_selects_padding():
    m = np.array([True, False, True])
    out = POS.pad_mask(m, 6)
    assert out.tolist() == [False, False, False, True, False, True]


def test_random_positions_deterministic_pool_restricted_count_matched_and_small_pool():
    pool = np.arange(10, 40)
    a, fa = POS.random_positions(pool, 9, "t", 3, 1); b, fb = POS.random_positions(pool, 9, "t", 3, 1)
    assert np.array_equal(a, b) and not fa and len(a) == 9 and len(set(a)) == 9 and set(a) <= set(pool)
    assert not np.array_equal(a, POS.random_positions(pool, 9, "t", 3, 2)[0]) and not np.array_equal(a, POS.random_positions(pool, 9, "t", 4, 1)[0])
    small, flag = POS.random_positions(np.arange(5), 9, "t", 0, 0)
    assert flag and np.array_equal(small, np.arange(5))
