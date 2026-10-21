import numpy as np
import pytest
import torch

from causal_subspaces.data.logic import INSTRUCTION, build_contents
from causal_subspaces.models.hooks import capture_residual, residual_patch
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.mvs.projection import orthonormalize
from causal_subspaces import phase05 as P


def batch(tok, texts):
    return encode(tok, [format_prompt(tok, t) for t in texts], "cpu")


def cached(tiny, tok, texts, idx):
    ids, m, pos = batch(tok, texts)
    sink = []
    with capture_residual(tiny, idx, sink):
        tiny.model(input_ids=ids, attention_mask=m, position_ids=pos)
    return sink[0], (ids, m, pos)


TEXTS_T = ["the red cat sat down", "the big dog ran away"]
TEXTS_S = ["a small cat sat down", "the old man ran away"]


def test_patch_full_rank_makes_target_equal_source(tiny, tok):
    d = tiny.config.hidden_size
    src, (ids_s, m_s, p_s) = cached(tiny, tok, TEXTS_S, 2)
    ids, m, pos = batch(tok, TEXTS_T)
    assert ids.shape == ids_s.shape
    U = orthonormalize(torch.randn(d, d))
    with residual_patch(tiny, 2, U, src):
        out = last_logits(tiny, ids, m, pos)
    assert torch.allclose(out, last_logits(tiny, ids_s, m_s, p_s), atol=1e-3)


def test_patch_with_self_or_rank0_is_noop(tiny, tok):
    d = tiny.config.hidden_size
    tgt, (ids, m, pos) = cached(tiny, tok, TEXTS_T, 2)
    base = last_logits(tiny, ids, m, pos)
    src, _ = cached(tiny, tok, TEXTS_S, 2)
    with residual_patch(tiny, 2, orthonormalize(torch.randn(d, 2)), tgt):  # source == target
        assert torch.allclose(last_logits(tiny, ids, m, pos), base, atol=1e-5)
    with residual_patch(tiny, 2, torch.zeros(d, 0), src):  # rank 0
        assert torch.allclose(last_logits(tiny, ids, m, pos), base, atol=1e-6)
    assert all(len(l._forward_hooks) == 0 for l in tiny.model.layers)


@pytest.mark.parametrize("position", ["all", "last"])
def test_patch_coordinates_inside_and_outside_subspace(tiny, tok, position):
    d = tiny.config.hidden_size
    U = orthonormalize(torch.randn(d, 3))
    src, _ = cached(tiny, tok, TEXTS_S, 2)
    tgt, (ids, m, pos) = cached(tiny, tok, TEXTS_T, 2)
    sink = []
    with residual_patch(tiny, 2, U, src, position), capture_residual(tiny, 2, sink):
        tiny.model(input_ids=ids, attention_mask=m, position_ids=pos)
    new = sink[0]
    sl = slice(None) if position == "all" else slice(-1, None)
    ort = lambda x: x - (x @ U) @ U.T
    assert torch.allclose((new[:, sl] @ U), (src[:, sl] @ U), atol=1e-5)       # MVS coordinates come from the source
    assert torch.allclose(ort(new[:, sl]), ort(tgt[:, sl]), atol=1e-5)          # orthogonal part stays the target's
    if position == "last":
        assert torch.allclose(new[:, :-1], tgt[:, :-1], atol=1e-6)              # earlier positions untouched


def test_complement_patch_is_the_dual(tiny, tok):
    d = tiny.config.hidden_size
    U = orthonormalize(torch.randn(d, 2))
    src, _ = cached(tiny, tok, TEXTS_S, 1)
    tgt, (ids, m, pos) = cached(tiny, tok, TEXTS_T, 1)
    sink = []
    with residual_patch(tiny, 1, U, src, complement=True), capture_residual(tiny, 1, sink):
        tiny.model(input_ids=ids, attention_mask=m, position_ids=pos)
    ort = lambda x: x - (x @ U) @ U.T
    assert torch.allclose(sink[0] @ U, tgt @ U, atol=1e-5) and torch.allclose(ort(sink[0]), ort(src), atol=1e-5)


def test_patch_shape_mismatch_raises(tiny, tok):
    d = tiny.config.hidden_size
    src, _ = cached(tiny, tok, ["short"], 1)
    _, (ids, m, pos) = cached(tiny, tok, ["a much longer prompt than before here"], 1)
    with pytest.raises(ValueError):
        with residual_patch(tiny, 1, orthonormalize(torch.randn(d, 2)), src[:1]):
            last_logits(tiny, ids[:1], m[:1], pos[:1])
    assert all(len(l._forward_hooks) == 0 for l in tiny.model.layers)


def test_format_variants(tok):
    assert P.build_prompt("num_orig", "S") == INSTRUCTION + "\n\nS"
    for f, v in P.FORMATS.items():
        assert v["valid"] != v["invalid"]
        P.single_token_id(tok, v["valid"]), P.single_token_id(tok, v["invalid"])
    assert P.FORMATS["num_rev"]["valid"] == P.FORMATS["num_orig"]["invalid"]
    assert P.FORMATS["ab_rev"]["valid"] == P.FORMATS["ab_orig"]["invalid"]
    assert P.FORMATS["ab_rev"]["lit_first"] == P.FORMATS["ab_orig"]["lit_first"]  # literal position cue is mapping independent


def test_auc_and_threshold():
    s, y = np.array([1., 2., 3., 4.]), np.array([False, False, True, True])
    assert P.auc(s, y) == 1.0 and P.auc(-s, y) == 0.0
    assert P.best_threshold_acc(s + 100, y) == 1.0   # a pure global shift cannot hide perfect discrimination
    assert P.auc(np.zeros(4), y) == 0.5


def test_answer_stats_constant_vs_shifted(tok):
    ia = P.single_token_id(tok, "1")
    gold = np.array([True] * 4 + [False] * 4)
    m = np.where(gold, 2.0, -2.0).astype(np.float32)
    shifted = dict(logit_aff=m + 10, logit_rej=np.zeros(8, np.float32), lse=np.full(8, 11., np.float32), greedy=np.full(8, ia))
    s = P.answer_stats(tok, shifted, gold, "1", "2")
    assert s["acc"] == 0.5 and s["pred_aff"] == 1.0 and s["auc"] == 1.0 and s["best_threshold_acc"] == 1.0
    assert s["margin_mean_gold_aff"] - s["margin_mean_gold_rej"] == pytest.approx(4.0)


def test_consensus_directions_find_shared_direction():
    torch.manual_seed(0)
    d = 64
    shared = torch.nn.functional.normalize(torch.randn(d, 1), dim=0)
    Us = {}
    for j in range(3):
        other = torch.randn(d, 1)
        other = other - shared * (shared.T @ other)
        Us[j] = torch.linalg.qr(torch.cat([shared * (1 if j else -1), other], 1))[0]
    out = P.consensus_directions(Us)
    for j in Us:
        sh, sp = out[j]["shared"].double(), out[j]["specific"].double()
        assert abs(float(sh @ shared[:, 0].double())) > 0.999 and abs(float(sh @ sp)) < 1e-5
        assert torch.allclose(Us[j].double() @ (Us[j].double().T @ sh), sh, atol=1e-5)  # lies inside its own MVS span


def test_matched_pairs_are_valid_minimal_swaps(tok):
    c = build_contents("ordinary")
    pairs = P.build_pairs(c, range(8), ["if_then", "reversed_if", "guarantee", "subset"], tok)
    assert len(pairs) > 20
    for p in pairs:
        sv, si = p["valid_syl"], p["invalid_syl"]
        assert sv != si
        # same multiset of lowercase words except the marker capitalisation / ordering: compare sorted words
        assert sorted(sv.lower().replace(".", "").split()) == sorted(si.lower().replace(".", "").split())
    ctl = P.lexical_controls(c, pairs, tok)
    assert all(x["base"]["content_id"] != x["other"]["content_id"] and x["base"]["ptype"] == x["other"]["ptype"] for x in ctl)
