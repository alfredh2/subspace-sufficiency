import pytest
import torch

from causal_subspaces.evaluation.metrics import recovery, summarize, wilson
from causal_subspaces.evaluation.scoring import evaluate_prompts, label_token_ids, score_first_token, sequence_logprob
import pandas as pd


def test_single_token_labels(tok):
    ids = label_token_ids(tok, ["1", "2"])
    assert len(set(ids)) == 2
    assert label_token_ids(tok, ["A", "B"])


def test_multi_token_label_rejected(tok):
    n = len(tok.encode("1234", add_special_tokens=False))
    assert n > 1
    with pytest.raises(ValueError):
        label_token_ids(tok, ["1234"])


def test_primary_vs_restricted_and_invalid(tok):
    one, two = label_token_ids(tok, ["1", "2"])
    V = len(tok)
    lg = torch.full((3, V), -10.0)
    lg[0, one] = 5            # greedy "1", correct
    lg[1, tok.encode("Sure", add_special_tokens=False)[0]] = 9; lg[1, one] = 5   # greedy invalid; restricted says 1
    lg[2, two] = 5
    rows = score_first_token(lg, tok, [one, two], ["1", "2"], ["1", "1", "1"])
    assert [r["correct"] for r in rows] == [True, False, False]
    assert [r["invalid"] for r in rows] == [False, True, False]
    assert [r["restricted_correct"] for r in rows] == [True, True, False]


def test_sequence_logprob_handles_multi_token(tiny, tok):
    res = sequence_logprob(tiny, tok, ["hello"], ["1", "1234"], device="cpu")[0]
    assert res["1"]["n_tokens"] == 1 and res["1234"]["n_tokens"] == 4
    assert res["1234"]["sum"] < 0 and abs(res["1234"]["mean"] * 4 - res["1234"]["sum"]) < 1e-4


def test_evaluate_prompts_batch_order_independent(tiny, tok):
    texts = ["short", "a considerably longer prompt than the first one", "mid length prompt"]
    kw = dict(device="cpu", batch_size=2)
    a = evaluate_prompts(tiny, tok, texts, ["1"] * 3, ["1", "2"], **kw)
    b = evaluate_prompts(tiny, tok, texts[::-1], ["1"] * 3, ["1", "2"], **kw)[::-1]
    assert [x["p_1"] for x in a] == pytest.approx([x["p_1"] for x in b], abs=1e-4)


def test_metrics():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    assert recovery(0.99, 0.99, 0.5) == 1 and recovery(0.5, 0.99, 0.5) == 0
    df = pd.DataFrame(dict(correct=[True, False], restricted_correct=[True, True], invalid=[False, True],
                           mass_on_labels=[1.0, .5], first_char=["1", "S"]))
    s = summarize(df)
    assert s["overall"]["acc"] == .5 and s["overall"]["invalid_rate"] == .5
