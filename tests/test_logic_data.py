import itertools

import pytest

from causal_subspaces.data.logic import (FAMILIES, FORMS, HELDOUT_FAMILIES, INSTRUCTION, TRAIN_FAMILIES, build_contents,
                                         build_splits, dataset_hash, make_example)


def entails(form):
    """Truth-table check of the FORMS schema for rule A=>Q (Q a literal): conclusion true in every world where
    rule and fact hold <=> valid."""
    f = FORMS[form]
    ok = True
    for A, Q in itertools.product([True, False], repeat=2):
        if (not A) or Q:  # rule A => Q
            val = {"A": A, "Q": Q}
            if val[f["fact"]] == f["fact_pos"] and val[f["concl"]] != f["concl_pos"]:
                ok = False
    return ok


@pytest.mark.parametrize("form", list(FORMS))
def test_validity_labels_match_truth_table(form):
    assert entails(form) == FORMS[form]["valid"]


def test_all_examples_labelled_consistently():
    for sp in build_splits(0).values():
        for e in sp:
            assert e.valid == FORMS[e.form.split("_")[0]]["valid"]
            assert e.label == ("1" if e.valid else "2")
            assert e.prompt_text.startswith(INSTRUCTION)


def test_negated_consequent_wording_present():
    c = build_contents("ordinary")[0]
    for fam in FAMILIES:
        pos = make_example("ordinary", 0, c, fam, "MP", False, "x").syllogism
        neg = make_example("ordinary", 0, c, fam, "MP", True, "x").syllogism
        assert pos != neg
        assert any(w in neg for w in ("not ", "No ", "never", "disjoint", "false")), (fam, neg)


def test_mt_conclusion_negates_antecedent():
    c = build_contents("ordinary")[0]
    s = make_example("ordinary", 0, c, "if_then", "MT", False, "x").syllogism.split("\n")
    assert s[1] == f"The {c['noun']} is not {c['adj_b']}." and s[2].endswith(f"{c['noun']} is not {c['adj_a']}.")


def test_split_sizes_and_family_separation():
    s = build_splits(0)
    assert all(len(s[k]) == 256 for k in ("items", "families", "nonwords"))
    assert {e.family for e in s["families"]} == set(HELDOUT_FAMILIES)
    assert not ({e.family for e in s["train"]} & set(HELDOUT_FAMILIES))
    assert len(s["train"]) == 36 * 11 * 8 and len(s["val"]) == 4 * 11 * 8


def test_no_content_leakage_between_optimisation_and_eval():
    s = build_splits(0)
    tr, va, it = ({e.content_id for e in s[k]} for k in ("train", "val", "items"))
    assert not (tr & va) and not (tr & it) and not (va & it)
    assert {e.lexicon for e in s["nonwords"]} == {"nonword"} and {e.lexicon for e in s["train"]} == {"ordinary"}
    ids = [e.id for sp in s.values() for e in sp if sp is not s["families"] and sp is not s["nonwords"]]
    assert len(set(e.id for e in s["train"])) == len(s["train"])


def test_seeded_reproducibility():
    assert dataset_hash(build_splits(0)) == dataset_hash(build_splits(0))
    assert dataset_hash(build_splits(0)) != dataset_hash(build_splits(1))


def test_label_balance_and_stratification():
    s = build_splits(0)
    for k in ("items", "families", "nonwords"):
        assert sum(e.label == "1" for e in s[k]) == 128
