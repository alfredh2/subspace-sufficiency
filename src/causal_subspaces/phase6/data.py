"""Phase 6 trial builders: token-identity partition of the position sets, same-label different-form (Fm) controls, set C with manifest verification."""
import hashlib
import json

import numpy as np

from causal_subspaces import phase05 as P5
from causal_subspaces.data.logic import FORMS, MARKERS, build_syllogism
from causal_subspaces.heads import attention as A
from causal_subspaces.heads import positions as POS
from causal_subspaces.models.loading import format_prompt
from causal_subspaces.upstream.data import build_trials, position_sets
from causal_subspaces.utils.config import ROOT


def augment(tok, trials):
    """Add KV = Fact|Concl|G1, the token-identity partition X_same / X_diff for X in {Concl, Fact, G1, Rule, KV}, Other (= before the first token difference),
    and tail0..tail8 (single tail rows). Shared `sets` dicts (both directions of a pair) are augmented once."""
    for t in trials:
        s = t["sets"]
        if "KV" in s:
            continue
        a = tok(format_prompt(tok, t["tgt"], False), add_special_tokens=False).input_ids; b = tok(format_prompt(tok, t["src"], False), add_special_tokens=False).input_ids
        same = np.array([x == y for x, y in zip(a, b)])
        s["KV"] = s["Fact"] | s["Concl"] | s["G1"]; s["Other"] = s["PreDiff"]; s["tokens_same"] = same
        for g in ("Concl", "Fact", "G1", "Rule", "KV"):
            s[g + "_same"] = s[g] & same; s[g + "_diff"] = s[g] & ~same
        g2 = np.where(s["G2"])[0]
        for k in range(9):
            m = np.zeros(len(same), bool)
            if k < len(g2): m[g2[k]] = True
            s[f"tail{k}"] = m
    return trials


def cond_mask6(t, cond: str, trial_idx: int = 0) -> np.ndarray:
    """Union of named sets with '+'; 'rk6:<d>': clean random key group, count-matched to |Concl|, drawn from the variable region outside the summary
    positions AND outside Fact, Concl and G1 (so it contains no informative group)."""
    s = t["sets"]
    if cond.startswith("rk6:"):
        pool = np.where(s["NonSummary"] & ~s["KV"])[0]
        pos, _ = POS.random_positions(pool, int(s["Concl"].sum()), "phase6|rk6", trial_idx, int(cond.split(":")[1]))
        m = np.zeros(len(s["All"]), bool); m[pos] = True; return m
    return np.logical_or.reduce([s[x] for x in cond.split("+")])


SAME_LABEL_FORM_PAIRS = [("MP", "MT"), ("AC", "DA")]


def build_fm_trials(tok, contents, content_ids, families, fmt="num_orig", tag="phase6Fm"):
    """Same-label, different-form pairs (MP<->MT valid, AC<->DA invalid; same content, family, marker, negation); both directions; alignment-validated.
    Validity is a function of form in this benchmark: these controls change form and polarity words but not the verdict."""
    f = P5.FORMATS[fmt]; ta, tb = P5.single_token_id(tok, f["valid"]), P5.single_token_id(tok, f["invalid"]); out = []; cen = dict(candidates=0, rejected=0)
    k = 0
    for cid in content_ids:
        for fam in families:
            for neg in (False, True):
                for pt, (fa, fb) in enumerate(SAME_LABEL_FORM_PAIRS):
                    mk = P5._pair_marker(cid, fam, neg, pt)
                    sa, sb = (build_syllogism(contents[cid], fam, ff, neg, mk) for ff in (fa, fb)); cen["candidates"] += 1
                    pa, pb = P5.build_prompt(fmt, sa), P5.build_prompt(fmt, sb)
                    ok, _ = A.pair_alignment(tok, format_prompt(tok, pa), sa.split("\n"), format_prompt(tok, pb), sb.split("\n"))
                    fa_, fb_ = format_prompt(tok, pa, False), format_prompt(tok, pb, False); ia, ib = tok(fa_, add_special_tokens=False).input_ids, tok(fb_, add_special_tokens=False).input_ids
                    if not ok or len(ia) != len(ib): cen["rejected"] += 1; continue
                    ca = POS.categorize(tok, fa_, pa, POS.spans_logic(pa, sa.split("\n"), fam)); cb = POS.categorize(tok, fb_, pb, POS.spans_logic(pb, sb.split("\n"), fam))
                    fd = next((j for j, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
                    if fd is None or ca != cb or POS.check_groups(POS.group_masks(ca)): cen["rejected"] += 1; continue
                    sets = position_sets(ca, fd, tag, k)
                    for tgt, src in ((pa, pb), (pb, pa)):
                        out.append(dict(pair=k, content=cid, family=fam, fmt=fmt, tgt=tgt, src=src, sigma=1.0, ta=ta, tb=tb, sets=sets, fd=fd, n_tok=len(ca), form_pair=f"{fa}/{fb}"))
                    k += 1
    cen["trials"] = len(out); cen["pairs"] = k
    return augment(tok, out), cen


def set_c_trials(tok, fmt):
    """Set C trials rebuilt from the frozen manifest; aborts if the rebuilt prompts differ from the committed hash."""
    from causal_subspaces.upstream.pipeline import FAMS
    man = json.load(open(ROOT / "results/phase6/set_c.json"))
    trials, census = build_trials(tok, man["contents"], list(range(48)), FAMS, fmt, max_pairs=200, seed=1, tag="phase6C")
    h = hashlib.sha256("\n".join(f"{t['tgt']}\x00{t['src']}" for t in trials).encode()).hexdigest()
    if h != man["formats"][fmt]["prompts_sha256"]:
        raise SystemExit(f"set C manifest mismatch for {fmt}: refusing to evaluate")
    return augment(tok, trials), census
