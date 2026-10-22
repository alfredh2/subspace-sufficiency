"""Aligned valid/invalid trials with validated token-span position sets for Phase 4 (docs/phase4_plan.md section 2-3)."""
import numpy as np

from causal_subspaces import phase05 as P5
from causal_subspaces.heads import attention as A
from causal_subspaces.heads import positions as POS
from causal_subspaces.models.loading import format_prompt

SET_NAMES = ["G1", "G2", "G1G2", "Self", "G1G2Self", "Rule", "Fact", "Concl", "NonSummary", "All", "AllMinusG1G2", "AllMinusG1G2Self", "PreDiff"]
N_RAND = 5


def position_sets(cats: list[str], fd: int, tag: str, trial: int) -> dict:
    c = np.array(cats); n = len(c); idx = np.arange(n)
    g1, g2, sf = c == "conclusion_final_token", c == "template_tail", c == "self"
    summ = g1 | g2 | sf
    var = idx >= fd
    s = dict(G1=g1, G2=g2, G1G2=g1 | g2, Self=sf, G1G2Self=g1 | g2 | sf, Rule=c == "rule", Fact=c == "fact", Concl=c == "conclusion",
             NonSummary=var & ~summ, All=np.ones(n, bool), AllMinusG1G2=~(g1 | g2), AllMinusG1G2Self=~summ, PreDiff=~var)
    pool = np.where(s["NonSummary"])[0]; cnt = int(s["G1G2"].sum()); flagged = 0
    for d in range(N_RAND):
        pos, flag = POS.random_positions(pool, cnt, tag, trial, d)
        m = np.zeros(n, bool); m[pos] = True; s[f"rand{d}"] = m; flagged += int(flag)
    s["_rand_flagged"] = flagged
    return s


def build_trials(tok, contents, content_ids, families, fmt: str, *, max_pairs: int | None = None, seed: int = 0, tag: str = "phase4"):
    """Aligned pairs (Phase 2.5 rule) -> two trials per pair (target invalid/source valid and vice versa). Returns (trials, census)."""
    f = P5.FORMATS[fmt]; ta, tb = P5.single_token_id(tok, f["valid"]), P5.single_token_id(tok, f["invalid"])
    pairs = P5.build_pairs(contents, content_ids, families, tok)
    census = dict(matched=len(pairs), alignment_rejected=0, mask_invalid=0, mask_mismatch=0)
    ok_pairs = []
    for p in pairs:
        pv, pi = P5.build_prompt(fmt, p["valid_syl"]), P5.build_prompt(fmt, p["invalid_syl"])
        ok, _ = A.pair_alignment(tok, format_prompt(tok, pv), p["valid_syl"].split("\n"), format_prompt(tok, pi), p["invalid_syl"].split("\n"))
        if not ok:
            census["alignment_rejected"] += 1; continue
        fv, fi = format_prompt(tok, pv, False), format_prompt(tok, pi, False)
        iv, ii = tok(fv, add_special_tokens=False).input_ids, tok(fi, add_special_tokens=False).input_ids
        if len(iv) != len(ii):
            census["alignment_rejected"] += 1; continue
        fd = next((k for k, (a, b) in enumerate(zip(iv, ii)) if a != b), None)
        cv = POS.categorize(tok, fv, pv, POS.spans_logic(pv, p["valid_syl"].split("\n"), p["family"]))
        ci = POS.categorize(tok, fi, pi, POS.spans_logic(pi, p["invalid_syl"].split("\n"), p["family"]))
        if fd is None or POS.check_groups(POS.group_masks(cv)) or POS.check_groups(POS.group_masks(ci)):
            census["mask_invalid"] += 1; continue
        if cv != ci:
            census["mask_mismatch"] += 1; continue
        ok_pairs.append((p, pv, pi, cv, fd))
    if max_pairs is not None and len(ok_pairs) > max_pairs:      # deterministic family-stratified sample
        rg = np.random.default_rng(seed); by = {}
        for k, (p, *_r) in enumerate(ok_pairs): by.setdefault(p["family"], []).append(k)
        keep = []; fams = sorted(by); per = max_pairs // len(fams) + 1
        for fm in fams:
            ks = by[fm]; keep += list(rg.permutation(ks)[:per])
        ok_pairs = [ok_pairs[k] for k in sorted(keep)[:max_pairs]]
    trials = []
    for k, (p, pv, pi, cats, fd) in enumerate(ok_pairs):
        sets_v = position_sets(cats, fd, f"{tag}|{fmt}", k)
        for tgt, src, src_valid in ((pi, pv, True), (pv, pi, False)):
            trials.append(dict(pair=k, content=p["content_id"], family=p["family"], fmt=fmt, tgt=tgt, src=src, tgt_aff=not src_valid, src_aff=src_valid, sigma=1.0 if src_valid else -1.0,
                               ta=ta, tb=tb, sets=sets_v, fd=fd, n_tok=len(cats)))
    census["pairs_kept"] = len(ok_pairs); census["trials"] = len(trials)
    census["rand_flagged"] = sum(t["sets"]["_rand_flagged"] for t in trials)
    return trials, census
