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
    s = dict(G1=g1, G2=g2, G1G2=g1 | g2, Self=sf, G1G2Self=g1 | g2 | sf, Rule=c == "rule", Fact=c == "fact", Concl=c == "conclusion", Instr=c == "instruction",
             NonSummary=var & ~summ, All=np.ones(n, bool), AllMinusG1G2=~(g1 | g2), AllMinusG1G2Self=~summ, PreDiff=~var)
    pool = np.where(s["NonSummary"])[0]; cnt = int(s["G1G2"].sum()); flagged = 0; flags = []
    for d in range(N_RAND):
        pos, flag = POS.random_positions(pool, cnt, tag, trial, d)
        m = np.zeros(n, bool); m[pos] = True; s[f"rand{d}"] = m; flagged += int(flag); flags.append(int(flag))
    s["_rand_flagged"] = flagged; s["_rand_flags"] = flags
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


def build_special_trials(tok, contents, content_ids, families, kind: str, *, tag: str):
    """kind='lexical': target and source carry the SAME label but different contents (equal token length, alignment-validated);
    kind='format': target = num_orig prompt, source = the SAME syllogism in the num_rev prompt (label mapping differs). Exploratory/diagnostic controls."""
    f = P5.FORMATS["num_orig"]; ta, tb = P5.single_token_id(tok, f["valid"]), P5.single_token_id(tok, f["invalid"])
    pairs = P5.build_pairs(contents, content_ids, families, tok); out = []; census = dict(candidates=0, rejected=0)

    def mk(syl_t, fmt_t, syl_s, fmt_s, pid, p):
        pt, ps = P5.build_prompt(fmt_t, syl_t), P5.build_prompt(fmt_s, syl_s)
        ft, fs = format_prompt(tok, pt, False), format_prompt(tok, ps, False)
        it, is_ = tok(ft, add_special_tokens=False).input_ids, tok(fs, add_special_tokens=False).input_ids
        census["candidates"] += 1
        if len(it) != len(is_): census["rejected"] += 1; return
        ct = POS.categorize(tok, ft, pt, POS.spans_logic(pt, syl_t.split("\n"), p["family"])); cs = POS.categorize(tok, fs, ps, POS.spans_logic(ps, syl_s.split("\n"), p["family"]))
        fd = next((k for k, (a, b) in enumerate(zip(it, is_)) if a != b), None)
        if fd is None or ct != cs or POS.check_groups(POS.group_masks(ct)): census["rejected"] += 1; return
        out.append(dict(pair=pid, content=p["content_id"], family=p["family"], fmt="num_orig", tgt=pt if fmt_t == "num_orig" else pt, src=ps, sigma=1.0, ta=ta, tb=tb,
                        sets=position_sets(ct, fd, tag, len(out)), fd=fd, n_tok=len(ct), fmt_src=fmt_s))
    if kind == "format":
        for k, p in enumerate(pairs):
            for syl in (p["valid_syl"], p["invalid_syl"]): mk(syl, "num_orig", syl, "num_rev", k, p)
    else:
        lc = P5.lexical_controls(contents, pairs, tok, "num_orig", seed=0)
        for k, d in enumerate(lc):
            for a, b in ((d["base"]["valid_syl"], d["other"]["valid_syl"]), (d["base"]["invalid_syl"], d["other"]["invalid_syl"])):
                pt, ps = P5.build_prompt("num_orig", a), P5.build_prompt("num_orig", b)
                ok, _ = A.pair_alignment(tok, format_prompt(tok, pt), a.split("\n"), format_prompt(tok, ps), b.split("\n"))
                census["candidates"] += 1
                if not ok: census["rejected"] += 1; continue
                mk(a, "num_orig", b, "num_orig", k, d["base"])
    census["trials"] = len(out)
    return out, census
