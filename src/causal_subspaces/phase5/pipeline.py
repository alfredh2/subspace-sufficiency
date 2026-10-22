"""Phase-5 setup, descriptive head statistics, null-set construction, scalar-model utilities (docs/phase5_plan.md)."""
import json
import zlib

import numpy as np
import torch

from causal_subspaces.heads.ablation import capture_z
from causal_subspaces.heads.decompose import head_dims
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.phase5.runner import LAYER, ROWS, Ctx5, _pad
from causal_subspaces.upstream import pipeline as UP
from causal_subspaces.upstream import stats as S
from causal_subspaces.utils.config import ROOT

OUT = ROOT / "results/phase5"
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
KEY_SINGLE = ["Rule", "Fact", "Concl", "G1", "G2", "Other"]
KEY_SETS = ["Fact+Concl", "Fact+G1", "Concl+G1", "Fact+Concl+G1", "Concl+G1+G2", "Fact+Concl+G1+G2", "Rule+Fact+Concl+G1+G2"]
KEY_ALL = "Rule+Fact+Concl+G1+G2"


def setup():
    P = UP.setup(); c = P["ctx"]
    P["ctx5"] = Ctx5(c.model, c.tok, c.d_c, c.dev)
    return P


def head_stats(ctx5, trials, bs=32):
    """Descriptive attribution on layer-19 rows G1G2 for each head (aggregated toward the source): s_h = gain_h . dz_h (signed d_c write),
    m_h = ||W_{O,h} dz_h|| (output magnitude), both averaged over the rows and trials; also mean over trials of the signed s_h."""
    model, tok, dev = ctx5.model, ctx5.tok, ctx5.dev; H, HD = ctx5.H, ctx5.HD
    W = model.model.layers[LAYER].self_attn.o_proj.weight.float()
    S_, M_ = [], []
    with torch.no_grad():
        for s0 in range(0, len(trials), bs):
            b = trials[s0:s0 + bs]
            ids_t, m_t, p_t = encode(tok, [format_prompt(tok, t["tgt"], False) for t in b], dev); ids_s, m_s, p_s = encode(tok, [format_prompt(tok, t["src"], False) for t in b], dev)
            zt, zs = {}, {}
            with capture_z(model, [LAYER], zt): last_logits(model, ids_t, m_t, p_t)
            with capture_z(model, [LAYER], zs): last_logits(model, ids_s, m_s, p_s)
            B, T = ids_t.shape; rows = _pad(b, ROWS, T, dev, s0).float(); n = rows.sum(1).clamp_min(1)
            dz = (zs[LAYER].float() - zt[LAYER].float()).view(B, T, H, HD)
            sgn = torch.tensor([t["sigma"] for t in b], device=dev)
            sh = ((dz * ctx5.gain[None, None]).sum(-1) * rows[..., None]).sum(1) / n[:, None]                 # (B,H)
            mh = torch.stack([(dz[:, :, h] @ W[:, h * HD:(h + 1) * HD].T).norm(dim=-1) for h in range(H)], 2)  # (B,T,H)
            mh = (mh * rows[..., None]).sum(1) / n[:, None]
            S_.append((sh * sgn[:, None]).cpu().numpy()); M_.append(mh.cpu().numpy())
    return np.concatenate(S_), np.concatenate(M_)


def null_sets(k, ref_heads, mags, n_random=20, n_matched=20, H=32, nn=6):
    """Deterministic (crc32) head sets of size k disjoint from ref_heads. random: uniform. magnitude-matched: per-head strata - each reference head (in
    decreasing output magnitude) is paired with a distinct non-reference head drawn at random from its `nn` nearest in output magnitude m_h. When the reference
    heads are the largest-magnitude heads (as here) the summed magnitude of any disjoint set is necessarily lower; `scale` = sum(m_ref)/sum(m_set) is the
    factor by which the matched set's patch is scaled in the magnitude-equalised variant."""
    ref = sorted(ref_heads, key=lambda h: -mags[h]); pool0 = [h for h in range(H) if h not in set(ref_heads)]
    rnd = []
    for d in range(n_random):
        rg = np.random.default_rng(zlib.crc32(f"phase5|rand|{k}|{d}".encode())); rnd.append(sorted(int(x) for x in rg.choice(pool0, k, replace=False)))
    mat, scales, seen = [], [], set()
    for t in range(1000):
        if len(mat) >= n_matched: break
        rg = np.random.default_rng(zlib.crc32(f"phase5|mm|{k}|{t}".encode())); chosen, pool = [], list(pool0)
        for hr in ref:
            near = sorted(pool, key=lambda h: abs(mags[h] - mags[hr]))[:nn]; c = int(rg.choice(near)); chosen.append(c); pool.remove(c)
        key = tuple(sorted(chosen))
        if key in seen: continue
        seen.add(key); mat.append(sorted(chosen)); scales.append(float(np.sum(mags[list(ref_heads)]) / np.sum(mags[chosen])))
    return rnd, mat, scales


def tstat(boot, tr, bt, bs, jobs, name, key):
    """Aggregated transfer of outcome `key` in {'margin','c','c20','w20'} for job `name`."""
    num, den = S.transfer_terms(tr["sigma"], jobs[name][key], bt[key], bs[key])
    return boot.ratio(num, den)


def interp_g(pi_grid, t_grid):
    pg, tg = np.asarray(pi_grid, float), np.asarray(t_grid, float); o = np.argsort(pg)
    return lambda pi: np.interp(pi, pg[o], tg[o])          # constant extrapolation outside the validation range


def build_cross_trials(tok, contents, content_ids, families, seed=0):
    """Exploratory (added after the Replication-B freeze): target and source have OPPOSITE labels but DIFFERENT contents (same family/form/negation, equal token
    length, alignment-validated). Transfer then cannot be carried by the lexical identity of the matched sibling."""
    from causal_subspaces import phase05 as P5
    from causal_subspaces.heads import attention as A
    from causal_subspaces.heads import positions as POS
    from causal_subspaces.upstream.data import position_sets
    f = P5.FORMATS["num_orig"]; ta, tb = P5.single_token_id(tok, f["valid"]), P5.single_token_id(tok, f["invalid"])
    pairs = P5.build_pairs(contents, content_ids, families, tok); by = {}
    for p in pairs: by.setdefault((p["family"], p["ptype"], p["neg"]), []).append(p)
    rg = np.random.default_rng(seed); out = []; cen = dict(candidates=0, rejected=0)
    for k, p in enumerate(pairs):
        cands = [q for q in by[(p["family"], p["ptype"], p["neg"])] if q["content_id"] != p["content_id"] and q["n_tokens"] == p["n_tokens"]]
        if not cands: continue
        q = cands[int(rg.integers(len(cands)))]
        for tsyl, ssyl, sv in ((p["invalid_syl"], q["valid_syl"], 1.0), (p["valid_syl"], q["invalid_syl"], -1.0)):
            pt, ps = P5.build_prompt("num_orig", tsyl), P5.build_prompt("num_orig", ssyl); cen["candidates"] += 1
            ok, _ = A.pair_alignment(tok, format_prompt(tok, pt), tsyl.split("\n"), format_prompt(tok, ps), ssyl.split("\n"))
            ft, fs = format_prompt(tok, pt, False), format_prompt(tok, ps, False); it, is_ = tok(ft, add_special_tokens=False).input_ids, tok(fs, add_special_tokens=False).input_ids
            if not ok or len(it) != len(is_): cen["rejected"] += 1; continue
            ct = POS.categorize(tok, ft, pt, POS.spans_logic(pt, tsyl.split("\n"), p["family"])); cs = POS.categorize(tok, fs, ps, POS.spans_logic(ps, ssyl.split("\n"), q["family"]))
            fd = next((j for j, (a, b) in enumerate(zip(it, is_)) if a != b), None)
            if fd is None or ct != cs or POS.check_groups(POS.group_masks(ct)): cen["rejected"] += 1; continue
            out.append(dict(pair=k, content=p["content_id"], family=p["family"], fmt="num_orig", tgt=pt, src=ps, sigma=sv, ta=ta, tb=tb, sets=position_sets(ct, fd, "phase5cross", len(out)), fd=fd, n_tok=len(ct)))
    cen["trials"] = len(out)
    return out, cen
