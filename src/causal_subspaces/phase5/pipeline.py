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


def null_sets(k, ref_heads, mags, n_random=20, n_matched=20, tol=0.10, H=32):
    """Deterministic (crc32) head sets of size k disjoint from ref_heads: random, and output-magnitude-matched (sum of m_h within tol of the reference)."""
    pool = [h for h in range(H) if h not in set(ref_heads)]; target = float(np.sum(mags[list(ref_heads)]))
    rnd = []
    for d in range(n_random):
        rg = np.random.default_rng(zlib.crc32(f"phase5|rand|{k}|{d}".encode())); rnd.append(sorted(int(x) for x in rg.choice(pool, k, replace=False)))
    mat, tries, d = [], 0, 0
    while len(mat) < n_matched and tries < 200000:
        rg = np.random.default_rng(zlib.crc32(f"phase5|mm|{k}|{tries}".encode())); c = sorted(int(x) for x in rg.choice(pool, k, replace=False)); tries += 1
        if abs(float(np.sum(mags[c])) - target) <= tol * target and c not in mat: mat.append(c)
    return rnd, mat


def tstat(boot, tr, bt, bs, jobs, name, key):
    """Aggregated transfer of outcome `key` in {'margin','c','c20','w20'} for job `name`."""
    num, den = S.transfer_terms(tr["sigma"], jobs[name][key], bt[key], bs[key])
    return boot.ratio(num, den)


def interp_g(pi_grid, t_grid):
    pg, tg = np.asarray(pi_grid, float), np.asarray(t_grid, float); o = np.argsort(pg)
    return lambda pi: np.interp(pi, pg[o], tg[o])          # constant extrapolation outside the validation range
