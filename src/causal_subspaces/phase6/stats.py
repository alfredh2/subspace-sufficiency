"""Curve statistics for Phase 6 (docs/phase6_plan.md section 11): signed transfer curves with pair/content replicates, F and R normalisations, onsets,
closing index, trajectory classes, RMS ratios."""
import numpy as np

from causal_subspaces.upstream import stats as S

DENOM_MIN = 0.08      # F(i,P) defined only if both lower bounds of |T*_V(19,P)| >= this (transfer units)
REF_MIN = 0.10        # R defined only if both lower bounds of T_ref >= this
CLOSE_THR = 0.10


def pct(a, q):
    return np.percentile(a, q, axis=0)


def tcurve(boot, sigma, bt, bs, jobs, names, key="margin"):
    """Signed transfer T(name) for each job name: point (nI,), rp/rc replicate arrays (nboot, nI), loco (nContent, nI)."""
    pt, rp, rc, lo = [], [], [], []
    for n in names:
        num, den = S.transfer_terms(sigma, jobs[n][key], bt[key], bs[key]); r = boot.ratio(num, den)
        pt.append(r["point"]); rp.append(r["rep_pair"]); rc.append(r["rep_content"]); lo.append(r["loco"])
    return dict(point=np.array(pt), rp=np.stack(rp, 1), rc=np.stack(rc, 1), loco=np.stack(lo, 1))


def scalar_curve(boot, sigma, jobs, names, key, base=None):
    """Mean sigma-signed change (key - base) per job (raw, unnormalised, e.g. logit change or readout change) with bootstrap replicates."""
    pt, rp, rc, lo = [], [], [], []
    for n in names:
        v = np.asarray(sigma) * (jobs[n][key] - (0 if base is None else base))
        r = boot.ratio(v, np.ones_like(v)); pt.append(r["point"]); rp.append(r["rep_pair"]); rc.append(r["rep_content"]); lo.append(r["loco"])
    return dict(point=np.array(pt), rp=np.stack(rp, 1), rc=np.stack(rc, 1), loco=np.stack(lo, 1))


def divide(cur, ref_cur, ref_index):
    """cur / ref where ref is column `ref_index` of ref_cur (replicate-wise)."""
    d = lambda a, b: a / b[:, [ref_index]]
    return dict(point=cur["point"] / ref_cur["point"][ref_index], rp=d(cur["rp"], ref_cur["rp"]), rc=d(cur["rc"], ref_cur["rc"]), loco=d(cur["loco"], ref_cur["loco"]))


def lower(c):
    return np.minimum(pct(c["rp"], 2.5), pct(c["rc"], 2.5))


def upper(c):
    return np.maximum(pct(c["rp"], 97.5), pct(c["rc"], 97.5))


def defined_denominator(ref_cur, ref_index, thr):
    lo = np.minimum(pct(np.abs(ref_cur["rp"][:, [ref_index]]), 2.5), pct(np.abs(ref_cur["rc"][:, [ref_index]]), 2.5))[0]
    return bool(lo >= thr), float(lo)


def onsets(F, grid, thetas=(0.1, 0.5, 0.9)):
    """first crossing (point) and sustained onset (both lower bounds >= theta for every later grid index, up to and including the last) per theta."""
    lo = lower(F); out = {}
    for th in thetas:
        first = next((grid[k] for k in range(len(grid)) if F["point"][k] >= th), None)
        sus = next((grid[k] for k in range(len(grid)) if (lo[k:] >= th).all()), None)
        out[th] = dict(first=first, sustained=sus)
    return out


def trajectory_class(F, grid, min_drop=0.15, drop_lb=0.05, rev_ub=-0.05, rise=0.5):
    """sign-reversing / non-monotone / monotone rise / flat (explicit rules of amendment 6). Differences are evaluated replicate-wise."""
    n = len(grid)
    up = upper(F)
    if any(up[k] < rev_ub for k in range(n - 1)):
        return "sign-reversing"
    for i in range(n):
        for j in range(i + 1, n):
            dp = F["rp"][:, i] - F["rp"][:, j]; dc = F["rc"][:, i] - F["rc"][:, j]
            if F["point"][i] - F["point"][j] >= min_drop and min(np.percentile(dp, 2.5), np.percentile(dc, 2.5)) >= drop_lb:
                return "non-monotone"
    return "monotone rise" if F["point"][-1] - F["point"][0] >= rise else "flat"


def closing_index(R, grid, thr=CLOSE_THR):
    """smallest grid index such that the upper bounds (pair and content) of R are <= thr for every grid index at or above it; None if not even the last
    index qualifies; 0-index result means 'no influence at any index'."""
    up = upper(R)
    for k in range(len(grid)):
        if (up[k:] <= thr).all():
            return grid[k]
    return None


def rms_ratio(boot_c, d_c, boot_m, d_m):
    """RMS(control)/RMS(matched) of per-trial changes with independent bootstrap replicates for the two data sets."""
    rc = boot_c.ratio(np.asarray(d_c) ** 2, np.ones(len(d_c))); rm = boot_m.ratio(np.asarray(d_m) ** 2, np.ones(len(d_m)))
    pt = float(np.sqrt(rc["point"] / rm["point"]))
    reps = lambda a, b: np.sqrt(np.maximum(a, 0) / np.maximum(b, 1e-12))
    return dict(point=pt, pair=(float(np.percentile(reps(rc["rep_pair"], rm["rep_pair"]), 2.5)), float(np.percentile(reps(rc["rep_pair"], rm["rep_pair"]), 97.5))),
                content=(float(np.percentile(reps(rc["rep_content"], rm["rep_content"]), 2.5)), float(np.percentile(reps(rc["rep_content"], rm["rep_content"]), 97.5))), rms_ctrl=float(np.sqrt(rc["point"])), rms_m=float(np.sqrt(rm["point"])))
