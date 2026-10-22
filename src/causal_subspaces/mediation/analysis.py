"""Aggregated interventional mediation quantities (docs/phase3_plan.md sections 2 and 7). Ratios are formed from AGGREGATED signed effects with paired
bootstrap CIs, never averaged per example, and are 'undefined' below the validity floors."""
import numpy as np
from scipy import stats

FLOORS = dict(coordinate=2.0, margin=5.0, gap=5.0)


def shifts(y00, y11, y10, y01, sgn=None):
    """Signed shifts relative to the clean cell (sgn = +-1 per trial for patching; None = +1)."""
    s = np.ones_like(y00) if sgn is None else sgn
    return s * (y11 - y00), s * (y10 - y00), s * (y01 - y00)


def ratios_from_means(te, d10, d01):
    """TE = mean(y11-y00); d10 = mean(y10-y00); d01 = mean(y01-y00). R = (TE-d10)/TE, B = d10/TE, S = d01/TE, J = R-S, Phi = (R+S)/2."""
    R = (te - d10) / te; B = d10 / te; S = d01 / te
    return dict(R=R, B=B, S=S, J=R - S, Phi=0.5 * (R + S))


def mediation_effects(y00, y11, y10, y01, sgn=None, floor=2.0, noise_floor=0.0, n_boot=500, seed=0, rows=None):
    """Aggregated effects + paired-bootstrap CIs. y*: arrays over trials. Returns dict with point estimates, CIs and an `undefined` flag/reason."""
    t, d10, d01 = shifts(y00, y11, y10, y01, sgn)
    if rows is not None:
        t, d10, d01 = t[rows], d10[rows], d01[rows]
    te, m10, m01 = float(t.mean()), float(d10.mean()), float(d01.mean())
    out = dict(TE=te, effect_fixed=m10, effect_transplant=m01, n=len(t))
    if abs(te) < floor or abs(te) < 3 * noise_floor:
        out.update(undefined=True, reason=f"|TE|={abs(te):.3f} below floor {max(floor, 3 * noise_floor):.3f}")
        return out
    out.update(undefined=False, **ratios_from_means(te, m10, m01))
    rg = np.random.default_rng(seed)
    n = len(t)
    boots, discarded = [], 0
    for _ in range(n_boot):
        r = rg.integers(0, n, n)
        tb = float(t[r].mean())
        if abs(tb) < 0.5 * floor:
            discarded += 1
            continue
        boots.append(ratios_from_means(tb, float(d10[r].mean()), float(d01[r].mean())))
    out["bootstrap_discarded"] = discarded
    if discarded > 0.05 * n_boot or len(boots) < 50:
        out.update(undefined=True, reason=f"{discarded}/{n_boot} bootstrap replicates have an unstable TE")
        return out
    for k in ("R", "B", "S", "J", "Phi"):
        v = np.array([b[k] for b in boots])
        out[k + "_ci"] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    return out


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k]); m = len(keys); adj, prev = {}, 0.0
    for i, k in enumerate(keys):
        prev = max(prev, min(1.0, pvals[k] * (m - i))); adj[k] = prev
    return adj


def compare_to_controls(native: float, controls) -> dict:
    """Empirical rank (p_min = 1/(n+1), not significance), z-score with Shapiro check, one-sided normal p."""
    c = np.asarray(controls, float)
    z = (native - c.mean()) / c.std(ddof=1) if len(c) > 2 and c.std(ddof=1) > 0 else float("nan")
    return dict(n=len(c), control_mean=float(c.mean()), control_max=float(c.max()), above_all=bool(native > c.max()), empirical_p=float((1 + (c >= native).sum()) / (len(c) + 1)),
                z=float(z), p_normal=float(stats.norm.sf(z)) if np.isfinite(z) else float("nan"), shapiro_p=float(stats.shapiro(c).pvalue) if len(c) >= 8 else float("nan"))


def gap_effects(y00, y11, y10, y01, accept, floor=5.0, noise_floor=0.0, n_boot=500, seed=0):
    """Class-gap version of the mediation quantities for ablation experiments: the 'effect' of a cell is the reduction of the class gap
    gap(y) = mean(y|accept) - mean(y|reject) relative to the clean cell; ratios from aggregated gaps, bootstrap over examples (stratified by class)."""
    acc = np.asarray(accept, bool)

    def eff(rows):
        g = lambda y: float(y[rows][acc[rows]].mean() - y[rows][~acc[rows]].mean())
        g00 = g(y00)
        return g00 - g(y11), g00 - g(y10), g00 - g(y01)
    allr = np.arange(len(y00)); te, e10, e01 = eff(allr)
    out = dict(TE=te, effect_fixed=e10, effect_transplant=e01, n=len(y00))
    if abs(te) < floor or abs(te) < 3 * noise_floor:
        out.update(undefined=True, reason=f"|TE|={abs(te):.3f} below floor {max(floor, 3 * noise_floor):.3f}")
        return out
    out.update(undefined=False, **ratios_from_means(te, e10, e01))
    rg = np.random.default_rng(seed); pos, neg = np.where(acc)[0], np.where(~acc)[0]; boots, disc = [], 0
    for _ in range(n_boot):
        rows = np.concatenate([rg.choice(pos, len(pos)), rg.choice(neg, len(neg))])
        tb, b10, b01 = eff(rows)
        if abs(tb) < 0.5 * floor: disc += 1; continue
        boots.append(ratios_from_means(tb, b10, b01))
    out["bootstrap_discarded"] = disc
    if disc > 0.05 * n_boot or len(boots) < 50:
        out.update(undefined=True, reason=f"{disc}/{n_boot} bootstrap replicates have an unstable TE"); return out
    for k in ("R", "B", "S", "J", "Phi"):
        v = np.array([b[k] for b in boots]); out[k + "_ci"] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    return out
