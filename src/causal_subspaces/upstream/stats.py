"""Transfer ratios with pair-level and content-clustered bootstrap, leave-one-content-out ranges, onset detection (docs/phase4_plan.md sections 1-2)."""
import numpy as np


class Boot:
    """pair_idx / content_idx: integer cluster ids per trial (both directions of a pair share a pair id). All statistics that share the denominator
    (the source-target gap) use the SAME resamples, so derived quantities (D_S8, interactions) can be formed from replicate arrays."""

    def __init__(self, pair_idx, content_idx, nboot=2000, seed=0):
        self.p = np.unique(pair_idx, return_inverse=True)[1]; self.c = np.unique(content_idx, return_inverse=True)[1]
        self.P, self.C, self.n = self.p.max() + 1, self.c.max() + 1, len(self.p)
        rg = np.random.default_rng(seed)
        self.Wp = rg.multinomial(self.P, np.full(self.P, 1 / self.P), size=nboot).astype(float)
        self.Wc = rg.multinomial(self.C, np.full(self.C, 1 / self.C), size=nboot).astype(float)

    def sums(self, v):
        v = np.asarray(v, float)
        return np.bincount(self.p, v, self.P), np.bincount(self.c, v, self.C)

    def ratio(self, num, den):
        """Aggregated ratio sum(num)/sum(den): point, bootstrap replicates (pair, content) and the leave-one-content-out values."""
        Np, Nc = self.sums(num); Dp, Dc = self.sums(den)
        return dict(point=float(Np.sum() / Dp.sum()), rep_pair=(self.Wp @ Np) / (self.Wp @ Dp), rep_content=(self.Wc @ Nc) / (self.Wc @ Dc),
                    loco=(Nc.sum() - Nc) / (Dc.sum() - Dc))

    @staticmethod
    def ci(rep, q=(2.5, 97.5)):
        return [float(np.percentile(rep, q[0])), float(np.percentile(rep, q[1]))]


def transfer_terms(sigma, y_patch, y_tgt, y_src):
    """Per-trial numerator and denominator of the aggregated transfer T = sum(sigma(Y_patch-Y_tgt)) / sum(sigma(Y_src-Y_tgt))."""
    sigma = np.asarray(sigma, float)
    return sigma * (np.asarray(y_patch) - np.asarray(y_tgt)), sigma * (np.asarray(y_src) - np.asarray(y_tgt))


def onsets(point, lo_pair, lo_content, plateau_idx=(19, 20, 21, 22), frac=0.5, last=22):
    """Curves over hook indices 0..23 (arrays). threshold = frac * median(point[plateau_idx]).
    first crossing: smallest i<=last with point >= threshold.  sustained (per interval type): smallest i<=last such that the lower bound >= threshold
    for ALL i' in [i, last].  sustained_both: both lower bounds. None = no such onset (never forced)."""
    point, lo_pair, lo_content = (np.asarray(a, float) for a in (point, lo_pair, lo_content))
    plat = float(np.median(point[list(plateau_idx)])); thr = frac * plat

    def sustained(lo):
        ok = lo[:last + 1] >= thr
        for i in range(last + 1):
            if ok[i:].all():
                return i
        return None
    first = next((i for i in range(last + 1) if point[i] >= thr), None)
    sp, sc = sustained(lo_pair), sustained(lo_content)
    both = None if sp is None or sc is None else max(sp, sc)
    return dict(plateau=plat, threshold=thr, first_crossing=first, sustained_pair=sp, sustained_content=sc, sustained_both=both)
