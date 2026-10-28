"""Unit populations for restoration / transplant (docs/plans/phase3_plan.md section 4). All draws are deterministic (crc32 seeds) and exclude the native set.
Pre-intervention-defined: random, activation-matched, alignment-matched, complement, full layer, dose-response (top-k in Phase-1 rank).
Response-selected (diagnostic only): delta-matched, oracle."""
import zlib

import numpy as np


def rng_for(tag: str, k: int, d: int):
    return np.random.default_rng(zlib.crc32(f"{tag}|{k}|{d}".encode()))


def random_set(m: int, native: np.ndarray, draw: int) -> np.ndarray:
    pool = np.setdiff1d(np.arange(m), native)
    return np.sort(rng_for("random", len(native), draw).choice(pool, len(native), replace=False))


def nearest_set(feat: np.ndarray, native: np.ndarray, tag: str, draw: int, n_cand: int = 5) -> np.ndarray:
    """For each native unit pick, without replacement, one of its n_cand nearest non-native units in feature space (feat: (m, f), standardised)."""
    rng = rng_for(tag, len(native), draw)
    used = np.zeros(feat.shape[0], bool); used[native] = True
    out = []
    for j in rng.permutation(native):
        d = ((feat - feat[j]) ** 2).sum(1); d[used] = np.inf
        c = int(rng.choice(np.argsort(d)[:n_cand])); used[c] = True; out.append(c)
    return np.sort(np.array(out))


def complement(m: int, native: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.arange(m), native)


def oracle_set(response: np.ndarray, native: np.ndarray, k: int) -> np.ndarray:
    """Top-k units by |response| (e.g. induced |Delta s|) over ALL units of the layer (diagnostic; native units are not excluded)."""
    return np.sort(np.argsort(-np.abs(response))[:k])


def delta_matched_set(mean_abs_delta_a: np.ndarray, native: np.ndarray, draw: int = 0, n_cand: int = 5) -> np.ndarray:
    f = np.log(mean_abs_delta_a + 1e-9)[:, None]; f = (f - f.mean()) / f.std()
    return nearest_set(f, native, "delta", draw, n_cand)


def matching_quality(feats: dict, native: np.ndarray, control: np.ndarray) -> dict:
    """Standardised mean difference per feature between native and control units."""
    out = {}
    for k, f in feats.items():
        f = np.asarray(f, float); sd = f.std() + 1e-12
        out[k] = float((f[control].mean() - f[native].mean()) / sd)
    return out
