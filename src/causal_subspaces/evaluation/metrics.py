"""Aggregate metrics, binomial confidence intervals, chance-normalised recovery."""
import math

import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(df: pd.DataFrame, by: list[str] | None = None) -> dict:
    """df needs boolean columns correct, restricted_correct, invalid. Returns overall + per-group stats."""
    def stats(g: pd.DataFrame) -> dict:
        n, k = len(g), int(g.correct.sum())
        lo, hi = wilson(k, n)
        return dict(n=n, acc=k / n if n else float("nan"), ci95=[lo, hi],
                    acc_restricted=float(g.restricted_correct.mean()) if n else float("nan"),
                    invalid_rate=float(g.invalid.mean()) if n else float("nan"),
                    mass_on_labels=float(g.mass_on_labels.mean()) if n else float("nan"))
    out = {"overall": stats(df)}
    for col in by or []:
        out[f"by_{col}"] = {str(k): stats(g) for k, g in df.groupby(col)}
    out["response_distribution"] = df.first_char.value_counts().to_dict()
    return out


def recovery(acc: float, acc_intact: float, chance: float) -> float:
    """Chance-normalised recovery (paper App. C, Eq. 3): (a - c) / (a0 - c)."""
    return (acc - chance) / (acc_intact - chance) if acc_intact != chance else float("nan")
