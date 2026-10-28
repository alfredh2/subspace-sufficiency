"""Concentration (|Delta| only) and signed reconstruction / cancellation metrics. See docs/plans/phase1_plan.md section 2."""
import numpy as np
import torch


def _p(x: torch.Tensor) -> torch.Tensor:
    a = x.abs().double()
    return a / a.sum()


def topk_abs_share(delta: torch.Tensor, k) -> torch.Tensor:
    """Share of sum|Delta| held by the k largest |Delta| (k may be a list). Works on (..., m)."""
    a = delta.abs().double()
    s, _ = a.sort(-1, descending=True)
    c = s.cumsum(-1) / a.sum(-1, keepdim=True)
    return c[..., (torch.as_tensor(k) - 1).long()] if not isinstance(k, int) else c[..., k - 1]


def n_eff(delta: torch.Tensor) -> torch.Tensor:
    p = _p(delta) if delta.dim() == 1 else delta.abs().double() / delta.abs().double().sum(-1, keepdim=True)
    return 1.0 / (p ** 2).sum(-1)


def exp_entropy(delta: torch.Tensor) -> torch.Tensor:
    p = delta.abs().double() / delta.abs().double().sum(-1, keepdim=True)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum(-1))


def gini(delta: torch.Tensor) -> torch.Tensor:
    a, _ = delta.abs().double().sort(-1)
    m = a.shape[-1]
    idx = torch.arange(1, m + 1, device=a.device, dtype=torch.float64)
    return (2 * (idx * a).sum(-1) / (m * a.sum(-1))) - (m + 1) / m


def concentration_summary(delta: torch.Tensor, fracs=(0.001, 0.01, 0.05, 0.10)) -> dict:
    m = delta.shape[-1]
    out = {f"top{f * 100:g}%": topk_abs_share(delta, max(1, int(round(f * m)))) for f in fracs}
    out.update(n_eff=n_eff(delta), exp_entropy=exp_entropy(delta), gini=gini(delta))
    return out


def signed_reconstruction(delta: torch.Tensor, idx: torch.Tensor) -> dict:
    """Top-set signed partial sum vs net class difference D = sum Delta."""
    D = delta.double().sum()
    part = delta.double()[idx].sum()
    return dict(D=float(D), partial=float(part), frac=float(part / D) if D != 0 else float("nan"), error=float(abs(D - part)))


def cancellation_index(delta: torch.Tensor) -> dict:
    d = delta.double()
    pos, neg = d.clamp_min(0).sum(), (-d).clamp_min(0).sum()
    return dict(kappa=float(d.abs().sum() / (pos - neg).abs()) if pos != neg else float("inf"), pos_mass=float(pos), neg_mass=float(neg))


def per_example_r2(S_train: torch.Tensor, S_test: torch.Tensor, idx: torch.Tensor) -> dict:
    """S_*: (n, m) per-example unit contributions s_ij. Target y_i = sum_j s_ij; prediction = sum_{j in S} s_ij + train-mean of the rest."""
    mask = torch.zeros(S_train.shape[1], dtype=torch.bool, device=S_train.device)
    mask[idx] = True
    c = S_train[:, ~mask].double().sum(1).mean()
    y = S_test.double().sum(1)
    yhat = S_test[:, mask].double().sum(1) + c
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    corr = torch.corrcoef(torch.stack([y, yhat]))[0, 1]
    return dict(r2=float(r2), corr=float(corr))


def per_example_kappa(S: torch.Tensor) -> torch.Tensor:
    return S.abs().double().sum(1) / S.double().sum(1).abs().clamp_min(1e-12)


def empirical_p(native: float, controls) -> float:
    """One-sided empirical p that a control draw >= native: (1 + #{c >= native}) / (n + 1)."""
    c = np.asarray(controls, float)
    return float((1 + (c >= native).sum()) / (len(c) + 1))


def bootstrap_ci(x, stat=np.mean, n=1000, seed=0, alpha=0.05):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    v = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))
