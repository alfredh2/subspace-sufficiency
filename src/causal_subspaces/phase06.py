"""Phase 0.6 helpers: residual collection, linear probes / INLP, subspaces orthogonal to the verdict direction,
and the logic-selective search objective (pre-registered in docs/phase06_plan.md)."""
import numpy as np
import torch
import torch.nn.functional as F

from causal_subspaces import phase05 as P05
from causal_subspaces.models.hooks import capture_residual, residual_intervention
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.mvs.checkpoint import load_checkpoint
from causal_subspaces.mvs.projection import orthonormalize
from causal_subspaces.utils.config import ROOT


# ---- the frozen verdict direction ----------------------------------------------------------------------------
def load_consensus(seeds=(0, 1, 2)) -> torch.Tensor:
    """Unit rank-1 consensus verdict direction from the three Phase-0 checkpoints (identical construction to Phase 0.5)."""
    Us = {s: load_checkpoint(ROOT / f"results/mvs_rank2_seed{s}/checkpoints/step_1600.pt")["U"] for s in seeds}
    cons = P05.consensus_directions(Us)
    return F.normalize(sum(cons[s]["shared"] for s in seeds), dim=0)


def perp_basis(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (d x k) of span((I - d d^T) W): a candidate subspace guaranteed orthogonal to the unit vector d."""
    d = d.to(W.device, W.dtype)
    Wp = W - d[:, None] * (d @ W)[None]
    return orthonormalize(Wp)


def random_orth_subspace(Qm: torch.Tensor, d: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Random rank-k subspace inside the top-variance space (init_V-style), orthogonal to d."""
    g = torch.Generator().manual_seed(seed)
    m = Qm.shape[1]
    return perp_basis(Qm @ (torch.randn(m, k, generator=g) / m ** 0.5), d.cpu())


def subspace_variance(U: torch.Tensor, Qm: torch.Tensor, eig: torch.Tensor) -> float:
    """Approximate total neutral-corpus variance inside span(U), using the top-m eigenpairs (99.85 % of variance)."""
    c = Qm.double().T @ U.double()  # m x k
    return float((eig.double()[:, None] * c ** 2).sum())


def variance_matched_random(Qm, eig, d, k, target_var, n, seed0=10_000, pool=400):
    cands = [random_orth_subspace(Qm, d, k, seed0 + i) for i in range(pool)]
    v = np.array([subspace_variance(u, Qm, eig) for u in cands])
    order = np.argsort(np.abs(np.log(v / target_var)))[:n]
    return [cands[i] for i in order], v[order]


# ---- residual collection -----------------------------------------------------------------------------------------
@torch.no_grad()
def collect_residuals(model, tok, texts, residual_index, *, device="cuda", batch_size=32) -> dict:
    """Residual stream at `residual_index` at (last) the final prompt token and (concl) the last token of the user message
    (the sentence-final '.' of the conclusion). Layers above the hook are skipped (temporarily truncated, no copy)."""
    d = model.config.hidden_size
    prompts = [format_prompt(tok, t, False) for t in texts]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    last = np.zeros((len(prompts), d), np.float32)
    concl = np.zeros_like(last)
    layers = model.model.layers
    model.model.layers = layers[:residual_index]
    try:
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            ids, mask, pos = encode(tok, [prompts[i] for i in idx], device)
            sink = []
            with capture_residual(model, residual_index, sink):
                model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
            h = sink[0].float()
            first_end = (ids == im_end).float().argmax(-1)
            last[idx] = h[:, -1].cpu().numpy()
            concl[idx] = h[torch.arange(len(idx)), first_end - 1].cpu().numpy()
    finally:
        model.model.layers = layers
    return dict(last=last, concl=concl)


# ---- probes ------------------------------------------------------------------------------------------------------
def project_out(X: torch.Tensor, D: torch.Tensor | None) -> torch.Tensor:
    """Remove the span of the orthonormal columns D (d x r) from the rows of X."""
    if D is None or D.shape[1] == 0:
        return X
    D = D.to(X.dtype)
    return X - (X @ D) @ D.T


def fit_probe(X: torch.Tensor, y: torch.Tensor, l2: float, steps: int = 100):
    """L2 logistic regression on standardised features (LBFGS). Returns a dict with mean/sd/w/b (original-space weights)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")
    yf = y.float()

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Z @ w + b, yf) + l2 * (w ** 2).sum() / 2
        loss.backward()
        return loss
    opt.step(closure)
    return dict(mu=mu, sd=sd, w=w.detach(), b=b.detach())


def probe_scores(p: dict, X: torch.Tensor) -> torch.Tensor:
    return ((X - p["mu"]) / p["sd"]) @ p["w"] + p["b"]


def probe_eval(p: dict, X: torch.Tensor, y: np.ndarray) -> dict:
    s = probe_scores(p, X).cpu().numpy()
    yb = y.astype(bool)
    return dict(acc=float(((s > 0) == yb).mean()), auc=P05.auc(s, yb))


def probe_direction(p: dict) -> torch.Tensor:
    """Unit direction in the ORIGINAL feature space that the probe reads (w / sd)."""
    v = p["w"] / p["sd"]
    return v / v.norm()


def inlp(Xtr, ytr, tests: dict, l2: float, n_iter: int, steps: int = 100) -> list:
    """Iterative null-space projection: fit, record held-out performance, remove the probe direction, repeat.
    tests: name -> (X, y). Returns per-iteration list of {name: probe_eval}."""
    out, D = [], None
    for it in range(n_iter):
        Xt = project_out(Xtr, D)
        p = fit_probe(Xt, ytr, l2, steps)
        out.append({n: probe_eval(p, project_out(X, D), y) for n, (X, y) in tests.items()})
        v = probe_direction(p)[:, None]
        if D is not None:  # keep the removed set orthonormal
            v = v - D @ (D.T @ v)
            v = v / v.norm()
        D = v if D is None else torch.cat([D, v], 1)
    return out


# ---- search objective ---------------------------------------------------------------------------------------------
def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    a = x.abs()
    return torch.where(a <= delta, 0.5 * x ** 2, delta * (a - 0.5 * delta))


def pair_loss(m_valid, m_invalid, m_valid0, m_invalid0, lpm_valid, lpm_invalid, lpm_valid0, lpm_invalid0, gap_scale: float, w_bias: float = 0.5):
    """Pre-registered logic-pair terms. Inputs are semantic margins (valid-label logit minus invalid-label logit) of the ablated
    model and cached intact values; lpm_* = log P(the two label tokens) at the answer position.
      L_disc: ((m_valid - m_invalid)/gap_scale)^2                                   drives validity discrimination to 0
      L_bias: huber(pair-mean margin shift / 5) + huber(label-mass shift)           forbids global shift / off-format collapse"""
    l_disc = (((m_valid - m_invalid) / gap_scale) ** 2).mean()
    s, s0 = (m_valid + m_invalid) / 2, (m_valid0 + m_invalid0) / 2
    l_bias = huber((s - s0) / 5).mean() + 0.5 * (huber(lpm_valid - lpm_valid0) + huber(lpm_invalid - lpm_invalid0)).mean()
    return dict(total=l_disc + w_bias * l_bias, disc=l_disc, bias=l_bias)


def ctrl_loss(m, m0, lpm, lpm0):
    """L_ctrl: huber(margin shift / 5) + huber(label-mass shift), for localizer, PrOntoQA, MMLU-binary ('1' minus '2')."""
    return huber((m - m0) / 5).mean() + huber(lpm - lpm0).mean()


def search_loss(m_valid, m_invalid, m_valid0, m_invalid0, lpmass_pair, lpmass_pair0, ctrl_m, ctrl_m0, ctrl_lpm, ctrl_lpm0,
                gap_scale: float, w_bias: float = 0.5, w_ctrl: float = 1.0) -> dict:
    """Full objective = L_disc + w_bias*L_bias + w_ctrl*L_ctrl (weights fixed in docs/phase06_plan.md)."""
    pl = pair_loss(m_valid, m_invalid, m_valid0, m_invalid0, lpmass_pair, lpmass_pair, lpmass_pair0, lpmass_pair0, gap_scale, w_bias)
    lc = ctrl_loss(ctrl_m, ctrl_m0, ctrl_lpm, ctrl_lpm0)
    return dict(total=pl["total"] + w_ctrl * lc, disc=pl["disc"], bias=pl["bias"], ctrl=lc)
