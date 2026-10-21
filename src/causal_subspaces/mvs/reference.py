"""Reference mean mu and initialisation covariance, computed in ONE streaming pass.

Paper (Sec. 3, App. C): mu_l = mean of h_l over all token positions of a neutral corpus disjoint from all
tasks; V0 = Q_m xi with Q_m the leading m=2048 eigenvectors of the layer's mu-centred activation covariance.
Corpus, token count and special-token handling are NOT stated -> configurable (data/reference in the
experiment config) and logged.

Memory: hidden states are never stored. Per batch we accumulate sum(x) and sum((x-s)(x-s)^T) with a fixed
shift s (mean of the first batch) so that fp32 batch matmuls are well conditioned (Qwen has coordinates ~800x
the median); the per-batch d x d matrices are summed in float64. Then
    mu = s + E[x - s],   Cov = E[(x-s)(x-s)^T] - E[x-s] E[x-s]^T   (exact identities).
"""
from pathlib import Path

import torch

from causal_subspaces.models.hooks import capture_residual


def load_neutral_corpus_ids(tok, name: str, config: str, split: str, max_tokens: int, window: int) -> torch.Tensor:
    """(n_windows, window) token ids, non-overlapping windows, no special tokens, no chat template."""
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split)
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    n = min(len(ids), max_tokens) // window * window
    return ids[:n].view(-1, window)


class StreamingMoments:
    def __init__(self, d: int, device):
        self.shift = None
        self.n = 0
        self.s1 = torch.zeros(d, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(d, d, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor):  # x: (n, d)
        x = x.float()
        if self.shift is None:
            self.shift = x.mean(0)
        c = x - self.shift
        self.n += c.shape[0]
        self.s1 += c.sum(0, dtype=torch.float64)
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # keep true fp32 for the second moment
        self.s2 += (c.T @ c).double()
        torch.backends.cuda.matmul.allow_tf32 = old

    def finalize(self):
        m1 = self.s1 / self.n
        mu = self.shift.double() + m1
        cov = self.s2 / self.n - torch.outer(m1, m1)
        return mu, cov


@torch.no_grad()
def compute_reference(model, windows: torch.Tensor, residual_index: int, m: int, batch_size: int = 16,
                      device="cuda") -> dict:
    """Run the first `residual_index` modules only (layers truncated temporarily; no copy), stream moments."""
    layers = model.model.layers
    d = model.config.hidden_size
    mom = StreamingMoments(d, device)
    sink: list = []
    model.model.layers = layers[:residual_index]  # ModuleList slice shares the same modules
    try:
        with capture_residual(model, residual_index, sink):
            for s in range(0, len(windows), batch_size):
                ids = windows[s:s + batch_size].to(device)
                sink.clear()
                model.model(input_ids=ids, use_cache=False)
                h = sink[0] if residual_index else model.model.embed_tokens(ids)
                mom.update(h.reshape(-1, d))
    finally:
        model.model.layers = layers
    mu, cov = mom.finalize()
    evals, evecs = torch.linalg.eigh(cov.cpu())  # fp64, ascending
    top = evecs[:, -m:].flip(-1).float()
    return dict(mu=mu.float().cpu(), Qm=top, eigvals=evals[-m:].flip(-1).float(), n_tokens=mom.n,
                total_var=float(evals.sum()), top_m_var=float(evals[-m:].sum()))


def cached_reference(path: Path, **kw) -> dict:
    if path.exists():
        return torch.load(path)
    ref = compute_reference(**kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, path)
    return ref


def init_V(Qm: torch.Tensor, k: int, seed: int, mode: str = "onmanifold") -> torch.Tensor:
    """V0 = Q_m xi, xi_ij ~ N(0, 1/m) (paper App. C). 'gaussian' = plain N(0,1/d) baseline for ablation studies."""
    g = torch.Generator().manual_seed(seed)
    if mode == "gaussian":
        d = Qm.shape[0]
        return torch.randn(d, k, generator=g) / d**0.5
    m = Qm.shape[1]
    return Qm @ (torch.randn(m, k, generator=g) / m**0.5)
