"""Phase 8 S2 library: world loading, extended forward runner (layerwise lens, displacement), random-subspace samplers and matching,
gap/necessity statistics. Operational definitions: docs/plans/phase8_s2_protocol.md."""
import json, pickle
import numpy as np, torch
from contextlib import ExitStack
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.hooks import capture_residual, residual_intervention
from causal_subspaces.models.loading import encode, last_logits, load_model
from causal_subspaces.phase7.stage1 import cand_array, class_gap, texts_of
from causal_subspaces.phase7.stage1b import build_1b
from causal_subspaces.utils.config import ROOT, load_config

R1 = ROOT / "results/phase8/stage1"
R2 = ROOT / "results/phase8/stage2"
KS = (1, 2, 3, 4, 6, 8)                  # control ranks 4k
FAMS = ("haar", "onm", "energy", "disp")
N_DRAW = {1: 8, 2: 8, 3: 8, 4: 8, 6: 6, 8: 6}
E_TOL, D_TOL = 0.10, 0.15
NOISE_FLOOR = 0.132


class World:
    def __init__(self, hook, device="cuda"):
        self.hook = hook
        self.cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
        self.model, self.tok, self.rev = load_model(self.cfg["model"], device)
        self.cfg["mvs"]["residual_index"] = hook
        self.ref = get_reference(self.model, self.tok, self.cfg, self.rev, device)
        self.mu = self.ref["mu"].cuda()
        d1b = build_1b("M", self.tok)["letters"]
        fresh = pickle.load(open(R1 / "fresh_items.pkl", "rb"))
        self.dev = d1b["val"] + fresh["devf"]["items"]
        self.conf, self.pairs = fresh["conf"]["items"], fresh["conf"]["pairs"]
        self.dev64 = [self.dev[i] for i in np.linspace(0, len(self.dev) - 1, 64).astype(int)]
        self.fr = json.load(open(R1 / "discovery_freeze.json"))
        self.W = self.model.lm_head.weight.detach().float()
        self.cids = torch.tensor(cand_array(self.dev[:1])[0]).cuda()
        self.Wc = self.W[self.cids]                      # (4, d)
        self.norm = self.model.model.norm
        self.chains = {}
        for nm, ch in self.fr["chains"].items():
            if not nm.startswith(f"h{hook}_"):
                continue
            s = int(nm.split("_s")[1]); K = ch["kstar_dev"]
            Us = [torch.load(ROOT / ch["carriers"]["1"]["file"] if k == 1 else R1 / "carriers" / f"h{hook}_s{s}_k{k}.pt").cuda().float() for k in range(1, K + 1)]
            self.chains[s] = Us

    def union(self, Us, idx):
        return torch.cat([Us[i - 1] for i in sorted(idx)], 1)


def key_of(idx):
    return "+".join(str(i) for i in sorted(idx))


@torch.no_grad()
def run_ext(w: World, items, U=None, bs=32, lens_layers=(), want_hL=False, ln_frozen_from=None, extra=None):
    """Forward pass with an optional natural-strength ablation of span(U) at w.hook. Returns candidate logits (N,4), greedy token, logsumexp;
    lens[l] = (N,4) logit-lens candidate logits of block l's output (l in lens_layers; l=36 is block 35 before the final norm) at the answer position;
    hL = (N,d) float32 block-35 output at the answer position. If ln_frozen_from (a dict from a previous call with want_hL=True) is given, an extra
    'lens_lnfrozen' is computed with the final-norm scale taken from the intact run."""
    model, tok = w.model, w.tok
    texts = texts_of(tok, items)
    N = len(items); order = sorted(range(N), key=lambda i: len(texts[i]))
    out = dict(cand_logits=np.zeros((N, 4), np.float32), greedy=np.zeros(N, np.int64), lse=np.zeros(N, np.float32))
    lens = {l: np.zeros((N, 4), np.float32) for l in lens_layers}
    rms = {l: np.zeros(N, np.float32) for l in lens_layers}
    lens_f = {l: np.zeros((N, 4), np.float32) for l in lens_layers} if ln_frozen_from is not None else None
    hL = np.zeros((N, model.config.hidden_size), np.float32) if want_hL else None
    cids = w.cids
    gw = w.norm.weight.float()
    for s in range(0, N, bs):
        ii = order[s:s + bs]
        ids, mask, pos = encode(tok, [texts[i] for i in ii], "cuda")
        sinks = {l: [] for l in set(lens_layers) | ({36} if want_hL else set())}
        with ExitStack() as st:
            if U is not None:
                st.enter_context(residual_intervention(model, w.hook, "ablate", U, w.mu))
            for l in sinks:
                st.enter_context(_block_capture(model, l, sinks[l]))
            lg = last_logits(model, ids, mask, pos)
        out["cand_logits"][ii] = lg.gather(1, cids[None].expand(len(ii), -1)).cpu().numpy()
        out["greedy"][ii] = lg.argmax(-1).cpu().numpy()
        out["lse"][ii] = torch.logsumexp(lg, -1).cpu().numpy()
        for l in lens_layers:
            hl = sinks[l][0][:, -1].float()
            lens[l][ii] = (w.norm(hl.to(torch.bfloat16)).float() @ w.Wc.T).cpu().numpy()
            rms[l][ii] = hl.pow(2).mean(-1).add(w.norm.variance_epsilon).sqrt().cpu().numpy()
            if lens_f is not None:
                rms_i = torch.tensor(ln_frozen_from["rms"][l][ii], device="cuda")
                lens_f[l][ii] = (((hl / rms_i[:, None]) * gw) @ w.Wc.T).cpu().numpy()
        if want_hL:
            hl = sinks[36][0][:, -1].float()
            hL[ii] = hl.cpu().numpy()
    if lens_layers:
        out["lens"] = lens; out["rms"] = rms
    if lens_f is not None:
        out["lens_lnfrozen"] = lens_f
    if want_hL:
        out["hL"] = hL
    return out


from contextlib import contextmanager


@contextmanager
def _block_capture(model, l, sink):
    """Capture the output of decoder block l-1 (= HF hidden_states[l]) for l in 1..36 (36: block 35 before the final norm)."""
    def hook(_m, _i, o):
        sink.append((o[0] if isinstance(o, tuple) else o).detach())
    h = model.model.layers[l - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def resid_matrices(w: World, items, all_positions=True, max_tok=40000, bs=32, seed=0):
    """Centered residuals at the hook: final-position matrix (N,d) and a sample of all-position tokens (<=max_tok, d)."""
    texts = texts_of(w.tok, items)
    order = sorted(range(len(items)), key=lambda i: len(texts[i]))
    fin = torch.zeros(len(items), w.model.config.hidden_size, device="cuda")
    allp = []
    for s in range(0, len(items), bs):
        ii = order[s:s + bs]
        ids, mask, pos = encode(w.tok, [texts[i] for i in ii], "cuda")
        sink = []
        with capture_residual(w.model, w.hook, sink):
            last_logits(w.model, ids, mask, pos)
        h = sink[0].float() - w.mu
        fin[ii] = h[:, -1]
        if all_positions:
            allp.append(h[mask.bool()])
    allp = torch.cat(allp) if allp else None
    if allp is not None and len(allp) > max_tok:
        g = torch.Generator().manual_seed(seed)
        allp = allp[torch.randperm(len(allp), generator=g)[:max_tok].cuda()]
    return fin, allp


def energy(U, X):
    """Mean over rows of ||U^T x||^2 / ||x||^2."""
    return float(((X @ U).pow(2).sum(1) / X.pow(2).sum(1)).mean())


def pert_norm(U, X):
    return float((X @ U).norm(dim=1).mean())


def haar(d, r, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, r, generator=g, dtype=torch.float64))[0].float().cuda()


def covdraw(ref, r, seed, a):
    """V = Q_m diag(lambda^(a/2)) xi; a=0 is the flat on-manifold sampler of Phase 7, a=1 is covariance-weighted."""
    g = torch.Generator().manual_seed(seed)
    Q, lam = ref["Qm"].double(), ref["eigvals"].double().clamp_min(1e-12)
    xi = torch.randn(Q.shape[1], r, generator=g, dtype=torch.float64)
    V = Q @ ((lam / lam.mean()).pow(a / 2)[:, None] * xi)
    return torch.linalg.qr(V)[0].float().cuda()


# ---------------------------------------------------------------- statistics
def gap_of(cl, gold):
    return class_gap(cl, gold)


def te(cl_int, cl_cond, gold, ix=None):
    """(relative, absolute) necessity of a condition: (G - gap)/G and G - gap."""
    if ix is not None:
        cl_int, cl_cond, gold = cl_int[ix], cl_cond[ix], gold[ix]
    G, g = class_gap(cl_int, gold), class_gap(cl_cond, gold)
    return (G - g) / G, G - g


def pcadraw(basis, r, n, seed):
    """Random rank-r subspace inside the span of the first n columns of `basis` (orthonormal, columns ordered by variance)."""
    g = torch.Generator().manual_seed(seed)
    V = basis[:, :n].double().cpu() @ torch.randn(n, r, generator=g, dtype=torch.float64)
    return torch.linalg.qr(V)[0].float().cuda()


def dev_pca_basis(Xfin, n=192):
    """Top principal directions of the dev final-position centered residuals (dev data only)."""
    U, S, Vt = torch.linalg.svd(Xfin.double(), full_matrices=False)
    return Vt[:n].T.float().cuda()


def pool_draw(w, basis, r, seed, avoid=None):
    """One candidate for the matched pools: a mix of covariance-weighted (a in [0,3]), PCA-restricted (n log-uniform in [r,192]) and flat draws."""
    rng = np.random.default_rng(seed)
    u = rng.random()
    if u < 0.4:
        U = covdraw(w.ref, r, seed, float(rng.uniform(0, 3)))
    elif u < 0.9:
        n = int(np.exp(rng.uniform(np.log(max(r, 5)), np.log(192))))
        U = pcadraw(basis, r, max(n, r), seed)
    else:
        U = covdraw(w.ref, r, seed, float(rng.uniform(0, 1)))
    if avoid is not None:
        V = U - avoid @ (avoid.T @ U)
        U = torch.linalg.qr(V.double())[0].float()
    return U
