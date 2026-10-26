"""Phase 8 S2 step 2 (CONFIRM DATA): natural-strength ablations (single / pair / cumulative), all control families, conditional co-ablation controls,
transplants (single / union / conditional) with controls. Run only after match_h{h}.json + controls are committed. Saves raw per-item arrays (no verdicts here)."""
import argparse, json, pickle, time
import numpy as np, torch
from causal_subspaces.models.hooks import capture_residual, residual_intervention, residual_patch
from causal_subspaces.models.loading import encode, last_logits
from causal_subspaces.phase7.stage1 import pair_trials, transplant, texts_of
from causal_subspaces.phase8.s2lib import *

ap = argparse.ArgumentParser(); ap.add_argument("--hook", type=int, default=25); ap.add_argument("--chains", nargs="*", type=int)
a = ap.parse_args()
h = a.hook; w = World(h)
ctrl = torch.load(R2 / f"controls_h{h}.pt"); M = json.load(open(R2 / f"match_h{h}.json"))
ks = M["ks"]; gold = np.array([i.gold for i in w.conf]); groups = np.array([i.group for i in w.conf])
trials = pair_trials(w.pairs); tgroups = np.array([t["tgt"].group for t in trials])
intact = run_ext(w, w.conf)
meta = dict(gold=gold, groups=groups, tgroups=tgroups, cl_int=intact["cand_logits"], ks=ks)
cuda = lambda U: U.cuda().float()


def abl(U):
    return run_ext(w, w.conf, U)["cand_logits"]


@torch.no_grad()
def cond_transplant(UA_list, UB_dict, bs=16):
    """For each A in UA_list (name->U): m with A ablated; for each (A,B): m with A ablated and B patched from the intact source. Returns dict of (n,) margins."""
    tt = texts_of(w.tok, [t["tgt"] for t in trials]); ts = texts_of(w.tok, [t["src"] for t in trials])
    lens = np.array([len(w.tok(x, add_special_tokens=False).input_ids) for x in tt])
    sc = np.array([t["src"].gold for t in trials]); tc = np.array([t["tgt"].gold for t in trials])
    tcand = torch.tensor(cand_array([t["tgt"] for t in trials])).cuda()
    n = len(trials); res = {}
    def marg(lg, ii):
        v = lg.gather(1, tcand[ii]); r = torch.arange(len(ii), device=lg.device)
        return (v[r, torch.tensor(sc[ii]).cuda()] - v[r, torch.tensor(tc[ii]).cuda()]).cpu().numpy()
    for L in np.unique(lens):
        idxs = np.where(lens == L)[0]
        for s0 in range(0, len(idxs), bs):
            ii = idxs[s0:s0 + bs]
            ids_t, m_t, p_t = encode(w.tok, [tt[i] for i in ii], "cuda"); ids_s, m_s, p_s = encode(w.tok, [ts[i] for i in ii], "cuda")
            sink = []
            with capture_residual(w.model, h, sink):
                last_logits(w.model, ids_s, m_s, p_s)
            src = sink[0]
            for an, UA in UA_list.items():
                with residual_intervention(w.model, h, "ablate", UA, w.mu):
                    lg = last_logits(w.model, ids_t, m_t, p_t)
                res.setdefault(("A", an), np.zeros(n))[ii] = marg(lg, ii)
                for bn, UB in UB_dict.items():
                    if bn == an:
                        continue
                    with residual_intervention(w.model, h, "ablate", UA, w.mu), residual_patch(w.model, h, UB, src, "all", False):
                        lg = last_logits(w.model, ids_t, m_t, p_t)
                    res.setdefault(("AB", an, bn), np.zeros(n))[ii] = marg(lg, ii)
    return res


def tr_subs(prefix, subs):
    return transplant(w.model, w.tok, trials, h, subs, bs=16, positions=("all",))


# ---------------------------------------------------------------- shared controls (haar, on-manifold): ablation + transplant
sh_path = R2 / f"conf_h{h}_shared.pkl"
if not sh_path.exists():
    t0 = time.time(); S = dict(meta=meta, abl={}, tr={})
    for fam in ("haar", "onm"):
        for k in ks:
            for j in range(N_DRAW[k]):
                S["abl"][(fam, k, j)] = abl(cuda(ctrl[(fam, k, "all", j)]))
    subs = {f"{fam}|{k}|{j}": cuda(ctrl[(fam, k, "all", j)]) for fam in ("haar", "onm") for k in ks if k in (1, 2, 4, 8) for j in range(6)}
    res = tr_subs("sh", subs); S["tr"] = res
    pickle.dump(S, open(sh_path, "wb")); print("shared done", f"{time.time()-t0:.0f}s", flush=True)

for s, Us in w.chains.items():
    if a.chains and s not in a.chains:
        continue
    pth = R2 / f"conf_h{h}_s{s}.pkl"
    if pth.exists():
        continue
    t0 = time.time(); K = len(Us); C = dict(meta=meta, abl={}, cond={}, tr={}, ctr={})
    # carrier ablations: singles, pairs among 1..4, cumulative
    keys = [(i,) for i in range(1, K + 1)] + [(i, j) for i in range(1, min(4, K) + 1) for j in range(i + 1, min(4, K) + 1)] + [tuple(range(1, k + 1)) for k in range(3, K + 1)]
    for key in keys:
        C["abl"][key_of(key)] = abl(w.union(Us, key))
    for k in ks:
        for fam in ("energy", "disp"):
            for j in range(N_DRAW[k]):
                C["abl"][f"{fam}|{k}|{j}"] = abl(cuda(ctrl[(fam, k, s, j)]))
    print(f"chain {s} ablations done {time.time()-t0:.0f}s", flush=True)
    # conditional co-ablation controls: TE(A u R) for R orthogonal to A
    for A in range(1, min(4, K) + 1):
        for B in range(1, min(4, K) + 1):
            if A == B:
                continue
            for fam in ("haar", "energy", "disp"):
                for j in range(6):
                    R = cuda(ctrl[("c" + fam, (A, B), s, j)])
                    C["cond"][(A, B, fam, j)] = abl(torch.cat([Us[A - 1], R], 1))
    print(f"chain {s} conditional controls done {time.time()-t0:.0f}s", flush=True)
    # transplants: singles, unions k in {2,4,8}, matched-control families (first 6 draws at k in {1,2,4,8})
    subs = {f"c{k}": Us[k - 1] for k in range(1, K + 1)}
    for k in (2, 4, 8):
        if k <= K:
            subs[f"u{k}"] = w.union(Us, range(1, k + 1))
    for fam in ("energy", "disp"):
        for k in (1, 2, 4, 8):
            if k in ks:
                for j in range(6):
                    subs[f"{fam}|{k}|{j}"] = cuda(ctrl[(fam, k, s, j)])
    C["tr"] = tr_subs("c", subs)
    print(f"chain {s} transplants done {time.time()-t0:.0f}s", flush=True)
    # conditional transplant among the first four carriers
    q = min(4, K); C["ctr"] = cond_transplant({f"c{i}": Us[i - 1] for i in range(1, q + 1)}, {f"c{i}": Us[i - 1] for i in range(1, q + 1)})
    pickle.dump(C, open(pth, "wb"))
    print(f"chain {s} DONE {time.time()-t0:.0f}s", flush=True)
print("S2 CONF DONE", h, flush=True)
