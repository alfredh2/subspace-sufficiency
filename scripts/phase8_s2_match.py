"""Phase 8 S2 step 1 (DEV DATA ONLY): carrier targets, random / on-manifold / energy-matched / displacement-matched control subspaces, conditional controls.
Saves controls to results/phase8/stage2/controls_h{h}.pt and a match-quality json. Never loads confirm items."""
import argparse, json, time
import numpy as np, torch
from causal_subspaces.phase8.s2lib import *

ap = argparse.ArgumentParser(); ap.add_argument("--hook", type=int, default=25); ap.add_argument("--pool", type=int, default=300); ap.add_argument("--cpool", type=int, default=100)
a = ap.parse_args()
w = World(a.hook); h = a.hook
w.conf = None; w.pairs = None                                  # confirm items are not used in this script
Kc = max(len(v) for v in w.chains.values())
ks = [k for k in KS if k <= Kc]
fin, allp = resid_matrices(w, w.dev)
basis = dev_pca_basis(fin)
base64 = run_ext(w, w.dev64, want_hL=True, bs=64)
gold_dev = np.array([i.gold for i in w.dev])
base_dev = run_ext(w, w.dev)


def measure(U):
    ab = run_ext(w, w.dev64, U, want_hL=True, bs=64)
    D = float(np.linalg.norm(ab["hL"] - base64["hL"], axis=1).mean())
    return dict(E=energy(U, fin), Eall=energy(U, allp), P=pert_norm(U, fin), D=D)


t0 = time.time()
out = dict(hook=h, ks=ks, targets={}, pools={}, sel={}, cond={}, dev_carrier_te={}, params=dict(E_TOL=E_TOL, D_TOL=D_TOL, N_DRAW=N_DRAW, pool=a.pool, cpool=a.cpool))
ctrl = {}
# ---- carrier targets and dev necessity (descriptive; dev only)
for s, Us in w.chains.items():
    K = len(Us); tg = {}
    singles = [measure(Us[k]) for k in range(K)]
    tg[1] = {x: float(np.mean([m[x] for m in singles])) for x in singles[0]}; tg["singles"] = singles
    for k in ks:
        if k > 1:
            tg[k] = measure(w.union(Us, range(1, k + 1)))
    out["targets"][s] = tg
    te_ = {}
    for k in range(1, K + 1):
        ab = run_ext(w, w.dev, Us[k - 1]); te_[f"single{k}"] = te(base_dev["cand_logits"], ab["cand_logits"], gold_dev)
        ab = run_ext(w, w.dev, w.union(Us, range(1, k + 1))); te_[f"cum{k}"] = te(base_dev["cand_logits"], ab["cand_logits"], gold_dev)
    out["dev_carrier_te"][s] = te_
    print("target", s, {k: (round(v["E"], 3), round(v["D"], 2)) for k, v in tg.items() if k != "singles"}, "dev cum TE", [round(te_[f"cum{k}"][0], 3) for k in range(1, K + 1)], flush=True)

# ---- shared pools per k
pool_u = {}
for k in ks:
    r = 4 * k; cand = []
    for j in range(a.pool):
        U = pool_draw(w, basis, r, 700000 + 1000 * k + j)
        m = measure(U); m["id"] = j; cand.append(m); pool_u[(k, j)] = U.cpu()
    out["pools"][k] = cand
    Es = np.array([c["E"] for c in cand]); Ds = np.array([c["D"] for c in cand])
    print("pool k", k, "E range", Es.min().round(4), Es.max().round(4), "D range", Ds.min().round(2), Ds.max().round(2), f"{time.time()-t0:.0f}s", flush=True)

# ---- haar / on-manifold per k (shared across chains) and selection of matched draws per (chain, k)
for k in ks:
    for fam in ("haar", "onm"):
        for j in range(N_DRAW[k]):
            U = haar(4096, 4 * k, 100000 + 10 * k + j) if fam == "haar" else covdraw(w.ref, 4 * k, 200000 + 10 * k + j, 0.0)
            ctrl[(fam, k, "all", j)] = U.cpu()
            out.setdefault("shared", {})[f"{fam}|{k}|{j}"] = measure(U)
for s in w.chains:
    for k in ks:
        tgt = out["targets"][s][k]
        for fam, key, tol in (("energy", "E", E_TOL), ("disp", "D", D_TOL)):
            cand = out["pools"][k]
            rat = np.array([c[key] / tgt[key] for c in cand])
            inside = np.where(np.abs(rat - 1) <= tol)[0]
            rng = np.random.default_rng(900 + 100 * s + 10 * k + (fam == "disp"))
            n = N_DRAW[k]
            if len(inside) >= n:
                pick = rng.choice(inside, n, replace=False); flag = "matched"
            else:
                pick = np.argsort(np.abs(rat - 1))[:n]; flag = "unmatched"
            for j, c in enumerate(pick):
                ctrl[(fam, k, s, j)] = pool_u[(k, int(c))]
            out["sel"][f"{fam}|{k}|{s}"] = dict(flag=flag, n_inside=int(len(inside)), pick=[int(c) for c in pick], ratio_E=[float(out["pools"][k][c]["E"] / tgt["E"]) for c in pick],
                                              ratio_D=[float(out["pools"][k][c]["D"] / tgt["D"]) for c in pick], ratio_P=[float(out["pools"][k][c]["P"] / tgt["P"]) for c in pick])
    print("selected chain", s, {f"{fam}{k}": out["sel"][f"{fam}|{k}|{s}"]["flag"][0] + str(out["sel"][f"{fam}|{k}|{s}"]["n_inside"]) for k in ks for fam in ("energy", "disp")}, flush=True)

# ---- conditional controls for ordered pairs among carriers 1..4: rank-4 subspaces orthogonal to A (haar-in-complement, energy- and disp-matched to B)
for s, Us in w.chains.items():
    for A in range(1, min(4, len(Us)) + 1):
        cpool = []
        for j in range(a.cpool):
            U = pool_draw(w, basis, 4, 800000 + 100 * s + 10 * A + j * 1000, avoid=Us[A - 1])
            m = measure(U); m["id"] = j; cpool.append((m, U.cpu()))
        for B in range(1, min(4, len(Us)) + 1):
            if B == A:
                continue
            tb = out["targets"][s]["singles"][B - 1]; rng = np.random.default_rng(5000 + 100 * s + 10 * A + B)
            for j in range(6):
                V = torch.randn(4096, 4, generator=torch.Generator().manual_seed(300000 + 1000 * s + 100 * A + 10 * B + j), dtype=torch.float64).cuda()
                V = V - Us[A - 1].double() @ (Us[A - 1].double().T @ V)
                U = torch.linalg.qr(V)[0].float()
                ctrl[("chaar", (A, B), s, j)] = U.cpu(); out["cond"][f"haar|{A}|{B}|{s}|{j}"] = measure(U)
            for fam, key, tol in (("energy", "E", E_TOL), ("disp", "D", D_TOL)):
                rat = np.array([m[key] / tb[key] for m, _ in cpool]); inside = np.where(np.abs(rat - 1) <= tol)[0]
                pick = rng.choice(inside, 6, replace=False) if len(inside) >= 6 else np.argsort(np.abs(rat - 1))[:6]
                for j, c in enumerate(pick):
                    ctrl[("c" + fam, (A, B), s, j)] = cpool[int(c)][1]
                out["cond"][f"{fam}|{A}|{B}|{s}"] = dict(flag="matched" if len(inside) >= 6 else "unmatched", n_inside=int(len(inside)),
                                                       ratio_E=[float(cpool[c][0]["E"] / tb["E"]) for c in pick], ratio_D=[float(cpool[c][0]["D"] / tb["D"]) for c in pick])
    print("conditional controls chain", s, f"{time.time()-t0:.0f}s", flush=True)
torch.save(ctrl, R2 / f"controls_h{h}.pt")
json.dump(out, open(R2 / f"match_h{h}.json", "w"), indent=0, default=float)
print("S2 MATCH DONE", h, f"{time.time()-t0:.0f}s", flush=True)
