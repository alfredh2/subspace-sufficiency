"""Phase 8 S1: carrier discovery. Chains of mutually orthogonal rank-4 carriers (frozen rules B4). Loads TRAIN and DEV items only (never confirm).
Usage: python scripts/phase8_s1_fit.py --hook 25 --seeds 0 1 2 3 4 --kmax 8 [--parallel]"""
import argparse, json, pickle, time
import numpy as np, torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.phase7.stage1 import run_stats, primary_accuracy_vec
from causal_subspaces.phase7.stage1b import build_1b, fit_b
from causal_subspaces.utils.config import ROOT, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--hook", type=int, default=25)
ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--kmax", type=int, default=8)
ap.add_argument("--parallel", action="store_true")
ap.add_argument("--determinism", action="store_true")
a = ap.parse_args()
R = ROOT / "results/phase8/stage1"
C = R / "carriers"
C.mkdir(exist_ok=True)
S1B = ROOT / "results/phase7/stage1b"
KL_MAX, ACC_RATIO = 0.05, 0.9           # frozen sufficiency thresholds (B4)

cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
h = a.hook
cfg["mvs"]["residual_index"] = h
ref = get_reference(model, tok, cfg, rev, "cuda")
mu = ref["mu"].cuda()
d = build_1b("M", tok)["letters"]
train = d["train"]
fresh = pickle.load(open(R / "fresh_items.pkl", "rb"))
dev = d["val"] + fresh["devf"]["items"]
intact = run_stats(model, tok, dev)
acc_i = float(primary_accuracy_vec(intact, dev).mean())


def dev_metrics(U):
    st = run_stats(model, tok, dev, h, "retain", U, mu, with_kl=True)
    kl = float(st["kl"].mean()); acc = float(primary_accuracy_vec(st, dev).mean())
    return dict(dev_kl=kl, dev_acc=acc, dev_acc_intact=acc_i, sufficient=bool(kl <= KL_MAX and acc >= ACC_RATIO * acc_i))


def u1_path(s):
    f = S1B / f"final/M_letters_h{h}_r4_s{s}.pt"
    return f if f.exists() else S1B / f"armA/M_h{h}_r4_s{s}.pt"


def orth_err(Us, U):
    return float(max((V.T @ U).abs().max() for V in Us)) if Us else 0.0


def fit_carrier(perp, init_seed):
    torch.manual_seed(init_seed)
    V0 = init_V(ref["Qm"], 4, init_seed, "onmanifold")
    P = torch.cat(perp, 1).cpu()
    V0 = V0 - P @ (P.T @ V0)
    return fit_b(model, tok, train, ref["mu"], V0, h, seed=init_seed, perp=torch.cat(perp, 1))


for s in a.seeds:
    Us = []
    u1 = torch.load(u1_path(s)).cuda()
    m1 = dev_metrics(u1)
    json.dump(dict(hook=h, seed=s, k=1, source=str(u1_path(s).relative_to(ROOT)), **m1), open(C / f"h{h}_s{s}_k1.json", "w"), indent=1)
    print(f"h{h} s{s} k1 (existing U1)", {k: round(v, 5) if isinstance(v, float) else v for k, v in m1.items()}, flush=True)
    Us.append(u1)
    kstar = 1 if m1["sufficient"] else 0
    for k in range(2, a.kmax + 1):
        if not m1["sufficient"] and k == 2:
            break
        fj = C / f"h{h}_s{s}_k{k}.json"; fp = C / f"h{h}_s{s}_k{k}.pt"
        if fj.exists():
            meta = json.load(open(fj))
            if meta["sufficient"]:
                Us.append(torch.load(fp).cuda()); kstar = k; continue
            break
        attempts = []
        for retry in (0, 1):
            init_seed = 100 * k + s + 1000 * retry
            U, summ = fit_carrier(Us, init_seed)
            m = dev_metrics(U)
            oe = orth_err(Us, U)
            attempts.append(dict(init_seed=init_seed, retry=retry, train_kl=summ["kl_last5_median"], seconds=summ["seconds"], orth_err=oe,
                                 self_orth_err=float((U.T @ U - torch.eye(4, device=U.device)).abs().max()), **m))
            print(f"h{h} s{s} k{k} retry{retry}", {k_: round(v, 5) if isinstance(v, float) else v for k_, v in attempts[-1].items()}, flush=True)
            if oe > 1e-6:
                raise RuntimeError(f"orthogonality violation {oe}")
            if m["sufficient"]:
                break
        ok = attempts[-1]["sufficient"]
        torch.save(U.cpu(), fp)
        json.dump(dict(hook=h, seed=s, k=k, sufficient=ok, retry_used=len(attempts) > 1, attempts=attempts, **{x: attempts[-1][x] for x in ("dev_kl", "dev_acc", "dev_acc_intact")}),
                  open(fj, "w"), indent=1)
        if not ok:
            print(f"h{h} s{s}: chain stops at k={k}", flush=True)
            break
        Us.append(U); kstar = k
    print(f"h{h} s{s} KSTAR_DEV {kstar}", flush=True)
    if a.parallel and len(Us) >= 2:
        fpp = C / f"h{h}_s{s}_k2par.pt"
        if not fpp.exists():
            U, summ = fit_carrier([Us[0]], 200 + s + 2000)
            m = dev_metrics(U)
            torch.save(U.cpu(), fpp)
            json.dump(dict(hook=h, seed=s, parallel=True, orth_err_to_U1=orth_err([Us[0]], U), train_kl=summ["kl_last5_median"], **m), open(C / f"h{h}_s{s}_k2par.json", "w"), indent=1)
            print(f"h{h} s{s} parallel U2'", m, flush=True)
    if a.determinism and s == a.seeds[0] and len(Us) >= 2:
        U, summ = fit_carrier([Us[0]], 100 * 2 + s)
        cs = torch.linalg.svdvals(U.T @ Us[1].to(U.device))
        m = dev_metrics(U)
        json.dump(dict(hook=h, seed=s, k=2, refit_cosines_to_original=cs.tolist(), refit_train_kl=summ["kl_last5_median"], **m), open(R / "determinism.json", "w"), indent=1)
        print("determinism refit cosines", cs.tolist(), m, flush=True)
print("S1 FIT DONE hook", h, flush=True)
