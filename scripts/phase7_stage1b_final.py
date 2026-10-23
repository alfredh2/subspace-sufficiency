"""Stage 1b confirm runs: for every task with a sufficient dev cell (selection.json), fit seeds 0-4 at the frozen (hook, rank) per format (seeds 0,1 of the
primary format are the Arm A screening fits, reused), evaluate on the CONFIRM contents (retain/ablate/random ablations/transplantation), save arrays.
Second formats of G, N, M, A are run only if the primary-format sufficiency, necessity and transplantation gates all PASS (amendment A5 ii); L and S
always run all formats (Arm C). Usage: python scripts/phase7_stage1b_final.py --tasks L S ..."""
import argparse, json, time
import numpy as np, torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.phase7.stage1 import run_stats
from causal_subspaces.phase7.stage1b import build_1b, FORMATS_1B, fit_b, eval_confirm, cell_gates
from causal_subspaces.utils.config import ROOT, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="*", required=True)
ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--steps", type=int, default=1600)
ap.add_argument("--force-all-formats", action="store_true")
a = ap.parse_args()
R = ROOT / "results/phase7/stage1b"
(R / "final").mkdir(exist_ok=True)
sel = json.load(open(R / "selection.json"))["tasks"]
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
for t in a.tasks:
    ch = sel[t]["choice"]
    if ch is None:
        print(t, "no sufficient dev cell: not run on confirm (1b-FAIL sufficiency)", flush=True); continue
    h, r = ch["hook"], ch["rank"]
    cfg["mvs"]["residual_index"] = h
    ref = get_reference(model, tok, cfg, rev, "cuda")
    mu = ref["mu"].cuda()
    data = build_1b(t, tok)
    for fi, fmt in enumerate(FORMATS_1B[t]):
        zf = R / "final" / f"{t}_{fmt}.npz"
        if zf.exists():
            continue
        if fi > 0 and t not in ("L", "S") and not a.force_all_formats:
            g0 = json.load(open(R / "final" / f"{t}_{FORMATS_1B[t][0]}.json"))["gates"]
            if not all(g0[k]["verdict"] == "PASS" for k in ("S", "N", "T")):
                print(t, fmt, "skipped: primary-format gates not all PASS (A5 ii)", flush=True); continue
        d = data[fmt]
        Us, fits = {}, {}
        for seed in a.seeds:
            pf = R / "final" / f"{t}_{fmt}_h{h}_r{r}_s{seed}.pt"
            ga = R / f"armA/{t}_h{h}_r{r}_s{seed}.pt"
            if fi == 0 and seed in (0, 1) and ga.exists():
                Us[seed] = torch.load(ga).cuda(); fits[seed] = dict(reused="armA"); continue
            if pf.exists():
                Us[seed] = torch.load(pf).cuda(); continue
            torch.manual_seed(seed)
            V0 = init_V(ref["Qm"], r, seed, "onmanifold")
            U, summ = fit_b(model, tok, d["train"], ref["mu"], V0, h, steps=a.steps, seed=seed)
            torch.save(U.cpu(), pf); Us[seed] = U; fits[seed] = summ
            print(t, fmt, "fit seed", seed, "trainKL %.4f %.0fs" % (summ["kl_last5_median"], summ["seconds"]), flush=True)
        intact = run_stats(model, tok, d["test"])
        z = eval_confirm(model, tok, d["test"], d["pairs"], h, Us, mu, intact, r)
        gates = cell_gates(z, a.seeds, intact_min=0.8 if t in ("M", "A") else 0.9)
        np.savez(zf, **z)
        json.dump(dict(task=t, fmt=fmt, hook=h, rank=r, seeds=a.seeds, fits=fits, n_test=len(d["test"]), n_pairs=len(d["pairs"]), gates=gates),
                  open(R / "final" / f"{t}_{fmt}.json", "w"), indent=1, default=float)
        print("CONFIRM", t, fmt, "h%d r%d" % (h, r), "intact acc %.3f" % gates["acc_intact"],
              "S", gates["S"]["verdict"], gates["S"]["n_seeds_ok"], "N", gates["N"]["verdict"], gates["N"]["n_seeds_ok"],
              "T", gates.get("T", {}).get("verdict"), gates.get("T", {}).get("n_seeds_ok"),
              "median kl %.4f rel %.2f abs %.1f transfer %.2f" % (gates["median"]["kl"], gates["median"]["rel"], gates["median"]["abs"], gates["median"].get("transfer", float("nan"))), flush=True)
print("FINAL DONE", flush=True)
