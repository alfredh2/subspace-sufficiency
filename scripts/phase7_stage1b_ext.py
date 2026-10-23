"""Stage 1b Arm A extensions (dev only), run ONLY for tasks with no sufficient grid cell (amendment A5): at the best cell (lowest median val KL)
(a) 4800-step frozen recipe, (b) LDA initialisation, (c) non-fitted LDA reference, and tier-2 diagnostics (never counted toward the panel):
(d) candidate-restricted objective (G, N-cont), (e) N-fixedpool split. 2 seeds each. Writes stage1b/ext/*.json."""
import argparse, json, time
import numpy as np, torch
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.reference import init_V
from causal_subspaces.phase7.stage1 import retention_metrics, run_stats
from causal_subspaces.phase7.stage1b import build_1b, build_N_fixedpool, cap_items, FORMATS_1B, fit_b, lda_directions, cand_kl, curve_slope
from causal_subspaces.utils.config import ROOT, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="*", default=None)
ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
a = ap.parse_args()
R = ROOT / "results/phase7/stage1b"
(R / "ext").mkdir(exist_ok=True)
sel = json.load(open(R / "selection.json"))["tasks"]
tasks = a.tasks or [t for t, s in sel.items() if s["choice"] is None]
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")


def dev_eval(items, intact, h, U, mu):
    m, ret, abl = retention_metrics(model, tok, items, h, U, mu, intact)
    m["cand_kl_retain"] = float(cand_kl(intact["cand_logits"], ret["cand_logits"]).mean())
    return m


def run(name, t, data, h, r, ref, kind, objective="full", init="onmanifold", steps=1600):
    d = data
    val = cap_items(d["val"], 256)
    intact = run_stats(model, tok, val)
    mu = ref["mu"].cuda()
    for seed in a.seeds:
        f = R / "ext" / f"{name}_{kind}_s{seed}.json"
        if f.exists():
            continue
        torch.manual_seed(seed)
        V0 = init_V(ref["Qm"], r, seed, "onmanifold")
        if init == "lda":
            L = lda_directions(model, tok, d["train"], h, ref["Qm"])[:, :r]
            V0 = torch.cat([L.cpu(), V0[:, L.shape[1]:]], 1) if L.shape[1] < r else L.cpu()
        U, summ = fit_b(model, tok, d["train"], ref["mu"], V0, h, steps=steps, seed=seed, objective=objective)
        m = dev_eval(val, intact, h, U, mu)
        json.dump(dict(task=t, name=name, kind=kind, hook=h, rank=r, seed=seed, steps=steps, objective=objective, init=init, train=summ, val=m,
                       train_slope_last25=curve_slope(summ["kl_curve"])), open(f, "w"), indent=1)
        torch.save(U.cpu(), f.with_suffix(".pt"))
        print(name, kind, seed, "trainKL %.3f valKL %.4f candKL %.4f accR %.2f gap %.1f->%.1f slope %.2f %.0fs" % (
            summ["kl_last5_median"], m["kl_retain"], m["cand_kl_retain"], m["acc_retain"], m["gap_intact"], m["gap_ablate"], curve_slope(summ["kl_curve"]), summ["seconds"]), flush=True)


for t in tasks:
    h, r = sel[t]["best_cell"]["hook"], sel[t]["best_cell"]["rank"]
    cfg["mvs"]["residual_index"] = h
    ref = get_reference(model, tok, cfg, rev, "cuda")
    data = build_1b(t, tok)
    fmt0 = FORMATS_1B[t][0]
    print("TASK", t, "best cell", h, r, flush=True)
    run(t, t, data[fmt0], h, r, ref, "long4800", steps=4800)
    run(t, t, data[fmt0], h, r, ref, "lda_init", init="lda")
    # non-fitted LDA reference
    f = R / "ext" / f"{t}_lda_ref.json"
    if not f.exists():
        d = data[fmt0]; val = cap_items(d["val"], 256); intact = run_stats(model, tok, val)
        L = lda_directions(model, tok, d["train"], h, ref["Qm"])[:, :r].cuda()
        m = dev_eval(val, intact, h, L, ref["mu"].cuda())
        json.dump(dict(task=t, rank=L.shape[1], val=m), open(f, "w"), indent=1); print(t, "LDA reference rank", L.shape[1], "valKL %.3f accR %.2f" % (m["kl_retain"], m["acc_retain"]), flush=True)
    if t in ("G", "N"):
        run(t, t, data[fmt0], h, r, ref, "tier2_candobj", objective="cand")
    if t == "N":
        fp = build_N_fixedpool(tok)
        run("Nfix", t, fp[fmt0], h, r, ref, "tier2_fixedpool")
print("EXT DONE", flush=True)
