"""Phase 5 staged runner (docs/phase5_plan.md). Usage: python scripts/phase5_run.py {pilot|select|stage1|stage2|stage3|freeze|stage4}"""
import json
import os
import sys
import time

import numpy as np
import torch

from causal_subspaces.phase5 import pipeline as PL
from causal_subspaces.phase5.runner import ROWS, run_jobs
from causal_subspaces.upstream import pipeline as UP
from causal_subspaces.upstream import stats as S
from causal_subspaces.utils.config import ROOT

stage = sys.argv[1]; T0 = time.time(); OUT = PL.OUT; OUT.mkdir(exist_ok=True)
FMTS = ("num_orig", "num_rev")


def log_budget(name, secs):
    p = OUT / "budget.json"; b = json.load(open(p)) if p.exists() else {"soft_budget_gpu_hours": [3, 4], "stages": {}}
    b["stages"][name] = secs; b["total_hours"] = sum(b["stages"].values()) / 3600; json.dump(b, open(p, "w"), indent=1)


def tm(boot, tr, bt, bs, jobs, name, key="margin"):
    return PL.tstat(boot, tr, bt, bs, jobs, name, key)["point"]


def save(path, trials, bt, bs, res, extra=None):
    UP.save_raw(path, trials, bt, bs, res)


if stage == "pilot":
    P = PL.setup(); c5 = P["ctx5"]; H = c5.H; out = {}
    allh = list(range(H))
    # (a) regression gate on set A: all-32 Z-patch at G1G2 rows reproduces Phase 4 comp|attn|L19|G1G2 (0.360 / 0.390)
    ref = json.load(open(ROOT / "results/phase4/stage4.json"))
    for fmt in FMTS:
        trials, _ = UP.trial_sets(P, "A", fmt); boot = UP.boot_for(trials, 500)
        jobs = [dict(name="z32", kind="z", heads=allh, rows=ROWS)]
        bt, bs_, res = run_jobs(c5, trials, jobs, bs=32)
        t = tm(boot, dict(sigma=UP.sigma_of(trials)), bt, bs_, res, "z32"); r4 = ref[fmt]["jobs"]["comp|attn|L19|G1G2"]["margin"]["T"]
        out[f"regression_{fmt}"] = dict(z32=t, phase4=r4, diff=t - r4, ok=bool(abs(t - r4) <= 0.01), vz=float(res["z32"]["vz"])); print("regression", fmt, out[f"regression_{fmt}"], flush=True)
    # (b) delta-recompute acceptance on validation (num_orig): AV all keys, all heads == model-native Z-patch; per-head recompute error
    trials, census = UP.trial_sets(P, "val", "num_orig"); tr = dict(sigma=UP.sigma_of(trials)); boot = UP.boot_for(trials, 500); print("val", census, flush=True)
    jobs = [dict(name="z32", kind="z", heads=allh, rows=ROWS), dict(name="av32", kind="av", heads=allh, patch="AV", rows=ROWS), dict(name="v32_all", kind="av", heads=allh, patch="V", rows=ROWS, keys=PL.KEY_ALL),
            dict(name="v32_other", kind="av", heads=allh, patch="V", rows=ROWS, keys="Other"), dict(name="z32_pin", kind="z", heads=allh, rows=ROWS, pin=dict(layers="22", which=("attn", "mlp")))]
    t1 = time.time(); bt, bs_, res = run_jobs(c5, trials, jobs, bs=32); dt = time.time() - t1
    tz, tav = tm(boot, tr, bt, bs_, res, "z32"), tm(boot, tr, bt, bs_, res, "av32")
    out["recompute_acceptance"] = dict(T_z32=tz, T_av32=tav, diff=tav - tz, ok=bool(abs(tav - tz) <= 0.03), T_v_allkeys=tm(boot, tr, bt, bs_, res, "v32_all"), T_v_other=tm(boot, tr, bt, bs_, res, "v32_other"),
                                       other_noop_max_abs=float(np.abs(res["v32_other"]["margin"] - bt["margin"]).max()), z32_pin_T=tm(boot, tr, bt, bs_, res, "z32_pin"), pin_verify=float(res["z32_pin"]["vp"]), z_verify=float(res["z32"]["vz"]),
                                       seconds_per_forward=dt / (len(jobs) * len(trials)))
    # per-head recompute error at rows G1G2 (relative to the head's own z std), validation trials
    from causal_subspaces.heads.attention import AttnStash, recompute
    from causal_subspaces.heads.ablation import capture_z
    from causal_subspaces.models.loading import encode, format_prompt, last_logits
    from causal_subspaces.phase5.runner import _pad
    errs, stds = [], []
    with torch.no_grad():
        for s0 in range(0, len(trials), 32):
            b = trials[s0:s0 + 32]; ids, m, p = encode(P["tok"], [format_prompt(P["tok"], t["tgt"], False) for t in b], c5.dev); z = {}; st = AttnStash(c5.model, [19])
            with capture_z(c5.model, [19], z): last_logits(c5.model, ids, m, p)
            rec = recompute(c5.model, 19, *st.data[19], m); st.close(); B, T = ids.shape; rows = _pad(b, ROWS, T, c5.dev, s0)
            zn = z[19].float().view(B, T, 32, 128); zr = rec["z"].float()
            d = ((zr - zn) ** 2).sum(-1).sqrt(); errs.append(torch.stack([(d[:, :, h][rows] ** 2).mean().sqrt() for h in range(32)]).cpu().numpy() if False else np.array([float(((zr - zn)[:, :, h][rows] ** 2).mean().sqrt()) for h in range(32)]))
            stds.append(np.array([float(zn[:, :, h][rows].std()) for h in range(32)]))
    rel = np.mean(errs, 0) / np.mean(stds, 0); out["recompute_rel_rms_error"] = rel.tolist(); out["recompute_heads_failing_1pct"] = [int(h) for h in np.where(rel > 0.01)[0]]
    out["noise_floor_batch"] = float(np.abs(run_jobs(c5, trials, [], bs=32)[0]["margin"] - run_jobs(c5, trials, [], bs=5)[0]["margin"]).max())
    out["pass"] = bool(all(out[f"regression_{f}"]["ok"] for f in FMTS) and out["recompute_acceptance"]["ok"] and out["recompute_acceptance"]["other_noop_max_abs"] < 1e-3 and out["recompute_acceptance"]["pin_verify"] < 1e-3 and out["recompute_acceptance"]["z_verify"] == 0.0)
    json.dump(out, open(OUT / "pilot.json", "w"), indent=1, default=float); print(json.dumps({k: v for k, v in out.items() if k != "recompute_rel_rms_error"}, indent=1, default=float), flush=True)
    print("rel err per head", np.round(rel, 4).tolist(), flush=True); log_budget("pilot", time.time() - T0)
elif stage == "select":
    P = PL.setup(); c5 = P["ctx5"]; H = c5.H; allh = list(range(H))
    trials, census = UP.trial_sets(P, "val", "num_orig"); tr = dict(sigma=UP.sigma_of(trials)); boot = UP.boot_for(trials, 1000); print("val", census, flush=True)
    # pass 0: base + scalar direction w (validation mean of sigma * (h20_src - h20_tgt) over rows G1G2, normalised)
    bt0, bs0, _, dh = run_jobs(c5, trials, [], bs=32, keep_dh=True)
    wv = (tr["sigma"][:, None] * dh).sum(0); wv = wv / np.linalg.norm(wv); c5.w = torch.tensor(wv, device=c5.dev, dtype=torch.float32)
    # pass 1: single-head Z-patches (R1 ranking) + all-32
    jobs = [dict(name=f"h{h}", kind="z", heads=[h], rows=ROWS) for h in allh] + [dict(name="z32", kind="z", heads=allh, rows=ROWS)]
    bt, bs_, res = run_jobs(c5, trials, jobs, bs=32)
    R1 = [float(tm(boot, tr, bt, bs_, res, f"h{h}")) for h in allh]; T32 = tm(boot, tr, bt, bs_, res, "z32")
    order = [int(h) for h in np.argsort(-np.array(R1))]
    # descriptive attribution (R2) and output magnitudes
    S_h, M_h = PL.head_stats(c5, trials); r2 = S_h.mean(0); mags = M_h.mean(0)
    order_r2 = [int(h) for h in np.argsort(-np.abs(r2))]
    d3 = np.load(ROOT / "results/phase2/stage1_arrays.npz")["delta_logic"][19]; order_r3 = [int(h) for h in np.argsort(-np.abs(d3))]
    # pass 2: nested top-k curve (k = 1..32) and KV groups
    jobs = [dict(name=f"top{k}", kind="z", heads=sorted(order[:k]), rows=ROWS) for k in range(1, H + 1)] + [dict(name=f"kv{g}", kind="z", heads=list(range(4 * g, 4 * g + 4)), rows=ROWS) for g in range(8)]
    bt, bs2, res2 = run_jobs(c5, trials, jobs, bs=32)
    curve = [float(tm(boot, tr, bt, bs2, res2, f"top{k}")) for k in range(1, H + 1)]
    kstar = next(k for k in range(1, H + 1) if curve[k - 1] >= 0.8 * T32)
    kv = [float(tm(boot, tr, bt, bs2, res2, f"kv{g}")) for g in range(8)]
    fallback = bool(kstar > 4)
    Hs = sorted(order[:kstar]) if not fallback else allh
    # pass 3: R2/R3 sets of size k*, L19H10, dose-response of H* on validation, null sets on validation
    rnd, mat = ([], []) if fallback else PL.null_sets(kstar, Hs, mags)
    jobs = [dict(name="r2set", kind="z", heads=sorted(order_r2[:kstar]), rows=ROWS), dict(name="r3set", kind="z", heads=sorted(order_r3[:kstar]), rows=ROWS), dict(name="l19h10", kind="z", heads=[10], rows=ROWS)]
    jobs += [dict(name=f"dose{a}", kind="z", heads=Hs, rows=ROWS, alpha=a) for a in PL.ALPHAS]
    jobs += [dict(name=f"rnd{i}", kind="z", heads=h, rows=ROWS) for i, h in enumerate(rnd)] + [dict(name=f"mm{i}", kind="z", heads=h, rows=ROWS) for i, h in enumerate(mat)]
    bt, bs3, res3 = run_jobs(c5, trials, jobs, bs=32)
    g = lambda key, n_: [float(tm(boot, tr, bt, bs3, res3, f"dose{a}", key)) for a in PL.ALPHAS]
    Tm_dose, Tw_dose, Tc_dose = g("margin", 0), g("w20", 0), g("c20", 0)
    pi_w = [x / Tw_dose[PL.ALPHAS.index(1.0)] for x in Tw_dose]; pi_c = [x / Tc_dose[PL.ALPHAS.index(1.0)] for x in Tc_dose]
    sel = dict(census=census, n_trials=len(trials), T_all32=T32, R1_single_head_T=R1, R1_order=order, nested_topk_T=curve, k_star=kstar, fallback_all32=fallback, H_star=Hs,
               kv_group_T=kv, R2_local_write=r2.tolist(), R2_order=order_r2, R3_phase2_lastrow_delta=d3.tolist(), R3_order=order_r3, head_output_magnitude=mags.tolist(),
               sets=dict(R2=sorted(order_r2[:kstar]), R3=sorted(order_r3[:kstar]), L19H10=[10]), validation_T=dict(R2=tm(boot, tr, bt, bs3, res3, "r2set"), R3=tm(boot, tr, bt, bs3, res3, "r3set"), L19H10=tm(boot, tr, bt, bs3, res3, "l19h10"),
                   random=[tm(boot, tr, bt, bs3, res3, f"rnd{i}") for i in range(len(rnd))], matched=[tm(boot, tr, bt, bs3, res3, f"mm{i}") for i in range(len(mat))]),
               null_random=rnd, null_matched=mat, matched_mag_sum=dict(ref=float(np.sum(np.array(mags)[Hs])), matched=[float(np.sum(np.array(mags)[h])) for h in mat]),
               dose=dict(alphas=PL.ALPHAS, T_margin=Tm_dose, T_w20=Tw_dose, T_c20=Tc_dose, pi_w=pi_w, pi_c=pi_c, g_w=dict(pi=pi_w, T=Tm_dose), g_dc=dict(pi=pi_c, T=Tm_dose)),
               scalar_baselines=["proportional: T_hat = pi * T_Z(H*)", "d_c scalar: T_hat = g_dc(pi_c)"], w_norm=float(np.linalg.norm(wv)))
    torch.save(torch.tensor(wv), OUT / "w_direction.pt"); json.dump(sel, open(OUT / "selection.json", "w"), indent=1, default=float)
    print("T_all32", T32, "k*", kstar, "H*", Hs, "fallback", fallback, flush=True); print("nested curve", np.round(curve, 3).tolist(), flush=True)
    print("R1 top8", order[:8], np.round(np.array(R1)[order[:8]], 3).tolist(), "| R2 top8", order_r2[:8], "| R3 top8", order_r3[:8], flush=True)
    print("kv", np.round(kv, 3).tolist(), "dose", np.round(Tm_dose, 3).tolist(), "pi_w", np.round(pi_w, 3).tolist(), flush=True)
    print("val nulls rnd mean", np.mean(sel["validation_T"]["random"]) if rnd else None, "matched mean", np.mean(sel["validation_T"]["matched"]) if mat else None, "R2", sel["validation_T"]["R2"], "R3", sel["validation_T"]["R3"], flush=True)
    log_budget("select", time.time() - T0)

else:
    raise SystemExit("not implemented: " + stage)
