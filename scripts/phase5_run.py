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
else:
    raise SystemExit("not implemented: " + stage)
