"""Phase 4 staged runner (docs/phase4_plan.md). Usage: python scripts/phase4_run.py {pilot|stage2|stage3|freeze|repl|stage5|stage4}"""
import json
import sys
import time

import numpy as np

from causal_subspaces.upstream import pipeline as PL
from causal_subspaces.upstream.runner import run_jobs
from causal_subspaces.utils.config import ROOT

stage = sys.argv[1]; T0 = time.time(); OUT = PL.OUT; OUT.mkdir(exist_ok=True)
CAP = dict(pilot=12 * 60, stage2=44 * 60, stage3=17 * 60, repl=14 * 60, stage5=15 * 60, stage4=6 * 60)[stage] if stage != "freeze" else 60
PILOT_IDX = [0, 2, 4, 8, 12, 16, 18, 19, 20, 21, 22, 23]
GRID_B = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23]


def log_budget(name, secs):
    p = OUT / "budget.json"; b = json.load(open(p)) if p.exists() else {"cap_gpu_hours": 1.5, "stages": {}}
    b["stages"][name] = secs; b["total_hours"] = sum(b["stages"].values()) / 3600; json.dump(b, open(p, "w"), indent=1)


def onsets_A():
    out = {}
    for fmt in ("num_orig", "num_rev"):
        tr, bt, bs, jobs = PL.load_raw(OUT / f"stage2_{fmt}.npz")
        class T: pass
        bo = PL.S.Boot(tr["pair"], tr["content"], nboot=2000, seed=7)
        out[fmt] = {c: PL.onset_of(PL.curve(bo, tr["sigma"], bt, bs, jobs, c, "margin")) for c in ("G1", "G2", "G1G2")}
    return out


def pathway_indices(on):
    lon = on["sustained_both"]
    if lon is None: return [16, 19, 22, 23], None
    return sorted({max(0, lon - 3), lon, min(22, lon + 3), 19, 22, 23}), lon


if stage == "pilot":
    P = PL.setup(); ctx = P["ctx"]
    census = {}
    for which in ("A", "B"):
        for fmt in ("num_orig", "num_rev"):
            t, c = PL.trial_sets(P, which, fmt); census[f"{which}|{fmt}"] = c; print("census", which, fmt, c, flush=True)
    trials, c = PL.trial_sets(P, "val", "num_orig"); census["val|num_orig"] = c; print("pilot trials", len(trials), c, flush=True)
    conds = ["G1", "G2", "G1G2", "Self", "NonSummary", "AllMinusG1G2", "All", "PreDiff", "rand0", "rand1"]
    jobs = PL.job_list_patch(conds, PILOT_IDX)
    jobs += [dict(name=f"patch|G1G2|{i}|{f}", kind="patch", index=i, cond="G1G2", freeze=f) for i in (16, 19, 22, 23) for f in ("S8", "nonS8", "late", "all", "rnd0")]
    # batch-invariance noise floor of the clean target forward
    b32 = run_jobs(ctx, trials, [], bs=32)[0]["margin"]; b5 = run_jobs(ctx, trials, [], bs=5)[0]["margin"]; noise = float(np.abs(b32 - b5).max())
    t1 = time.time(); bt, bs_, res = run_jobs(ctx, trials, jobs, bs=32, t0=T0, cap=CAP); dt = time.time() - t1
    boot = PL.boot_for(trials, 500); summ = {j: PL.summarize(boot, trials, bt, bs_, r) for j, r in res.items()}
    PL.save_raw(OUT / "pilot_raw.npz", trials, bt, bs_, res)
    sig = PL.sigma_of(trials)
    all_T = {i: summ[f"patch|All|{i}"]["margin"]["T"] for i in PILOT_IDX}
    noop = {i: float(np.abs(res[f"patch|PreDiff|{i}"]["margin"] - bt["margin"]).max()) for i in PILOT_IDX}
    verify = {j: summ[j]["verify_max"] for j in summ if "verify_max" in summ[j] and "|S8" in j or "|all" in j or "|late" in j or "|nonS8" in j or "|rnd0" in j}
    gaps = {k: summ["patch|All|0"][k]["gap"] for k in ("margin", "c", "rj", "r22")}
    gate = dict(all_T_exact=bool(all(abs(v - 1) <= 0.02 for v in all_T.values())), noop_exact=bool(max(noop.values()) <= max(noise, 1e-3) + 1e-3), verify_ok=bool(max(summ[j]["verify_max"] for j in summ if "verify_max" in summ[j] and any(s in j for s in ("S8", "nonS8", "late", "all", "rnd0")) and j.count("|") == 3) <= 1e-3),
                gaps_ok=bool(abs(gaps["margin"]) >= 10 and abs(gaps["c"]) >= 2.0), n_jobs=len(jobs), n_trials=len(trials), seconds_jobs=dt, sec_per_forward=dt / (len(jobs) * len(trials)))
    gate["pass"] = bool(gate["all_T_exact"] and gate["noop_exact"] and gate["verify_ok"] and gate["gaps_ok"])
    json.dump(dict(census=census, noise_floor_margin=noise, gate=gate, all_T=all_T, noop_max_abs=noop, gaps=gaps, summary={j: summ[j] for j in summ}), open(OUT / "pilot.json", "w"), indent=1, default=float)
    est = gate["sec_per_forward"]; print("PILOT noise", noise, "gate", gate, flush=True)
    print("T(All) by index", {i: round(v, 4) for i, v in all_T.items()}, flush=True)
    print("projected stage-2 hours ~", est * (14 * 24 + 11) * 868 / 3600, flush=True)
    for c in ("G1", "G2", "G1G2", "Self", "NonSummary", "AllMinusG1G2", "rand0"):
        print(c, [round(summ[f"patch|{c}|{i}"]["margin"]["T"], 2) for i in PILOT_IDX], flush=True)
    log_budget("pilot", time.time() - T0)

elif stage == "stage2":
    P = PL.setup(); ctx = P["ctx"]; summary = {}
    for fmt in ("num_orig", "num_rev"):
        trials, c = PL.trial_sets(P, "A", fmt); print(fmt, c, flush=True)
        jobs = PL.job_list_patch(PL.STAGE2_CONDS, PL.IDX_ALL) + PL.job_list_patch(["All"], PL.SPARSE_ALL) + PL.job_list_patch(["PreDiff"], PL.NOOP_IDX)
        bt, bs_, res = run_jobs(ctx, trials, jobs, bs=32, t0=T0, cap=CAP)
        PL.save_raw(OUT / f"stage2_{fmt}.npz", trials, bt, bs_, res)
        boot = PL.boot_for(trials); summary[fmt] = dict(census=c, jobs={j: PL.summarize(boot, trials, bt, bs_, r) for j, r in res.items()})
        print(fmt, "done", f"{time.time() - T0:.0f}s", flush=True)
        json.dump(summary, open(OUT / "stage2.json", "w"), indent=1, default=float)
    on = onsets_A(); json.dump(on, open(OUT / "onsets_A.json", "w"), indent=1, default=float); print("onsets", json.dumps(on, default=float), flush=True)
    log_budget("stage2", time.time() - T0)

elif stage == "stage3":
    P = PL.setup(); ctx = P["ctx"]; on = onsets_A(); summary = {}
    for fmt in ("num_orig", "num_rev"):
        trials, c = PL.trial_sets(P, "A", fmt)
        idx, lon = pathway_indices(on[fmt]["G1G2"]); idx_single = sorted({lon, 22} - {None}) if lon is not None else [19, 22]
        jobs = [dict(name=f"path|G1G2|{i}|{f}", kind="patch", index=i, cond="G1G2", freeze=(None if f == "total" else f)) for i in idx for f in ["total"] + PL.FREEZES]
        jobs += [dict(name=f"path|{cd}|{i}|{f}", kind="patch", index=i, cond=cd, freeze=(None if f == "total" else f)) for cd in ("G1", "G2") for i in idx_single for f in ["total"] + PL.FREEZES]
        bt, bs_, res = run_jobs(ctx, trials, jobs, bs=32, t0=T0, cap=CAP, diag=False)
        PL.save_raw(OUT / f"stage3_{fmt}.npz", trials, bt, bs_, res)
        boot = PL.boot_for(trials); summary[fmt] = dict(indices=idx, onset=lon, jobs={j: PL.summarize(boot, trials, bt, bs_, r, keys=("margin", "c", "rj", "r22")) for j, r in res.items()})
        print(fmt, "stage3 done", idx, f"{time.time() - T0:.0f}s", flush=True)
        json.dump(summary, open(OUT / "stage3.json", "w"), indent=1, default=float)
    log_budget("stage3", time.time() - T0)

elif stage == "freeze":
    on = onsets_A(); cfgf = dict(files=PL.file_hashes(PL.FROZEN_FILES), onsets_A=on, grid_B=GRID_B, note="frozen before any Replication-B computation (docs/phase4_plan.md section 8)")
    json.dump(cfgf, open(OUT / "frozen_config.json", "w"), indent=1, default=float); print("frozen", json.dumps(on, default=float))

elif stage == "repl":
    fz = json.load(open(OUT / "frozen_config.json"))
    now = PL.file_hashes(PL.FROZEN_FILES)
    if now != fz["files"]: raise SystemExit("frozen files changed since the freeze: " + str([k for k in now if now[k] != fz["files"].get(k)]))
    P = PL.setup(); ctx = P["ctx"]; summary = {}
    for fmt in ("num_orig", "num_rev"):
        trials, c = PL.trial_sets(P, "B", fmt); print("B", fmt, c, flush=True)
        lon = fz["onsets_A"][fmt]["G1G2"]["sustained_both"]; idx = sorted({i for i in ([lon] if lon is not None else [16]) + [19, 22, 23]})
        jobs = PL.job_list_patch(["G1", "G2", "G1G2", "NonSummary", "AllMinusG1G2", "rand0", "rand1", "rand2"], GRID_B) + PL.job_list_patch(["All"], [0, 8, 16, 22, 23])
        jobs += [dict(name=f"path|G1G2|{i}|{f}", kind="patch", index=i, cond="G1G2", freeze=(None if f == "total" else f)) for i in idx for f in ["total", "S8", "nonS8", "late", "rnd0", "rnd1", "rnd2"]]
        bt, bs_, res = run_jobs(ctx, trials, jobs, bs=32, t0=T0, cap=CAP)
        PL.save_raw(OUT / f"repl_{fmt}.npz", trials, bt, bs_, res)
        boot = PL.boot_for(trials); summary[fmt] = dict(census=c, path_indices=idx, jobs={j: PL.summarize(boot, trials, bt, bs_, r) for j, r in res.items()})
        json.dump(summary, open(OUT / "repl.json", "w"), indent=1, default=float); print("B", fmt, "done", f"{time.time() - T0:.0f}s", flush=True)
    log_budget("repl", time.time() - T0)
else:
    raise SystemExit("stage not implemented yet: " + stage)
