"""Phase 6 staged runner (docs/plans/phase6_plan.md). Usage: python scripts/phase6_run.py {pilot|stage2|stage3|freeze|stage4|stage5}"""
import json
import os
import sys
import time

import numpy as np
import torch

from causal_subspaces.data.logic import build_contents
from causal_subspaces.phase5 import pipeline as PL5
from causal_subspaces.phase6 import data as D6
from causal_subspaces.phase6 import runner as R6
from causal_subspaces.phase6 import stats as S6
from causal_subspaces.upstream import pipeline as UP
from causal_subspaces.upstream import stats as S
from causal_subspaces.upstream.data import build_special_trials
from causal_subspaces.utils.config import ROOT

stage = sys.argv[1]; OUT = ROOT / "results/phase6"; OUT.mkdir(exist_ok=True); T0 = time.time()
FMTS = ("num_orig", "num_rev")
SPARSE = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19]
DENSE = list(range(20))
NULL_IDX = [0, 8, 14, 17, 19]
FROZEN = ["src/causal_subspaces/phase6/early.py", "src/causal_subspaces/phase6/runner.py", "src/causal_subspaces/phase6/data.py", "src/causal_subspaces/phase6/stats.py",
          "src/causal_subspaces/phase5/patching.py", "src/causal_subspaces/phase5/runner.py", "src/causal_subspaces/upstream/patching.py", "src/causal_subspaces/upstream/stats.py",
          "src/causal_subspaces/upstream/data.py", "scripts/phase6_run.py", "docs/plans/phase6_plan.md", "results/phase6/set_c.json", "results/phase5/selection.json"]
sel = json.load(open(ROOT / "results/phase5/selection.json")); Hs = sel["H_star"]; mats = sel["null_matched"]
ACC = [(l, c) for l in (20, 21, 22) for c in ("attn", "mlp")]


def log_budget(name, secs):
    p = OUT / "budget.json"; b = json.load(open(p)) if p.exists() else {"soft_budget_gpu_hours": [3, 4], "stages": {}}
    b["stages"][name] = secs; b["total_hours"] = sum(b["stages"].values()) / 3600; json.dump(b, open(p, "w"), indent=1)


def O(group, i, typ="resid", prefix="o", **kw): return dict(name=f"{prefix}|{group}|{i}", kind="origin", spec=dict(type=typ, index=i, group=group), **kw)
def U(group, i): return dict(name=f"u|{group}|{i}", kind="u", spec=dict(type="resid", index=i, group=group))
def C(kind, typ, l, comps, group="Concl"):
    return dict(name=f"{kind}|{'+'.join(comps)}|{l}|{group}", kind=kind_of(kind), spec=dict(type=typ, layer=l, comps=tuple(comps), group=group))
def kind_of(k): return "origin" if k in ("ct", "cd") else "u"


def jobs_main(full=True):
    J = [O("Concl", i) for i in DENSE] + [U("Concl", i) for i in DENSE]
    grp = ["KV", "Fact", "G1", "Concl+G1", "Rule"] if full else ["KV", "Fact"]
    J += [O(g, i) for g in grp for i in SPARSE] + [U(g, i) for g in ("Fact", "KV") for i in SPARSE]
    J += [O(g, i) for g in ("Concl_same", "Concl_diff") for i in SPARSE]
    J += [O("Other", i) for i in (0, 10, 19)] + [O(f"rk6:{d}", i) for d in range(3) for i in NULL_IDX] + [O("Concl", i, typ="noise", prefix="on") for i in NULL_IDX]
    if full:
        J += [dict(O("Concl", i, prefix=f"oh{d}"), heads=mats[d]) for d in range(3) for i in (0, 14, 19)]
    for l in range(19):
        for comps in (("attn",), ("mlp",), ("attn", "mlp")):
            if not full and len(comps) == 2: continue
            J.append(C("ct", "comp_total", l, comps)); J.append(C("cd", "comp_direct", l, comps))
    return J


def jobs_ctrl():     # controls: groups Concl and KV on the sparse grid
    return [O(g, i) for g in ("Concl", "KV") for i in SPARSE]


def jobs_core_indep():    # sets B and C (frozen configuration)
    J = [O(g, i) for g in ("Concl", "KV", "Fact", "Concl_same", "Concl_diff") for i in SPARSE] + [U("Concl", i) for i in SPARSE]
    J += [O("Other", i) for i in (0, 10, 19)] + [O("rk6:0", i) for i in NULL_IDX] + [O("Concl", i, typ="noise", prefix="on") for i in NULL_IDX]
    for l in range(19):
        J.append(C("ct", "comp_total", l, ("attn",))); J.append(C("ct", "comp_total", l, ("mlp",)))
    return J


def pin_jobs():
    P = lambda name, pins, **kw: dict(name=name, kind="pin", heads=Hs, pins=pins, **kw)
    J = [P("free", [])]
    J += [P(f"pin1|{l}{c}", [(l, c)]) for l, c in ACC] + [P("pin_all", ACC)]
    J += [P(f"layer|{l}|both", [(l, "attn"), (l, "mlp")]) for l in (20, 21, 22)]
    J += [P("cum|20", [(20, "attn"), (20, "mlp")]), P("cum|2021", [(20, "attn"), (20, "mlp"), (21, "attn"), (21, "mlp")])]
    J += [P(f"loo|{l}{c}", [x for x in ACC if x != (l, c)]) for l, c in ACC]
    J += [P(f"tail|{k}", ACC, pin_rows=f"tail{k}") for k in range(9)] + [P("rows|G1", ACC, pin_rows="G1"), P("rows|G2", ACC, pin_rows="G2")]
    for a in PL5.ALPHAS:
        if a != 1.0: J += [dict(name=f"dose|free|{a}", kind="pin", heads=Hs, pins=[], alpha=a), dict(name=f"dose|pin|{a}", kind="pin", heads=Hs, pins=ACC, alpha=a)]
    J += [dict(name=f"dc|{part}|{tag}", kind="dcomp", part=part, pins=pins) for part in ("full", "dc", "orth") for tag, pins in (("free", []), ("pin", ACC))]
    return J


def prep(P, which, fmt):
    tok = P["tok"]
    if which == "C": return D6.set_c_trials(tok, fmt)
    tr, cen = UP.trial_sets(P, which, fmt); return D6.augment(tok, tr), cen


def run_set(ctx, trials, jobs, name, cap=None, comp_layers=range(19)):
    t = time.time(); bt, bs, res = R6.run_jobs(ctx, Hs, trials, jobs, bs=32, t0=t, cap=cap, comp_layers=comp_layers); UP.save_raw(OUT / f"{name}.npz", trials, bt, bs, res)
    print(name, len(trials), "trials", len(jobs), "jobs", f"{time.time() - t:.0f}s", flush=True); log_budget(name, time.time() - t); return bt, bs, res


def setup():
    P = PL5.setup(); c5 = P["ctx5"]; c5.w = torch.load(ROOT / "results/phase5/w_direction.pt").float().to(c5.dev); return P, c5


def tT(boot, trials, bt, bs, res, name, key="margin"):
    num, den = S.transfer_terms(UP.sigma_of(trials), res[name][key], bt[key], bs[key]); return boot.ratio(num, den)["point"]


if stage == "pilot":
    P, c5 = setup(); out = {}
    # (1) regression against Phase 5 on set A num_orig: value patch of Concl and of KV at index 19, and the residual patch of Concl at index 19 (unrestricted)
    trials, _ = prep(P, "A", "num_orig"); boot = UP.boot_for(trials, 300)
    J = [O("Concl", 19), O("KV", 19), U("Concl", 19), O("Other", 0), O("Other", 19), O("Other", 10), O("Concl_same", 0), O("Concl_same", 19)]
    bt, bs, res = run_set(c5, trials, J, "pilot_A")
    ph5 = UP.load_raw(ROOT / "results/phase5/stage2_num_orig.npz")[3]; ph5t = UP.load_raw(ROOT / "results/phase5/stage2_num_orig.npz")[1]
    reg = {}
    for nm, ref in (("o|Concl|19", "V|Hstar|Concl"), ("o|KV|19", "V|Hstar|Fact+Concl+G1"), ("u|Concl|19", "res19|Concl")):
        a = tT(boot, trials, bt, bs, res, nm); num, den = S.transfer_terms(UP.sigma_of(trials), ph5[ref]["margin"], ph5t["margin"], UP.load_raw(ROOT / "results/phase5/stage2_num_orig.npz")[2]["margin"])
        b = float(boot.ratio(num, den)["point"]); reg[nm] = dict(new=a, phase5=b, diff=a - b, max_abs_margin_diff=float(np.abs(res[nm]["margin"] - ph5[ref]["margin"]).max()), ok=bool(abs(a - b) <= 0.005))
    out["regression"] = reg
    out["equality"] = {nm: dict(maxdiff=float(res[nm]["maxdiff"].max()), eq_frac=float(res[nm]["eq_frac"].mean())) for nm in ("o|Other|0", "o|Other|10", "o|Other|19", "o|Concl_same|0", "o|Concl_same|19")}
    out["other_exactly_equal"] = bool(all(out["equality"][f"o|Other|{i}"]["maxdiff"] == 0 for i in (0, 10, 19)))
    # (2) real-model identities: early-exit stash vs full forward; direct-write identity; accounting; pin verification (validation contents)
    from causal_subspaces.heads.attention import AttnStash
    from causal_subspaces.models.loading import encode, format_prompt, last_logits
    from causal_subspaces.phase6.early import exit_before_attention, run_partial
    val, cv = prep(P, "val", "num_orig"); b = val[:32]
    ids, m, p = encode(P["tok"], [format_prompt(P["tok"], t["tgt"], False) for t in b], c5.dev); st = AttnStash(c5.model, [19])
    with torch.no_grad(): lg0 = last_logits(c5.model, ids, m, p)
    full = st.data[19][0].clone(); st.close(); store = {}; run_partial(c5.model, ids, m, p, exit_before_attention(c5.model, 19, store))
    with torch.no_grad(): lg1 = last_logits(c5.model, ids, m, p)
    out["early_exit"] = dict(stash_equal=bool(torch.equal(store["h"], full)), logits_after_equal=bool(torch.equal(lg0, lg1)))
    Jv = [O("KV", 19), O("Concl", 10), U("Concl", 10), C("cd", "comp_direct", 18, ("attn",)), C("ct", "comp_total", 18, ("attn",)), C("cd", "comp_direct", 12, ("mlp",))] + pin_jobs()[:10]
    bt2, bs2, res2 = run_set(c5, val, Jv, "pilot_val")
    acc = {nm: float(np.abs((res2[nm]["dC"][:, -1] - res2[nm]["dC"][:, 0]) - res2[nm]["W"].sum(1)).max()) for nm in ("free", "pin_all")}
    scale = float(np.abs(res2["free"]["dC"][:, 0] - bt2["dC"][:, 0]).mean())
    out["accounting_max_abs_error"] = acc; out["accounting_ok"] = bool(all(v <= 0.05 * max(scale, 1e-6) + 0.05 for v in acc.values()))
    out["pin_verify_max_abs"] = float(np.abs(res2["pin_all"]["W"] - bt2["W"]).max()); out["pin_ok"] = bool(out["pin_verify_max_abs"] < 1e-3)
    gap = float(np.abs(bs2["margin"] - bt2["margin"]).mean()); out["margin_gap_validation"] = gap
    # curves on validation (coarse) for the record
    bv = UP.boot_for(val, 300); out["validation_coarse"] = {nm: tT(bv, val, bt2, bs2, res2, nm) for nm in ("o|KV|19", "o|Concl|10", "u|Concl|10", "cd|attn|18|Concl", "ct|attn|18|Concl", "cd|mlp|12|Concl")}
    out["pass"] = bool(all(v["ok"] for v in reg.values()) and out["other_exactly_equal"] and out["early_exit"]["stash_equal"] and out["early_exit"]["logits_after_equal"] and out["accounting_ok"] and out["pin_ok"] and gap >= 10)
    # (3) control builders: yields on set A (num_orig) and set C
    ctrl = {}
    ids_a = sorted({e.content_id for e in UP.get_splits(P["cfg"])["items"]}); cons = build_contents("ordinary")
    fm, cfm = D6.build_fm_trials(P["tok"], cons, ids_a, UP.FAMS, "num_orig"); ctrl["Fm_A"] = dict(cfm, n_trials=len(fm))
    lx, clx = build_special_trials(P["tok"], cons, ids_a, UP.FAMS, "lexical", tag="phase6L"); ctrl["L_A"] = dict(clx, n_trials=len(lx))
    xs, cxs = PL5.build_cross_trials(P["tok"], cons, ids_a, UP.FAMS); ctrl["X_A"] = dict(cxs, n_trials=len(xs))
    for fmt in FMTS: tc, cc = prep(P, "C", fmt); ctrl[f"C_{fmt}"] = dict(n_trials=len(tc), underpowered=bool(len(tc) < 150))
    out["control_builders"] = ctrl
    json.dump(out, open(OUT / "pilot.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float)); log_budget("pilot", time.time() - T0)

elif stage == "stage2":
    P, c5 = setup()
    for fmt, full in (("num_orig", True), ("num_rev", False)):
        trials, cen = prep(P, "A", fmt); run_set(c5, trials, jobs_main(full), f"stage2_{fmt}")

elif stage == "stage3":
    P, c5 = setup(); cons = build_contents("ordinary"); ids_a = sorted({e.content_id for e in UP.get_splits(P["cfg"])["items"]})
    sets = {}
    for fmt in FMTS:
        fm, cfm = D6.build_fm_trials(P["tok"], cons, ids_a, UP.FAMS, fmt); sets[f"Fm_{fmt}"] = fm
    lx, _ = build_special_trials(P["tok"], cons, ids_a, UP.FAMS, "lexical", tag="phase6L"); sets["L_num_orig"] = D6.augment(P["tok"], lx)
    xs, _ = PL5.build_cross_trials(P["tok"], cons, ids_a, UP.FAMS); sets["X_num_orig"] = D6.augment(P["tok"], xs)
    for name, tr in sets.items(): run_set(c5, tr, jobs_ctrl(), f"stage3_{name}")

elif stage == "freeze":
    fz = dict(files=UP.file_hashes(FROZEN), note="frozen before any Replication-B / set-C computation (docs/plans/phase6_plan.md)"); json.dump(fz, open(OUT / "frozen_config.json", "w"), indent=1); print("frozen", len(fz["files"]))

elif stage == "stage4":
    fz = json.load(open(OUT / "frozen_config.json")); now = UP.file_hashes(FROZEN); amend = OUT / "frozen_config_amendment.json"
    expect = json.load(open(amend))["new_files"] if amend.exists() else fz["files"]
    if now != expect: raise SystemExit("frozen files changed: " + str([k for k in now if now[k] != expect.get(k)]))
    P, c5 = setup(); cons_by = {"B": build_contents("nonword", seed=P["cfg"].get("data_seed", P["cfg"]["seed"])), "C": json.load(open(OUT / "set_c.json"))["contents"]}
    for which in ("B", "C"):
        for fmt in FMTS:
            trials, cen = prep(P, which, fmt); run_set(c5, trials, jobs_core_indep(), f"stage4_{which}_{fmt}")
            if fmt == "num_orig":
                cs = cons_by[which]; lx, _ = build_special_trials(P["tok"], cs, list(range(48)), UP.FAMS, "lexical", tag=f"phase6L{which}"); xs, _ = PL5.build_cross_trials(P["tok"], cs, list(range(48)), UP.FAMS)
                for nm, tr in (("L", D6.augment(P["tok"], lx)), ("X", D6.augment(P["tok"], xs))): run_set(c5, tr, jobs_ctrl(), f"stage4_{which}_{nm}_{fmt}")

elif stage == "stage5":
    P, c5 = setup()
    for which, fmts in (("A", FMTS), ("B", FMTS)):
        for fmt in fmts:
            trials, cen = prep(P, which, fmt); run_set(c5, trials, pin_jobs(), f"stage5_{which}_{fmt}")
elif stage == "shapley":
    import zlib
    P, c5 = setup()
    def perm_jobs(n_perm=24):
        J = []
        for k in range(n_perm):
            rg = np.random.default_rng(zlib.crc32(f"phase6|shapley|{k}".encode())); order = [ACC[i] for i in rg.permutation(len(ACC))]
            for m in range(1, len(ACC)):
                J.append(dict(name=f"shap|{k}|{m}", kind="pin", heads=Hs, pins=order[:m], order=[f"{l}{c}" for l, c in order]))
        return J
    for fmt in FMTS:
        trials, cen = prep(P, "A", fmt); jobs = perm_jobs() + [dict(name="free", kind="pin", heads=Hs, pins=[]), dict(name="pin_all", kind="pin", heads=Hs, pins=ACC)]
        run_set(c5, trials, jobs, f"shapley_A_{fmt}")
    json.dump({f"{k}": [f"{l}{c}" for l, c in [ACC[i] for i in np.random.default_rng(zlib.crc32(f"phase6|shapley|{k}".encode())).permutation(len(ACC))]] for k in range(24)}, open(OUT / "shapley_orders.json", "w"))

elif stage == "q5":
    P, c5 = setup(); SPIN = lambda name, heads: dict(name=name, kind="pin", heads=Hs, zpins=[(20, list(heads))], pin_rows=ROWSN)
    ROWSN = "G1G2"
    val, cv = prep(P, "val", "num_orig"); jobs = [SPIN(f"h{h}", [h]) for h in range(32)] + [dict(name="free", kind="pin", heads=Hs, pins=[])]
    bt, bs, res = run_set(c5, val, jobs, "q5_val"); bv = UP.boot_for(val, 300)
    eff = [tT(bv, val, bt, bs, res, f"h{h}") - tT(bv, val, bt, bs, res, "free") for h in range(32)]; order = [int(h) for h in np.argsort(-np.array(eff))]
    json.dump(dict(effect_single_pin_validation=eff, order=order), open(OUT / "q5_selection.json", "w"), indent=1)
    KS = (1, 2, 3, 4, 6, 8, 16, 32)
    J = [SPIN(f"h{h}", [h]) for h in range(32)] + [SPIN(f"top{k}", sorted(order[:k])) for k in KS] + [dict(name="free", kind="pin", heads=Hs, pins=[]), dict(name="attn20", kind="pin", heads=Hs, pins=[(20, "attn")])]
    for which in ("A", "B"):
        for fmt in FMTS:
            trials, cen = prep(P, which, fmt); run_set(c5, trials, J, f"q5_{which}_{fmt}")

else:
    raise SystemExit("unknown stage")
