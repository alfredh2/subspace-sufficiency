"""Assemble Stage 1b verdicts and tables from the stored result files (no model). Writes stage1b/stage1b_verdicts.json and stage1b_report_tables.md."""
import json, glob
import numpy as np
from causal_subspaces.phase7.stage1b import FORMATS_1B, TASKS_1B
from causal_subspaces.utils.config import ROOT

R = ROOT / "results/phase7/stage1b"
sel = json.load(open(R / "selection.json"))["tasks"]
md, V = [], {}
f3 = lambda x: "nan" if x is None else "%.3f" % x
# ---- per task verdicts
md += ["## Task verdicts (tier 1, confirm contents, 5 seeds)", "", "| task | cell | intact acc (a) | 1b-S sufficiency | 1b-N necessity | 1b-G(i) transplant | 1b-G(ii) second format 1b-S | 1b-G | **1b verdict** |", "|---|---|---|---|---|---|---|---|---|"]
cellrows = ["", "## Confirm-set cell results (median over 5 seeds; intervals are hierarchical 95 %)", "",
            "| task/format | hook,rank | intact acc | KL med [CI] (seeds ok) | S | rel red. med [CI] | abs red. med [CI] (logits) | literal-10 ok | seeds N ok | N | transfer med [CI] | rand max | seeds T ok | T |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t in TASKS_1B:
    ch = sel[t]["choice"]
    if ch is None:
        V[t] = dict(verdict="FAIL (no sufficient dev cell)", S="not reached", N="not run", G="not run")
        md.append(f"| {t} | none | – | no dev cell with val KL <= 0.05 in both seeds (best dev cell h{sel[t]['best_cell']['hook']} r{sel[t]['best_cell']['rank']}) | not run | not run | not run | not run | **1b-FAIL (sufficiency)** |")
        continue
    gs = {}
    for fmt in FORMATS_1B[t]:
        f = R / "final" / f"{t}_{fmt}.json"
        if f.exists():
            gs[fmt] = json.load(open(f))["gates"]
    f0 = FORMATS_1B[t][0]; g0 = gs[f0]
    for fmt, g in gs.items():
        cellrows.append("| %s/%s | %d,%d | %.3f | %.4f [%.4f, %.4f] (%d/5) | %s | %.2f [%.2f, %.2f] | %.1f [%.1f, %.1f] | %s | %d/5 | %s | %s [%s] | %s | %s/5 | %s |" % (
            t, fmt, ch["hook"], ch["rank"], g["acc_intact"], g["median"]["kl"], g["S"]["ci_kl"][0], g["S"]["ci_kl"][1], g["S"]["n_seeds_ok"], g["S"]["verdict"],
            g["median"]["rel"], g["N"]["ci_rel"][0], g["N"]["ci_rel"][1], g["median"]["abs"], g["N"]["ci_abs"][0], g["N"]["ci_abs"][1], g["N"]["literal10_median_ok"],
            g["N"]["n_seeds_ok"], g["N"]["verdict"], f3(g["median"].get("transfer")), "%.2f, %.2f" % tuple(g["T"]["ci"]), f3(g["T"]["rand_max"]), g["T"]["n_seeds_ok"], g["T"]["verdict"]))
    second = [fm for fm in FORMATS_1B[t][1:] if fm in gs]
    s2 = gs[second[0]]["S"]["verdict"] if second else "NOT RUN"
    if second and t in ("L", "S"):
        s2 = "PASS" if all(gs[fm]["S"]["verdict"] == "PASS" for fm in second) else ("FAIL" if any(gs[fm]["S"]["verdict"] == "FAIL" for fm in second) else "INCONCLUSIVE")
    Tv = g0["T"]["verdict"]
    G = "FAIL" if (Tv == "FAIL" or s2 == "FAIL") else ("PASS" if (Tv == "PASS" and s2 == "PASS") else "INCONCLUSIVE")
    a_ok = all(g["intact_ok"] for g in gs.values())
    vs = [g0["S"]["verdict"], g0["N"]["verdict"], G]
    ver = "PASS" if (all(v == "PASS" for v in vs) and a_ok) else ("FAIL" if "FAIL" in vs or not a_ok else "INCONCLUSIVE")
    V[t] = dict(cell=ch, S=g0["S"]["verdict"], N=g0["N"]["verdict"], T=Tv, second_format_S=s2, G=G, intact_ok=a_ok, verdict=ver)
    md.append(f"| {t} | h{ch['hook']} r{ch['rank']} | {'ok' if a_ok else 'FAIL'} ({g0['acc_intact']:.3f}) | {g0['S']['verdict']} | {g0['N']['verdict']} | {Tv} | {s2} | {G} | **1b-{ver}** |")
n_pass = sum(1 for v in V.values() if v["verdict"] == "PASS")
md += ["", f"**Tier-1 count: {n_pass} of {len(TASKS_1B)} candidate tasks are 1b-PASS.** Four-task gate: " + ("MET (proposal for a multi-task comparison would need your approval)." if n_pass >= 4 else "NOT MET -> stop the multi-task fixed-basis programme and consolidate.")]
md += cellrows
# ---- Arm A diagnostics for G, N (extensions)
md += ["", "## Arm A diagnostics for tasks without a sufficient cell (dev data; plan section 3 labels)", "", "| task | run | train KL | val KL | cand-KL | retained acc | gap intact -> ablated | note |", "|---|---|---|---|---|---|---|---|"]
for t in ("G", "N"):
    b = sel[t]["best_cell"]; c = sel[t]["cells"][f"h{b['hook']}_r{b['rank']}"]
    md.append(f"| {t} | grid best cell h{b['hook']} r{b['rank']} (1600 steps, 2 seeds) | {np.median(c['train_kl']):.3f} | {np.median(c['val_kl']):.3f} | – | – | – | train slope {np.median(c['slope']):.2f} |")
    for kind in ("long4800", "lda_init", "tier2_candobj"):
        rows = [json.load(open(f)) for f in sorted(glob.glob(str(R / f"ext/{t}_{kind}_s*.json")))]
        if rows:
            md.append("| %s | %s | %.3f | %.3f | %.3f | %.2f | %.1f -> %.1f | %s |" % (t, kind + (" (TIER-2, diagnostic)" if kind.startswith("tier2") else ""), np.median([x["train"]["kl_last5_median"] for x in rows]),
                      np.median([x["val"]["kl_retain"] for x in rows]), np.median([x["val"]["cand_kl_retain"] for x in rows]), np.median([x["val"]["acc_retain"] for x in rows]),
                      rows[0]["val"]["gap_intact"], np.median([x["val"]["gap_ablate"] for x in rows]), "slope %.2f" % np.median([x["train_slope_last25"] for x in rows])))
    lr = R / f"ext/{t}_lda_ref.json"
    if lr.exists():
        x = json.load(open(lr)); md.append("| %s | non-fitted LDA reference (rank %d) | – | %.3f | %.3f | %.2f | – | |" % (t, x["rank"], x["val"]["kl_retain"], x["val"]["cand_kl_retain"], x["val"]["acc_retain"]))
rows = [json.load(open(f)) for f in sorted(glob.glob(str(R / "ext/Nfix_tier2_fixedpool_s*.json")))]
if rows:
    md.append("| N | N-fixedpool (TIER-2, own split with shared names) | %.3f | %.3f | %.3f | %.2f | %.1f -> %.1f | |" % (np.median([x["train"]["kl_last5_median"] for x in rows]), np.median([x["val"]["kl_retain"] for x in rows]),
              np.median([x["val"]["cand_kl_retain"] for x in rows]), np.median([x["val"]["acc_retain"] for x in rows]), rows[0]["val"]["gap_intact"], np.median([x["val"]["gap_ablate"] for x in rows])))
# ---- labels
md += ["", "### Three-way diagnostic labels (frozen rules, plan section 3 + A8)", ""]
for t in ("G", "N"):
    b = sel[t]["best_cell"]; c = sel[t]["cells"][f"h{b['hook']}_r{b['rank']}"]
    base = np.median(c["val_kl"]); ext = [np.median([json.load(open(f))["val"]["kl_retain"] for f in glob.glob(str(R / f"ext/{t}_{k}_s*.json"))]) for k in ("long4800", "lda_init")]
    ext_gain = max((base - e) / base for e in ext)
    trk = [np.median([json.load(open(f))["train"]["kl_last5_median"] for f in glob.glob(str(R / f"ext/{t}_{k}_s*.json"))]) for k in ("long4800",)]
    opt = bool(c["opt_flags"]["not_converged"] or c["opt_flags"]["seed_spread"] or ext_gain >= 0.30)
    gen = bool(np.median(c["train_kl"]) <= 0.25 * base and base - np.median(c["train_kl"]) >= 0.1)
    gen_long = bool(trk[0] <= 0.25 * np.median([json.load(open(f))["val"]["kl_retain"] for f in glob.glob(str(R / f"ext/{t}_long4800_s*.json"))]))
    cap = bool(sel[t]["rank8_all_train_kl_above_0p05"] and not opt)
    md.append(f"* **{t}** (best cell h{b['hook']} r{b['rank']}): OPT-LIMITED={opt} (not_converged={c['opt_flags']['not_converged']}, seed_spread={c['opt_flags']['seed_spread']}, best extension gain on val KL={ext_gain:.1%}); GENERALIZATION-LIMITED={gen or gen_long} (1600-step={gen}, 4800-step={gen_long}); CAPACITY-LIMITED={cap}.")
    V[t]["labels"] = dict(opt=opt, gen=bool(gen or gen_long), cap=cap, ext_gain=ext_gain)
# ---- Arm B / C dumps
for nm in ("M_sel", "M_h25_r2", "L_sel", "S_sel"):
    f = R / f"armB/{nm}.json"
    if f.exists():
        x = json.load(open(f))
        md += ["", f"## Arm B subject {nm} ({x['task']}/{x['fmt']} h{x['hook']} r{x['rank']}, seeds {x['seeds']}, confirm contents; intact gap {x['intact_gap']:.1f})", "", "```", json.dumps({k: x[k] for k in x if k not in ("seed_results",)}, indent=1, default=float)[:6000], "```", "```", json.dumps(x["seed_results"], indent=1, default=float)[:9000], "```"]
for t in ("L", "S"):
    f = R / f"armC/{t}.json"
    if f.exists():
        x = json.load(open(f))
        md += ["", f"## Arm C {t} (h{x['hook']} r{x['rank']})", "", "| train -> eval | KL med | rel red. | abs red. | transfer | rand max | S | N | T |", "|---|---|---|---|---|---|---|---|---|"]
        for fj in sorted(glob.glob(str(R / f"armC/{t}_*_to_*.json"))):
            g = json.load(open(fj)); nm = fj.split(f"{t}_")[-1][:-5]
            md.append("| %s | %.4f | %.2f | %.1f | %.2f | %.3f | %s | %s | %s |" % (nm, g["median"]["kl"], g["median"]["rel"], g["median"]["abs"], g["median"].get("transfer", float("nan")), g["T"]["rand_max"], g["S"]["verdict"], g["N"]["verdict"], g["T"]["verdict"]))
        md += ["", "Principal angles:", "```", json.dumps(x["angles"], indent=1)[:5000], "```", "Cross-format transplantation:", "```", json.dumps(x["cross_transplant"], indent=1)[:4000], "```"]
json.dump(V, open(R / "stage1b_verdicts.json", "w"), indent=1, default=float)
open(R / "stage1b_report_tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md[:60]))
