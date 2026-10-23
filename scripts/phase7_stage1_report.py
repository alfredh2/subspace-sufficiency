"""Stage 1 report tables and gate verdicts from results/phase7/stage1 (frozen rules of plan section 12)."""
import json, glob
import numpy as np, torch
from causal_subspaces.phase7.tasks import FORMATS, REVERSED
from causal_subspaces.utils.config import ROOT

R = ROOT / "results/phase7/stage1"
base = json.load(open(R / "baseline_test.json"))
sweep = {t: json.load(open(R / f"sweep/choice_{t}.json")) for t in "LSGNM"}
med = lambda xs: float(np.median(xs))
out, lines = {}, []
P = lines.append
P("## Gate (a): intact test accuracy (frozen screen)\n\n| task/format | n | primary acc | 95% CI | threshold | pass |\n|---|---|---|---|---|---|")
for t in "LSGNM":
    for f in FORMATS[t]:
        b = base[f"{t}/{f}"]; th = 0.8 if t == "M" else 0.9
        P(f"| {t}/{f} | {b['n']} | {b['acc']:.3f} | {b['ci'][0]:.3f}-{b['ci'][1]:.3f} | {th} | {b['acc'] >= th} |")
P("\n## Sweep (val, 300 steps, seed 0): frozen choice\n\n| task | passing configs of 12 | choice (hook, rank) | passes literal 10-logit floor |\n|---|---|---|---|")
for t in "LSGNM":
    s = sweep[t]
    P(f"| {t} | {s['n_pass']} | {s['choice'] and (s['choice']['hook'], s['choice']['rank'])} | {s['literal_choice'] is not None} |")
verd = {}
for t in "LS":
    fm = {}
    for f in FORMATS[t]:
        j = json.load(open(R / f"final/{t}_{f}.json"))
        rows = j["rows"]
        gap_i = [r["test"]["gap_intact"] for r in rows]
        red = [r["test"]["gap_intact"] - r["test"]["gap_ablate"] for r in rows]
        rel = [a / b for a, b in zip(red, gap_i)]
        floor_ok = [x >= 0.5 * g and x >= min(10, 0.5 * g) for x, g in zip(red, gap_i)]
        lit_ok = [x >= 0.5 * g and x >= 10 for x, g in zip(red, gap_i)]
        klm = med([r["test"]["kl_retain"] for r in rows])
        acc_ratio = med([r["test"]["acc_retain"] / r["test"]["acc_intact"] for r in rows])
        tr_all = [j["transplant"][f"s{s}|all"] for s in j["seeds"]]
        tr_last = [j["transplant"][f"s{s}|last"] for s in j["seeds"]]
        rnd_all = [j["transplant"][f"rand{k}|all"][0] for k in range(20)]
        rnd_last = [j["transplant"][f"rand{k}|last"][0] for k in range(20)]
        cons = float(np.mean(list(j["consensus_cos"].values())))
        g = {}
        g["b_kl"] = klm <= 0.05; g["b_acc"] = acc_ratio >= 0.9
        g["b"] = g["b_kl"] and g["b_acc"]
        g["b_incon"] = g["b"] and (med([r["ci_kl_retain"][1] for r in rows]) > 0.05 or med([r["ci_acc_ratio"][0] for r in rows]) < 0.9)
        g["c"] = med(rel) >= 0.5 and sum(floor_ok) >= 2
        g["c_lit"] = sum(lit_ok) >= 2
        g["c_incon"] = g["c"] and med([r["ci_rel_reduction"][0] for r in rows]) < 0.5
        g["d_tr"] = med([x[0] for x in tr_all]) >= 0.25; g["d_rand"] = float(np.mean(rnd_all)) <= 0.05
        g["d"] = g["d_tr"] and g["d_rand"]
        g["d_incon"] = g["d"] and med([x[1] for x in tr_all]) < 0.25
        g["e_cons"] = cons >= 0.7
        fm[f] = dict(g=g, klm=klm, acc_i=med([r["test"]["acc_intact"] for r in rows]), acc_r=med([r["test"]["acc_retain"] for r in rows]),
                     acc_a=med([r["test"]["acc_ablate"] for r in rows]), rel=med(rel), red=med(red), gap=med(gap_i), tr_all=[x[0] for x in tr_all],
                     tr_all_ci=[x[1:] for x in tr_all], tr_last=[x[0] for x in tr_last], rnd_all=(float(np.mean(rnd_all)), float(np.max(rnd_all))),
                     rnd_last=(float(np.mean(rnd_last)), float(np.max(rnd_last))), cons=cons, cons_min=min(j["consensus_cos"].values()),
                     rel_seeds=rel, ci_rel=[r["ci_rel_reduction"] for r in rows], hook=j["hook"], rank=j["rank"])
    prim = FORMATS[t][0]
    nb_ok = [f for f in fm if fm[f]["g"]["b"] and fm[f]["g"]["d"]]
    e = len(nb_ok) >= 2 and (t not in REVERSED or REVERSED[t] in nb_ok) and all(fm[f]["g"]["e_cons"] for f in fm)
    a_ok = all(base[f"{t}/{f}"]["acc"] >= 0.9 for f in FORMATS[t])
    gp = fm[prim]["g"]
    allp = a_ok and gp["b"] and gp["c"] and gp["d"] and e
    anyf = (not a_ok) or (not gp["b"]) or (not gp["c"]) or (not gp["d"]) or (not e)
    incon = allp and (gp["b_incon"] or gp["c_incon"] or gp["d_incon"])
    verd[t] = dict(a=a_ok, e=e, nb_ok=nb_ok, status="INCONCLUSIVE" if incon else "PASS" if allp else "FAIL")
    out[t] = dict(formats=fm, verdict=verd[t])
P("\n## Final MVS gates on TEST (seed-median; 3 seeds)\n\n| task/format | hook,rank | KL ret | acc I / R / A | gap I | gap reduction (rel) | transplant all (3 seeds) | random all mean / max | transplant last | consensus cos mean / min | b | c | c literal | d |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in "LS":
    for f, x in out[t]["formats"].items():
        g = x["g"]
        P(f"| {t}/{f} | {x['hook']},{x['rank']} | {x['klm']:.4f} | {x['acc_i']:.3f} / {x['acc_r']:.3f} / {x['acc_a']:.3f} | {x['gap']:.1f} | {x['red']:.1f} ({x['rel']:.2f}; seeds {', '.join('%.2f' % r for r in x['rel_seeds'])}) | "
          f"{', '.join('%.2f' % v for v in x['tr_all'])} | {x['rnd_all'][0]:.3f} / {x['rnd_all'][1]:.3f} | {', '.join('%.2f' % v for v in x['tr_last'])} | {x['cons']:.2f} / {x['cons_min']:.2f} | {g['b']} | {g['c']} | {g['c_lit']} | {g['d']} |")
P("\n## Task verdicts\n\n| task | gate a | primary-format b,c,d | >=2 formats (b&d) incl. reversed, consensus>=0.7 | verdict |\n|---|---|---|---|---|")
for t in "LS":
    g = out[t]["formats"][FORMATS[t][0]]["g"]
    P(f"| {t} | {verd[t]['a']} | {g['b']}, {g['c']}, {g['d']} | {verd[t]['e']} ({verd[t]['nb_ok']}) | {verd[t]['status']} |")
for t in "GNM":
    P(f"| {t} | True | - | - | FAIL: no sweep configuration passed (KL and/or ablation) |")
# direction geometry
P("\n## Task-direction geometry (cosines of consensus directions; only comparable at the same hook)")
D = {}
for t in "LS":
    for f in FORMATS[t]:
        D[(t, f)] = (torch.load(R / f"final/{t}_{f}_directions.pt")["dtask"], out[t]["formats"][f]["hook"])
keys = list(D)
P("\n| | " + " | ".join(f"{t}/{f}" for t, f in keys) + " |\n|" + "---|" * (len(keys) + 1))
for a in keys:
    P(f"| {a[0]}/{a[1]} | " + " | ".join(("%.2f" % float(D[a][0] @ D[b][0])) if D[a][1] == D[b][1] else "n/a (hook %d vs %d)" % (D[a][1], D[b][1]) for b in keys) + " |")
json.dump(out, open(R / "stage1_verdicts.json", "w"), indent=1, default=float)
open(R / "stage1_report_tables.md", "w").write("\n".join(lines))
print("\n".join(lines))
