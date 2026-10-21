"""Phase-0 robustness analysis from saved eval outputs (no model needed).
Seeds' aggregate accuracy, ablation prediction diagnostics (confusion / class freq / entropy), control grouping,
language-localizer per-example consistency, and cross-seed subspace overlap.
Usage: python scripts/analyze_robustness.py --seeds 0 1 2"""
import argparse
import itertools
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
import yaml

from causal_subspaces.experiment import get_controls
from causal_subspaces.utils.config import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
a = ap.parse_args()
OUT = ROOT / "results" / "phase0_robustness"
OUT.mkdir(exist_ok=True, parents=True)
SPL = ("items", "families", "nonwords")
CONDS = ("intact", "retain", "ablate")
NONLOGIC = ["word_recall_8", "word_recall_48", "language_localizer", "mmlu_binary", "stroop", "msit", "multiplication_1digit"]
ADJACENT = ["prontoqa"]
md = []

def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else float('nan'):.3f}"

M, P, S = {}, {}, {}
for s in a.seeds:
    rd = ROOT / "results" / f"mvs_rank2_seed{s}"
    M[s] = json.load(open(rd / "eval_metrics.json"))
    S[s] = json.load(open(rd / "train_summary.json"))
    P[s] = pd.read_json(rd / "eval_predictions.jsonl", lines=True)
    env = json.load(open(rd / "environment.json"))
    S[s]["dataset_hash"] = env["dataset_hash"]

# 1/2. per-seed + aggregate logic accuracy
md.append("## 1. Logic accuracy per seed (first-character accuracy; invalid rate in brackets for ablate)\n")
md.append("| seed | final KL (median last 5) | dataset hash | " + " | ".join(f"{c}/{sp}" for c in CONDS for sp in SPL) + " |")
md.append("|---|---|---|" + "---|" * 9)
for s in a.seeds:
    row = [f"{M[s]['conditions'][c]['logic'][sp]['overall']['acc']:.3f}" for c in CONDS for sp in SPL]
    md.append(f"| {s} | {S[s]['final_kl_median_last5']:.4f} | {S[s]['dataset_hash']} | " + " | ".join(row) + " |")
agg = {}
row = []
for c in CONDS:
    for sp in SPL:
        v = [M[s]["conditions"][c]["logic"][sp]["overall"]["acc"] for s in a.seeds]
        agg[f"{c}/{sp}"] = dict(values=v, mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else None)
        row.append(ms(v))
kl = [S[s]["final_kl_median_last5"] for s in a.seeds]
md.append(f"| **mean ± std** | {ms(kl)} | | " + " | ".join(row) + " |\n")
md.append("Per-form accuracy (mean over seeds) for retain / ablate on Items:\n")
forms = sorted(M[a.seeds[0]]["conditions"]["intact"]["logic"]["items"]["by_form"])
md.append("| cond | " + " | ".join(forms) + " |\n|---|" + "---|" * len(forms))
for c in CONDS:
    md.append(f"| {c} | " + " | ".join(ms([M[s]['conditions'][c]['logic']['items']['by_form'][f]['acc'] for s in a.seeds]) for f in forms) + " |")
md.append("")

# 3. prediction distributions
def H(p):
    return -sum(x * math.log2(x) for x in p if x > 0)

md.append("## 3. Prediction distributions\nGold is balanced 128/128 in every split. Confusion rows = gold (1 valid / 2 invalid), cols = first generated char.\n")
diag = {}
for s in a.seeds:
    p = P[s]
    for cond_name, cond_key in (("intact", "noop"), ("retain", "retain"), ("ablate", "ablate")):
        for sp in SPL:
            # intact rows are stored under condition 'noop' in evaluate_condition's predictions
            d = p[(p.condition == cond_key) & (p.task == f"logic/{sp}")]
            gold = np.where(d.valid, "1", "2")
            conf = pd.crosstab(pd.Series(gold, name="gold"), pd.Series(d.first_char.values, name="pred"))
            freq = Counter(d.first_char)
            pe = np.array(list(freq.values())) / len(d)
            pr = np.clip(d.p_1.values / np.clip(d.p_1.values + d.p_2.values, 1e-12, None), 1e-12, 1 - 1e-12)
            ent = float(np.mean([H([x, 1 - x]) for x in pr]))
            diag[(s, cond_name, sp)] = dict(
                n=len(d), acc=float(d.correct.mean()), pred_freq=dict(freq), class_entropy_bits=H(pe),
                mean_example_entropy_bits=ent, mean_p1_normalised=float(pr.mean()), std_p1_normalised=float(pr.std()),
                constant_prediction=len(freq) == 1,
                confusion={str(r): {str(k): int(v) for k, v in row.items()} for r, row in conf.iterrows()})
md.append("### Ablation: predicted-class frequencies and entropies\n")
md.append("| seed | split | acc | predicted-char counts | constant? | empirical class entropy (bits) | mean per-example entropy over {1,2} (bits) | mean P(1) | std P(1) | confusion (gold→pred) |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for s in a.seeds:
    for sp in SPL:
        d = diag[(s, "ablate", sp)]
        md.append(f"| {s} | {sp} | {d['acc']:.3f} | {d['pred_freq']} | {d['constant_prediction']} | {d['class_entropy_bits']:.3f} | "
                  f"{d['mean_example_entropy_bits']:.3f} | {d['mean_p1_normalised']:.3f} | {d['std_p1_normalised']:.3f} | {d['confusion']} |")
md.append("\nFor contrast, intact / retain class entropy and per-example entropy (seed 0):\n")
for cn in ("intact", "retain"):
    for sp in SPL:
        d = diag[(a.seeds[0], cn, sp)]
        md.append(f"- {cn}/{sp}: acc {d['acc']:.3f}, preds {d['pred_freq']}, class entropy {d['class_entropy_bits']:.3f}, per-example entropy {d['mean_example_entropy_bits']:.3f}")
# per-form behaviour of ablate (constant label would give 100/0 on forms by label)
md.append("\nAblate: accuracy by logical form (Items, per seed) — a constant predictor gives 1.0 on one label's forms and 0.0 on the other's:\n")
md.append("| seed | " + " | ".join(forms) + " |\n|---|" + "---|" * len(forms))
for s in a.seeds:
    md.append(f"| {s} | " + " | ".join(f"{M[s]['conditions']['ablate']['logic']['items']['by_form'][f]['acc']:.2f}" for f in forms) + " |")
md.append("")

# 5/6. controls
md.append("## 5/6. Controls (accuracy; lm_kl in nats). Logic-adjacent PrOntoQA is kept separate from clearly non-logical controls.\n")
def ctrl(s, c, t):
    v = M[s]["conditions"][c]["controls"][t]
    return v if t == "lm_kl" else v["overall"]["acc"]
for title, tasks in (("Clearly non-logical controls", NONLOGIC + ["lm_kl"]), ("Logic-adjacent control", ADJACENT)):
    md.append(f"### {title}\n\n| task | intact (per seed) | retain (per seed) | ablate (per seed) | ablate mean ± std | Δ ablate−intact mean |\n|---|---|---|---|---|---|")
    for t in tasks:
        iv = [ctrl(s, "intact", t) for s in a.seeds]; rv = [ctrl(s, "retain", t) for s in a.seeds]; av = [ctrl(s, "ablate", t) for s in a.seeds]
        md.append(f"| {t} | {', '.join(f'{x:.3f}' for x in iv)} | {', '.join(f'{x:.3f}' for x in rv)} | {', '.join(f'{x:.3f}' for x in av)} | {ms(av)} | {np.mean(av) - np.mean(iv):+.3f} |")
    md.append("")

# localizer per-example
cfg0 = yaml.safe_load((ROOT / "results" / f"mvs_rank2_seed{a.seeds[0]}" / "config.yaml").read_text())
items = get_controls(cfg0)["language_localizer"]["items"]
loc = pd.DataFrame(dict(idx=range(len(items)), text=[x[0].split("\n\n", 1)[1] for x in items], gold=[x[1] for x in items]))
for s in a.seeds:
    for cond, key in (("intact", "noop"), ("ablate", "ablate"), ("retain", "retain")):
        d = P[s][(P[s].condition == key) & (P[s].task == "control/language_localizer")].reset_index(drop=True)
        assert len(d) == len(loc) and (d.gold.values == loc.gold.values).all(), "item alignment failed"
        loc[f"{cond}_pred_s{s}"] = d.first_char.values
        loc[f"{cond}_p1_s{s}"] = d.p_1.values.round(4)
loc.to_csv(OUT / "localizer_per_example.csv", index=False)
md.append("## 5. Language localizer under ablation (gold 1 = English sentence, 2 = pseudoword string)\n")
md.append("| seed | acc | acc on gold=1 | acc on gold=2 | predicted-char counts | wrong rate on pseudoword (gold 2) | n wrong |\n|---|---|---|---|---|---|---|")
for s in a.seeds:
    pr = loc[f"ablate_pred_s{s}"]
    ok = pr == loc.gold
    md.append(f"| {s} | {ok.mean():.3f} | {ok[loc.gold == '1'].mean():.3f} (n={int((loc.gold=='1').sum())}) | {ok[loc.gold == '2'].mean():.3f} (n={int((loc.gold=='2').sum())}) | {dict(Counter(pr))} | {1 - ok[loc.gold == '2'].mean():.3f} | {int((~ok).sum())} |")
wrong = np.stack([(loc[f"ablate_pred_s{s}"] != loc.gold).values for s in a.seeds])
iw = (loc["intact_pred_s%d" % a.seeds[0]] != loc.gold).values
md.append(f"\nIntact errors: {int(iw.sum())}/{len(loc)}. Items wrong under ablation in ≥1 seed: {int(wrong.any(0).sum())}; in all {len(a.seeds)} seeds: {int(wrong.all(0).sum())}.")
for s1, s2 in itertools.combinations(range(len(a.seeds)), 2):
    ps1, ps2 = loc[f"ablate_pred_s{a.seeds[s1]}"], loc[f"ablate_pred_s{a.seeds[s2]}"]
    md.append(f"- seeds {a.seeds[s1]}/{a.seeds[s2]}: prediction agreement {np.mean(ps1 == ps2):.3f}; correctness agreement {np.mean((ps1 == loc.gold) == (ps2 == loc.gold)):.3f}; both-wrong {int(((ps1 != loc.gold) & (ps2 != loc.gold)).sum())}")
md.append(f"\nExamples wrong in all seeds (first 8):\n")
for _, r in loc[wrong.all(0)].head(8).iterrows():
    md.append(f"- gold {r.gold}: {r.text}")
md.append("")

# answer-bias diagnostic across all label-based controls under ablation
md.append("## Answer-channel diagnostic: ablated accuracy by gold label and predicted-label counts (all label-based tasks)\n")
md.append("| task | seed | acc | acc | acc | predicted counts | mean P(first label) |\n|---|---|---|---|---|---|---|".replace("| acc | acc | acc |", "| acc | acc gold=first label | acc gold=second label |"))
for t in ["language_localizer", "prontoqa", "mmlu_binary", "stroop", "word_recall_8", "word_recall_48", "msit"]:
    for s in a.seeds:
        d = P[s][(P[s].condition == "ablate") & (P[s].task == f"control/{t}")]
        labs = sorted(d.gold.unique())
        g1, g2 = (d[d.gold == l].correct.mean() for l in labs[:2])
        md.append(f"| {t} | {s} | {d.correct.mean():.3f} | {g1:.3f} | {g2:.3f} | {dict(Counter(d.first_char))} | {d[[c for c in d.columns if c.startswith('p_')][0]].mean():.3f} |")
md.append("")

# random-subspace specificity control (written by scripts/random_subspace_control.py)
rpath = OUT / "random_ablation.json"
if rpath.exists():
    R = json.load(open(rpath))["random"]
    md.append("## 4. Random rank-2 ablations at the same residual location (unoptimised; seed-0 mu)\n")
    md.append("gaussian = uniform random directions; topvar = random directions inside the top-2048 variance subspace. Δ = ablated − intact accuracy.\n")
    md.append("| subspace | n | " + " | ".join(f"{sp} acc mean [min,max]" for sp in SPL) + " | max |Δ| over splits |\n|---|---|---|---|---|---|")
    intact_l = {sp: M[a.seeds[0]]["conditions"]["intact"]["logic"][sp]["overall"]["acc"] for sp in SPL}
    for kind in ("gaussian", "topvar"):
        rr = [r for r in R if r["kind"] == kind]
        cells = []
        for sp in SPL:
            v = np.array([r["logic"][sp] for r in rr])
            cells.append(f"{v.mean():.3f} [{v.min():.3f},{v.max():.3f}]")
        mx = max(abs(r["logic"][sp] - intact_l[sp]) for r in rr for sp in SPL)
        md.append(f"| {kind} | {len(rr)} | " + " | ".join(cells) + f" | {mx:.3f} |")
    for s in a.seeds:
        md.append(f"| learned MVS (seed {s}) | 1 | " + " | ".join(f"{M[s]['conditions']['ablate']['logic'][sp]['overall']['acc']:.3f}" for sp in SPL) + " | " +
                  f"{max(abs(M[s]['conditions']['ablate']['logic'][sp]['overall']['acc'] - intact_l[sp]) for sp in SPL):.3f} |")
    md.append("\nControls under random ablation (mean over subspaces [min,max]) vs learned MVS (per seed) vs intact:\n")
    md.append("| task | intact | gaussian | topvar | learned s" + " | learned s".join(map(str, a.seeds)) + " |\n|---|---|---|---|" + "---|" * len(a.seeds))
    for t in NONLOGIC + ADJACENT + ["lm_kl"]:
        cell = []
        for kind in ("gaussian", "topvar"):
            v = np.array([r["controls"][t] for r in R if r["kind"] == kind])
            cell.append(f"{v.mean():.3f} [{v.min():.3f},{v.max():.3f}]")
        md.append(f"| {t} | {ctrl(a.seeds[0], 'intact', t):.3f} | " + " | ".join(cell) + " | " + " | ".join(f"{ctrl(s, 'ablate', t):.3f}" for s in a.seeds) + " |")
    ov = [max(r["overlap_with_learned"]) for r in R]
    md.append(f"\nLargest principal-angle cosine between each random subspace and the seed-0 learned MVS: max {max(ov):.3f}, mean {np.mean(ov):.3f}.\n")

# subspace overlap
Us = {s: torch.load(ROOT / "results" / f"mvs_rank2_seed{s}" / "checkpoints" / "step_1600.pt", weights_only=False) for s in a.seeds}
md.append("## Cross-seed subspace overlap (cosines of principal angles between the learned rank-2 bases; 1 = identical, ~sqrt(2/4096)=0.022 typical random)\n")
for s1, s2 in itertools.combinations(a.seeds, 2):
    sv = torch.linalg.svdvals(Us[s1]["U"].double().T @ Us[s2]["U"].double())
    md.append(f"- seeds {s1}/{s2}: {[round(float(x), 4) for x in sv]}")
mus = [Us[s]["mu"] for s in a.seeds]
md.append(f"- reference mu identical across seeds: {all(torch.equal(mus[0], m) for m in mus)}")
(OUT / "analysis_tables.md").write_text("\n".join(md))
json.dump(dict(aggregate=agg, final_kl=kl, diag={f"{k[0]}/{k[1]}/{k[2]}": v for k, v in diag.items()}), open(OUT / "analysis_summary.json", "w"), indent=2, default=str)
print("\n".join(md))
