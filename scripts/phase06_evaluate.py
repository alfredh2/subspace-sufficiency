"""Phase 0.6 Part C: held-out evaluation of every search candidate + baselines (verdict direction, seed-0 MVS, rank/variance-matched
random subspaces orthogonal to d_c). Intervention = ablation of the candidate coordinates only (all other residual content intact).
Measures: AUC / class gap / mean margin / P(valid) per answer format on the held-out Items/Families/Nonwords, controls (localizer,
PrOntoQA, MMLU-binary, Stroop, word recall), WikiText KL, and matched counterfactual patching. Usage: python scripts/phase06_evaluate.py"""
import argparse
import json

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5, phase06 as P6
from causal_subspaces.data.logic import HELDOUT_FAMILIES, TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_controls, get_lm_passages, get_reference, get_splits, lm_kl
from causal_subspaces.models.loading import load_model
from causal_subspaces.mvs.checkpoint import load_checkpoint
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="search")
a = ap.parse_args()
OUT, SEARCH = ROOT / "results" / "phase06", ROOT / "results" / "phase06" / a.tag
EVAL = OUT / ("evaluation.json" if a.tag == "search" else f"evaluation_{a.tag}.json")
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx, bs = cfg["hardware"]["device"], cfg["mvs"]["residual_index"], cfg["hardware"]["eval_batch_size"]
model, tok, rev = load_model(cfg["model"], dev)
splits, controls = get_splits(cfg), get_controls(cfg)
ref = get_reference(model, tok, cfg, rev, dev)
mu = ref["mu"].to(dev)
d_c = P6.load_consensus()
passages = get_lm_passages(tok, cfg)
Qm, eig = ref["Qm"], ref["eigvals"]

# ---- candidates and baselines ------------------------------------------------------------------------------------
cands = {}
for run in sorted(SEARCH.glob("k*_s*")):
    if (run / "U.pt").exists():
        cands[f"cand_{run.name}"] = torch.load(run / "U.pt").float()
ranks = sorted({U.shape[1] for U in cands.values()})
base = {"consensus_dc": d_c[:, None], "mvs_s0_rank2": load_checkpoint(ROOT / "results/mvs_rank2_seed0/checkpoints/step_1600.pt")["U"]}
for k in ranks:
    tv = float(np.mean([P6.subspace_variance(U, Qm, eig) for n, U in cands.items() if U.shape[1] == k]))
    rs, vs = P6.variance_matched_random(Qm, eig, d_c, k, tv, 4)
    for i, (u, v) in enumerate(zip(rs, vs)):
        base[f"rand_k{k}_{i}"] = u
    print(f"rank {k}: target variance {tv:.1f}; matched random variances {np.round(vs, 1)}", flush=True)
allU = {**cands, **base}
meta = {n: dict(rank=int(U.shape[1]), variance=P6.subspace_variance(U.float(), Qm, eig), cos_to_dc=float((d_c @ U.float()).abs().max())) for n, U in allU.items()}

# ---- ablation metrics ---------------------------------------------------------------------------------------------
ex = [e for sp in ("items", "families", "nonwords") for e in splits[sp]]
gold = np.array([e.valid for e in ex]); spl = np.array([e.split for e in ex])
CTRL = ["language_localizer", "prontoqa", "mmlu_binary", "stroop", "word_recall_48"]

def evaluate(U):
    mode, Ud = ("noop", None) if U is None else ("ablate", U.to(dev))
    out = {}
    for fmt, f in P5.FORMATS.items():
        rec = P5.collect(model, tok, [P5.build_prompt(fmt, e.syllogism) for e in ex], f["valid"], f["invalid"], device=dev, batch_size=bs,
                         residual_index=idx, mode=mode, U=Ud, mu=mu)
        st = P5.answer_stats(tok, rec, gold, f["valid"], f["invalid"], f["lit_first"], spl)
        m = rec["logit_aff"] - rec["logit_rej"]
        st["auc_by_split"] = {s: P5.auc(m[spl == s], gold[spl == s]) for s in ("items", "families", "nonwords")}
        out[f"logic/{fmt}"] = st
    for t in CTRL:
        items = controls[t]["items"]
        rec = P5.collect(model, tok, [x[0] for x in items], "1", "2", device=dev, batch_size=bs, residual_index=idx, mode=mode, U=Ud, mu=mu)
        out[f"control/{t}"] = P5.answer_stats(tok, rec, np.array([x[1] == "1" for x in items]), "1", "2", "1")
    out["lm_kl"] = 0.0 if U is None else lm_kl(model, tok, passages, device=dev, residual_index=idx, mode="ablate", U=Ud, mu=mu)
    return out

results = {"meta": meta, "ablation": {}}
results["ablation"]["intact"] = evaluate(None)
for n, U in allU.items():
    results["ablation"][n] = evaluate(U)
    r = results["ablation"][n]
    print(f"{n:16s} k={meta[n]['rank']} logic AUC(num_orig)={r['logic/num_orig']['auc']:.3f} gap={r['logic/num_orig']['margin_mean_gold_aff'] - r['logic/num_orig']['margin_mean_gold_rej']:.1f} "
          f"P(valid)={r['logic/num_orig']['pred_aff']:.2f} loc_auc={r['control/language_localizer']['auc']:.3f} pronto_auc={r['control/prontoqa']['auc']:.3f} "
          f"mmlu_acc={r['control/mmlu_binary']['acc']:.3f} kl={r['lm_kl']:.3f}", flush=True)
json.dump(results, open(EVAL, "w"), indent=1)

# ---- matched patching ---------------------------------------------------------------------------------------------
contents = build_contents("ordinary")
test_ids = sorted({e.content_id for e in splits["items"]})
pairs = P5.build_pairs(contents, test_ids, TRAIN_FAMILIES + HELDOUT_FAMILIES, tok)
subs = {n: (U.to(dev), False) for n, U in allU.items()}
patch = {}
def go(group, trials, aff, rej, positions):
    res = P5.run_trials(model, tok, trials, aff, rej, subs, idx, positions=positions, device=dev, batch_size=16)
    ta, sa = np.array([t["tgt_aff"] for t in trials]), np.array([t["src_aff"] for t in trials])
    patch[group] = {f"{n}|{p}": P5.summarize_patching(res, ta, sa, n, p) for n in subs for p in positions}
    print("patched", group, flush=True)
for fmt in ("num_orig", "num_rev"):
    f = P5.FORMATS[fmt]
    tr = []
    for p in pairs:
        v, i = P5.build_prompt(fmt, p["valid_syl"]), P5.build_prompt(fmt, p["invalid_syl"])
        tr += [dict(tgt=i, src=v, tgt_aff=False, src_aff=True), dict(tgt=v, src=i, tgt_aff=True, src_aff=False)]
    go(f"match_{fmt}", tr, f["valid"], f["invalid"], ("all", "last"))
rng = np.random.default_rng(0)
def unmatched(a_, r_, n=150):
    tr = []
    for i, j in zip(rng.choice(len(a_), n), rng.choice(len(r_), n)):
        tr += [dict(tgt=r_[j], src=a_[i], tgt_aff=False, src_aff=True), dict(tgt=a_[i], src=r_[j], tgt_aff=True, src_aff=False)]
    return tr
for t in ("language_localizer", "prontoqa", "mmlu_binary", "stroop"):
    it = controls[t]["items"]
    go(f"unmatched_{t}", unmatched([x[0] for x in it if x[1] == "1"], [x[0] for x in it if x[1] == "2"]), "1", "2", ("last",))
lg = splits["items"] + splits["families"] + splits["nonwords"]
go("unmatched_logic", unmatched([P5.build_prompt("num_orig", e.syllogism) for e in lg if e.valid], [P5.build_prompt("num_orig", e.syllogism) for e in lg if not e.valid]), "1", "2", ("last",))
results["patching"] = patch
# subspace overlap between same-rank candidate solutions (search-seed reproducibility)
ov = {}
names = [n for n in cands]
for a_ in names:
    for b_ in names:
        if a_ < b_ and cands[a_].shape[1] == cands[b_].shape[1]:
            ov[f"{a_}|{b_}"] = torch.linalg.svdvals(cands[a_].double().T @ cands[b_].double()).tolist()
results["candidate_overlap"] = ov
json.dump(results, open(EVAL, "w"), indent=1)
print("saved")
