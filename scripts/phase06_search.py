"""Phase 0.6 Part B: small pilot search for logic-selective subspaces orthogonal to the frozen consensus verdict direction.
Objective and weights are pre-registered in docs/phase06_plan.md. Only the candidate basis is trained; the model is frozen.
Usage: python scripts/phase06_search.py --ranks 1 2 4 --seeds 0 1 --steps 300"""
import argparse
import json
import time

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5, phase06 as P6
from causal_subspaces.data.controls import build_controls
from causal_subspaces.data.logic import TRAIN_FAMILIES, build_contents
from causal_subspaces.experiment import get_reference, get_splits
from causal_subspaces.models.hooks import residual_intervention
from causal_subspaces.models.loading import encode, format_prompt, last_logits, load_model
from causal_subspaces.mvs.optimization import lr_at
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--tag", default="search")
ap.add_argument("--train-formats", nargs="+", default=["num_orig", "ab_orig"])
a = ap.parse_args()
OUT = ROOT / "results" / "phase06" / a.tag
OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
dev, idx = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
F_TRAIN = a.train_formats                   # search formats (pilot 1: num_orig, ab_orig; pilot 2: num_orig, ab_rev); all others are held out for evaluation
PAIRS_PER_STEP, CTRL_PER_TASK, MB = 12, 8, 8
SEARCH_SUBJECTS = ("anatomy", "philosophy")  # disjoint from the evaluation MMLU subjects
model, tok, rev = load_model(cfg["model"], dev)
splits = get_splits(cfg)
ref = get_reference(model, tok, cfg, rev, dev)
mu = ref["mu"].to(dev)
d_c = P6.load_consensus().to(dev)
contents = build_contents("ordinary")
tr_ids = sorted({e.content_id for e in splits["train"]}); va_ids = sorted({e.content_id for e in splits["val"]})
pairs = {"train": P5.build_pairs(contents, tr_ids, TRAIN_FAMILIES, tok), "val": P5.build_pairs(contents, va_ids, TRAIN_FAMILIES, tok)}
pairs["val"] = pairs["val"][::max(1, len(pairs["val"]) // 96)]
ctrl_sets = {"train": build_controls(101, 128, SEARCH_SUBJECTS, 16), "val": build_controls(102, 64, SEARCH_SUBJECTS, 8)}
CT = ("language_localizer", "prontoqa", "mmlu_binary")
print(f"{len(pairs['train'])} train pairs, {len(pairs['val'])} val pairs; controls " + str({k: len(v[CT[0]]['items']) for k, v in ctrl_sets.items()}), flush=True)

TOK = {f: (P5.single_token_id(tok, P5.FORMATS[f]["valid"]), P5.single_token_id(tok, P5.FORMATS[f]["invalid"])) for f in P5.FORMATS}
T12 = (P5.single_token_id(tok, "1"), P5.single_token_id(tok, "2"))

def margin_lpm(logits, ta, tb):
    la, lb = logits.gather(1, ta[:, None])[:, 0], logits.gather(1, tb[:, None])[:, 0]
    return la - lb, torch.logsumexp(torch.stack([la, lb], 1), 1) - torch.logsumexp(logits, -1)

@torch.no_grad()
def intact_pairs(plist):
    """cache[(i, fmt)] = (m_v, m_i, lpm_v, lpm_i) of the intact model."""
    cache = {}
    for f in F_TRAIN:
        for kind in ("valid_syl", "invalid_syl"):
            texts = [P5.build_prompt(f, p[kind]) for p in plist]
            rec = P5.collect(model, tok, texts, P5.FORMATS[f]["valid"], P5.FORMATS[f]["invalid"], device=dev, batch_size=32)
            m = rec["logit_aff"] - rec["logit_rej"]
            lpm = np.logaddexp(rec["logit_aff"], rec["logit_rej"]) - rec["lse"]
            for i in range(len(plist)):
                cache.setdefault((i, f), {})[kind] = (m[i], lpm[i])
    return cache

@torch.no_grad()
def intact_ctrl(items):
    rec = P5.collect(model, tok, [t for t, _ in items], "1", "2", device=dev, batch_size=32)
    return rec["logit_aff"] - rec["logit_rej"], np.logaddexp(rec["logit_aff"], rec["logit_rej"]) - rec["lse"]

cache = {s: intact_pairs(pairs[s]) for s in pairs}
ccache = {s: {t: intact_ctrl(ctrl_sets[s][t]["items"]) for t in CT} for s in ctrl_sets}
gap_scale = float(np.mean([c["valid_syl"][0] - c["invalid_syl"][0] for c in cache["train"].values()]))
print(f"intact mean pair gap {gap_scale:.1f}", flush=True)

def fwd(prompts, ta, tb, U):
    ids, mask, pos = encode(tok, prompts, dev)
    with residual_intervention(model, idx, "ablate", U, mu):
        lg = last_logits(model, ids, mask, pos)
    return margin_lpm(lg, ta, tb)

def pair_batch(pi, fm, split):
    prompts, ta, tb, m0, l0 = [], [], [], [], []
    for i, f in zip(pi, fm):
        for kind in ("valid_syl", "invalid_syl"):
            prompts.append(format_prompt(tok, P5.build_prompt(f, pairs[split][i][kind]), False))
            ta.append(TOK[f][0]); tb.append(TOK[f][1])
            m, l = cache[split][(i, f)][kind]; m0.append(m); l0.append(l)
    t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=dev)
    return prompts, t(ta), t(tb), t(m0, torch.float32), t(l0, torch.float32)

def ctrl_batch(split, task, ii):
    items = ctrl_sets[split][task]["items"]
    m0, l0 = ccache[split][task]
    prompts = [format_prompt(tok, items[i][0], False) for i in ii]
    ta = torch.full((len(ii),), T12[0], device=dev); tb = torch.full((len(ii),), T12[1], device=dev)
    return prompts, ta, tb, torch.tensor(m0[ii], device=dev, dtype=torch.float32), torch.tensor(l0[ii], device=dev, dtype=torch.float32)

def split_pair_terms(m, lpm, m0, l0):
    return m[0::2], m[1::2], m0[0::2], m0[1::2], lpm[0::2], lpm[1::2], l0[0::2], l0[1::2]

@torch.no_grad()
def validate(U):
    mv, mi, mv0, mi0, auc_n, sh = [], [], [], [], [], []
    losses = {"disc": [], "bias": []}
    for f in F_TRAIN:
        for s in range(0, len(pairs["val"]), 8):
            pi = list(range(s, min(s + 8, len(pairs["val"]))))
            pr, ta, tb, m0, l0 = pair_batch(pi, [f] * len(pi), "val")
            m, lpm = fwd(pr, ta, tb, U)
            L = P6.pair_loss(*split_pair_terms(m, lpm, m0, l0), gap_scale)
            losses["disc"].append(float(L["disc"])); losses["bias"].append(float(L["bias"]))
            mv += m[0::2].tolist(); mi += m[1::2].tolist(); mv0 += m0[0::2].tolist(); mi0 += m0[1::2].tolist()
    y = np.array([True] * len(mv) + [False] * len(mi))
    out = dict(val_disc=float(np.mean(losses["disc"])), val_bias=float(np.mean(losses["bias"])),
               val_auc_logic=P5.auc(np.array(mv + mi), y), val_auc_logic_intact=P5.auc(np.array(mv0 + mi0), y),
               val_gap_ratio=float((np.mean(mv) - np.mean(mi)) / (np.mean(mv0) - np.mean(mi0))),
               val_mean_margin_shift=float(np.mean(mv + mi) - np.mean(mv0 + mi0)))
    cl = []
    for t in CT:
        n = len(ctrl_sets["val"][t]["items"]); ms, m0s, ls, l0s = [], [], [], []
        for s in range(0, n, 8):
            ii = np.arange(s, min(s + 8, n)); pr, ta, tb, m0, l0 = ctrl_batch("val", t, ii)
            m, lpm = fwd(pr, ta, tb, U); ms += m.tolist(); m0s += m0.tolist(); ls += lpm.tolist(); l0s += l0.tolist()
        gold = np.array([g == "1" for _, g in ctrl_sets["val"][t]["items"]])
        cl.append(float(P6.ctrl_loss(torch.tensor(ms), torch.tensor(m0s), torch.tensor(ls), torch.tensor(l0s))))
        out[f"val_auc_{t}"] = P5.auc(np.array(ms), gold) if t != "mmlu_binary" or True else None
        out[f"val_auc_{t}_intact"] = P5.auc(np.array(m0s), gold)
        out[f"val_margin_shift_{t}"] = float(np.mean(ms) - np.mean(m0s))
    out["val_ctrl"] = float(np.mean(cl))
    out["val_total"] = out["val_disc"] + 0.5 * out["val_bias"] + 1.0 * out["val_ctrl"]
    return out

summary = []
for k in a.ranks:
    for sd in a.seeds:
        name = f"k{k}_s{sd}"
        if (OUT / name / "summary.json").exists():
            print("skip", name); continue
        (OUT / name).mkdir(exist_ok=True)
        seed_everything(1000 * sd + k)
        W = torch.nn.Parameter(P6.random_orth_subspace(ref["Qm"], d_c.cpu(), k, 500 + 10 * sd + k).to(dev).clone())
        opt = torch.optim.Adam([W], lr=1e-2)
        rng = np.random.default_rng(1000 * sd + k)
        log = open(OUT / name / "log.jsonl", "w")
        t0 = time.time()
        for step in range(a.steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.steps, 1e-2, 20.0)
            opt.zero_grad(set_to_none=True)
            U = P6.perp_basis(W, d_c); Ud = U.detach().requires_grad_(True)
            pi = rng.choice(len(pairs["train"]), PAIRS_PER_STEP, replace=False); fm = rng.choice(F_TRAIN, PAIRS_PER_STEP)
            rec = dict(step=step + 1, disc=0., bias=0., ctrl=0.)
            for s in range(0, PAIRS_PER_STEP, MB // 2):
                sl = slice(s, s + MB // 2)
                pr, ta, tb, m0, l0 = pair_batch(pi[sl], fm[sl], "train")
                m, lpm = fwd(pr, ta, tb, Ud)
                L = P6.pair_loss(*split_pair_terms(m, lpm, m0, l0), gap_scale)
                w = len(pi[sl]) / PAIRS_PER_STEP
                (w * L["total"]).backward()
                rec["disc"] += w * float(L["disc"]); rec["bias"] += w * float(L["bias"])
            for t in CT:
                ii = rng.choice(len(ctrl_sets["train"][t]["items"]), CTRL_PER_TASK, replace=False)
                pr, ta, tb, m0, l0 = ctrl_batch("train", t, ii)
                m, lpm = fwd(pr, ta, tb, Ud)
                Lc = P6.ctrl_loss(m, m0, lpm, l0)
                (Lc / len(CT)).backward()
                rec["ctrl"] += float(Lc) / len(CT)
            U.backward(Ud.grad)
            rec["grad_norm"] = float(W.grad.norm())
            opt.step()
            rec["total"] = rec["disc"] + 0.5 * rec["bias"] + rec["ctrl"]
            if (step + 1) % 50 == 0 or step + 1 == a.steps:
                rec.update(validate(P6.perp_basis(W.detach(), d_c)))
                print(name, {kk: round(v, 3) for kk, v in rec.items() if kk in ("step", "total", "disc", "bias", "ctrl", "val_total", "val_gap_ratio", "val_auc_logic", "val_auc_language_localizer", "val_auc_prontoqa", "val_mean_margin_shift")}, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        Uf = P6.perp_basis(W.detach(), d_c)
        torch.save(Uf.cpu(), OUT / name / "U.pt")
        fin = validate(Uf)
        fin.update(rank=k, seed=sd, steps=a.steps, seconds=time.time() - t0, orth_to_dc=float((d_c @ Uf).abs().max()))
        json.dump(fin, open(OUT / name / "summary.json", "w"), indent=1)
        summary.append(fin)
json.dump(dict(gap_scale=gap_scale, train_formats=F_TRAIN, runs=summary), open(OUT / "search_summary.json", "w"), indent=1)
print("done")
