"""Phase 8 S2: layerwise answer traces on confirm items (protocol section 5): logit-lens margins at blocks 26,27,29,31,33,35 and the real output, with the
final-norm scale frozen to the intact run as the passive-repair control, for intact / single / union ablations and matched controls (ranks 16, 32).
Candidate signatures only; no verdict."""
import argparse, json, pickle
import numpy as np, torch
from causal_subspaces.phase7.stage1 import class_gap
from causal_subspaces.phase8.s2lib import *

ap = argparse.ArgumentParser(); ap.add_argument("--hook", type=int, default=25); a = ap.parse_args(); h = a.hook
w = World(h); ctrl = torch.load(R2 / f"controls_h{h}.pt"); gold = np.array([i.gold for i in w.conf])
LAY = (26, 27, 29, 31, 33, 35, 36)
base = run_ext(w, w.conf, lens_layers=LAY)
gap = lambda cl: class_gap(cl, gold)
acc = lambda cl: float((cl.argmax(1) == gold).mean())
G = {l: gap(base["lens"][l]) for l in LAY}
res = dict(hook=h, layers=LAY, intact=dict(gap={l: G[l] for l in LAY}, acc={l: acc(base["lens"][l]) for l in LAY}), conds={})


def summarize(name, U):
    r = run_ext(w, w.conf, U, lens_layers=LAY, ln_frozen_from=base)
    res["conds"][name] = dict(gap={l: gap(r["lens"][l]) for l in LAY}, gap_lnfrozen={l: gap(r["lens_lnfrozen"][l]) for l in LAY},
                              acc={l: acc(r["lens"][l]) for l in LAY}, acc_lnfrozen={l: acc(r["lens_lnfrozen"][l]) for l in LAY},
                              final_gap=gap(r["cand_logits"]), rms_ratio={l: float(np.mean(r["rms"][l] / base["rms"][l])) for l in LAY})


Ks = sorted(w.chains)
for s in Ks:
    Us = w.chains[s]; K = len(Us)
    summarize(f"s{s}|carrier1", Us[0])
    for k in (4, 8):
        if k <= K:
            summarize(f"s{s}|union{k}", w.union(Us, range(1, k + 1)))
    for k in (4, 8):
        if k <= K:
            for fam in ("energy", "disp"):
                for j in range(4):
                    summarize(f"s{s}|{fam}{k}|{j}", ctrl[(fam, k, s, j)].cuda().float())
for k in (4, 8):
    for fam in ("haar", "onm"):
        for j in range(2):
            summarize(f"{fam}{k}|{j}", ctrl[(fam, k, "all", j)].cuda().float())
json.dump(res, open(R2 / f"trace_h{h}.json", "w"), indent=0, default=float)
print("S2 TRACE DONE", h)
