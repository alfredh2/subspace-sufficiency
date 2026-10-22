"""Phase 1: criterion G3 (signed reconstruction, k=512) for the three per-seed shared directions, to evaluate 'replicates for >= 3 of 4 directions'."""
import json

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5
from causal_subspaces.experiment import get_splits
from causal_subspaces.models.loading import load_model
from causal_subspaces.neurons import concentration as C
from causal_subspaces.neurons.decompose import layerwise_pass, unit_gain
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

OUT = ROOT / "results/phase1"
cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
LS = json.load(open(OUT / "stage1.json"))["selection"]["layers"]
dirs = {k: v.to(dev) for k, v in torch.load(OUT / "directions.pt").items()}
model, tok, rev = load_model(cfg["model"], dev)
splits = get_splits(cfg)
rng = np.random.default_rng(0)
tr = splits["train"]
val, inv = [e for e in tr if e.valid], [e for e in tr if not e.valid]
pick = [val[i] for i in rng.choice(len(val), 500, replace=False)] + [inv[i] for i in rng.choice(len(inv), 500, replace=False)]
ho = [e for sp in ("items", "families", "nonwords") for e in splits[sp]]
data = {}
for part, ex in (("tr", pick), ("te", ho)):
    o = layerwise_pass(model, tok, [P5.build_prompt("num_orig", e.syllogism) for e in ex], np.array([e.valid for e in ex]), dirs["consensus"], idx_hook, device=dev, keep_acts_layers=LS)
    data[part] = ({l: o["acts"][l].float().to(dev) for l in LS}, torch.tensor([e.valid for e in ex], device=dev))
res = {}
for dn, d in dirs.items():
    g = {l: unit_gain(model, l, d) for l in LS}
    S = {p: torch.cat([data[p][0][l] * g[l] for l in LS], 1) for p in data}
    D = {p: S[p][data[p][1]].mean(0) - S[p][~data[p][1]].mean(0) for p in data}
    order = D["tr"].abs().argsort(descending=True)
    gen = np.random.default_rng(1)
    rnd = [C.signed_reconstruction(D["te"], torch.as_tensor(gen.choice(D["te"].numel(), 512, replace=False), device=dev))["frac"] for _ in range(20)]
    res[dn] = dict(k512=dict(**C.signed_reconstruction(D["te"], order[:512]), **C.per_example_r2(S["tr"], S["te"], order[:512])),
                   k16=C.signed_reconstruction(D["te"], order[:16]), random_frac_max=float(np.max(rnd)), random_frac_mean=float(np.mean(rnd)),
                   net_D_test=float(D["te"].sum()), cancellation=C.cancellation_index(D["te"]))
    print(dn, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[dn]["k512"].items()}, "random max", res[dn]["random_frac_max"], flush=True)
json.dump(res, open(OUT / "g3_directions.json", "w"), indent=1)
