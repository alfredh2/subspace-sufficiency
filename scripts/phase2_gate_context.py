"""SUPPLEMENTARY context for the failed recompute gate (does not change the gate verdict): distribution of the d_c-contribution error of the recompute path."""
import json

import numpy as np
import torch
import yaml

from causal_subspaces import phase05 as P5
from causal_subspaces.data.controls import build_controls
from causal_subspaces.experiment import get_splits
from causal_subspaces.heads import attention as A
from causal_subspaces.heads.ablation import capture_z
from causal_subspaces.heads.decompose import head_dims, head_gain
from causal_subspaces.models.loading import encode, format_prompt, load_model
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev = cfg["hardware"]["device"]
model, tok, rev = load_model(cfg["model"], dev)
splits = get_splits(cfg); ctrl = build_controls(5, 64, ("anatomy",), 8)
texts = [P5.build_prompt("num_orig", e.syllogism) for e in splits["items"][:6]] + [t for t, _ in ctrl["language_localizer"]["items"][:5]] + [t for t, _ in ctrl["prontoqa"]["items"][:5]]
ids, mask, pos = encode(tok, [format_prompt(tok, t, False) for t in texts], dev)
H, HD, KV = head_dims(model); d = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev); LAY = [0, 10, 19, 22]
stash = A.AttnStash(model, LAY); zs = {}
with torch.no_grad(), capture_z(model, LAY, zs):
    saved = model.model.layers; model.model.layers = saved[:23]; model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False); model.model.layers = saved
real = mask.bool(); out = {}
last = torch.zeros_like(real); last[:, -1] = True
for l in LAY:
    h, pe = stash.data[l]
    with torch.no_grad():
        rec = A.recompute(model, l, h, pe, mask)
    B, T = h.shape[:2]; gain = head_gain(model, l, d).view(H, HD)
    so = (zs[l].float().view(B, T, H, HD) * gain).sum(-1); sn = (rec["z"].float() * gain).sum(-1)
    e = (sn - so)[real].abs(); sd = so[real].std(); el = (sn - so)[last].abs(); sdl = so[last].std()
    out[l] = dict(all_positions=dict(rms_over_std=float((e ** 2).mean().sqrt() / sd), p99_over_std=float(e.quantile(.99) / sd), max_over_std=float(e.max() / sd)),
                  last_position=dict(rms_over_std=float((el ** 2).mean().sqrt() / sdl), max_over_std=float(el.max() / sdl)))
    print(l, out[l], flush=True)
json.dump(out, open(ROOT / "results/phase2/recompute_gate_context.json", "w"), indent=1)
