"""Post-hoc diagnostic for the Stage-0 pilot (NOT part of the preregistered decision rule): are the top residual PCs behaviour-irrelevant? Reports cosine of each of the
top 20 residual PCs with d_c, their verdict AUC on held-out data, and the held-out concentration of their write by PC (ranked by cos)."""
import json
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from causal_subspaces import phase05 as P5
from causal_subspaces.experiment import get_splits
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt, load_model
from causal_subspaces.neurons.decompose import layerwise_pass, unit_gain
from causal_subspaces.phase7 import concentration as C7
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text()); seed_everything(0)
dev, idx = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]; LS = json.load(open(ROOT / "results/phase1/stage1.json"))["selection"]["layers"]; Q = 492
model, tok, rev = load_model(cfg["model"], dev); d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev).float()
splits = get_splits(cfg); rng = np.random.default_rng(0); tr = splits["train"]
val, inv = [e for e in tr if e.valid], [e for e in tr if not e.valid]
pick = [val[i] for i in rng.choice(len(val), 500, replace=False)] + [inv[i] for i in rng.choice(len(inv), 500, replace=False)]
ho = [e for sp in ("items", "families", "nonwords") for e in splits[sp]]
tx = {"tr": [P5.build_prompt("num_orig", e.syllogism) for e in pick], "te": [P5.build_prompt("num_orig", e.syllogism) for e in ho]}
y = {"tr": torch.tensor([e.valid for e in pick], device=dev), "te": torch.tensor([e.valid for e in ho], device=dev)}
acts = {p: {l: v.float().to(dev) for l, v in layerwise_pass(model, tok, tx[p], y[p].cpu().numpy(), d_c, idx, device=dev, keep_acts_layers=LS)["acts"].items()} for p in ("tr", "te")}
@torch.no_grad()
def resid(t):
    saved = model.model.layers; model.model.layers = saved[:idx]; res = []
    try:
        for s in range(0, len(t), 16):
            ids, mask, pos = encode(tok, [format_prompt(tok, a, False) for a in t[s:s + 16]], dev); sk = []
            with capture_residual(model, idx, sk): model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
            res.append(sk[0][:, -1].float())
    finally: model.model.layers = saved
    return torch.cat(res)
Htr, Hte = resid(tx["tr"]), resid(tx["te"]); mu = Htr.mean(0); _, S_, Vh = torch.linalg.svd(Htr - mu, full_matrices=False)
rows = []
for i in range(20):
    pc = Vh[i]; cos = float(pc @ d_c); auc = float(roc_auc_score(y["te"].cpu().numpy(), ((Hte - mu) @ pc).cpu().numpy())); auc = max(auc, 1 - auc)
    g = torch.stack([unit_gain(model, l, pc) for l in LS]); Dtr = torch.cat([(acts["tr"][l][y["tr"]].mean(0) - acts["tr"][l][~y["tr"]].mean(0)) * g[k] for k, l in enumerate(LS)])
    Dte = torch.cat([(acts["te"][l][y["te"]].mean(0) - acts["te"][l][~y["te"]].mean(0)) * g[k] for k, l in enumerate(LS)])
    s = C7.pooled_stats(Dtr, Dte, Q); rows.append(dict(pc=i, var_frac=float(S_[i] ** 2 / (S_ ** 2).sum()), cos_dc=cos, verdict_auc=auc, share_heldout=s["share_heldout"], neff=s["neff_heldout"], net_over_abs=s["net_over_abs"]))
    print(rows[-1], flush=True)
json.dump(rows, open(ROOT / "results/phase7/stage0_pc_diagnostic.json", "w"), indent=1)
