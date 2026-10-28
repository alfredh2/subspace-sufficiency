"""Phase 2 hard gate: attention recompute vs the original implementation on REAL Qwen3-8B padded batches (docs/plans/phase2_plan.md section 2).
Interpretation of the preregistered tolerances: per layer, relative L2 error over all real query positions and heads <= 0.02, median per-head relative
error <= 0.005, attention rows sum to 1 +- 1e-3, error of the d_c contribution <= 2 % of its std. Also measures the self-patch noise floor of
attention_patch on final-answer margins."""
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
from causal_subspaces.models.loading import encode, format_prompt, last_logits, load_model
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev, idx = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
splits = get_splits(cfg)
ctrl = build_controls(5, 64, ("anatomy",), 8)
texts = [P5.build_prompt("num_orig", e.syllogism) for e in splits["items"][:6]] + [t for t, _ in ctrl["language_localizer"]["items"][:5]] + [t for t, _ in ctrl["prontoqa"]["items"][:5]]
prompts = [format_prompt(tok, t, False) for t in texts]
ids, mask, pos = encode(tok, prompts, dev)
lens = mask.sum(1).tolist()
print("batch lengths", sorted(lens), "padded T", ids.shape[1], flush=True)
H, HD, KV = head_dims(model)
d = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
LAY = [0, 10, 19, 22]
res = {"batch_lengths": lens, "layers": {}}

def run(label):
    stash = A.AttnStash(model, LAY)
    zs = {}
    with torch.no_grad(), capture_z(model, LAY, zs):
        saved = model.model.layers; model.model.layers = saved[:max(LAY) + 1]
        model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
        model.model.layers = saved
    data = dict(stash.data); stash.close()
    return zs, data

zs, data = run("sdpa")
real = mask.bool()
for l in LAY:
    h, pe = data[l]
    with torch.no_grad():
        rec = A.recompute(model, l, h, pe, mask)
    B, T = h.shape[:2]
    orig = zs[l].float().view(B, T, H, HD)[real]            # (n_real, H, HD)
    new = rec["z"].float()[real]
    err = (new - orig)
    layer_rel = float(err.norm() / orig.norm())
    head_rel = (err.norm(dim=-1) / orig.norm(dim=-1).clamp_min(1e-9)).mean(0) if False else (err.reshape(-1, H, HD).norm(dim=-1).pow(2).sum(0).sqrt() / orig.reshape(-1, H, HD).norm(dim=-1).pow(2).sum(0).sqrt().clamp_min(1e-9))
    gain = head_gain(model, l, d)
    s_o = (orig * gain.view(H, HD)).sum(-1); s_n = (new * gain.view(H, HD)).sum(-1)
    dc_err = float((s_n - s_o).abs().max() / s_o.std())
    rows = float((rec["A"].sum(-1) - 1).abs().max())
    res["layers"][l] = dict(layer_rel_L2=layer_rel, median_head_rel_L2=float(head_rel.median()), max_head_rel_L2=float(head_rel.max()), dc_contribution_err_over_std=dc_err, attention_row_sum_max_dev=rows)
    print(l, {k: round(v, 5) for k, v in res["layers"][l].items()}, flush=True)

# eager cross-check (if the implementation can be switched at runtime)
res["eager"] = "not attempted"
try:
    model.config._attn_implementation = "eager"
    got = [getattr(m.self_attn.config, "_attn_implementation", None) for m in model.model.layers[:1]]
    if got[0] == "eager":
        zs_e, _ = run("eager")
        res["eager"] = {str(l): dict(eager_vs_sdpa_rel=float((zs_e[l].float()[real.unsqueeze(-1).expand(-1, -1, H * HD)] - zs[l].float()[real.unsqueeze(-1).expand(-1, -1, H * HD)]).norm() / zs[l].float()[real.unsqueeze(-1).expand(-1, -1, H * HD)].norm()),
                                     recompute_vs_eager_rel=float((A.recompute(model, l, *data[l], mask)["z"].float()[real] - zs_e[l].float().view(*zs_e[l].shape[:2], H, HD)[real]).norm() / zs_e[l].float().view(*zs_e[l].shape[:2], H, HD)[real].norm())) for l in LAY}
    else:
        res["eager"] = f"not switchable (reads {got[0]})"
    model.config._attn_implementation = "sdpa"
except Exception as e:  # noqa
    res["eager"] = f"error: {e!r}"
    model.config._attn_implementation = "sdpa"
print("eager:", res["eager"], flush=True)

# self-patch noise floor through attention_patch on the answer margins (heads of layers 19-22)
spec = {19: [0, 1, 2, 3], 20: [4, 5], 21: [8, 9], 22: [0, 1, 2, 3]}
ta, tb = P5.single_token_id(tok, "1"), P5.single_token_id(tok, "2")
with torch.no_grad():
    base = last_logits(model, ids, mask, pos)
    stash = A.AttnStash(model, list(spec)); saved = model.model.layers; model.model.layers = saved[:max(spec) + 1]
    model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False); model.model.layers = saved
    src = {l: A.recompute(model, l, *stash.data[l], mask) for l in spec}; stash.close()
    with A.attention_patch(model, spec, {l: dict(A=v["A"], v=v["v"]) for l, v in src.items()}, mask, idx, patch="AV"):
        selfp = last_logits(model, ids, mask, pos)
m0, m1 = (base[:, ta] - base[:, tb]), (selfp[:, ta] - selfp[:, tb])
res["self_patch_margin_noise"] = dict(max_abs=float((m1 - m0).abs().max()), mean_abs=float((m1 - m0).abs().mean()), margin_std=float(m0.std()))
print("self-patch noise", res["self_patch_margin_noise"], flush=True)
L = res["layers"]
res["pass"] = bool(all(v["layer_rel_L2"] <= 0.02 and v["median_head_rel_L2"] <= 0.005 and v["attention_row_sum_max_dev"] <= 1e-3 and v["dc_contribution_err_over_std"] <= 0.02 for v in L.values()))
json.dump(res, open(ROOT / "results/phase2/recompute_gate.json", "w"), indent=1)
print("GATE PASS" if res["pass"] else "GATE FAIL")
