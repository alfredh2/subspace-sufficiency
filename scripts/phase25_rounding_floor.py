"""Phase 2.5 step 1b: is the worst-case recompute error just bf16 rounding? (a) quantisation floor: bf16-rounding the fp32 recompute output and comparing it with the
fp32 output; (b) per-element noise scale: |s_model - s_recompute| / (2^-9/sqrt(3) * sqrt(sum_i (gain_i z_i)^2)) -- the rms effect of independent relative
rounding of every z_i on the contribution s = gain.z (large when z is large but s is a small projection of it)."""
import json

import numpy as np
import torch
import yaml
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from causal_subspaces import phase05 as P5
from causal_subspaces.data.controls import build_controls
from causal_subspaces.experiment import get_splits
from causal_subspaces.heads import attention as A
from causal_subspaces.heads.decompose import head_dims, head_gain
from causal_subspaces.models.loading import encode, format_prompt, load_model
from causal_subspaces.utils.config import ROOT
from causal_subspaces.utils.reproducibility import seed_everything

cfg = yaml.safe_load((ROOT / "results/mvs_rank2_seed0/config.yaml").read_text())
seed_everything(0)
dev = cfg["hardware"]["device"]
model, tok, rev = load_model(cfg["model"], dev)
H, HD, KV = head_dims(model)
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
LAYERS = [0, 10, 19, 20, 21, 22]
S8 = {(s["layer"], s["head"]) for s in json.load(open(ROOT / "results/phase2/selected_heads.json"))["S8"]}
GAIN = {l: head_gain(model, l, d_c).view(H, HD) for l in LAYERS}
splits = get_splits(cfg); ctrl = build_controls(5, 128, ("anatomy", "philosophy"), 16)
logic = [P5.build_prompt("num_orig", e.syllogism) for e in splits["items"] + splits["families"] + splits["nonwords"]]
loc = [t for t, _ in ctrl["language_localizer"]["items"]]; pro = [t for t, _ in ctrl["prontoqa"]["items"]]
rng = np.random.default_rng(0)
out = {l: dict(obs=[], floor=[], ratio=[], ratio_s8=[], z_rel_obs=[], z_rel_floor=[]) for l in LAYERS}
for b in range(3):
    texts = [logic[i] for i in rng.choice(len(logic), 6, replace=False)] + [loc[i] for i in rng.choice(len(loc), 5, replace=False)] + [pro[i] for i in rng.choice(len(pro), 5, replace=False)]
    ids, mask2d, pos = encode(tok, [format_prompt(tok, t, False) for t in texts], dev); real = mask2d.bool(); B, T = ids.shape
    cap = {}
    hs = []
    for l in LAYERS:
        hs.append(model.model.layers[l].self_attn.register_forward_pre_hook(lambda mod, args, kwargs, l=l: cap.__setitem__(l, (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0], kwargs["position_embeddings"])), with_kwargs=True))
        hs.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, i, l=l: cap.__setitem__(("z", l), i[0].detach().float().view(B, T, H, HD))))
    with torch.no_grad():
        saved = model.model.layers; model.model.layers = saved[:23]; model.model(input_ids=ids, attention_mask=mask2d, position_ids=pos, use_cache=False); model.model.layers = saved
    for h in hs: h.remove()
    for l in LAYERS:
        hidden, pe = cap[l]
        with torch.no_grad():
            z0 = A.recompute(model, l, hidden, pe, mask2d)["z"]                    # bf16-cast (the gate's quantity)
            attn = model.model.layers[l].self_attn
            q = attn.q_norm(attn.q_proj(hidden).view(B, T, H, HD)).transpose(1, 2); k = attn.k_norm(attn.k_proj(hidden).view(B, T, KV, HD)).transpose(1, 2); v = attn.v_proj(hidden).view(B, T, KV, HD).transpose(1, 2)
            qr, kr = apply_rotary_pos_emb(q, k, *pe)
            Af = torch.softmax((qr.float() @ repeat_kv(kr, H // KV).float().transpose(2, 3)) * attn.scaling + A.additive_mask(mask2d), -1)
            zf = (Af @ repeat_kv(v, H // KV).float()).transpose(1, 2)                  # fp32 recompute (pre-rounding)
        zm = cap[("z", l)]
        g = GAIN[l]
        s_m, s_f, s_q = (zm * g).sum(-1), (zf * g).sum(-1), (zf.to(torch.bfloat16).float() * g).sum(-1)
        sd = s_f[real].std()
        obs = (s_m - s_f).abs() * real.unsqueeze(-1)                                    # model output (bf16) vs fp32 recompute
        floor = (s_q - s_f).abs() * real.unsqueeze(-1)                                  # pure bf16 output quantisation of the fp32 result
        scale = (2 ** -9) / np.sqrt(3) * torch.sqrt(((g * zf) ** 2).sum(-1))            # rms effect of independent relative rounding of each z_i
        ratio = (obs / scale.clamp_min(1e-12))[real.unsqueeze(-1).expand(-1, -1, H)]
        out[l]["obs"].append((float(obs.pow(2).sum().div(real.sum() * H).sqrt() / sd), float(obs.max() / sd)))
        out[l]["floor"].append((float(floor.pow(2).sum().div(real.sum() * H).sqrt() / sd), float(floor.max() / sd)))
        out[l]["ratio"].append((float(ratio.pow(2).mean().sqrt()), float(ratio.quantile(.999)), float(ratio.max())))
        out[l]["z_rel_obs"].append(float((zm - zf)[real].norm() / zf[real].norm())); out[l]["z_rel_floor"].append(float((zf.to(torch.bfloat16).float() - zf)[real].norm() / zf[real].norm()))
        s8m = torch.stack([(obs[..., h] / scale[..., h].clamp_min(1e-12))[real] for (ll, h) in S8 if ll == l]) if any(ll == l for ll, _ in S8) else None
        if s8m is not None: out[l]["ratio_s8"].append((float(s8m.pow(2).mean().sqrt()), float(s8m.max())))
    print("batch", b, flush=True)
summ = {}
for l in LAYERS:
    o = out[l]; summ[l] = dict(observed_model_vs_fp32recompute=dict(rms_over_std=float(np.mean([x[0] for x in o["obs"]])), max_over_std=float(np.max([x[1] for x in o["obs"]]))),
                              quantisation_floor_of_bf16_output=dict(rms_over_std=float(np.mean([x[0] for x in o["floor"]])), max_over_std=float(np.max([x[1] for x in o["floor"]]))),
                              z_rel_L2=dict(observed=float(np.mean(o["z_rel_obs"])), quantisation_floor=float(np.mean(o["z_rel_floor"]))),
                              error_over_rounding_noise_scale=dict(rms=float(np.mean([x[0] for x in o["ratio"]])), p999=float(np.max([x[1] for x in o["ratio"]])), max=float(np.max([x[2] for x in o["ratio"]]))),
                              S8_error_over_noise_scale=dict(rms=float(np.mean([x[0] for x in o["ratio_s8"]])) if o["ratio_s8"] else None, max=float(np.max([x[1] for x in o["ratio_s8"]])) if o["ratio_s8"] else None))
    print(l, json.dumps(summ[l]), flush=True)
json.dump(summ, open(ROOT / "results/phase25/rounding_floor.json", "w"), indent=1)
