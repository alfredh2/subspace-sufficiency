"""Phase 2.5 step 1: diagnose the worst-case attention-recompute error on real padded Qwen3-8B batches (docs/phase25_plan.md).
All comparisons use the SAME bf16 layer input. References: the model's own SDPA output (captured at o_proj), HF eager on identical q/k/v, my own SDPA calls,
and an FP32 reference (projections, q/k norm, RoPE and attention in fp32)."""
import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward, repeat_kv

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
dev, idx_hook = cfg["hardware"]["device"], cfg["mvs"]["residual_index"]
model, tok, rev = load_model(cfg["model"], dev)
H, HD, KV = head_dims(model)
d_c = torch.load(ROOT / "results/phase1/directions.pt")["consensus"].to(dev)
LAYERS = [0, 10, 19, 20, 21, 22]
S8 = [(s["layer"], s["head"]) for s in json.load(open(ROOT / "results/phase2/selected_heads.json"))["S8"]]
GAIN = {l: head_gain(model, l, d_c).view(H, HD) for l in LAYERS}
print("model attention implementation:", model.config._attn_implementation, flush=True)

# ---- prompts: 3 mixed padded batches of 16 + 1 unpadded batch of equal-length logic prompts -------------------------------------------------
splits = get_splits(cfg); ctrl = build_controls(5, 128, ("anatomy", "philosophy"), 16)
logic = [P5.build_prompt("num_orig", e.syllogism) for e in splits["items"] + splits["families"] + splits["nonwords"]]
loc = [t for t, _ in ctrl["language_localizer"]["items"]]; pro = [t for t, _ in ctrl["prontoqa"]["items"]]
rng = np.random.default_rng(0)
batches = []
for b in range(3):
    texts = [logic[i] for i in rng.choice(len(logic), 6, replace=False)] + [loc[i] for i in rng.choice(len(loc), 5, replace=False)] + [pro[i] for i in rng.choice(len(pro), 5, replace=False)]
    batches.append(texts)
lens = Counter(len(tok(format_prompt(tok, t, False), add_special_tokens=False).input_ids) for t in logic)
mode_len = lens.most_common(1)[0][0]
eq = [t for t in logic if len(tok(format_prompt(tok, t, False), add_special_tokens=False).input_ids) == mode_len][:12]
batches.append(eq)

class Grab:
    def __init__(self, layers):
        self.d, self.h = {}, []
        for l in layers:
            self.h.append(model.model.layers[l].self_attn.register_forward_pre_hook(
                lambda mod, args, kwargs, l=l: self.d.__setitem__(l, dict(hidden=(args[0] if args else kwargs["hidden_states"]).detach(), pe=kwargs["position_embeddings"], mask=kwargs.get("attention_mask"))), with_kwargs=True))
            self.h.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, i, l=l: self.d.setdefault(("z", l), i[0].detach())))
    def close(self):
        for x in self.h: x.remove()

def rms32(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()

def variants(attn, hidden, pe, mask2d):
    B, T, _ = hidden.shape
    mask = A.additive_mask(mask2d)
    keep = (mask > -1).squeeze(1)                                                    # (B,T,T) bool, True = keep
    q = attn.q_norm(attn.q_proj(hidden).view(B, T, H, HD)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(hidden).view(B, T, KV, HD)).transpose(1, 2)
    v = attn.v_proj(hidden).view(B, T, KV, HD).transpose(1, 2)
    cos, sin = pe
    qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
    out = {}
    kk, vv = repeat_kv(kr, H // KV), repeat_kv(v, H // KV)
    A0 = torch.softmax((qr.float() @ kk.float().transpose(2, 3)) * attn.scaling + mask, -1)
    out["R0_recompute_fp32_attention"] = (A0 @ vv.float()).transpose(1, 2)
    out["R1_probs_cast_bf16"] = (A0.to(hidden.dtype) @ vv).float().transpose(1, 2)
    ez, _ = eager_attention_forward(attn, qr, kr, v, mask.to(hidden.dtype), scaling=attn.scaling)
    out["R2_hf_eager_same_qkv"] = ez.float()
    out["R3a_sdpa_float_mask"] = F.scaled_dot_product_attention(qr, kk, vv, attn_mask=mask.to(hidden.dtype), scale=attn.scaling).transpose(1, 2).float()
    out["R3b_sdpa_bool_mask"] = F.scaled_dot_product_attention(qr, kk, vv, attn_mask=keep[:, None], scale=attn.scaling).transpose(1, 2).float()
    # GQA: explicit per-group loop instead of repeat_kv (must be identical to R0)
    zl = torch.zeros_like(out["R0_recompute_fp32_attention"])
    for g in range(KV):
        qs = qr[:, g * (H // KV):(g + 1) * (H // KV)].float()
        a = torch.softmax((qs @ kr[:, g:g + 1].float().transpose(2, 3)) * attn.scaling + mask, -1)
        zl[:, :, g * (H // KV):(g + 1) * (H // KV)] = (a @ v[:, g:g + 1].float()).transpose(1, 2)
    out["R0g_gqa_group_loop"] = zl
    # RoPE in fp32 (inputs to RoPE are the bf16 normed q/k)
    qf, kf = apply_rotary_pos_emb(q.float(), k.float(), cos.float(), sin.float())
    A5 = torch.softmax((qf @ repeat_kv(kf, H // KV).transpose(2, 3)) * attn.scaling + mask, -1)
    out["R0r_rope_fp32"] = (A5 @ vv.float()).transpose(1, 2)
    # FP32 reference
    h32 = hidden.float(); eps = attn.q_norm.variance_epsilon
    q32 = rms32(F.linear(h32, attn.q_proj.weight.float()).view(B, T, H, HD), attn.q_norm.weight, eps).transpose(1, 2)
    k32 = rms32(F.linear(h32, attn.k_proj.weight.float()).view(B, T, KV, HD), attn.k_norm.weight, eps).transpose(1, 2)
    v32 = F.linear(h32, attn.v_proj.weight.float()).view(B, T, KV, HD).transpose(1, 2)
    q32, k32 = apply_rotary_pos_emb(q32, k32, cos.float(), sin.float())
    A32 = torch.softmax((q32 @ repeat_kv(k32, H // KV).transpose(2, 3)) * attn.scaling + mask, -1)
    out["FP32_reference"] = (A32 @ repeat_kv(v32, H // KV)).transpose(1, 2)
    return out

def contrib(Z, l):                                # (B,T,H,HD) -> (B,T,H)
    return (Z * GAIN[l]).sum(-1)

def stats(Zx, Zy, real, l, heads=None):
    e = (Zx - Zy)[real]; ref = Zy[real]
    sx, sy = contrib(Zx, l)[real], contrib(Zy, l)[real]
    de = (sx - sy).abs()
    out = dict(z_rel_L2=float(e.norm() / ref.norm()), z_rms=float(e.pow(2).mean().sqrt()), z_max=float(e.abs().max()),
               s_rms_over_std=float(de.pow(2).mean().sqrt() / sy.std()), s_p99_over_std=float(de.flatten().quantile(.99) / sy.std()) if de.numel() < 16_000_000 else float("nan"), s_max_over_std=float(de.max() / sy.std()),
               s_abs_max=float(de.max()), s_abs_rms=float(de.pow(2).mean().sqrt()))
    return out

res = {"attn_impl": model.config._attn_implementation, "layers": {}, "s8": {}, "worst": [], "mask_info": None}
worst_rows, per_head_max = [], {}
agg = {}
for bi, texts in enumerate(batches):
    ids, mask2d, pos = encode(tok, [format_prompt(tok, t, False) for t in texts], dev)
    real = mask2d.bool()
    grab = Grab(LAYERS)
    with torch.no_grad():
        saved = model.model.layers; model.model.layers = saved[:max(LAYERS) + 1]
        model.model(input_ids=ids, attention_mask=mask2d, position_ids=pos, use_cache=False)
        model.model.layers = saved
    for l in LAYERS:
        g = grab.d[l]
        if res["mask_info"] is None:
            m = g["mask"]; res["mask_info"] = dict(type=str(type(m)), dtype=str(getattr(m, "dtype", None)), shape=tuple(m.shape) if m is not None else None)
        with torch.no_grad():
            V = variants(model.model.layers[l].self_attn, g["hidden"], g["pe"], mask2d)
        B, T = ids.shape
        Zm = grab.d[("z", l)].float().view(B, T, H, HD)
        V["MODEL_sdpa_captured"] = Zm
        pairs = [("MODEL_sdpa_captured", "R0_recompute_fp32_attention"), ("MODEL_sdpa_captured", "R1_probs_cast_bf16"), ("MODEL_sdpa_captured", "R2_hf_eager_same_qkv"),
                 ("MODEL_sdpa_captured", "R3a_sdpa_float_mask"), ("MODEL_sdpa_captured", "R3b_sdpa_bool_mask"), ("R0_recompute_fp32_attention", "R0g_gqa_group_loop"),
                 ("R0_recompute_fp32_attention", "R0r_rope_fp32")] + [(x, "FP32_reference") for x in ("MODEL_sdpa_captured", "R0_recompute_fp32_attention", "R1_probs_cast_bf16", "R2_hf_eager_same_qkv", "R3a_sdpa_float_mask", "R3b_sdpa_bool_mask")]
        for a_, b_ in pairs:
            st = stats(V[a_], V[b_], real, l)
            agg.setdefault((l, a_, b_), []).append(st)
        # S8 heads: error of the verdict-coordinate contribution at all real positions and at the last position
        for (ll, h) in S8:
            if ll != l: continue
            for a_, b_ in (("MODEL_sdpa_captured", "R0_recompute_fp32_attention"), ("FP32_reference", "R0_recompute_fp32_attention"), ("FP32_reference", "MODEL_sdpa_captured")):
                sa, sb = contrib(V[a_], l)[..., h], contrib(V[b_], l)[..., h]
                d_all = (sa - sb)[real].abs(); d_last = (sa - sb)[:, -1].abs(); sd_all = sb[real].std(); sd_last = sb[:, -1].std()
                res["s8"].setdefault(f"L{l}H{h}|{a_}|vs|{b_}", []).append(dict(all_rms_over_std=float(d_all.pow(2).mean().sqrt() / sd_all), all_max_over_std=float(d_all.max() / sd_all), last_rms_over_std=float(d_last.pow(2).mean().sqrt() / sd_last),
                                                                       last_max_over_std=float(d_last.max() / sd_last), last_abs_max=float(d_last.max()), head_s_std_last=float(sd_last)))
        # localise the worst contribution errors for the gate comparison (model sdpa vs R0), across all heads
        sm, s0, sr = contrib(V["MODEL_sdpa_captured"], l), contrib(V["R0_recompute_fp32_attention"], l), contrib(V["FP32_reference"], l)
        sd = sr[real].std()
        err = (sm - s0).abs() * real.unsqueeze(-1)
        top = err.flatten().topk(15)
        for val, fi in zip(top.values.tolist(), top.indices.tolist()):
            b, t, h = fi // (T * H), (fi // H) % T, fi % H
            first_real = int((~real[b]).sum())
            worst_rows.append(dict(batch=bi, example=b, layer=l, head=h, in_S8=(l, h) in S8, abs_pos=t, pos_in_seq=t - first_real, seq_len=int(real[b].sum()), token=tok.decode(ids[b, t:t + 1]),
                                   err_over_std=val / float(sd), s_model=float(sm[b, t, h]), s_recompute=float(s0[b, t, h]), s_fp32=float(sr[b, t, h]),
                                   closer_to_fp32="model_sdpa" if abs(float(sm[b, t, h] - sr[b, t, h])) < abs(float(s0[b, t, h] - sr[b, t, h])) else "recompute"))
    grab.close()
    print(f"batch {bi} done", flush=True)

def mean_stats(lst): return {k: float(np.mean([s[k] for s in lst])) for k in lst[0]} | {f"max_{k}": float(np.max([s[k] for s in lst])) for k in ("z_max", "s_max_over_std", "s_abs_max")}
res["pairs"] = {f"L{l}|{a_}|vs|{b_}": mean_stats(v) for (l, a_, b_), v in agg.items()}
res["s8"] = {k: {m: float(np.max([x[m] for x in v])) for m in v[0]} for k, v in res["s8"].items()}
worst_rows.sort(key=lambda r: -r["err_over_std"])
res["worst"] = worst_rows[:60]
top100 = worst_rows[:100]
res["worst_summary"] = dict(n=len(top100), frac_first_real_token=float(np.mean([r["pos_in_seq"] == 0 for r in top100])), frac_in_S8=float(np.mean([r["in_S8"] for r in top100])),
                            frac_closer_to_fp32_is_model_sdpa=float(np.mean([r["closer_to_fp32"] == "model_sdpa" for r in top100])),
                            by_layer=dict(Counter(r["layer"] for r in top100)), by_head=dict(Counter(f"L{r['layer']}H{r['head']}" for r in top100).most_common(8)),
                            by_position_class=dict(Counter("first_real" if r["pos_in_seq"] == 0 else "last_5" if r["pos_in_seq"] >= r["seq_len"] - 5 else "middle" for r in top100)),
                            tokens=dict(Counter(r["token"] for r in top100).most_common(6)))
json.dump(res, open(ROOT / "results/phase25/recompute_diagnosis.json", "w"), indent=1)
print(json.dumps(res["worst_summary"], indent=1)); print(res["mask_info"])
for L in (19, 22):
    for key in ("MODEL_sdpa_captured|vs|R0_recompute_fp32_attention", "MODEL_sdpa_captured|vs|R1_probs_cast_bf16", "MODEL_sdpa_captured|vs|R2_hf_eager_same_qkv", "MODEL_sdpa_captured|vs|R3a_sdpa_float_mask", "MODEL_sdpa_captured|vs|R3b_sdpa_bool_mask",
                "MODEL_sdpa_captured|vs|FP32_reference", "R0_recompute_fp32_attention|vs|FP32_reference", "R1_probs_cast_bf16|vs|FP32_reference", "R2_hf_eager_same_qkv|vs|FP32_reference", "R0_recompute_fp32_attention|vs|R0g_gqa_group_loop", "R0_recompute_fp32_attention|vs|R0r_rope_fp32"):
        r = res["pairs"][f"L{L}|{key.replace('|vs|', '|')}|".rstrip("|").replace("|", "|", 1)] if False else res["pairs"].get(f"L{L}|{key.split('|vs|')[0]}|vs|{key.split('|vs|')[1]}")
        if r is None:
            r = res["pairs"].get(f"L{L}|{key.split('|vs|')[0]}|{key.split('|vs|')[1]}")
        print(L, key, {k: round(v, 5) for k, v in r.items() if k in ("z_rel_L2", "s_rms_over_std", "s_max_over_std")})
