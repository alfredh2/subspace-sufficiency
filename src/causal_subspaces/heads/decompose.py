"""Per-head decomposition of attention writes into the verdict coordinate (Phase 2, Stage 1).
Head h of layer l writes W_{O,h} z_h with z_h the h-th 128-dim slice of the input of o_proj; its contribution to c = d.h is gain_h . z_h with
gain = W_O^T d.  GQA only affects how z_h is computed (query head h reads KV head h // (H/KV)); the o_proj slice is per query head."""
import numpy as np
import torch

from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt
from causal_subspaces.neurons.decompose import _at, layers_below


def head_dims(model):
    c = model.config
    return c.num_attention_heads, c.head_dim, c.num_key_value_heads


def kv_head_of(h: int, model) -> int:
    H, _, KV = head_dims(model)
    return h // (H // KV)


def head_gain(model, layer: int, d: torch.Tensor) -> torch.Tensor:
    W = model.model.layers[layer].self_attn.o_proj.weight  # (hidden, H*HD)
    return W.float().T @ d.to(W.device).float()


@torch.no_grad()
def head_pass(model, tok, texts, accept, d, residual_index, *, device="cuda", batch_size=16, position="last", keep_z=False):
    """One truncated forward. Returns
       S: (N, L, H) per-example head contributions s_h to c at the chosen position
       attn_dot: (N, L) d . attention_output (for the reconstruction identity)
       mean_acc, mean_rej: (L, H*HD) class means of z
       z: (N, L, H*HD) bf16 (optional)"""
    L = layers_below(model, residual_index)
    H, HD, _ = head_dims(model)
    d = d.to(device).float()
    gains = {l: head_gain(model, l, d) for l in L}
    N = len(texts)
    accept = np.asarray(accept, bool)
    S = np.zeros((N, len(L), H), np.float32)
    ad = np.zeros((N, len(L)), np.float32)
    sums = {c: torch.zeros(len(L), H * HD, dtype=torch.float64, device=device) for c in (True, False)}
    Z = torch.zeros(N, len(L), H * HD, dtype=torch.bfloat16) if keep_z else None
    prompts = [format_prompt(tok, t, False) for t in texts]
    order = sorted(range(N), key=lambda i: len(prompts[i]))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cur, st, handles = {}, {}, []
    layers = model.model.layers
    for li, l in enumerate(L):
        handles.append(layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, i, li=li: cur.__setitem__(("z", li), _at(i[0], st["pos"]).float())))
        handles.append(layers[l].self_attn.register_forward_hook(lambda mod, i, o, li=li: cur.__setitem__(("a", li), _at(o[0] if isinstance(o, tuple) else o, st["pos"]).float() @ d)))
    saved = model.model.layers
    model.model.layers = saved[:residual_index]
    try:
        for s in range(0, N, batch_size):
            idx = order[s:s + batch_size]
            ids, mask, pos = encode(tok, [prompts[i] for i in idx], device)
            st["pos"] = (ids.shape[1] - 1) * torch.ones(len(idx), dtype=torch.long, device=device) if position == "last" else (ids == im_end).float().argmax(-1) - 1
            sink = []
            with capture_residual(model, residual_index, sink):
                model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
            acc_t = torch.tensor(accept[idx], device=device)
            for li, l in enumerate(L):
                zp = cur[("z", li)]
                S[idx, li] = (zp * gains[l]).view(len(idx), H, HD).sum(-1).cpu().numpy()
                ad[idx, li] = cur[("a", li)].cpu().numpy()
                sums[True][li] += zp[acc_t].double().sum(0)
                sums[False][li] += zp[~acc_t].double().sum(0)
                if keep_z:
                    Z[idx, li] = zp.to(torch.bfloat16).cpu()
    finally:
        model.model.layers = saved
        for h in handles:
            h.remove()
    n_a, n_r = int(accept.sum()), int((~accept).sum())
    out = dict(S=S, attn_dot=ad, mean_acc=(sums[True] / n_a).float().cpu(), mean_rej=(sums[False] / n_r).float().cpu(), layers=L, gains=gains)
    if keep_z:
        out["z"] = Z
    return out


def class_diff_heads(S: np.ndarray, accept) -> np.ndarray:
    """Delta_h (L, H) = E[s_h | accept] - E[s_h | reject]."""
    accept = np.asarray(accept, bool)
    return S[accept].mean(0) - S[~accept].mean(0)
