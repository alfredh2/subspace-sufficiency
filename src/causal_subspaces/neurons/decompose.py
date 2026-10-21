"""Layerwise decomposition of the verdict coordinate c = d.h at the hook, and per-unit class statistics.
Exact identity (additive residual stream): c = d.emb + sum_l d.attn_l + sum_l d.mlp_l for l < residual_index.
Memory: layers above the hook are skipped; only last-position vectors are reduced/kept."""
import numpy as np
import torch

from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, format_prompt


def _at(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x[b, pos[b]] for x of shape (B, T, ...)."""
    return x[torch.arange(x.shape[0], device=x.device), pos]


def layers_below(model, residual_index: int):
    if not 1 <= residual_index <= model.config.num_hidden_layers - 1:
        raise ValueError(residual_index)
    return list(range(residual_index))  # module indices 0..residual_index-1 write into hidden_states[residual_index]


def unit_gain(model, layer: int, d: torch.Tensor) -> torch.Tensor:
    """g^l = W_down^T d  (fp32): direct contribution of unit j is a_j * g_j."""
    W = model.model.layers[layer].mlp.down_proj.weight
    return W.float().T @ d.to(W.device).float()


@torch.no_grad()
def layerwise_pass(model, tok, texts, accept, d, residual_index, *, device="cuda", batch_size=16, keep_acts_layers=(), enable_thinking=False, position="last"):
    """One truncated forward over `texts`. Returns
       comps: emb (N,), attn (N,L), mlp (N,L), resid (N,)   -- d . (component at last position), fp32
       mean_acc/mean_rej: (L, m) class means of the down_proj input (last position)
       acts: {layer: (N, m) bf16 cpu} for keep_acts_layers
    accept: bool array (N,) of semantic labels. position: "last" (final prompt token) or "concl" (last token of the user message, the
    sentence-final token of the conclusion).
    """
    L = layers_below(model, residual_index)
    d = d.to(device).float()
    N, m = len(texts), model.config.intermediate_size
    accept = np.asarray(accept, bool)
    comps = dict(emb=np.zeros(N), attn=np.zeros((N, len(L))), mlp=np.zeros((N, len(L))), resid=np.zeros(N))
    sums = {c: torch.zeros(len(L), m, dtype=torch.float64, device=device) for c in (True, False)}
    acts = {l: torch.zeros(N, m, dtype=torch.bfloat16) for l in keep_acts_layers}
    prompts = [format_prompt(tok, t, enable_thinking) for t in texts]
    order = sorted(range(N), key=lambda i: len(prompts[i]))
    cur = {}
    st = {}
    handles = []
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    layers = model.model.layers
    handles.append(model.model.embed_tokens.register_forward_hook(lambda mod, i, o: cur.__setitem__("emb", (_at(o, st["pos"]).float() @ d))))
    for li, l in enumerate(L):
        handles.append(layers[l].self_attn.register_forward_hook(
            lambda mod, i, o, li=li: cur.__setitem__(("attn", li), _at(o[0] if isinstance(o, tuple) else o, st["pos"]).float() @ d)))
        handles.append(layers[l].mlp.register_forward_hook(lambda mod, i, o, li=li: cur.__setitem__(("mlp", li), _at(o, st["pos"]).float() @ d)))
        handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, i, li=li: cur.__setitem__(("a", li), _at(i[0], st["pos"]).float())))
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
            comps["resid"][idx] = (_at(sink[0], st["pos"]).float() @ d).cpu().numpy()
            comps["emb"][idx] = cur["emb"].cpu().numpy()
            acc_t = torch.tensor(accept[idx], device=device)
            for li, l in enumerate(L):
                comps["attn"][idx, li] = cur[("attn", li)].cpu().numpy()
                comps["mlp"][idx, li] = cur[("mlp", li)].cpu().numpy()
                a = cur[("a", li)]
                sums[True][li] += a[acc_t].double().sum(0)
                sums[False][li] += a[~acc_t].double().sum(0)
                if l in acts:
                    acts[l][idx] = a.to(torch.bfloat16).cpu()
    finally:
        model.model.layers = saved
        for h in handles:
            h.remove()
    n_acc, n_rej = int(accept.sum()), int((~accept).sum())
    return dict(comps=comps, mean_acc=(sums[True] / n_acc).float().cpu(), mean_rej=(sums[False] / n_rej).float().cpu(), acts=acts, layers=L)


def class_difference_table(comps, accept) -> dict:
    """Signed class differences (accept - reject) of each component's contribution to c, per layer, plus identity error."""
    accept = np.asarray(accept, bool)
    diff = lambda x: x[accept].mean(0) - x[~accept].mean(0)
    recon = comps["emb"] + comps["attn"].sum(1) + comps["mlp"].sum(1)
    return dict(emb=float(diff(comps["emb"])), attn=diff(comps["attn"]), mlp=diff(comps["mlp"]), resid=float(diff(comps["resid"])),
                identity_max_abs_err=float(np.abs(recon - comps["resid"]).max()), resid_std=float(comps["resid"].std()))
