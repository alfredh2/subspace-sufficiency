"""Validated recomputation of Qwen3 attention for head-level analysis (Phase 2): attention weights, per-head value projections, routing/content
decomposition and A/V patching. The causal+padding mask is built here from the 2D attention mask (never from HF internals).
Everything here is only used after the recompute gate (docs/phase2_plan.md section 2) has passed on real Qwen3 batches."""
from contextlib import contextmanager

import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from causal_subspaces.heads.decompose import head_dims


def additive_mask(mask2d: torch.Tensor) -> torch.Tensor:
    """(B,T) 1=real token (left padded) -> additive (B,1,T,T) fp32. Keys must be real and causal; padded query rows attend to themselves
    (their outputs are never used) so softmax stays finite."""
    B, T = mask2d.shape
    dev = mask2d.device
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))[None] & mask2d.bool()[:, None, :]
    keep = keep | torch.eye(T, dtype=torch.bool, device=dev)[None]
    m = torch.zeros(B, 1, T, T, device=dev)
    return m.masked_fill(~keep[:, None], -1e30)


class AttnStash:
    """Stash (hidden_states, position_embeddings) fed to each layer's self_attn (the decoder layer calls it with keyword arguments)."""

    def __init__(self, model, layers):
        self.data, self.handles = {}, []
        for l in layers:
            self.handles.append(model.model.layers[l].self_attn.register_forward_pre_hook(
                lambda mod, args, kwargs, l=l: self.data.__setitem__(l, (
                    (args[0] if args else kwargs["hidden_states"]).detach(), kwargs["position_embeddings"] if "position_embeddings" in kwargs else args[1])), with_kwargs=True))

    def close(self):
        for h in self.handles:
            h.remove()


def recompute(model, layer: int, hidden: torch.Tensor, pos_emb, mask2d: torch.Tensor, *, A_override: dict | None = None, V_override: dict | None = None):
    """Recompute one layer's per-head outputs from its attention input.
    A_override[h]: (B,T,T) attention weights to use for query head h; V_override[h]: (B,T,HD) values to use for query head h only.
    Returns dict(z=(B,T,H,HD) in hidden dtype, A=(B,H,T,T) fp32, v=(B,H,T,HD) fp32 per-query-head values)."""
    attn = model.model.layers[layer].self_attn
    H, HD, KV = head_dims(model)
    B, T, _ = hidden.shape
    q = attn.q_norm(attn.q_proj(hidden).view(B, T, H, HD)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(hidden).view(B, T, KV, HD)).transpose(1, 2)
    v = attn.v_proj(hidden).view(B, T, KV, HD).transpose(1, 2)
    cos, sin = pos_emb
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    kr, vr = repeat_kv(k, H // KV).float(), repeat_kv(v, H // KV).float()
    A = torch.softmax((q.float() @ kr.transpose(2, 3)) * attn.scaling + additive_mask(mask2d), dim=-1)
    if A_override:
        A = A.clone()
        for h, a_ in A_override.items():
            A[:, h] = a_.to(A.dtype)
    if V_override:
        vr = vr.clone()
        for h, v_ in V_override.items():
            vr[:, h] = v_.float()
    z = (A @ vr).transpose(1, 2)
    return dict(z=z.to(hidden.dtype), A=A, v=vr)


def source_projection(v: torch.Tensor, gain: torch.Tensor, h: int, HD: int) -> torch.Tensor:
    """u_s = gain_h . v_s for query head h: the scalar each source position adds to the verdict coordinate per unit attention weight. v: (B,H,T,HD)."""
    return (v[:, h] * gain.view(-1, HD)[h].to(v.device)).sum(-1)


def routing_content(A_a: torch.Tensor, A_b: torch.Tensor, u_a: torch.Tensor, u_b: torch.Tensor):
    """Symmetric exact decomposition of c_a - c_b for c = sum_s A_s u_s over the last-query row:
         routing = sum_s (A_a - A_b) * (u_a + u_b)/2 ,  content = sum_s (A_a + A_b)/2 * (u_a - u_b),  routing + content = c_a - c_b."""
    routing = ((A_a - A_b) * (u_a + u_b) / 2).sum(-1)
    content = ((A_a + A_b) / 2 * (u_a - u_b)).sum(-1)
    return routing, content, (A_a * u_a).sum(-1) - (A_b * u_b).sum(-1)


# ---- pair alignment (amendment 8) ---------------------------------------------------------------------------------
def line_spans(tok, prompt: str, lines: list[str]):
    """Token index ranges [a, b) of each syllogism line inside the formatted prompt (via character offsets)."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    spans, start = [], 0
    for ln in lines:
        c0 = prompt.find(ln, start)
        if c0 < 0:
            return enc["input_ids"], None
        c1 = c0 + len(ln)
        toks = [i for i, (a, b) in enumerate(offs) if b > c0 and a < c1]
        spans.append((toks[0], toks[-1] + 1))
        start = c1
    return enc["input_ids"], spans


def pair_alignment(tok, prompt_a: str, lines_a: list[str], prompt_b: str, lines_b: list[str]) -> tuple[bool, str]:
    """A pair may be used for all-position / attention-weight / value patching and routing-content only if: equal length; every syllogism line
    occupies the SAME token span in both prompts (same role at the same position, hence the same RoPE position); all tokens outside the lines are
    identical; identical lines have identical tokens. Otherwise it is rejected (never realigned heuristically)."""
    ia, sa = line_spans(tok, prompt_a, lines_a)
    ib, sb = line_spans(tok, prompt_b, lines_b)
    if sa is None or sb is None:
        return False, "line not found"
    if len(ia) != len(ib):
        return False, "length"
    if sa != sb:
        return False, "line spans differ"
    inside = np.zeros(len(ia), bool)
    for a, b in sa:
        inside[a:b] = True
    if any(x != y for x, y, ins in zip(ia, ib, inside) if not ins):
        return False, "tokens outside lines differ"
    for (a, b), la, lb in zip(sa, lines_a, lines_b):
        if la == lb and ia[a:b] != ib[a:b]:
            return False, "identical line tokenised differently"
    return True, "ok"


# ---- attention-weight / value patching ---------------------------------------------------------------------------------
@contextmanager
def attention_patch(model, spec: dict, source: dict, mask2d: torch.Tensor, residual_index: int, *, patch="AV", position="all"):
    """Recompute the listed heads with source attention weights (patch contains 'A') and/or source values (patch contains 'V').
    spec[layer] = heads; source[layer] = dict(A=(B,H,T,T), v=(B,H,T,HD)) from `recompute` on the SOURCE forward (same batch order, same shapes).
    position 'last' only replaces the final query row. The recompute path must have passed the gate; self-patching (source = target) measures its noise floor."""
    if position not in ("all", "last") or not set(patch) <= {"A", "V"} or not patch:
        raise ValueError((patch, position))
    H, HD, _ = head_dims(model)
    stash = AttnStash(model, list(spec))
    handles = []
    for l, heads in spec.items():
        if not 0 <= l < residual_index:
            raise ValueError(f"layer {l} cannot write into hidden_states[{residual_index}]")

        def hook(mod, inp, l=l, heads=list(heads)):
            z = inp[0]
            B, T, _ = z.shape
            hidden, pe = stash.data[l]
            if source[l]["A"].shape[-1] != T or source[l]["v"].shape[0] != B:
                raise ValueError("source/target shapes differ; pair rejected")
            base = recompute(model, l, hidden, pe, mask2d)
            A_ov, V_ov = {}, {}
            for h in heads:
                if "A" in patch:
                    a = base["A"][:, h].clone()
                    if position == "all":
                        a = source[l]["A"][:, h]
                    else:
                        a[:, -1] = source[l]["A"][:, h, -1]
                    A_ov[h] = a
                if "V" in patch:
                    V_ov[h] = source[l]["v"][:, h]
            rec = recompute(model, l, hidden, pe, mask2d, A_override=A_ov or None, V_override=V_ov or None)
            new = z.float().view(B, T, H, HD).clone()
            for h in heads:
                if position == "all":
                    new[:, :, h] = rec["z"][:, :, h].float()
                else:
                    new[:, -1, h] = rec["z"][:, -1, h].float()
            return (new.view(B, T, H * HD).to(z.dtype),) + tuple(inp[1:])

        handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
        stash.close()
