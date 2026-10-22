"""Early-exit forward passes for Phase 6 (docs/phase6_plan.md): stop right after the input of a layer's attention (or after a layer's output) and return the
activations. Hooks are always removed; a controlled exception is the only control flow."""
from contextlib import ExitStack, contextmanager

import torch


class StopForward(Exception):
    pass


@contextmanager
def exit_before_attention(model, layer: int, store: dict):
    """Pre-hook on layers[layer].self_attn (registered last by the caller so patch hooks have already fired): store the attention input (post input-layernorm
    hidden states) and the position embeddings, then raise StopForward."""
    def hook(mod, args, kwargs):
        store["h"] = (args[0] if args else kwargs["hidden_states"]).detach()
        store["pe"] = kwargs["position_embeddings"] if "position_embeddings" in kwargs else args[1]
        raise StopForward()
    handle = model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def exit_after_layer(model, layer: int, store: dict):
    """Forward hook on layers[layer] (registered after patch/capture hooks): store the layer output (= hidden_states[layer+1]) and raise."""
    def hook(mod, args, kwargs, out):
        store["out"] = (out[0] if isinstance(out, tuple) else out).detach()
        raise StopForward()
    handle = model.model.layers[layer].register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def run_partial(model, ids, mask, pos, exit_cm, patch_cms=()):
    """Run model.model up to the exit hook. patch_cms are entered before the exit hook so they act first."""
    with ExitStack() as st:
        for c in patch_cms:
            st.enter_context(c)
        st.enter_context(exit_cm)
        try:
            with torch.no_grad():
                model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
        except StopForward:
            pass
