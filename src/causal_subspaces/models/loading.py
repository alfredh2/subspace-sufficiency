"""Frozen Qwen3 loading, non-thinking prompt formatting and batched last-position forward."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(cfg: dict, device: str = "cuda"):
    """Returns (model, tokenizer, revision). Parameters frozen, eval mode, no copies made."""
    m = cfg["model"] if "model" in cfg else cfg
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m.get("revision"), padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        m["name"], revision=m.get("revision"), dtype=DTYPES[m.get("dtype", "bfloat16")]
    ).to(device)
    model.eval().requires_grad_(False)
    if m.get("gradient_checkpointing"):
        # Weights are frozen, so the input to the checkpointed blocks must require grad for backprop to
        # reach the (hook-injected) trainable subspace; use_reentrant=False handles this.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    revision = m.get("revision")
    if revision is None:  # resolve the exact snapshot hash actually loaded, for logging
        from huggingface_hub import snapshot_download
        try:
            revision = snapshot_download(m["name"], local_files_only=True).rstrip("/").split("/")[-1]
        except Exception:
            revision = getattr(model.config, "_commit_hash", None)
    return model, tok, revision


def format_prompt(tok, user_text: str, enable_thinking: bool = False) -> str:
    """Chat template, non-thinking: the Qwen3 template then appends an empty <think></think> block.
    (The paper only says 'non-thinking mode'; system prompt / template details are not stated.)"""
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )


def encode(tok, prompts: list[str], device: str):
    """Left-padded batch with explicit position_ids so padding does not shift RoPE positions."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    return enc.input_ids, enc.attention_mask, pos


def last_logits(model, input_ids, attention_mask, position_ids) -> torch.Tensor:
    """Next-token logits at the final prompt position (fp32), shape (B, V). Only the last position is
    projected through lm_head (saves memory); gradients still flow through the whole stack."""
    out = model.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    return model.lm_head(out.last_hidden_state[:, -1]).float()
