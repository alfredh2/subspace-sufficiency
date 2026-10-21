import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


@pytest.fixture(scope="session")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", padding_side="left")


@pytest.fixture()
def tiny(tok):
    """Randomly initialised 4-layer Qwen3 with the real tokenizer's vocab: tests hooks/scoring without the 8B model."""
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
    m = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    return m
