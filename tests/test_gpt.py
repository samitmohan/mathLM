import math
import pytest
import torch
from gpt import GPTConfig, MLP, GPT, precompute_rotary


def small_config(**kwargs):
    defaults = dict(
        vocab_size=100,
        seq_len=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        use_moe=False,
        n_experts=4,
    )
    defaults.update(kwargs)
    return GPTConfig(**defaults)


def test_mlp_output_shape():
    config = small_config()
    mlp = MLP(config)
    x = torch.randn(2, 10, 128)
    out = mlp(x)
    assert out.shape == (2, 10, 128)


def test_mlp_hidden_dim_multiple_of_64():
    config = small_config(n_embd=128)
    mlp = MLP(config)
    assert mlp.fc1.out_features % 64 == 0


def test_mlp_is_residual_marked():
    config = small_config()
    mlp = MLP(config)
    assert getattr(mlp.fc2, "_is_residual", False) is True
