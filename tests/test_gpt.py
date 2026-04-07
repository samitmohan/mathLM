import math
import pytest
import torch
from gpt import GPTConfig, MLP, Block, SparseMoE, GPT, precompute_rotary


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


def test_block_returns_tuple_dense():
    config = small_config()
    block = Block(config, 0)
    cos, sin = precompute_rotary(32, 128 // 4, "cpu")
    x = torch.randn(1, 10, 128)
    out, aux = block(x, cos, sin)
    assert out.shape == (1, 10, 128)
    assert aux == 0.0


def test_moe_output_shape_and_aux():
    config = small_config(use_moe=True)
    moe = SparseMoE(config)
    x = torch.randn(2, 10, 128)
    out, aux = moe(x)
    assert out.shape == (2, 10, 128)
    assert isinstance(aux, torch.Tensor) and aux.ndim == 0


def test_block_returns_tuple_moe():
    config = small_config(use_moe=True)
    block = Block(config, 0)
    cos, sin = precompute_rotary(32, 128 // 4, "cpu")
    x = torch.randn(1, 10, 128)
    out, aux = block(x, cos, sin)
    assert out.shape == (1, 10, 128)
    assert isinstance(aux, torch.Tensor)


def test_gpt_forward_dense():
    config = small_config()
    model = GPT(config)
    idx = torch.zeros((1, 10), dtype=torch.long)
    logits, loss = model(idx, idx)
    assert logits.shape == (1, 10, 100)
    assert loss is not None and loss.ndim == 0


def test_gpt_forward_moe():
    config = small_config(use_moe=True)
    model = GPT(config)
    idx = torch.zeros((1, 10), dtype=torch.long)
    logits, loss = model(idx, idx)
    assert logits.shape == (1, 10, 100)
    assert loss is not None and loss.ndim == 0
