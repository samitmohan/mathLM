"""Smoke test: build a tiny GPT, run forward + generate. Exits 0 on success.

Catches package-wiring breakage. Runs in <5s on CPU.

    python scripts/smoke_test.py
"""

import torch

from mathlm.model.gpt import GPT, GPTConfig


def main():
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=256,
        sequence_length=64,
        number_layers=2,
        number_heads=4,
        number_kv_heads=2,
        embedding_dim=64,
    )
    model = GPT(config)
    model.eval()

    x = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 16, config.vocab_size), logits.shape
    assert loss is None

    y = torch.randint(0, config.vocab_size, (2, 16))
    _, loss = model(x, y)
    assert loss is not None and torch.isfinite(loss), loss

    # KV-cached generation: 4 new tokens
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=10)
    assert out.shape == (1, 12), out.shape

    print("smoke test passed")


if __name__ == "__main__":
    main()
