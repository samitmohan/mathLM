import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

from gpt import GPT, GPTConfig


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_data(path):
    with open(path, "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    data = np.array(encode(text), dtype=np.int64)
    return data, vocab_size, encode, decode


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # Phase 1: linear warmup from 0 to max_lr. Starting at zero avoids large
    # random gradient updates before the model has any useful structure.
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # Phase 2: cosine decay from max_lr to min_lr. Cosine avoids the abrupt drop
    # you'd get from a step schedule and smoothly approaches the final learning rate.
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, vocab_size, encode, decode = load_data("input.txt")
    config = GPTConfig(
        vocab_size=vocab_size,
        seq_len=128,
        n_layer=6,
        n_head=6,
        n_kv_head=2,
        n_embd=384
    )

    model = GPT(config).to(device)

    dataset = CharDataset(data, config.seq_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    max_steps = 20000
    warmup_steps = 1000
    max_lr = 3e-4
    min_lr = max_lr * 0.1

    model.train()
    step = 0

    # bfloat16 only on CUDA: Metal and CPU lack hardware bfloat16 support.
    # bfloat16 keeps the float32 exponent range while halving memory and compute.
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    for epoch in range(10):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with torch.autocast(device_type=device, dtype=autocast_dtype):
                logits, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping: rescale gradients so their norm never exceeds 1.0.
            # Prevents large random batches early in training from destabilizing weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if step % 100 == 0:
                print(f"step {step} | loss {loss.item():.4f} | lr {lr:.6f}")

            step += 1
            if step > max_steps:
                break

        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=200)

        print(decode(out[0].tolist()))

        model.train()

        if step > max_steps:
            break


if __name__ == "__main__":
    train()