import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from gpt import GPT, GPTConfig
from tokenizer import BPETokenizer


class TokenDataset(Dataset):
    """Dataset of fixed-length token sequences drawn from a 1D token array."""

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


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


def evaluate(model, val_loader, device, autocast_dtype):
    """Run full validation pass, return mean cross-entropy loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("input.txt") as f:
        text = f.read()

    # Tokenizer: train once on the corpus, reuse on subsequent runs.
    if os.path.exists("tokenizer.json"):
        tok = BPETokenizer()
        tok.load("tokenizer.json")
        print(f"loaded tokenizer: {len(tok)} tokens")
    else:
        print("training tokenizer...")
        tok = BPETokenizer()
        tok.train(text, vocab_size=4096)
        tok.save("tokenizer.json")
        print(f"tokenizer saved: {len(tok)} tokens")

    data = np.array(tok.encode(text), dtype=np.int64)

    # 90/10 train/val split at a token boundary (preserves document order).
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    config = GPTConfig(
        vocab_size=len(tok),
        seq_len=1024,
        n_layer=12,
        n_head=12,
        n_kv_head=3,
        n_embd=768,
    )

    model = GPT(config).to(device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Resume from checkpoint if one exists.
    step = 0
    if os.path.exists("checkpoint.pt"):
        ckpt = torch.load("checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resuming from step {step}")

    # torch.compile: fuses ops into optimized kernels (~30% throughput gain on CUDA).
    model = torch.compile(model)

    max_steps    = 20000
    warmup_steps = 1000
    max_lr       = 3e-4
    min_lr       = max_lr * 0.1
    batch_size   = 8
    grad_accum_steps = 8    # effective batch = batch_size * grad_accum_steps sequences
    eval_interval    = 500

    # bfloat16 only on CUDA: Metal and CPU lack hardware bfloat16 support.
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    train_dataset = TokenDataset(train_data, config.seq_len)
    val_dataset   = TokenDataset(val_data,   config.seq_len)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size)

    train_iter = iter(train_loader)

    model.train()
    while step <= max_steps:
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation: accumulate over grad_accum_steps micro-batches
        # before stepping, simulating a larger effective batch without extra memory.
        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps  # scale before backward to average correctly
            loss.backward()
            loss_accum += loss.item()

        # Gradient clipping: rescale gradients so their norm never exceeds 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.6f}")

        if step > 0 and step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, autocast_dtype)
            print(f"step {step:5d} | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f}")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": config,
            }, "checkpoint.pt")
            print(f"checkpoint saved")

        step += 1


if __name__ == "__main__":
    train()
