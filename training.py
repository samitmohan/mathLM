'''
Pipeline:
Text -> tokenise -> token ids -> dataset -> batches -> GPT -> loss -> optimize
'''
import math
import os
import time
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False
from torch.utils.data import DataLoader, Dataset

from gippity import GPT, GPTConfig
import tiktoken
# from tokenizer import BPETokenizer

# dataset: tokendataset (turns long sequence into training examples)
# [token0, token1, token2, token3, token4, token5, token6]
# x = [token0, token1, token2, token3, token4]
# y = [token1, token2, token3, token4, token5]

class TokenDataset(Dataset):
    def __init__(self, data, sequence_length):
        # data is a list of token ids, sequence_length is the length of the input sequence for the model
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        # the number of training examples is the total number of tokens minus the sequence length (since we need sequence_length tokens for input and 1 token for output)
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # return input and target sequences for the given index
        # input sequence of length sequence_length
        # target sequence is the input sequence shifted by one token to the right
        x = torch.tensor(self.data[idx:idx+self.sequence_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.sequence_length+1], dtype=torch.long)
        return x, y


# learning rate scheduler:
# what is a LR scheduler? it is a function that takes the current step and returns the learning rate for that step.
# it is used to adjust the learning rate during training to improve convergence and prevent overfitting.

def get_lr(step, warmup, max_steps, max_lr, min_lr):
    # we want a function that maps training step to learning rate which we call it every iteration to decide how big of an update we need
    # early training is unstable; we want to start with a small learning rate and gradually increase it to the maximum learning rate over the first few thousand steps (linear warmup)
    # ideally we want to use cosine decay instead of step decay to have a smoother learning rate schedule that gradually decreases the learning rate over time instead of abruptly dropping it every 10000 steps (avoids getting stuck in local minima)

    # 1. warmup
    if step < warmup:
        return max_lr * step / warmup

    # 2. cosine decay
    if step > max_steps:
        return min_lr

    # we want to start the cosine decay after the warmup period, so we subtract the warmup steps from the current step and divide by the total number of steps minus the warmup steps to get a ratio that goes from 0 to 1
    ratio = (step - warmup) / (max_steps - warmup)
    # cosine goes from 1 to -1, we want it to go from 1 to 0, so we add 1 and divide by 2
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))

    # we want the learning rate to go from max_lr to min_lr, so we multiply the coefficient by the difference between max_lr and min_lr and add min_lr to scale
    return min_lr + coeff * (max_lr - min_lr)


def evaluate(model, val_loader, device, autocast_dtype):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, y)
            total_loss += loss.item() * x.size(0)  # accumulate sample-weighted sum
            n += x.size(0)
    model.train()
    return total_loss / n


enc = tiktoken.get_encoding("gpt2")
def train():
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('high')  # enables TF32 on CUDA, free speedup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # tokeniser = BPETokenizer()
    # tokeniser.load("tokenizer.json")
    data = np.memmap("train.bin", dtype=np.int32, mode="r")

    # Augment with synthetic math Q&A data if available (run gen_math_data.py to generate)
    if os.path.exists("math_qa.bin"):
        math_data = np.fromfile("math_qa.bin", dtype=np.int32)
        # repeat math data 15x so it's well-represented alongside the large openwebmath corpus
        data = np.concatenate([data, np.tile(math_data, 15)])
        console.print(f"  [dim]math_qa.bin loaded — {len(math_data):,} tokens × 5 appended[/dim]")

    # train test split
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    config = GPTConfig(
        vocab_size=50304,  # 50257 base + headroom for special tokens, rounded to multiple of 64
        sequence_length=256,
        number_layers=8,
        number_heads=8,
        number_kv_heads=4,  # GQA: 2 query heads per KV head
        embedding_dim=512,  # ~50M params — fits comfortably in 24 GB VRAM
    )

    model = GPT(config).to(device)

    # weight decay only on 2D params (weight matrices); not on biases, norms, or embeddings
    # betas=(0.9, 0.95) is Karpathy's recommendation over the default (0.9, 0.999)
    decay_params    = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=3e-4, betas=(0.9, 0.95)) # karpathy!

    step = 0
    if os.path.exists("checkpoint.pt"):
        ckpt = torch.load("checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resuming from step {step}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    model = torch.compile(model) # optimises model for faster training (PyTorch 2.0 feature)

    max_steps = 100000
    warmup = 2000
    max_lr = 3e-4
    min_lr = 3e-5
    grad_accum_steps = 4    # effective batch = 128 × 256 × 4 ≈ 131k tokens/step; peak VRAM ~16.5 GB on RTX 3090
    eval_interval = 5000

    # bfloat16 only on CUDA: Metal and CPU lack hardware bfloat16 support
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    num_workers = 4

    batch_size = 128
    train_loader = DataLoader(TokenDataset(train_data, config.sequence_length), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(TokenDataset(val_data, config.sequence_length), batch_size=128, num_workers=num_workers)
    train_iter = iter(train_loader)

    console.print(Panel(
        f"  [bold]layers[/bold] {config.number_layers}  [bold]dim[/bold] {config.embedding_dim}"
        f"  [bold]heads[/bold] {config.number_heads}  [bold]params[/bold] {params:.1f}M"
        f"  [bold]device[/bold] {device}  [bold]vocab[/bold] {config.vocab_size}",
        title="[bold cyan]nanochat — pretraining[/bold cyan]",
        border_style="cyan",
    ))

    if WANDB:
        wandb.init(
            project="nanochat",
            name=f"pretrain-{config.number_layers}L-{config.embedding_dim}d-{params:.1f}M",
            config={
                "vocab_size": config.vocab_size, "sequence_length": config.sequence_length,
                "number_layers": config.number_layers, "number_heads": config.number_heads,
                "number_kv_heads": config.number_kv_heads, "embedding_dim": config.embedding_dim,
                "use_moe": config.use_moe, "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps, "max_lr": max_lr,
                "min_lr": min_lr, "warmup": warmup, "max_steps": max_steps, "params_M": params,
            },
            resume="allow",
        )
    else:
        console.print("[dim]wandb not installed — run [bold]pip install wandb[/bold] for experiment tracking[/dim]")

    # train loop
    model.train()
    t_last_log = time.time()
    tokens_since_log = 0
    with Progress(
        TextColumn("[bold cyan]training[/bold cyan]"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%  step {task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task("train", total=max_steps, completed=step)
        while step < max_steps:
            lr = get_lr(step, warmup, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.zero_grad()

            loss_total = 0.0
            for _ in range(grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    _, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss.backward()
                loss_total += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens_since_log += config.sequence_length * batch_size * grad_accum_steps
            progress.update(task, advance=1)

            if step % 100 == 0:
                now = time.time()
                elapsed = now - t_last_log
                tok_per_sec = tokens_since_log / elapsed if elapsed > 0 else 0
                loss_color = "green" if loss_total < 2.0 else "yellow" if loss_total < 4.0 else "red"
                progress.console.print(
                    f"  [dim]step[/dim] [bold]{step:5d}[/bold]  │"
                    f"  [{loss_color}]loss {loss_total:.4f}[/{loss_color}]  │"
                    f"  [dim]lr[/dim] {lr:.2e}  │  [dim]gnorm[/dim] {grad_norm:.3f}  │"
                    f"  [cyan]tok/s {tok_per_sec:.0f}[/cyan]"
                )
                if WANDB:
                    wandb.log({"train/loss": loss_total, "train/lr": lr, "train/grad_norm": float(grad_norm), "train/tok_per_sec": tok_per_sec}, step=step)
                t_last_log = now
                tokens_since_log = 0

            if step > 0 and step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, autocast_dtype)
                progress.console.print(f"\n  [bold yellow]val_loss {val_loss:.4f}[/bold yellow] at step {step}\n")
                if WANDB:
                    wandb.log({"val/loss": val_loss}, step=step)

                # generation sample — use uncompiled model to avoid torch.compile + dynamic KV cache shape issues
                raw_for_gen = model._orig_mod if hasattr(model, "_orig_mod") else model
                raw_for_gen.eval()
                prompt = "Q: What is the derivative of x^5?\nA:"
                tokens = torch.tensor([enc.encode(prompt)], device=device)
                sample = ""
                with torch.no_grad():
                    for tok in raw_for_gen.stream_generate(tokens, max_new_tokens=40, temperature=0.8, top_k=50):
                        try:
                            sample += enc.decode([tok])
                        except Exception:
                            pass
                progress.console.print(Panel(sample.strip() or "[dim](empty)[/dim]", title=f"[dim]sample @ step {step}[/dim]", border_style="dim"))
                raw_for_gen.train()
                model.train()

                # checkpoint
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                }, "checkpoint.pt")
                progress.console.print("  [dim]checkpoint saved[/dim]")

            step += 1

    if WANDB:
        wandb.finish()


if __name__ == "__main__":
    train()
