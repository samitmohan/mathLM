'''
Pretraining pipeline:
    Text → tokenize → token IDs → dataset → batches → GPT → loss → optimizer
'''
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from mathlm.model.gpt import GPT, GPTConfig
from mathlm.model.tokenizer import MathTokenizer
import tiktoken

console = Console()

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


# TokenDataset turns a flat list of token IDs into (input, target) pairs for next-token prediction.
#
# Example with sequence_length=4:
#   tokens : [A, B, C, D, E, F]
#   pair 0 : x=[A,B,C,D]  y=[B,C,D,E]   ← model sees A,B,C,D and must predict B,C,D,E
#   pair 1 : x=[B,C,D,E]  y=[C,D,E,F]
class TokenDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.sequence_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.sequence_length + 1], dtype=torch.long)
        return x, y


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    '''
    Learning rate schedule: linear warmup → cosine decay.

    Warmup: ramp from ~0 up to max_lr over warmup_steps.
            Prevents large, unstable updates at the very start of training.
    Cosine decay: smoothly reduce max_lr → min_lr for the rest of training.
                  Smoother than step decay; avoids abrupt drops that can hurt convergence.
    '''
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    # how far through the decay phase (0.0 = just started, 1.0 = finished)
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1.0 → 0.0
    return min_lr + coeff * (max_lr - min_lr)


def evaluate(model, val_loader, device, autocast_dtype, max_batches=50):
    '''Compute average validation loss over up to max_batches batches.'''
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    model.train()
    return total_loss / n


def train():
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('high')  # use TF32 on CUDA — faster, nearly same accuracy
    torch.backends.cudnn.benchmark = True        # auto-tune CUDA kernels for fixed input shapes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer — math-specific BPE if trained, otherwise fall back to GPT-2 vocab
    if MathTokenizer.is_available():
        enc = MathTokenizer()
        enc.load()
        console.print("  [dim]Using math tokenizer[/dim]")
    else:
        console.print("  [yellow]math_tokenizer/ not found — using GPT-2 tiktoken[/yellow]")
        enc = tiktoken.get_encoding("gpt2")

    # Data loading strategy:
    # - Large files (>50 MB): kept as np.memmap — the OS pages them on demand, zero RAM copy.
    # - Small files: loaded fully and tiled to boost their sampling weight.
    # - Each source is split 90/10 independently so the val set is representative of all sources.
    # - Sources are combined with ConcatDataset — avoids a single 2+ GB np.concatenate call.
    #
    # Peak RAM: only the small tiled arrays (~300 MB total) + model weights.

    seq_len = 512  # must match GPTConfig.sequence_length below

    def _add_source(fname, tile, train_ds, val_ds, *, large=False):
        """Load fname, optionally tile, split 90/10, append TokenDatasets to lists."""
        if not os.path.exists(fname):
            return
        if large:
            # memmap: no RAM copy; OS loads pages on first access
            data = np.memmap(fname, dtype=np.int32, mode="r")
        else:
            data = np.fromfile(fname, dtype=np.int32)
            if tile > 1:
                data = np.tile(data, tile)
        n = len(data)
        cut = int(0.9 * n)
        train_ds.append(TokenDataset(data[:cut], seq_len))
        val_ds.append(TokenDataset(data[cut:], seq_len))
        tag = "memmap" if large else f"×{tile}"
        console.print(f"  [dim]{fname} — {n:,} tokens ({tag})[/dim]")

    train_ds_list: list = []
    val_ds_list:   list = []

    # Large files — memmap, no tiling needed (DataLoader cycles through them naturally)
    _add_source("train.bin",        1,  train_ds_list, val_ds_list, large=True)
    _add_source("openr1_math.bin",  1,  train_ds_list, val_ds_list, large=True)
    _add_source("numina_math.bin",  1,  train_ds_list, val_ds_list, large=True)

    # Small files — load into RAM and tile to give them adequate weight
    _add_source("math_qa.bin",      15, train_ds_list, val_ds_list)
    _add_source("gsm8k_train.bin",  20, train_ds_list, val_ds_list)
    _add_source("math_train.bin",   15, train_ds_list, val_ds_list)
    _add_source("ml_math.bin",      30, train_ds_list, val_ds_list)

    train_dataset = ConcatDataset(train_ds_list)
    val_dataset   = ConcatDataset(val_ds_list)

    # Round vocab size up to the nearest multiple of 64 for tensor-core efficiency
    raw_vocab = enc.vocab_size if hasattr(enc, 'vocab_size') else enc.n_vocab
    vocab_size = ((raw_vocab + 63) // 64) * 64

    config = GPTConfig(
        vocab_size=vocab_size,
        sequence_length=512,         # longer sequences give the model more context for reasoning
        number_layers=8,
        number_heads=8,
        number_kv_heads=4,           # GQA: 4 KV heads shared across 8 query heads → saves memory
        embedding_dim=512,           # ~40M params total
        gradient_checkpointing=True, # saves ~40% activation memory; required at batch=64, vocab=32k
    )

    model = GPT(config).to(device)

    # Apply weight decay only to weight matrices (2D tensors), not to biases or norm scales.
    # Decaying biases/norms provides no regularisation benefit and can hurt training.
    decay_params    = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=3e-4,
        betas=(0.9, 0.95),  # 0.95 forgets old gradients faster than the default 0.999 — better for LLMs
    )

    # Resume from checkpoint if one exists
    step = 0
    if os.path.exists("checkpoint.pt"):
        ckpt = torch.load("checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        console.print(f"  [dim]Resuming from step {step}[/dim]")

    params = sum(p.numel() for p in model.parameters()) / 1e6

    # torch.compile traces the model into an optimised computation graph.
    # "default" mode is used (not "reduce-overhead") because gradient checkpointing introduces
    # dynamic control flow that breaks CUDA graph capture used by reduce-overhead.
    model = torch.compile(model, mode="default")

    # --- Hyperparameters ---
    max_steps        = 100_000
    warmup_steps     = 2_000
    max_lr           = 3e-4
    min_lr           = 3e-5
    batch_size       = 128
    grad_accum_steps = 8 # effective batch ≈ 128 × 512 × 8 = 524k tokens/step (same as before)
    eval_interval    = 500
    num_workers      = 4

    # bfloat16 has the same exponent range as float32 but uses half the memory — ideal for training
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,           # discard the partial last batch so every batch is the same shape
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True,  # keep worker processes alive between epochs (avoids respawn cost)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )
    train_iter = iter(train_loader)

    console.print(
        f"[dim]pretrain  layers={config.number_layers}  dim={config.embedding_dim}  "
        f"heads={config.number_heads}  params={params:.1f}M  device={device}  "
        f"vocab={config.vocab_size}[/dim]"
    )

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
                "min_lr": min_lr, "warmup_steps": warmup_steps, "max_steps": max_steps,
                "params_M": params,
            },
            resume="allow",
        )
    else:
        console.print("[dim]wandb not installed — pip install wandb for experiment tracking[/dim]")

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
            lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()

            # Gradient accumulation: run grad_accum_steps mini-batches before one optimizer step.
            # This simulates a much larger batch without needing more GPU memory.
            # We divide each loss by grad_accum_steps so the accumulated gradient equals
            # what we'd get from one big batch of (batch_size × grad_accum_steps) samples.
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

            # Gradient clipping prevents a single bad batch from causing a huge destructive update
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
                    wandb.log({
                        "train/loss": loss_total, "train/lr": lr,
                        "train/grad_norm": float(grad_norm), "train/tok_per_sec": tok_per_sec,
                    }, step=step)
                t_last_log = now
                tokens_since_log = 0

            if step > 0 and step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, autocast_dtype)
                progress.console.print(f"\n  [bold yellow]val_loss {val_loss:.4f}[/bold yellow] at step {step}\n")
                if WANDB:
                    wandb.log({"val/loss": val_loss}, step=step)

                # torch.compile wraps the model; _orig_mod is the original unwrapped model.
                # We need the raw model to (a) run generation with dynamic shapes that CUDA
                # graphs can't handle, and (b) save weights without the compiled wrapper.
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

                # Print a sample generation to see qualitative progress
                raw_model.eval()
                prompt = "Q: What is the derivative of x^5?\nA:"
                tokens = torch.tensor([enc.encode(prompt)], device=device)
                sample = ""
                with torch.no_grad():
                    for tok_id in raw_model.stream_generate(tokens, max_new_tokens=40, temperature=0.8, top_k=50):
                        try:
                            sample += enc.decode([tok_id])
                        except Exception:
                            pass
                progress.console.print(f"[dim]sample @ step {step}:[/dim] {sample.strip() or '(empty)'}")
                raw_model.train()
                model.train()

                # Save checkpoint so training can be resumed if interrupted
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                    "meta": {
                        "val_loss": val_loss,
                        "train_loss": loss_total,
                        "params_M": round(params, 1),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "data": "openwebmath + math_qa + openr1",
                        "max_steps": max_steps,
                    },
                }, "checkpoint.pt")
                progress.console.print("  [dim]checkpoint saved[/dim]")

            step += 1

    if WANDB:
        wandb.finish()


if __name__ == "__main__":
    train()
