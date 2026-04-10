# SFT on chat data — loss computed only on assistant tokens (user/system masked with -100)
import random
import json, math, os, time, torch
from torch.utils.data import DataLoader, Dataset
from gippity import GPT
import tiktoken
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

# Special tokens appended after GPT-2's 50257-token base vocab.
# IDs must stay in sync with inference.py.
SPECIAL_TOKENS = {
    "<|system|>":    50257,
    "<|user|>":      50258,
    "<|assistant|>": 50259,
    "<|end|>":       50260,
    "<|pad|>":       50261,
}

def make_tokenizer():
    """GPT-2 tiktoken encoding extended with chat special tokens."""
    base = tiktoken.get_encoding("gpt2")
    return tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **SPECIAL_TOKENS},
    )

'''
Chat format: <|system|>You are helpful.<|end|><|user|>What is 2+2?<|end|><|assistant|>4<|end|>
Data format (chat_data.json):
[
    {"system": "You are helpful.", "user": "What is 2+2?", "assistant": "4"},
    ...
]
'''

SYSTEM_TOK  = "<|system|>"
USER_TOK    = "<|user|>"
ASST_TOK    = "<|assistant|>"
END_TOK     = "<|end|>"
PAD_TOK     = "<|pad|>"
DEFAULT_SYSTEM = "You are a helpful assistant."

def format_chat(system, user, assistant):
    return f"{SYSTEM_TOK}{system}{END_TOK}{USER_TOK}{user}{END_TOK}{ASST_TOK}{assistant}{END_TOK}"

class ChatDataset(Dataset):
    def __init__(self, examples, enc, sequence_length):
        asst_id = SPECIAL_TOKENS[ASST_TOK]
        end_id  = SPECIAL_TOKENS[END_TOK]
        pad_id  = SPECIAL_TOKENS[PAD_TOK]
        self.samples = []
        for example in examples:
            system = example.get("system", DEFAULT_SYSTEM)
            text = format_chat(system, example["user"], example["assistant"])
            ids = enc.encode(text, allowed_special="all")
            if len(ids) < 2 or len(ids) > sequence_length + 1:
                continue
            input_ids  = ids[:-1]
            target_ids = ids[1:]

            # mask everything except assistant response tokens
            # cross_entropy ignores -100 targets so loss is only computed on what the assistant produced
            masked_target_ids = [-100] * len(target_ids)
            in_assistant = False
            for i, tok in enumerate(ids[:-1]):
                if tok == asst_id:
                    in_assistant = True
                    continue
                if tok == end_id and in_assistant:
                    masked_target_ids[i] = target_ids[i]  # include closing <|end|> in loss
                    in_assistant = False
                    continue
                if in_assistant:
                    masked_target_ids[i] = target_ids[i]

            if all(t == -100 for t in masked_target_ids):
                continue  # no assistant tokens → skip

            pad = sequence_length - len(input_ids)
            input_ids         += [pad_id] * pad
            masked_target_ids += [-100]   * pad
            self.samples.append((torch.tensor(input_ids), torch.tensor(masked_target_ids)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / warmup
    if step > max_steps:
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)

def finetune():
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = make_tokenizer()

    if not os.path.exists("chat_data.json"):
        console.print("[bold red]Error:[/bold red] chat_data.json not found.")
        return
    with open("chat_data.json") as f:
        examples = json.load(f)
    random.seed(42)
    random.shuffle(examples)
    split = int(0.9 * len(examples))
    train_examples, val_examples = examples[:split], examples[split:]

    if not os.path.exists("checkpoint.pt"):
        console.print("[bold red]Error:[/bold red] checkpoint.pt not found — run training.py first.")
        return
    ckpt = torch.load("checkpoint.pt", map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    # strict=False: special-token embedding rows (IDs 50257–50261) will be randomly initialized
    # since they were never seen during pretraining — finetuning will train them from scratch
    model.load_state_dict(ckpt["model"], strict=False)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(Panel(
        f"  [bold]params[/bold] {params:.1f}M  [bold]device[/bold] {device}"
        f"  [bold]vocab[/bold] {config.vocab_size}  [bold]seq_len[/bold] {config.sequence_length}",
        title="[bold cyan]nanochat — finetuning[/bold cyan]",
        border_style="cyan",
    ))

    decay    = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay,    "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=1e-4, betas=(0.9, 0.95))

    model = torch.compile(model)
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    batch_size    = 128
    max_steps     = 2000
    warmup        = 100
    max_lr        = 1e-4
    min_lr        = 1e-5
    grad_accum    = 2
    eval_interval = 1000

    train_loader = DataLoader(ChatDataset(train_examples, enc, config.sequence_length), batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(ChatDataset(val_examples,   enc, config.sequence_length), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))
    train_iter = iter(train_loader)

    if WANDB:
        wandb.init(
            project="nanochat",
            name=f"finetune-{config.number_layers}L-{config.embedding_dim}d",
            config={
                "params_M": params, "max_steps": max_steps, "max_lr": max_lr,
                "batch_size": batch_size, "grad_accum": grad_accum,
            },
            resume="allow",
        )
    else:
        console.print("  [dim]wandb not installed — run [bold]pip install wandb[/bold] for experiment tracking[/dim]")

    step = 0
    model.train()
    t_last_log = time.time()
    tokens_since_log = 0
    with Progress(
        TextColumn("[bold cyan]finetuning[/bold cyan]"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%  step {task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task("finetune", total=max_steps)
        while step < max_steps:
            lr = get_lr(step, warmup, max_steps, max_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad()

            loss_total = 0.0
            for _ in range(grad_accum):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    _, loss = model(x, y)
                loss = loss / grad_accum
                loss.backward()
                loss_total += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens_since_log += config.sequence_length * batch_size * grad_accum
            progress.update(task, advance=1)

            if step % 200 == 0:
                now = time.time()
                elapsed = now - t_last_log
                tok_per_sec = tokens_since_log / elapsed if elapsed > 0 else 0
                loss_color = "green" if loss_total < 1.0 else "yellow" if loss_total < 2.0 else "red"
                progress.console.print(
                    f"  [dim]step[/dim] [bold]{step:4d}[/bold]  │"
                    f"  [{loss_color}]loss {loss_total:.4f}[/{loss_color}]  │"
                    f"  [dim]lr[/dim] {lr:.2e}  │  [dim]gnorm[/dim] {grad_norm:.3f}  │"
                    f"  [cyan]tok/s {tok_per_sec:.0f}[/cyan]"
                )
                if WANDB:
                    wandb.log({"train/loss": loss_total, "train/lr": lr, "train/grad_norm": float(grad_norm), "train/tok_per_sec": tok_per_sec}, step=step)
                t_last_log = now
                tokens_since_log = 0

            if step % eval_interval == 0:
                model.eval()
                total, count = 0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        with torch.autocast(device_type=device, dtype=autocast_dtype):
                            _, loss = model(x, y)
                        total += loss.item() * x.size(0)
                        count += x.size(0)
                val_loss = total / count if count > 0 else float("nan")
                progress.console.print(f"\n  [bold yellow]val_loss {val_loss:.4f}[/bold yellow] at step {step}\n")
                if WANDB:
                    wandb.log({"val/loss": val_loss}, step=step)
                model.train()

                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "step": step}, "checkpoint_chat.pt")
                progress.console.print("  [dim]checkpoint saved → checkpoint_chat.pt[/dim]")

            step += 1

    if WANDB:
        wandb.finish()


if __name__ == "__main__":
    finetune()
