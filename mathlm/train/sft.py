"""Unified SFT across calculus, ML math, GSM8K, and Hendrycks MATH.

Starts from pretrained checkpoint.pt and fine-tunes on the mixed dataset
with loss masked to assistant tokens only (the "Q: ... A: " prefix gets
-100). Mixed-domain training at low LR (3e-5) prevents the catastrophic
forgetting that single-domain SFT produced.

    python -m mathlm.train.sft
    python -m mathlm.train.sft --epochs 2 --lr 2e-5
"""

import math
import os
import time
import random
import argparse

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from mathlm.model.gpt import GPT
from mathlm.model.tokenizer import MathTokenizer
import tiktoken

console = Console()

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


def make_tokenizer():
    if MathTokenizer.is_available():
        tok = MathTokenizer()
        tok.load()
        return tok
    return tiktoken.get_encoding("gpt2")


def _encode(enc, text: str) -> list[int]:
    try:
        return enc.encode(text)
    except TypeError:
        return enc.encode(text, disallowed_special=())



def parse_qa_strings(formatted: list[str]) -> list[tuple[str, str]]:
    """
    Convert formatted Q/A strings ("Q: ...\\nA: ...\\n\\n") back to (question, answer) tuples.
    The generator files (gen_math_data.py, gen_ml_math_data.py) return formatted strings,
    not raw tuples. We need tuples so the SFTDataset can apply per-pair loss masking.
    """
    pairs = []
    for text in formatted:
        if not text.startswith("Q: "):
            continue
        body = text[3:]  # strip "Q: "
        parts = body.split("\nA: ", 1)
        if len(parts) != 2:
            continue
        question = parts[0].strip()
        answer = parts[1].strip()
        if question and answer:
            pairs.append((question, answer))
    return pairs


def load_gsm8k() -> list[tuple[str, str]]:
    """Load GSM8K training split. Answer field already contains '#### N' format."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [(ex["question"], ex["answer"]) for ex in ds]


def load_math_dataset() -> list[tuple[str, str]]:
    """
    Load Hendrycks MATH training split (7 subjects).
    Includes subject and difficulty level as a prefix hint so the model
    learns difficulty-awareness — e.g. "[algebra, Level 3]" before the question.
    """
    from datasets import load_dataset
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    pairs = []
    for ex in ds:
        problem  = (ex.get("problem") or "").strip()
        solution = (ex.get("solution") or "").strip()
        subject  = ex.get("type", "")
        level    = ex.get("level", "")
        if not problem or not solution:
            continue
        header = f"[{subject}, {level}] " if subject and level else ""
        pairs.append((f"{header}{problem}", solution))
    return pairs



class SFTDataset(Dataset):
    """
    Converts (question, answer) pairs into masked training examples.

    Format: "Q: {question}\\nA: {answer}\\n\\n"
    Mask:   tokens in "Q: {question}\\nA: " → -100 (ignored by cross-entropy)
            tokens in "{answer}\\n\\n"       → actual target ids

    The prefix masking is critical: without it, most of the loss comes from
    predicting the question text (which is already given at inference time),
    wasting gradient budget and degrading answer quality.
    """

    def __init__(
        self,
        examples: list[tuple[str, str]],
        enc,
        sequence_length: int,
        source_name: str = "unknown",
    ):
        self.samples = []
        skipped = 0

        for question, answer in examples:
            full_text   = f"Q: {question}\nA: {answer}\n\n"
            prefix_text = f"Q: {question}\nA: "  # everything to mask

            full_ids   = _encode(enc, full_text)
            prefix_ids = _encode(enc, prefix_text)

            if len(full_ids) < 4 or len(full_ids) > sequence_length + 1:
                skipped += 1
                continue

            input_ids  = full_ids[:-1]
            target_ids = full_ids[1:]

            # input_ids[i] predicts target_ids[i] = full_ids[i+1].
            # Answer starts at full_ids[len(prefix_ids)].
            # First position with a loss gradient: i = len(prefix_ids) - 1
            # (input sees last prefix token, target is first answer token).
            answer_start = len(prefix_ids) - 1

            masked_targets = [-100] * len(target_ids)
            for i in range(max(0, answer_start), len(target_ids)):
                masked_targets[i] = target_ids[i]

            if all(t == -100 for t in masked_targets):
                skipped += 1
                continue

            # Pad to sequence_length with zeros (masked → no loss)
            pad = sequence_length - len(input_ids)
            if pad > 0:
                input_ids      = input_ids      + [0] * pad
                masked_targets = masked_targets + [-100] * pad

            self.samples.append((
                torch.tensor(input_ids,      dtype=torch.long),
                torch.tensor(masked_targets, dtype=torch.long),
            ))

        if skipped:
            console.print(f"  [dim]{source_name}: {len(self.samples)} samples ({skipped} skipped — too long or empty)[/dim]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    # Short linear warmup then cosine decay.
    # SFT warmup is short (50 steps) because the model is already pretrained —
    # we don't need to "ramp up" from a cold start.
    if step < warmup:
        return max_lr * step / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)



def evaluate_domain(model, loader, device, autocast_dtype) -> float:
    """Average loss over a domain's validation loader. Model already in eval mode."""
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, y)
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count if count > 0 else float("nan")



# One test prompt per domain — checked after every epoch.
# Watch for:
#   - Calculus:  a correct derivative expression (no forum threads)
#   - ML math:   a gradient expression with correct structure
#   - GSM8K:     answer ending in "#### N"
DOMAIN_PROBES = {
    "calculus":  ("Q: What is the derivative of 4x^5 - 3x^2 + 7?\nA:", "20x^4 - 6x"),
    "ml_math":   ("Q: What is the gradient of L=||Xw-y||^2 w.r.t. w?\nA:", "2X^T(Xw-y)"),
    "gsm8k":     ("Q: Tom has 15 apples. He gives 4 to his friend. How many does he have?\nA:", "#### 11"),
}


def sample_responses(raw_model, enc, device, max_new_tokens: int = 60) -> dict[str, str]:
    """Generate one response per domain probe. Returns {domain: response_text}."""
    responses = {}
    raw_model.eval()
    with torch.no_grad():
        for domain, (prompt, _expected) in DOMAIN_PROBES.items():
            ids = _encode(enc, prompt)
            tok = torch.tensor([ids], device=device)
            out = []
            for tid in raw_model.stream_generate(tok, max_new_tokens=max_new_tokens,
                                                  temperature=0.1, top_k=1):
                try:
                    out.append(enc.decode([tid]))
                except Exception:
                    pass
            responses[domain] = "".join(out).strip()
    raw_model.train()
    return responses



def main():
    parser = argparse.ArgumentParser(description="MathLM Unified SFT")
    parser.add_argument("--checkpoint", default="checkpoint.pt",
                        help="Pretrained checkpoint (default: checkpoint.pt)")
    parser.add_argument("--output",     default="checkpoint_mathlm.pt")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Peak LR (default 3e-5; deliberately low to prevent forgetting)")
    parser.add_argument("--batch-size", type=int,   default=32)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    enc = make_tokenizer()

    console.print("\n[bold cyan]Loading data sources...[/bold cyan]")

    # Calculus Q/A — generated in-memory, no disk I/O needed
    from mathlm.data.generate_calculus import build_pairs as calc_build
    calc_raw = parse_qa_strings(calc_build())
    console.print(f"  [dim]Calculus: {len(calc_raw):,} pairs[/dim]")

    # ML/DL math Q/A — generated in-memory
    from mathlm.data.generate_ml_math import build_pairs as ml_build
    ml_raw = parse_qa_strings(ml_build())
    console.print(f"  [dim]ML/DL math: {len(ml_raw):,} pairs[/dim]")

    # GSM8K word problems — from HuggingFace
    try:
        gsm8k_raw = load_gsm8k()
        console.print(f"  [dim]GSM8K: {len(gsm8k_raw):,} pairs[/dim]")
    except Exception as e:
        console.print(f"  [yellow]GSM8K load failed ({e}) — skipping[/yellow]")
        gsm8k_raw = []

    # Hendrycks MATH — from HuggingFace
    try:
        math_raw = load_math_dataset()
        console.print(f"  [dim]MATH dataset: {len(math_raw):,} pairs[/dim]")
    except Exception as e:
        console.print(f"  [yellow]MATH dataset load failed ({e}) — skipping[/yellow]")
        math_raw = []

    # Split each source 90/10 independently.
    # This gives a val set representative of all domains — we can track per-domain
    # val_loss to detect catastrophic forgetting (any domain rising > 0.3 = warning).

    random.seed(42)

    def split_90_10(pairs):
        pairs = list(pairs)
        random.shuffle(pairs)
        cut = int(0.9 * len(pairs))
        return pairs[:cut], pairs[cut:]

    calc_train,  calc_val  = split_90_10(calc_raw)
    ml_train,    ml_val    = split_90_10(ml_raw)
    gsm8k_train, gsm8k_val = split_90_10(gsm8k_raw) if gsm8k_raw else ([], [])
    math_train,  math_val  = split_90_10(math_raw)  if math_raw  else ([], [])

    # ML/DL math is the smallest source (354 pairs) so we oversample it (2×).
    # GSM8K gets 1.5× to ensure the "#### N" format gets enough gradient signal.
    # Weights are implemented by repeating/tiling the source lists before shuffling.
    def weighted(pairs, w):
        """Tile pairs by weight w (integer or .5 multiples)."""
        full = int(w)
        result = pairs * full
        if w - full >= 0.5:
            random.shuffle(pairs)
            result += pairs[:len(pairs) // 2]
        return result

    train_all = (
        weighted(calc_train,  1.0)   # 10,575 effective
        + weighted(ml_train,  2.0)   # 638 effective (small set, needs oversampling)
        + weighted(gsm8k_train, 1.5) # 10,087 effective
        + weighted(math_train, 1.0)  # 6,750 effective
    )
    random.shuffle(train_all)

    config_seq = None  # will be set after loading model

    if not os.path.exists(args.checkpoint):
        console.print(f"[bold red]Error:[/bold red] {args.checkpoint} not found.")
        return

    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model  = GPT(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    seq_len = config.sequence_length

    params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(
        f"[dim]SFT  params={params:.1f}M  device={device}  seq_len={seq_len}  "
        f"from={args.checkpoint}  epochs={args.epochs}  lr={args.lr:.0e}[/dim]"
    )

    train_ds = SFTDataset(train_all, enc, seq_len, "train_all")

    # Per-domain val datasets for forgetting detection
    val_ds_calc  = SFTDataset(calc_val,  enc, seq_len, "val_calc")
    val_ds_ml    = SFTDataset(ml_val,    enc, seq_len, "val_ml")
    val_ds_gsm8k = SFTDataset(gsm8k_val, enc, seq_len, "val_gsm8k")
    val_ds_math  = SFTDataset(math_val,  enc, seq_len, "val_math")

    console.print(f"  [dim]Train: {len(train_ds):,} samples total[/dim]")

    loader_kwargs = dict(
        num_workers=4,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loaders = {
        "calc":  DataLoader(val_ds_calc,  batch_size=args.batch_size, **loader_kwargs),
        "ml":    DataLoader(val_ds_ml,    batch_size=args.batch_size, **loader_kwargs),
        "gsm8k": DataLoader(val_ds_gsm8k, batch_size=args.batch_size, **loader_kwargs),
        "math":  DataLoader(val_ds_math,  batch_size=args.batch_size, **loader_kwargs),
    }

    # Lower weight decay for SFT (0.05 vs pretraining's 0.1) — we're nudging the
    # pretrained model, not regularising hard. Strong decay would push weights
    # toward zero and fight the pretrained knowledge we want to preserve.
    decay    = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    model = torch.compile(model, mode="default")

    steps_per_epoch = len(train_loader)
    max_steps       = args.epochs * steps_per_epoch
    warmup          = 50   # short warmup — model is already pretrained
    min_lr          = args.lr / 10
    eval_interval   = steps_per_epoch  # evaluate once per epoch

    console.print(
        f"  [dim]{steps_per_epoch} steps/epoch × {args.epochs} epochs = {max_steps} total steps[/dim]"
    )

    if WANDB:
        wandb.init(
            project="nanochat",
            name=f"sft-unified-{params:.0f}M-{args.epochs}ep",
            config={
                "epochs": args.epochs, "max_steps": max_steps,
                "lr": args.lr, "batch_size": args.batch_size,
                "params_M": params, "warmup": warmup,
            },
            resume="allow",
        )

    step       = 0
    train_iter = iter(train_loader)
    t_last_log = time.time()
    tokens_since_log = 0

    # Track initial val losses to detect catastrophic forgetting later
    initial_val_losses: dict[str, float] = {}

    with Progress(
        TextColumn("[bold cyan]sft-unified[/bold cyan]"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%  step {task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task("sft", total=max_steps)

        model.train()
        while step < max_steps:
            lr = get_lr(step, warmup, max_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()

            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, y)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens_since_log += seq_len * args.batch_size
            progress.update(task, advance=1)

            if step % 50 == 0:
                now     = time.time()
                elapsed = now - t_last_log
                tok_per_sec = tokens_since_log / elapsed if elapsed > 0 else 0
                loss_val   = loss.item()
                loss_color = "green" if loss_val < 1.0 else "yellow" if loss_val < 2.0 else "red"
                progress.console.print(
                    f"  [dim]step[/dim] [bold]{step:4d}[/bold]  │"
                    f"  [{loss_color}]loss {loss_val:.4f}[/{loss_color}]  │"
                    f"  [dim]lr[/dim] {lr:.2e}  │"
                    f"  [dim]gnorm[/dim] {grad_norm:.3f}  │"
                    f"  [cyan]tok/s {tok_per_sec:.0f}[/cyan]"
                )
                if WANDB:
                    wandb.log({"train/loss": loss_val, "train/lr": lr,
                               "train/grad_norm": float(grad_norm)}, step=step)
                t_last_log       = now
                tokens_since_log = 0

            if step > 0 and step % eval_interval == 0:
                epoch = step // steps_per_epoch
                model.eval()

                # Per-domain validation losses
                domain_losses = {}
                for domain, vloader in val_loaders.items():
                    if len(vloader) == 0:
                        continue
                    domain_losses[domain] = evaluate_domain(model, vloader, device, autocast_dtype)

                # Store initial losses at epoch 1 for forgetting detection
                if epoch == 1:
                    initial_val_losses = dict(domain_losses)

                # Build a rich table for the per-domain report
                table = Table(title=f"Epoch {epoch} — Per-domain val loss", show_header=True)
                table.add_column("Domain", style="bold")
                table.add_column("Val loss")
                table.add_column("Δ from epoch 1")
                forgetting_alarm = False
                for domain, vloss in domain_losses.items():
                    init = initial_val_losses.get(domain)
                    if init is not None:
                        delta = vloss - init
                        delta_str = f"[red]+{delta:.3f} ⚠[/red]" if delta > 0.3 else f"[green]{delta:+.3f}[/green]"
                        if delta > 0.3:
                            forgetting_alarm = True
                    else:
                        delta_str = "[dim]—[/dim]"
                    color = "green" if vloss < 1.0 else "yellow" if vloss < 2.0 else "red"
                    table.add_row(domain, f"[{color}]{vloss:.4f}[/{color}]", delta_str)
                progress.console.print(table)

                if forgetting_alarm:
                    progress.console.print(
                        "[bold red]WARNING: catastrophic forgetting detected (Δ > 0.3). "
                        "Consider stopping and using --lr 1e-5.[/bold red]"
                    )
                if WANDB:
                    wandb.log({f"val/{d}": l for d, l in domain_losses.items()}, step=step)

                # Qualitative domain probes — verify all three domains still work
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                responses = sample_responses(raw_model, enc, device)
                for domain, response in responses.items():
                    expected = DOMAIN_PROBES[domain][1]
                    ok = expected.lower() in response.lower()
                    tag = "[green]ok[/green]" if ok else "[red]fail[/red]"
                    progress.console.print(
                        f"[dim]{domain} @ epoch {epoch} ({tag}):[/dim] {response or '(empty)'}"
                    )

                model.train()

                # Checkpoint
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                    "meta": {
                        "val_losses": domain_losses,
                        "sft_epoch": epoch,
                        "sft_type": "unified_math",
                        "domains": ["calculus", "ml_math", "gsm8k", "math"],
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                }, args.output)
                progress.console.print(f"  [dim]saved → {args.output}[/dim]")

            step += 1

    if WANDB:
        wandb.finish()

    console.print(f"\nSFT complete. {args.output} saved.")


if __name__ == "__main__":
    main()
