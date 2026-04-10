import os, sys, torch
import tiktoken
from rich.console import Console
from rich.panel import Panel
from gippity import GPT

console = Console()

# Must stay in sync with finetuning.py
SPECIAL_TOKENS = {
    "<|system|>":    50257,
    "<|user|>":      50258,
    "<|assistant|>": 50259,
    "<|end|>":       50260,
    "<|pad|>":       50261,
}
END_ID = SPECIAL_TOKENS["<|end|>"]

def make_tokenizer(chat=False):
    base = tiktoken.get_encoding("gpt2")
    if not chat:
        return base
    return tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **SPECIAL_TOKENS},
    )

def decode_token(enc, token_id):
    """Decode a single token id; returns '' for special tokens (IDs >= 50257)."""
    if token_id >= 50257:
        return ""
    try:
        return enc.decode([token_id])
    except Exception:
        return ""


def chat_loop(model, enc, device, config, temperature=0.8, top_k=50, top_p=0.9):
    system = "You are a helpful assistant that answers math questions and explains concepts clearly."
    context = enc.encode(f"<|system|>{system}<|end|>", allowed_special="all")

    console.print(Panel(
        "[dim]Type your message and press Enter  •  [bold]exit[/bold] to quit[/dim]",
        title="[bold cyan]nanochat[/bold cyan]",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = console.input("\n[bold cyan]You ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye![/dim]")
            break
        if user_input.lower() in {"exit", "quit"}:
            console.print("[dim]bye![/dim]")
            break
        if not user_input:
            continue

        user_tokens = enc.encode(f"<|user|>{user_input}<|end|><|assistant|>", allowed_special="all")
        context = context + user_tokens
        # truncate from the left to stay within the model's sequence_length; keep the system prefix by trimming older turns
        context = context[-config.sequence_length:]
        idx = torch.tensor([context], dtype=torch.long, device=device)

        console.print("[bold green]Assistant ›[/bold green] ", end="")
        new_tokens = []
        for token_id in model.stream_generate(idx, max_new_tokens=200, temperature=temperature, top_k=top_k, top_p=top_p):
            if token_id == END_ID:
                break
            new_tokens.append(token_id)
            chunk = decode_token(enc, token_id)
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

        context = context + new_tokens + [END_ID]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_chat = True
    prompt = "Calculate the derivative of sin(x) with respect to x"
    max_new_tokens = 100
    temperature = 0.8
    top_k = 50
    top_p = 0.9

    # prefer chat checkpoint if it exists, fall back to pretrain checkpoint with a warning
    ckpt_file = "checkpoint_chat.pt" if use_chat else "checkpoint.pt"
    if not os.path.exists(ckpt_file):
        fallback = "checkpoint.pt"
        if use_chat and os.path.exists(fallback):
            console.print(f"[yellow]Warning:[/yellow] {ckpt_file} not found, falling back to {fallback} (run finetuning.py for proper chat mode)")
            ckpt_file = fallback
        else:
            console.print(f"[bold red]Error:[/bold red] {ckpt_file} not found — run training.py first.")
            return

    enc = make_tokenizer(chat=use_chat)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(Panel(
        f"  [bold]params[/bold] {params:.1f}M    [bold]device[/bold] {device}"
        f"    [bold]checkpoint[/bold] {ckpt_file}    [bold]step[/bold] {ckpt.get('step', '?')}",
        title="[bold cyan]nanochat[/bold cyan]",
        border_style="cyan",
    ))

    if use_chat:
        chat_loop(model, enc, device, config, temperature, top_k, top_p)
        return

    console.print(f"\n[bold yellow]Prompt:[/bold yellow] {prompt}\n")
    console.print("[bold green]Response:[/bold green] ", end="")
    prompt_ids = enc.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for token_id in model.stream_generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p):
        chunk = decode_token(enc, token_id)
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
