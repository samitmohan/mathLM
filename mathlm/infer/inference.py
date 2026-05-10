from __future__ import annotations

"""mathLM inference — library and CLI.

Library:
    from mathlm.infer.inference import load_model, generate_text, chat_response
    model, enc, config, device = load_model()
    print(generate_text(model, enc, device, "Q: derivative of x^5?\nA:"))

CLI:
    python -m mathlm.infer.inference --prompt "Q: 2+2? A:"
    python -m mathlm.infer.inference --chat
"""

import argparse
import os
import sys
import torch
import tiktoken
from rich.console import Console
from mathlm.model.gpt import GPT

console = Console()

# GPT-2 fallback special token IDs (used only when math_tokenizer/ is absent)
_GPT2_SPECIAL_TOKENS = {
    "<|system|>": 50257, "<|user|>": 50258, "<|assistant|>": 50259,
    "<|end|>": 50260, "<|pad|>": 50261,
}


def make_tokenizer(chat: bool = False):
    from mathlm.model.tokenizer import MathTokenizer
    if MathTokenizer.is_available():
        tok = MathTokenizer()
        tok.load()
        return tok
    # fall back to GPT-2 tiktoken
    base = tiktoken.get_encoding("gpt2")
    if not chat:
        return base
    return tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **_GPT2_SPECIAL_TOKENS},
    )


def _special_ids(enc) -> dict:
    """Return {token_str: id} for all special tokens, regardless of tokenizer type."""
    from mathlm.model.tokenizer import MathTokenizer
    if isinstance(enc, MathTokenizer):
        return enc.special_token_ids
    return _GPT2_SPECIAL_TOKENS


def decode_token(enc, token_id: int) -> str:
    """Decode a single token id; returns '' for special tokens."""
    special_ids = set(_special_ids(enc).values())
    if token_id in special_ids:
        return ""
    try:
        return enc.decode([token_id])
    except Exception:
        return ""


def load_model(ckpt_file: str | None = None, device: str | None = None):
    """Load a checkpoint and return (model, tokenizer, config, device).

    Auto-detects checkpoint_chat.pt → checkpoint.pt if ckpt_file is None.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if ckpt_file is None:
        for candidate in ("checkpoint_chat.pt", "checkpoint.pt"):
            if os.path.exists(candidate):
                ckpt_file = candidate
                break
        if ckpt_file is None:
            raise FileNotFoundError("No checkpoint found — train one with mathlm.train.pretrain first.")

    use_chat = "chat" in ckpt_file
    enc = make_tokenizer(chat=use_chat)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(
        f"[dim]loaded {ckpt_file}  params={params:.1f}M  device={device}  step={ckpt.get('step', '?')}[/dim]"
    )

    return model, enc, config, device


def generate_text(
    model,
    enc,
    device: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    stop_at_newline: bool = False,
) -> str:
    """Run greedy/sampled generation on a raw text prompt. Returns the generated text (not including the prompt)."""
    end_id = _special_ids(enc)["<|end|>"]
    prompt_ids = enc.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = []
    for token_id in model.stream_generate(idx, max_new_tokens, temperature, top_k, top_p):
        if token_id == end_id:
            break
        chunk = decode_token(enc, token_id)
        if stop_at_newline and chunk == "\n":
            break
        result.append(chunk)
    return "".join(result)


def chat_response(
    model,
    enc,
    device: str,
    config,
    user_input: str,
    history: list[tuple[str, str]] | None = None,
    system: str = "You are a helpful assistant that answers math questions clearly.",
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    max_new_tokens: int = 200,
) -> str:
    """
    Single-turn or multi-turn chat response.
    history: list of (user, assistant) pairs from previous turns.
    Returns only the assistant's new response as a string.
    """
    end_id = _special_ids(enc)["<|end|>"]
    context = enc.encode(f"<|system|>{system}<|end|>", allowed_special="all")
    for user_turn, asst_turn in (history or []):
        context += enc.encode(
            f"<|user|>{user_turn}<|end|><|assistant|>{asst_turn}<|end|>",
            allowed_special="all",
        )
    context += enc.encode(f"<|user|>{user_input}<|end|><|assistant|>", allowed_special="all")
    context = context[-config.sequence_length:]

    idx = torch.tensor([context], dtype=torch.long, device=device)
    tokens = []
    for token_id in model.stream_generate(idx, max_new_tokens, temperature, top_k, top_p):
        if token_id == end_id:
            break
        tokens.append(token_id)
    return "".join(decode_token(enc, t) for t in tokens)


def chat_loop(model, enc, device, config, temperature=0.8, top_k=50, top_p=0.9):
    system = "You are a helpful assistant that answers math questions and explains concepts clearly."
    history: list[tuple[str, str]] = []
    console.print("[dim]Type your message; 'exit' to quit.[/dim]")

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

        console.print("[bold green]Assistant ›[/bold green] ", end="")
        response = chat_response(model, enc, device, config, user_input, history,
                                  system=system, temperature=temperature, top_k=top_k, top_p=top_p)
        sys.stdout.write(response + "\n")
        sys.stdout.flush()
        history.append((user_input, response))


def main():
    parser = argparse.ArgumentParser(description="mathLM inference")
    parser.add_argument("--prompt",      default="Q: What is the derivative of x^5?\nA:",
                        help="Prompt for single-shot generation")
    parser.add_argument("--max-tokens",  type=int,   default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k",       type=int,   default=50)
    parser.add_argument("--top-p",       type=float, default=0.9)
    parser.add_argument("--checkpoint",  default=None, help="Path to checkpoint file")
    parser.add_argument("--chat",        action="store_true", help="Interactive chat mode")
    args = parser.parse_args()

    model, enc, config, device = load_model(ckpt_file=args.checkpoint)

    if args.chat:
        chat_loop(model, enc, device, config, args.temperature, args.top_k, args.top_p)
        return

    console.print(f"\n[bold yellow]Prompt:[/bold yellow] {args.prompt}\n")
    console.print("[bold green]Response:[/bold green] ", end="")
    end_id = _special_ids(enc)["<|end|>"]
    prompt_ids = enc.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for token_id in model.stream_generate(idx, args.max_tokens, args.temperature, args.top_k, args.top_p):
        if token_id == end_id:
            break
        chunk = decode_token(enc, token_id)
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
