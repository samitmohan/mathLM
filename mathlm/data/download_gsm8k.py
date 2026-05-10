"""Download GSM8K train split and tokenize to gsm8k_train.bin.

GSM8K (Grade School Math 8K): 7,473 word problems with step-by-step
solutions ending in '#### <number>' — the format mathlm.eval.gsm8k extracts.

    python -m mathlm.data.download_gsm8k
    python -m mathlm.data.download_gsm8k --check   # preview 5 samples, no write
"""

import argparse
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()


def format_example(example: dict) -> str:
    q = example["question"].strip()
    a = example["answer"].strip()
    return f"Q: {q}\nA: {a}\n\n"


def main():
    parser = argparse.ArgumentParser(description="Tokenise GSM8K train split → gsm8k_train.bin")
    parser.add_argument("--output", default="gsm8k_train.bin")
    parser.add_argument("--check", action="store_true", help="Preview 5 samples, no file written")
    args = parser.parse_args()

    from datasets import load_dataset

    from mathlm.model.tokenizer import MathTokenizer
    if MathTokenizer.is_available():
        tok = MathTokenizer(); tok.load()
        encode = tok.encode
    else:
        import tiktoken
        console.print("[yellow]math_tokenizer/ not found — falling back to GPT-2 tiktoken[/yellow]")
        _enc = tiktoken.get_encoding("gpt2")
        encode = lambda text: _enc.encode(text, disallowed_special=())

    ds = load_dataset("openai/gsm8k", "main", split="train")
    console.print(f"Loaded {len(ds):,} problems")

    if args.check:
        for i in range(min(5, len(ds))):
            console.print(format_example(ds[i]))
            console.print("[dim]---[/dim]")
        return

    all_tokens: list[int] = []
    for example in track(ds, description="Tokenising", console=console):
        all_tokens.extend(encode(format_example(example)))

    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(args.output)
    console.print(f"{len(all_tokens):,} tokens → {args.output} ({arr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
