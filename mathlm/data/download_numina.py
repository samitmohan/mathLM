"""Download NuminaMath-CoT and tokenize to numina_math.bin.

859k math problems with chain-of-thought solutions, ranging from
school-level to olympiad. Largest single CoT math dataset.

    python -m mathlm.data.download_numina
    python -m mathlm.data.download_numina --max 200000   # first 200k
"""

import argparse
import sys
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()

DATASET_NAME = "AI-MO/NuminaMath-CoT"


def format_example(example: dict) -> str:
    problem = (example.get("problem") or "").strip()
    solution = (example.get("solution") or "").strip()
    if not problem or not solution:
        return ""
    return f"Q: {problem}\nA: {solution}\n\n"


def main():
    parser = argparse.ArgumentParser(description="Tokenise NuminaMath-CoT → numina_math.bin")
    parser.add_argument("--output", default="numina_math.bin")
    parser.add_argument("--max", type=int, default=None, help="Max examples (default: all 859k)")
    parser.add_argument("--check", action="store_true")
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

    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    if args.check:
        for i, ex in enumerate(ds):
            if i >= 5:
                break
            console.print(format_example(ex)[:300])
            console.print("[dim]---[/dim]")
        return

    all_tokens: list[int] = []
    skipped = 0
    count = 0
    for example in track(ds, description="Tokenising", console=console, total=args.max):
        if args.max and count >= args.max:
            break
        text = format_example(example)
        if not text:
            skipped += 1
            continue
        all_tokens.extend(encode(text))
        count += 1

    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(args.output)
    console.print(
        f"{count:,} examples ({skipped} skipped), {len(all_tokens):,} tokens → "
        f"{args.output} ({arr.nbytes / 1e6:.0f} MB)"
    )


if __name__ == "__main__":
    main()
