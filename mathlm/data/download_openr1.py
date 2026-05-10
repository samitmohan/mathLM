"""Download OpenR1-Math-220k and tokenize to openr1_math.bin.

220k math problems with chain-of-thought reasoning, covering competition
(AMC, AIME, olympiad) and standard curriculum math.

    python -m mathlm.data.download_openr1
    python -m mathlm.data.download_openr1 --max 50000   # quick test
"""

import argparse
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()

DATASET_NAME = "open-r1/OpenR1-Math-220k"


def format_example(example: dict) -> str:
    problem = (example.get("problem") or example.get("question") or "").strip()
    solution = (example.get("solution") or example.get("answer") or "").strip()
    if not problem or not solution:
        return ""
    return f"Q: {problem}\nA: {solution}\n\n"


def main():
    parser = argparse.ArgumentParser(description="Tokenise OpenR1-Math-220k → openr1_math.bin")
    parser.add_argument("--max", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--output", default="openr1_math.bin")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=args.split, streaming=True)

    from mathlm.model.tokenizer import MathTokenizer
    if MathTokenizer.is_available():
        tok = MathTokenizer(); tok.load()
        encode = tok.encode
    else:
        import tiktoken
        console.print("[yellow]math_tokenizer/ not found — falling back to GPT-2 tiktoken[/yellow]")
        _enc = tiktoken.get_encoding("gpt2")
        encode = lambda text: _enc.encode(text, disallowed_special=())

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
