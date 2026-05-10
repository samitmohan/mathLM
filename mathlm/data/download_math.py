"""Download Hendrycks MATH dataset and tokenize to math_train.bin.

7 subjects (algebra, number_theory, counting_and_probability, geometry,
intermediate_algebra, prealgebra, precalculus) at difficulty levels 1-5.
Each problem has a step-by-step LaTeX solution.

    python -m mathlm.data.download_math
    python -m mathlm.data.download_math --subject algebra --check
"""

import argparse
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()

DATASET_NAME = "DigitalLearningGmbH/MATH-lighteval"
SUBJECTS = [
    "algebra", "number_theory", "counting_and_probability",
    "geometry", "intermediate_algebra", "prealgebra", "precalculus",
]


def format_example(example: dict) -> str:
    problem = (example.get("problem") or "").strip()
    solution = (example.get("solution") or "").strip()
    if not problem or not solution:
        return ""
    level = example.get("level", "")
    subject = example.get("type", "")
    header = f"[{subject}, {level}] " if subject and level else ""
    return f"Q: {header}{problem}\nA: {solution}\n\n"


def main():
    parser = argparse.ArgumentParser(description="Tokenise Hendrycks MATH → math_train.bin")
    parser.add_argument("--output", default="math_train.bin")
    parser.add_argument("--check", action="store_true", help="Preview samples, no file written")
    parser.add_argument("--subject", default=None, help=f"Filter to one subject: {SUBJECTS}")
    parser.add_argument("--split", default="train", choices=["train", "test"])
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

    ds = load_dataset(DATASET_NAME, split=args.split)
    if args.subject:
        ds = ds.filter(lambda x: x.get("type") == args.subject)
    console.print(f"Loaded {len(ds):,} problems")

    if args.check:
        for i in range(min(5, len(ds))):
            console.print(format_example(ds[i]))
            console.print("[dim]---[/dim]")
        return

    all_tokens: list[int] = []
    skipped = 0
    for example in track(ds, description="Tokenising", console=console):
        text = format_example(example)
        if not text:
            skipped += 1
            continue
        all_tokens.extend(encode(text))

    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(args.output)
    console.print(f"{len(all_tokens):,} tokens → {args.output} ({arr.nbytes / 1e6:.1f} MB), {skipped} skipped")


if __name__ == "__main__":
    main()
