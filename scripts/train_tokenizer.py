"""Train a byte-level BPE tokenizer on openwebmath.txt.

    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 32768 --input openwebmath.txt

Output: math_tokenizer/{vocab.json, merges.txt}.
"""

import argparse
import os

SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"]
DEFAULT_VOCAB_SIZE = 32768
DEFAULT_OUTPUT_DIR = "math_tokenizer"


def train(input_file: str, vocab_size: int, output_dir: str):
    from tokenizers import ByteLevelBPETokenizer

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"{input_file} not found. Place a corpus there or run "
            "`python -m mathlm.data.build_pretrain` after acquiring openwebmath."
        )

    os.makedirs(output_dir, exist_ok=True)
    size_mb = os.path.getsize(input_file) / 1e6
    print(f"training BPE on {input_file} ({size_mb:.0f} MB), vocab={vocab_size}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[input_file],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(output_dir)

    print(f"saved → {output_dir}/{{vocab.json, merges.txt}}, vocab_size={tokenizer.get_vocab_size()}")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok} → id {tokenizer.token_to_id(tok)}")


def main():
    parser = argparse.ArgumentParser(description="Train math BPE tokenizer")
    parser.add_argument("--input", default="openwebmath.txt")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    train(args.input, args.vocab_size, args.output)


if __name__ == "__main__":
    main()
