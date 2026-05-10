"""Tokenise openwebmath.txt → train.bin (int32 token ids).

Run scripts/train_tokenizer.py first to build math_tokenizer/; otherwise
falls back to GPT-2 tiktoken.
"""

import numpy as np
from mathlm.model.tokenizer import MathTokenizer

INPUT_FILE = "openwebmath.txt"
OUTPUT_FILE = "train.bin"
CHUNK_SIZE = 5_000_000


def read_in_chunks(file_path, chunk_size):
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def main():
    if MathTokenizer.is_available():
        tok = MathTokenizer(); tok.load()
        encode = tok.encode
    else:
        import tiktoken
        print("math_tokenizer/ not found — falling back to GPT-2 tiktoken")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda text: enc.encode(text, disallowed_special=())

    total_tokens = 0
    with open(OUTPUT_FILE, "wb") as f:
        for i, chunk in enumerate(read_in_chunks(INPUT_FILE, CHUNK_SIZE)):
            ids = encode(chunk)
            total_tokens += len(ids)
            np.array(ids, dtype=np.int32).tofile(f)
            if i % 10 == 0:
                print(f"  chunk {i}  {total_tokens:,} tokens so far")

    print(f"Done. Total tokens: {total_tokens:,} → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
