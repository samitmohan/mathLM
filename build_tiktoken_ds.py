import torch
import tiktoken

INPUT_FILE = "openwebmath.txt"
OUTPUT_FILE = "train.bin"
CHUNK_SIZE = 5_000_000  # 5MB chunks

def read_in_chunks(file_path, chunk_size):
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def main():
    enc = tiktoken.get_encoding("gpt2")
    total_tokens = 0

    with open(OUTPUT_FILE, "wb") as f:
        for i, chunk in enumerate(read_in_chunks(INPUT_FILE, CHUNK_SIZE)):
            ids = enc.encode(chunk, disallowed_special=())
            total_tokens += len(ids)
            torch.tensor(ids, dtype=torch.int32).numpy().tofile(f)

    print(f"Total tokens: {total_tokens}")

if __name__ == "__main__":
    main()