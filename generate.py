"""
Demo inference script. Loads checkpoint.pt and tokenizer.json, generates text.

Usage:
    python generate.py
    python generate.py --prompt "To be or not"
    python generate.py --prompt "CHAPTER I" --tokens 500 --top_p 0.9 --temperature 0.8
"""
import argparse
import torch
from gpt import GPT
from tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",      type=str,   default="")
    parser.add_argument("--tokens",      type=int,   default=200)
    parser.add_argument("--top_k",       type=int,   default=50)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = BPETokenizer()
    tok.load("tokenizer.json")

    # weights_only=False is required because the checkpoint contains a GPTConfig
    # dataclass, not just tensors. Only load checkpoints you produced yourself.
    ckpt = torch.load("checkpoint.pt", map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {n_params:.1f}M parameters")
    print(f"tokenizer: {len(tok)} tokens")
    print()

    prompt_ids = tok.encode(args.prompt) if args.prompt else [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print("--- prompt ---")
    print(args.prompt if args.prompt else "(empty)")
    print("--- generated ---")

    out = model.generate(
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Print only the newly generated tokens, not the prompt.
    new_ids = out[0].tolist()[len(prompt_ids):]
    print(tok.decode(new_ids))


if __name__ == "__main__":
    main()
