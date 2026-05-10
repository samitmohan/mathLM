"""Upload mathLM weights, tokenizer, and source to HuggingFace Hub.

    pip install huggingface_hub
    huggingface-cli login
    python scripts/upload_hf.py
    python scripts/upload_hf.py --model checkpoint_mathlm.pt
    python scripts/upload_hf.py --dry-run
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ID = "samitmohan/mathlm"
REPO_TYPE = "model"

PACKAGE_FILES = [
    "mathlm/__init__.py",
    "mathlm/model/__init__.py",
    "mathlm/model/gpt.py",
    "mathlm/model/tokenizer.py",
    "mathlm/infer/__init__.py",
    "mathlm/infer/inference.py",
]
TOKENIZER_FILES = [
    "math_tokenizer/vocab.json",
    "math_tokenizer/merges.txt",
]


def main():
    parser = argparse.ArgumentParser(description="Upload mathLM to HuggingFace Hub")
    parser.add_argument("--model", default="checkpoint_mathlm_grpo.pt",
                        help="Checkpoint file to upload")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    files = [(args.model, os.path.basename(args.model))]
    files += [(p, p) for p in PACKAGE_FILES + TOKENIZER_FILES]

    missing = [loc for loc, _ in files if not os.path.exists(loc)]
    if missing:
        print(f"Missing files: {missing}")
        sys.exit(1)

    if args.dry_run:
        print(f"Would upload to {args.repo_id}:")
        for loc, dest in files:
            size = Path(loc).stat().st_size / 1e6
            print(f"  {loc} ({size:.2f} MB) → {dest}")
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=REPO_TYPE, exist_ok=True)
    for local_path, repo_path in files:
        size = Path(local_path).stat().st_size / 1e6
        print(f"  uploading {local_path} ({size:.2f} MB) → {repo_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
        )

    print(f"\nDone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
