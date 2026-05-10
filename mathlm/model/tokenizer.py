import os


class MathTokenizer:
    """Byte-level BPE tokenizer backed by HuggingFace tokenizers.

    Drop-in for tiktoken: same .encode()/.decode() surface.
    Train via `python scripts/train_tokenizer.py`; load via `tok = MathTokenizer(); tok.load()`.
    """

    SPECIAL_TOKEN_NAMES = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"]

    def __init__(self):
        self._tok = None

    def load(self, dir_path: str = "math_tokenizer"):
        from tokenizers import ByteLevelBPETokenizer
        self._tok = ByteLevelBPETokenizer.from_file(
            vocab_filename=f"{dir_path}/vocab.json",
            merges_filename=f"{dir_path}/merges.txt",
        )
        self._tok.add_special_tokens(self.SPECIAL_TOKEN_NAMES)

    def encode(self, text: str, allowed_special: str = "none", disallowed_special=()) -> list:
        return self._tok.encode(text).ids

    def decode(self, ids) -> str:
        return self._tok.decode(list(ids))

    def token_to_id(self, token: str) -> int:
        return self._tok.token_to_id(token)

    @property
    def special_token_ids(self) -> dict:
        return {tok: self._tok.token_to_id(tok) for tok in self.SPECIAL_TOKEN_NAMES}

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    @staticmethod
    def is_available(dir_path: str = "math_tokenizer") -> bool:
        return (os.path.exists(f"{dir_path}/vocab.json") and
                os.path.exists(f"{dir_path}/merges.txt"))
