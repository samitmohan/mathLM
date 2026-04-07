import json
from collections import defaultdict


class BPETokenizer:
    """
    Byte-level BPE tokenizer trained from scratch.
    Starts with 256 byte tokens and learns merge rules until vocab_size is reached.
    """

    def __init__(self):
        self.merges = {}   # (int, int) -> int: merge rules in training order
        self.vocab = {}    # int -> bytes: token id to byte sequence

    def _count_pairs(self, ids):
        counts = defaultdict(int)
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
        return counts

    def _apply_merge(self, ids, pair, new_id):
        """Replace every occurrence of pair in ids with new_id."""
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def train(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256, "vocab_size must be at least 256 (byte alphabet)"

        # Initialize: each byte is its own token
        ids = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        n_merges = vocab_size - 256
        for merge_idx in range(n_merges):
            counts = self._count_pairs(ids)
            if not counts:
                # Pad vocab to vocab_size with empty byte sequences when
                # training text is exhausted before reaching the target size.
                for pad_id in range(256 + merge_idx, vocab_size):
                    self.vocab[pad_id] = b""
                break

            # Pick most frequent adjacent pair
            best = max(counts, key=counts.get)
            new_id = 256 + merge_idx

            ids = self._apply_merge(ids, best, new_id)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

    def encode(self, text: str) -> list:
        if not text:
            return []
        ids = list(text.encode("utf-8"))
        # Apply merges in the order they were learned
        for pair, new_id in self.merges.items():
            ids = self._apply_merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list) -> str:
        tokens = b"".join(self.vocab[i] for i in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        data = {
            "merges": [[a, b] for (a, b) in self.merges.keys()],
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        self.merges = {}
        new_id = 256
        for a, b in data["merges"]:
            self.merges[(a, b)] = new_id
            new_id += 1

    def __len__(self) -> int:
        return len(self.vocab)
