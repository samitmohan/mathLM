
'''
Walkthrough
text = "low low"

ids = list("low low".encode("utf-8"))
So ASCII we get:
l = 108
o = 111
w = 119
space = 32
: [108, 111, 119, 32, 108, 111, 119 ]

Iteration 1:
Counter(zip(ids, ids, ids[1:]))

(108,111) -> "l o" -> 2 times
(111,119) -> "o w" -> 2 times
(119,32) -> "w space" -> 1
(32,108) -> "space l" -> 1

Most frequent is (108,111) = "l o" -> we assign new_id = 256 to this pair
Replace every occurrence of (108,111) with 256:

Before ids: [108, 111, 119, 32, 108, 111, 119 ]
After ids: [256, 119, 32, 256, 119]

Update vocab: vocab[256] = b"l" + b"o" = b"lo "

Iteration 2:
Counter(zip(ids, ids[1:]))
(256,119) -> "lo w" -> 2
(119,32)  -> "w space" -> 1
(32,256)  -> "space lo" -> 1

Most frequent is (256,119) = "lo w" -> assign new_id = 257
Replace every occurrence of (256,119) with 257:
Before ids: [256, 119, 32, 256, 119]
After ids: [257, 32, 257]
Updated vocab[257] = b"lo" + b"w" = b"low"

Iteration 3:
Counter(zip(ids, ids[1:]))
(257,32) → "low space"
(32,257) → "space low"
Pick one :: (257,32) -> assign new_id = 258
Before ids: [257, 32, 257]
After ids: [258, 257]
Update vocab 258 → b"low "


[
 ((108,111), 256),   # l + o → lo
 ((256,119), 257),   # lo + w → low
 ((257,32), 258)     # low + space → "low "
]

Now encode: "low"
[108,111,119] -> apply merges -> [256, 119] -> apply merges -> [257]

Decoding [257] -> vocab[257] = b"low" -> "low"

Find most common pattern -> compress it -> repeat
'''


import re
import json
from collections import Counter

try:
    import rustbpe as _rustbpe  # Rust-backed BPE for fast training; pip install rustbpe
    _RUSTBPE_AVAILABLE = True
except ImportError:
    _RUSTBPE_AVAILABLE = False


def _extract_merges_from_rustbpe_ranks(ranks_list):
    '''
    Convert rustbpe's mergeable ranks [(bytes, rank), ...] back into Python BPE merge format.
    For each merged token (rank >= 256) we find how it splits into two previously-known tokens,
    giving us the ordered merge sequence needed for encoding.
    '''
    bytes_to_rank = {token_bytes: rank for token_bytes, rank in ranks_list}
    merges = []  # list of ((left_token_id, right_token_id), merged_token_id)
    for token_bytes, rank in sorted(ranks_list, key=lambda item: item[1]):
        if rank < 256:
            continue  # base byte tokens are not merges
        # find the unique split point where both halves are already in the vocab at a lower rank
        for split_index in range(1, len(token_bytes)):
            left_bytes = token_bytes[:split_index]
            right_bytes = token_bytes[split_index:]
            if left_bytes in bytes_to_rank and right_bytes in bytes_to_rank:
                left_rank = bytes_to_rank[left_bytes]
                right_rank = bytes_to_rank[right_bytes]
                if left_rank < rank and right_rank < rank:
                    merges.append(((left_rank, right_rank), rank))
                    break
    return merges


class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.vocab = {}
        self.special_tokens = {}  # str -> id; bypass BPE entirely

    def add_special_tokens(self, tokens):
        # special tokens are added after the BPE vocab and get their own IDs
        # they are never split by BPE - encode() handles them before BPE runs
        for token in tokens:
            if token not in self.special_tokens:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.special_tokens[token] = new_id
                self.vocab[new_id] = token.encode("utf-8")

    def train(self, text, vocab_size):
        assert vocab_size >= 256

        if _RUSTBPE_AVAILABLE:
            # use Rust-backed BPE for fast parallel training; much faster than pure Python on large corpora
            rust_tokenizer = _rustbpe.Tokenizer()
            rust_tokenizer.train_from_iterator([text], vocab_size=vocab_size)
            ranks_list = rust_tokenizer.get_mergeable_ranks()  # list of (bytes, rank) in rank order
            # build vocab from ranks (rank -> byte sequence)
            self.vocab = {rank: token_bytes for token_bytes, rank in ranks_list}
            # extract the ordered merge pairs so _encode_chunk can apply them correctly
            self.merges = _extract_merges_from_rustbpe_ranks(ranks_list)
        else:
            # pure Python fallback: slower but correct if rustbpe is not installed
            ids = list(text.encode("utf-8"))

            # base vocab: bytes
            self.vocab = {i: bytes([i]) for i in range(256)}
            self.merges = []

            for new_id in range(256, vocab_size):
                pair_counts = Counter(zip(ids, ids[1:])) # count pairs
                best_pair = max(pair_counts, key=pair_counts.get)

                # merge in one pass
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1

                ids = new_ids

                # store merge
                self.merges.append((best_pair, new_id))

                # build vocab
                self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        print(f"Training complete; final vocab size: {len(self.vocab)}")

    def _encode_chunk(self, text):
        # encode a plain text chunk (no special tokens) using BPE merges
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            i = 0
            new_ids = []
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def encode(self, text):
        if not self.special_tokens:
            return self._encode_chunk(text)

        # split on special tokens first so they bypass BPE entirely
        # re.split with a capturing group keeps the delimiters in the result
        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        parts = re.split(pattern, text)

        result = []
        for part in parts:
            if part in self.special_tokens:
                result.append(self.special_tokens[part])  # special token -> single id
            elif part:
                result.extend(self._encode_chunk(part))   # normal text -> BPE
        return result

    def decode(self, ids):
        # use conditional expression (not dict.get) so bytes([i]) is only evaluated when i is not in vocab
        # dict.get(key, default) always evaluates default eagerly; bytes([i>=256]) would raise ValueError
        return b''.join(self.vocab[i] if i in self.vocab else bytes([i]) for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "merges": [(list(pair), new_id) for pair, new_id in self.merges],
                "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
                "special_tokens": self.special_tokens,
            }, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)

        self.merges = [(tuple(pair), new_id) for pair, new_id in data["merges"]]
        self.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        self.special_tokens = data.get("special_tokens", {})

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.vocab)
