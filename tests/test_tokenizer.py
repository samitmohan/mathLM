import pytest
from tokenizer import BPETokenizer


def test_vocab_size():
    # Use a large varied corpus so all merges are learned from real data
    tok = BPETokenizer()
    import string
    # Generate text with enough variety for 44+ unique pairs
    text = "".join(string.ascii_lowercase) * 500 + " the quick brown fox " * 200
    tok.train(text, vocab_size=300)
    assert len(tok) <= 300
    assert len(tok) >= 256  # always has at least the byte alphabet


def test_encode_decode_roundtrip():
    tok = BPETokenizer()
    tok.train("hello world " * 200, vocab_size=300)
    assert tok.decode(tok.encode("hello world")) == "hello world"


def test_merges_reduce_length():
    tok = BPETokenizer()
    text = "ab" * 1000
    tok.train(text, vocab_size=260)
    encoded = tok.encode("abababab")
    # BPE should merge "ab" into one token, so fewer ids than bytes
    assert len(encoded) < len("abababab".encode("utf-8"))


def test_save_load(tmp_path):
    tok = BPETokenizer()
    tok.train("hello world " * 200, vocab_size=300)
    path = str(tmp_path / "tok.json")
    tok.save(path)

    tok2 = BPETokenizer()
    tok2.load(path)

    assert len(tok) == len(tok2)
    assert tok.encode("hello world") == tok2.encode("hello world")


def test_encode_empty():
    tok = BPETokenizer()
    tok.train("hello " * 100, vocab_size=270)
    assert tok.encode("") == []


def test_decode_bytes():
    tok = BPETokenizer()
    tok.train("hello " * 100, vocab_size=270)
    ids = tok.encode("hello")
    assert isinstance(tok.decode(ids), str)
