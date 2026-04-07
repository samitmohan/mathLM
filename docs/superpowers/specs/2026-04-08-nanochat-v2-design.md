# nanochat v2: Architecture + Pipeline Design

**Date:** 2026-04-08
**Goal:** Extend nanochat with BPE tokenizer, SwiGLU MLP, top-p sampling, train/val split with perplexity, checkpoint save/load, and a demo `generate.py`.

---

## Scope

Five improvements across three files, plus two new files:

| Change | File |
|--------|------|
| BPE tokenizer (train, encode, decode, save, load) | `tokenizer.py` (new) |
| SwiGLU MLP | `gpt.py` |
| Top-p (nucleus) sampling | `gpt.py` |
| Train/val split + perplexity logging | `train.py` |
| Checkpoint save/load + resume | `train.py` |
| Demo inference script | `generate.py` (new) |

---

## 1. BPE Tokenizer (`tokenizer.py`)

### Algorithm

Byte-level BPE starting from 256 base tokens (one per byte value). Training runs
`vocab_size - 256` merge steps (4096 - 256 = 3840 merges for the default).

Each merge step:
1. Count all adjacent pairs in the current token sequence
2. Find the most frequent pair
3. Replace every occurrence of that pair with a new token ID
4. Record the merge rule `(a, b) -> new_id`

### Interface

```python
class BPETokenizer:
    def train(self, text: str, vocab_size: int) -> None: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

`save` writes a JSON file with two keys:
- `"merges"`: list of `[a, b]` pairs in training order
- `"vocab"`: dict mapping token id (str) to byte sequence (list of ints)

`load` reconstructs both from the JSON file.

### Usage in `train.py`

```python
if os.path.exists("tokenizer.json"):
    tok = BPETokenizer()
    tok.load("tokenizer.json")
else:
    tok = BPETokenizer()
    tok.train(text, vocab_size=4096)
    tok.save("tokenizer.json")
```

Encoding the corpus after tokenizer is ready:
```python
data = np.array(tok.encode(text), dtype=np.int64)
```

`GPTConfig.vocab_size` is set to `len(tok)` (number of tokens in the trained vocabulary).

---

## 2. SwiGLU MLP (`gpt.py`)

Replace the two-layer GELU MLP with a three-layer gated MLP.

### Architecture

```python
class MLP(nn.Module):
    """Gated feed-forward block (SwiGLU): two parallel projections, one gates the other."""

    def __init__(self, config):
        # hidden_dim = (8/3) * n_embd, rounded up to nearest multiple of 64
        # keeps parameter count ~equal to the old 4x GELU MLP
        hidden_dim = round_to_multiple(int(8 * config.n_embd / 3), 64)
        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.fc1  = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.fc2  = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.gate(x)) * self.fc1(x))
```

`round_to_multiple(n, m)` is a one-liner helper: `((n + m - 1) // m) * m`.

`GPTConfig` does not need a new field - `hidden_dim` is derived from `n_embd` at `MLP.__init__` time.

---

## 3. Top-p Sampling (`gpt.py`)

Add `top_p: float | None = None` parameter to `generate()`. Applied after temperature
scaling and after `top_k` filtering (if both are set, `top_k` runs first).

```python
if top_p is not None:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # shift cumprobs right by one so the token that pushes over the threshold is kept
    remove = (cumprobs - F.softmax(sorted_logits, dim=-1)) > top_p
    sorted_logits[remove] = -float("inf")
    logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
```

`top_p=None` by default - no behavior change for existing callers.

---

## 4. Train/Val Split + Perplexity (`train.py`)

### Data split

Split encoded data 90/10 before creating datasets. Split at a token boundary, no
shuffling across the boundary (preserve document order):

```python
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
```

### Validation loop

At the end of each epoch, evaluate on the full val set with `model.eval()` and
`torch.no_grad()`. Compute mean cross-entropy loss over all val batches, then:

```python
val_ppl = math.exp(val_loss)
print(f"epoch {epoch} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
```

Training step logs (`step N | loss X | lr Y`) are unchanged.

---

## 5. Checkpoint Save/Load (`train.py`)

### Save

At end of each epoch, after validation:

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
    "epoch": epoch,
    "config": config,
}, "checkpoint.pt")
```

### Resume

At the start of `train()`, before the training loop:

```python
start_epoch, start_step = 0, 0
if os.path.exists("checkpoint.pt"):
    ckpt = torch.load("checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]
    start_epoch = ckpt["epoch"] + 1
```

The training loop starts from `start_epoch` - prior epochs are never entered.
`step` is initialized to `start_step` so the LR schedule continues at the correct point
in the cosine decay without rewarming.

---

## 6. Demo Script (`generate.py`)

Standalone script. Loads `checkpoint.pt` and `tokenizer.json`, accepts an optional
prompt from the command line, generates and prints output.

```
python generate.py                          # prompt = empty string
python generate.py --prompt "To be or not"
python generate.py --prompt "BRUTUS:" --tokens 500 --top_p 0.9
```

Output format - separates prompt from generated text clearly:

```
--- prompt ---
To be or not
--- generated ---
To be or not to die: 'tis nobler in the mind...
```

Arguments:
- `--prompt`: seed text (default: empty)
- `--tokens`: number of tokens to generate (default: 200)
- `--top_k`: top-k filtering (default: 50)
- `--top_p`: nucleus sampling threshold (default: 0.9)
- `--temperature`: sampling temperature (default: 1.0)

No external dependencies (just argparse from stdlib).

---

## File Map

```
tokenizer.py   new   BPETokenizer class (~150 lines)
generate.py    new   demo inference script (~40 lines)
gpt.py         mod   SwiGLU MLP, top-p sampling
train.py       mod   BPE integration, train/val split, perplexity, checkpointing
```

## What Does Not Change

- `GPTConfig` field names and defaults (except `vocab_size` is now set from tokenizer)
- `KVCache`, `CausalSelfAttention`, `Block`, `GPT` forward pass
- Training hyperparameters
- `input.txt` stays in repo
- `.gitignore` already covers `*.pt` and `tokenizer.json` does not need to be committed
