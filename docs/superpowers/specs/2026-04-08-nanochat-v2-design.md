# nanochat v2: Architecture + Pipeline Design

**Date:** 2026-04-08
**Goal:** Extend nanochat with BPE tokenizer, SwiGLU MLP, top-p sampling, train/val
split with perplexity, checkpoint save/load, MoE (optional), and a polished `generate.py`.

---

## Scope

| Change | File |
|--------|------|
| BPE tokenizer (train, encode, decode, save, load) | `tokenizer.py` (new) |
| SwiGLU MLP | `gpt.py` |
| Sparse MoE (optional, off by default) | `gpt.py` |
| Top-p (nucleus) sampling | `gpt.py` |
| Scaled residual init, torch.compile, param count | `gpt.py` |
| Train/val split + perplexity logging | `train.py` |
| Gradient accumulation | `train.py` |
| Checkpoint save/load + resume | `train.py` |
| Reproducibility seed | `train.py` |
| Demo inference script | `generate.py` (new) |

---

## Recommended Model Config (24GB VRAM, small corpus)

```python
GPTConfig(
    vocab_size=4096,   # set from tokenizer
    seq_len=1024,
    n_layer=12,
    n_head=12,
    n_kv_head=3,
    n_embd=768,
    use_moe=False,
    n_experts=8,
)
```

~124M parameters (GPT-2 small scale). Fits comfortably under 4GB VRAM at bfloat16
with batch_size=8 and grad_accum_steps=8 (effective batch = 64 sequences = ~65k tokens/step).
Larger models will overfit on a small corpus - this is the sweet spot.

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
    def __len__(self) -> int: ...   # returns vocab_size
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

data = np.array(tok.encode(text), dtype=np.int64)
config = GPTConfig(vocab_size=len(tok), ...)
```

---

## 2. SwiGLU MLP (`gpt.py`)

Replace the two-layer GELU MLP with a three-layer gated MLP.

```python
class MLP(nn.Module):
    """Gated feed-forward block (SwiGLU): two parallel projections, one gates the other."""

    def __init__(self, config):
        # hidden_dim = (8/3) * n_embd rounded to nearest multiple of 64.
        # Keeps parameter count ~equal to the old 4x GELU MLP.
        hidden_dim = _round_to_multiple(int(8 * config.n_embd / 3), 64)
        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.fc1  = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.fc2  = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.gate(x)) * self.fc1(x))
```

`_round_to_multiple(n, m)` is a module-level helper: `((n + m - 1) // m) * m`.

---

## 3. Sparse MoE (`gpt.py`)

`GPTConfig` gets two new fields: `use_moe: bool = False`, `n_experts: int = 8`.
When `use_moe=False` (default), `Block` uses `MLP` exactly as before - zero behavior change.

When `use_moe=True`, `Block` uses `SparseMoE` instead of `MLP`:

```python
class SparseMoE(nn.Module):
    """Sparse mixture-of-experts: routes each token to top-2 of n_experts SwiGLU experts."""

    def __init__(self, config):
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_experts)])
        self.router  = nn.Linear(config.n_embd, config.n_experts, bias=False)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        x_flat = x.view(B * T, C)

        # router: (B*T, n_experts) -> top-2 weights and indices
        logits = self.router(x_flat)
        weights, indices = torch.topk(logits, 2, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # dispatch to experts and weighted sum
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)   # tokens routed to expert i
            if not mask.any():
                continue
            expert_weights = weights[mask] * (indices[mask] == i).float()
            out[mask] += expert_weights.sum(dim=-1, keepdim=True) * expert(x_flat[mask])

        # auxiliary load-balancing loss: penalize unequal expert utilization
        # computed as the dot product of mean router probs and mean expert usage fraction
        router_probs = F.softmax(logits, dim=-1)
        expert_usage = torch.zeros(config.n_experts, device=x.device)
        for i in range(config.n_experts):
            expert_usage[i] = (indices == i).float().mean()
        aux_loss = (router_probs.mean(0) * expert_usage).sum() * config.n_experts

        return out.view(B, T, C), aux_loss
```

`Block.forward` always returns `(x, aux_loss)` - a tensor and a scalar. When `use_moe=False`,
`Block` calls `MLP` and returns `(mlp_out, 0.0)`. When `use_moe=True`, it calls `SparseMoE`
and returns its `(out, aux_loss)` directly. `GPT.forward` always unpacks the tuple, sums
`aux_loss` across all blocks, and adds `0.01 * total_aux_loss` to the cross-entropy loss.

---

## 4. Top-p Sampling (`gpt.py`)

Add `top_p: float | None = None` to `generate()`. Applied after temperature scaling
and after `top_k` (if both set, `top_k` runs first).

```python
if top_p is not None:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # shift right so the token that pushes past top_p is kept, not removed
    remove = (cumprobs - F.softmax(sorted_logits, dim=-1)) > top_p
    sorted_logits[remove] = -float("inf")
    logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
```

---

## 5. Karpathy Additions (`gpt.py` + `train.py`)

### Scaled residual init (`gpt.py`)

In `GPT._init_weights`, output projection layers (`self.proj` in attention, `self.fc2`
in MLP) are initialized with a smaller std to prevent gradient explosion at depth:

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        std = 0.02
        if hasattr(module, "_is_residual"):
            std = 0.02 / math.sqrt(2 * self.config.n_layer)
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

`self.proj` and `self.fc2` get `module._is_residual = True` set in their respective
`__init__` methods.

### `torch.compile` + param count + seed (`train.py`)

```python
torch.manual_seed(1337)
model = GPT(config).to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
model = torch.compile(model)
```

### Gradient accumulation (`train.py`)

```python
grad_accum_steps = 8   # effective batch = batch_size * grad_accum_steps

optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    x, y = next(train_iter)
    with torch.autocast(device_type=device, dtype=autocast_dtype):
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps   # scale loss before backward
    loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

The training loop becomes iterator-based rather than epoch-based to support clean
micro-batch stepping. Step count drives everything; epoch is derived from steps.

---

## 6. Train/Val Split + Perplexity (`train.py`)

Split encoded data 90/10 at a token boundary:

```python
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
```

At end of each eval interval (every 500 steps), run full val set with `model.eval()`:

```python
val_loss = mean cross-entropy over val_loader
print(f"step {step} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f}")
```

Eval interval is configurable: `eval_interval = 500`.

---

## 7. Checkpoint Save/Load (`train.py`)

Save every `eval_interval` steps (same cadence as validation):

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
    "config": config,
}, "checkpoint.pt")
```

Resume at start of `train()`:

```python
step = 0
if os.path.exists("checkpoint.pt"):
    ckpt = torch.load("checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    print(f"resuming from step {step}")
```

LR schedule continues from restored `step` - cosine decay picks up exactly where it left off.

---

## 8. Demo Script (`generate.py`)

Loads `checkpoint.pt` + `tokenizer.json`. Clean output, all sampling params exposed.

```
python generate.py
python generate.py --prompt "To be or not"
python generate.py --prompt "BRUTUS:" --tokens 500 --top_p 0.9 --temperature 0.8
```

Output:

```
model: 124.4M parameters
tokenizer: 4096 tokens

--- prompt ---
BRUTUS:
--- generated ---
BRUTUS: I have done; and therein I come to speak
What I have done is done; and what I have done
...
```

Arguments: `--prompt` (default: ""), `--tokens` (default: 200), `--top_k` (default: 50),
`--top_p` (default: 0.9), `--temperature` (default: 1.0).

---

## File Map

```
tokenizer.py   new   BPETokenizer: byte-level BPE, encode/decode, save/load (~160 lines)
generate.py    new   demo inference script (~45 lines)
gpt.py         mod   SwiGLU, SparseMoE, top-p, scaled init, _is_residual markers
train.py       mod   BPE, train/val split, grad accum, perplexity, checkpointing, compile
```

## What Does Not Change

- `KVCache`, `CausalSelfAttention` internals, RoPE
- `GPT.generate` logic (only adds top_p parameter)
- `input.txt` stays in repo
- `.gitignore` updated to cover `*.pt` and `tokenizer.json` (both are generated artifacts)
