# nanochat GitHub Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish nanochat for GitHub publication as a pedagogical reference implementation of a modern GPT.

**Architecture:** Two targeted code fixes (consistent pre-norm, consistent RMSNorm), pedagogical inline comments in both source files, and a full README rewrite. No functional or architectural changes beyond the norm fixes.

**Tech Stack:** Python, PyTorch >= 2.1.0

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `gpt.py` | Modify | Pre-norm fix, remove `ln_f`, module docstring, inline comments |
| `train.py` | Modify | Inline comments on LR schedule, autocast, grad clip |
| `README.md` | Rewrite | Full pedagogical README |
| `.gitignore` | Create | Python + checkpoint patterns |
| `requirements.txt` | Create | torch, numpy with version pins |

---

## Task 1: Fix norm inconsistencies in `gpt.py`

**Files:**
- Modify: `gpt.py`

- [ ] **Step 1: Add pre-norm before attention in `Block.forward`**

In `gpt.py`, find `Block.forward` (currently line ~147) and change:

```python
def forward(self, x, cos, sin, kv_cache=None):
    x = x + self.attn(x, cos, sin, kv_cache, self.layer_idx)
    x = x + self.mlp(norm(x))
    return x
```

to:

```python
def forward(self, x, cos, sin, kv_cache=None):
    x = x + self.attn(norm(x), cos, sin, kv_cache, self.layer_idx)
    x = x + self.mlp(norm(x))
    return x
```

- [ ] **Step 2: Remove `ln_f` from `GPT.__init__`**

In `GPT.__init__`, remove this line:

```python
self.ln_f = nn.LayerNorm(config.n_embd)
```

- [ ] **Step 3: Replace `self.ln_f(x)` with `norm(x)` in `GPT.forward`**

In `GPT.forward`, change:

```python
x = self.ln_f(x)
logits = self.lm_head(x)
```

to:

```python
x = norm(x)
logits = self.lm_head(x)
```

- [ ] **Step 4: Smoke test - verify forward pass still works**

```bash
python -c "
import torch
from gpt import GPT, GPTConfig
config = GPTConfig(vocab_size=65, seq_len=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=128)
model = GPT(config)
idx = torch.zeros((1, 10), dtype=torch.long)
logits, loss = model(idx, idx)
print('forward pass OK, logits shape:', logits.shape)

# test generation (KV cache path)
out = model.generate(idx, max_new_tokens=5)
print('generate OK, output shape:', out.shape)
"
```

Expected output:
```
forward pass OK, logits shape: torch.Size([1, 10, 65])
generate OK, output shape: torch.Size([1, 15])
```

- [ ] **Step 5: Commit**

```bash
git add gpt.py
git commit -m "fix: consistent pre-norm and rms_norm throughout"
```

---

## Task 2: Add comments to `gpt.py`

**Files:**
- Modify: `gpt.py`

- [ ] **Step 1: Add module-level docstring at top of `gpt.py`**

Insert after the imports (after `import torch.nn.functional as F`):

```python
"""
Minimal GPT with modern transformer features: RMSNorm, RoPE positional embeddings,
Grouped Query Attention (GQA), QK-norm, and KV cache for efficient autoregressive
generation. ~225 lines. No custom CUDA, no external dependencies beyond PyTorch.
"""
```

- [ ] **Step 2: Add docstring to `KVCache`**

Change:

```python
class KVCache:
    def __init__(self, n_layer):
```

to:

```python
class KVCache:
    """Stores past keys and values per layer to skip recomputing them each decode step."""

    def __init__(self, n_layer):
```

- [ ] **Step 3: Add docstrings to `CausalSelfAttention`, `MLP`, `Block`, `GPT`**

```python
class CausalSelfAttention(nn.Module):
    """Multi-head attention with GQA, RoPE, and QK-norm. Supports KV cache at inference."""
```

```python
class MLP(nn.Module):
    """Position-wise feed-forward block: linear -> GELU -> linear, 4x expansion."""
```

```python
class Block(nn.Module):
    """Single transformer layer: Pre-LN attention + Pre-LN MLP, both with residual."""
```

```python
class GPT(nn.Module):
    """Full transformer language model with weight-tied embeddings and precomputed RoPE."""
```

- [ ] **Step 4: Add inline comment to `precompute_rotary`**

Change:

```python
def precompute_rotary(seq_len, head_dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
```

to:

```python
def precompute_rotary(seq_len, head_dim, device):
    # Base 10000 from original RoPE paper (Su et al. 2021). Larger base = slower
    # rotation per dimension = longer effective context range.
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
```

- [ ] **Step 5: Add inline comment to `apply_rotary`**

Change:

```python
def apply_rotary(x, cos, sin):
    # x: (B, T, H, D)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
```

to:

```python
def apply_rotary(x, cos, sin):
    # x: (B, T, H, D). Split last dim into pairs: each pair (x1_i, x2_i) is a 2D
    # point rotated by angle theta_i. Standard 2D rotation: x*cos - y*sin, x*sin + y*cos.
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
```

- [ ] **Step 6: Add inline comment to `repeat_kv`**

Change:

```python
def repeat_kv(x, n_rep):
    # x: (B, H_kv, T, D) → (B, H, T, D)
    if n_rep == 1:
        return x

    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :]                  # (B, H_kv, 1, T, D)
    x = x.expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)
```

to:

```python
def repeat_kv(x, n_rep):
    # x: (B, H_kv, T, D) → (B, H, T, D)
    # expand() creates a view (no memory copy); reshape materializes only when needed.
    # Gives each query head a matching key/value without duplicating data.
    if n_rep == 1:
        return x

    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :]                  # (B, H_kv, 1, T, D)
    x = x.expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)
```

- [ ] **Step 7: Add comment on separate projections in `CausalSelfAttention.__init__`**

Change:

```python
        # separate projections (needed for MQA/GQA)
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
```

to:

```python
        # Q has n_head outputs; K/V have n_kv_head outputs - different sizes require
        # separate projections. Fusing them would need awkward slicing.
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
```

- [ ] **Step 8: Add QK-norm and is_causal comments in `CausalSelfAttention.forward`**

Change:

```python
        q, k = norm(q), norm(k)
```

to:

```python
        # QK-norm: normalizing Q and K prevents attention logit explosion at long sequences.
        q, k = norm(q), norm(k)
```

Change:

```python
        else:
            # single-token decoding (already causal via cache)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

to:

```python
        else:
            # Single-token decode: KV cache holds all prior context, no future tokens
            # exist to mask, so causal masking is both unnecessary and incorrect here.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

- [ ] **Step 9: Add weight tying comment in `GPT.__init__`**

Change:

```python
        # weight tying
        self.lm_head.weight = self.wte.weight
```

to:

```python
        # Weight tying: embedding rows and unembedding columns encode the same token
        # similarity geometry. Sharing weights halves parameters with no quality loss.
        self.lm_head.weight = self.wte.weight
```

- [ ] **Step 10: Smoke test**

```bash
python -c "import gpt; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 11: Commit**

```bash
git add gpt.py
git commit -m "docs: add pedagogical comments to gpt.py"
```

---

## Task 3: Add comments to `train.py`

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add comment to `get_lr`**

Change:

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # linear warmup
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # cosine decay
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)
```

to:

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # Phase 1: linear warmup from 0 to max_lr. Starting at zero avoids large
    # random gradient updates before the model has any useful structure.
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # Phase 2: cosine decay from max_lr to min_lr. Cosine avoids the abrupt drop
    # you'd get from a step schedule and smoothly approaches the final learning rate.
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)
```

- [ ] **Step 2: Add comment to `autocast_dtype`**

Change:

```python
    # bf16 works on newer GPUs (Ampere+), fallback is safe
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
```

to:

```python
    # bfloat16 only on CUDA: Metal and CPU lack hardware bfloat16 support.
    # bfloat16 keeps the float32 exponent range while halving memory and compute.
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
```

- [ ] **Step 3: Add comment to `clip_grad_norm_`**

Change:

```python
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

to:

```python
            # Gradient clipping: rescale gradients so their norm never exceeds 1.0.
            # Prevents large random batches early in training from destabilizing weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

- [ ] **Step 4: Smoke test**

```bash
python -c "import train; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "docs: add pedagogical comments to train.py"
```

---

## Task 4: Rewrite README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the full content of `README.md`**

Replace the entire file with:

```markdown
# nanochat

Minimal GPT trained on Shakespeare. Implements the key ideas behind modern LLMs
(GQA, RoPE, KV cache) in ~350 lines of PyTorch.

## What this is

A from-scratch transformer that covers the core architecture of LLaMA and GPT-2
without the production complexity. Character-level language model: learns to generate
Shakespeare-like text, one character at a time.

Designed to be read alongside any modern LLM explainer. Each design choice maps
directly to what you'll find in production models.

## Architecture

| Component     | Choice         | Why                                                              |
|---------------|----------------|------------------------------------------------------------------|
| Normalization | RMSNorm        | No learned scale/bias - cheaper, matches LayerNorm quality       |
| Position enc. | RoPE           | Position encoded as rotation angle, so relative distance appears in the dot product |
| Attention     | GQA (6Q / 2KV) | K/V are the memory bottleneck at inference; share them across query groups |
| Inference     | KV cache       | Skip recomputing past keys/values at each generation step        |
| Embeddings    | Weight-tied    | Embedding and unembedding rows encode the same token geometry    |

## Quick start

```bash
pip install torch numpy
python train.py
```

Trains for up to 20k steps (~10 epochs on the included Shakespeare text). Prints a
generated sample after each epoch. Runs on CPU; GPU (CUDA) significantly faster.

## Files

```
gpt.py    model definition: config, attention, MLP, transformer blocks, generation
train.py  data loading, learning rate schedule, training loop
```

## Training

Default config: 6 layers, 6 heads, 2 KV heads, 384-dim embeddings (~10M parameters).

```
step 0    | loss 4.1847 | lr 0.000000
step 100  | loss 2.4123 | lr 0.000030
step 1000 | loss 1.5823 | lr 0.000300
step 3000 | loss 1.2068 | lr 0.000293
step 5000 | loss 1.0624 | lr 0.000272
step 5800 | loss 0.9066 | lr 0.000260
```

## References

- Vaswani et al. (2017) - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Su et al. (2021) - [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Ainslie et al. (2023) - [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245)
- Karpathy - [nanoGPT](https://github.com/karpathy/nanoGPT) (inspiration)
```

- [ ] **Step 2: Verify the file looks correct**

```bash
python -c "
with open('README.md') as f:
    lines = f.readlines()
print(f'{len(lines)} lines')
print('first line:', lines[0].strip())
print('last line:', lines[-1].strip())
"
```

Expected: reasonable line count, first line is `# nanochat`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as pedagogical reference"
```

---

## Task 5: Add project hygiene files

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`

- [ ] **Step 1: Create `.gitignore`**

Create `/Users/samit/personal/nanochat/.gitignore` with:

```
__pycache__/
*.pyc
*.pyo
*.pth
*.pt
.DS_Store
```

- [ ] **Step 2: Create `requirements.txt`**

Create `/Users/samit/personal/nanochat/requirements.txt` with:

```
torch>=2.1.0
numpy
```

`torch>=2.1.0` is required for `F.rms_norm` and `F.scaled_dot_product_attention`.

- [ ] **Step 3: Verify `.gitignore` excludes `__pycache__`**

```bash
git check-ignore __pycache__/ && echo "gitignore OK"
```

Expected: `__pycache__/` followed by `gitignore OK`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore requirements.txt
git commit -m "chore: add .gitignore and requirements.txt"
```

---

## Self-Review

**Spec coverage:**
- [x] Code fix: pre-norm before attention - Task 1
- [x] Code fix: remove `nn.LayerNorm`, use `norm()` everywhere - Task 1
- [x] Module docstring for `gpt.py` - Task 2
- [x] Class docstrings - Task 2
- [x] Inline comments: `precompute_rotary`, `apply_rotary`, `repeat_kv`, QK-norm, `is_causal=False`, weight tying, separate projections - Task 2
- [x] Inline comments: `get_lr`, `autocast_dtype`, `clip_grad_norm_` - Task 3
- [x] README rewrite with architecture table, quick start, training results, references - Task 4
- [x] `.gitignore`, `requirements.txt` - Task 5
- [x] `input.txt` stays in repo - no task needed

**Placeholder scan:** No TBDs, no "handle edge cases", all code blocks complete.

**Type consistency:** No shared types across tasks - each task is self-contained edits.
