# nanochat: GitHub Polish Design

**Date:** 2026-04-07
**Goal:** Make nanochat ready to publish on GitHub as a pedagogical reference implementation of a modern GPT.

---

## Context

nanochat is a from-scratch GPT implementation: character-level LM trained on Shakespeare, ~350 lines across two files. It implements modern transformer features (GQA, RoPE, KV cache, QK-norm) that most toy implementations skip.

Target audience: people learning transformers from scratch. The code should be readable alongside any LLaMA/GPT blog post and immediately make sense.

---

## Scope

Three work areas:

1. **Code fixes** - two norm inconsistencies that would confuse a learner
2. **Comments** - pedagogical inline comments explaining *why*, not *what*
3. **README** - full rewrite as a clean reference guide
4. **Project hygiene** - `.gitignore`, `requirements.txt`

`input.txt` stays in the repo (self-contained, convenient).

---

## 1. Code Fixes (`gpt.py`)

### 1a. Consistent pre-norm in `Block.forward`

**Current:**
```python
x = x + self.attn(x, cos, sin, ...)    # no norm before attention
x = x + self.mlp(norm(x))              # norm before MLP
```

**Fix:**
```python
x = x + self.attn(norm(x), cos, sin, ...)
x = x + self.mlp(norm(x))
```

Why: Standard Pre-LN transformer pattern. Every modern LLM tutorial shows this. A learner reading the current code alongside a LLaMA explainer will be confused by the asymmetry. The QK-norm inside attention is additive - it stays.

### 1b. Remove `nn.LayerNorm`, use `F.rms_norm` everywhere

**Current:** `self.ln_f = nn.LayerNorm(config.n_embd)` (has learned scale and bias) applied in `forward()`.

**Fix:** Remove `ln_f` attribute. Call `norm(x)` directly in `GPT.forward()` before the LM head.

Why: Every other norm in the model is parameter-free `F.rms_norm`. Using `nn.LayerNorm` only for the final norm creates a hidden inconsistency - different behavior, extra learned parameters, different class. One `norm()` function, used everywhere.

---

## 2. Comments Strategy

**Philosophy:** One crisp sentence per non-obvious decision. No restating what the code already says clearly.

### `gpt.py` module docstring (top of file)
3-4 lines: what this implements, what makes it modern (GQA, RoPE, KV cache, QK-norm).

### Class docstrings (one line each)
- `KVCache`: what it stores and why (avoid recomputing keys/values at each decode step)
- `CausalSelfAttention`: GQA with QK-norm and RoPE
- `MLP`: feed-forward block
- `Block`: single transformer layer (Pre-LN)
- `GPT`: full model

### Inline comments for non-obvious lines

**`precompute_rotary`:**
- The `10000` base: frequency scale from original RoPE paper (Su et al. 2023) - controls how fast each dimension rotates with position

**`apply_rotary`:**
- Why split into two halves: each pair `(x1_i, x2_i)` is a 2D point rotated by an angle - classic 2D rotation formula

**`repeat_kv`:**
- Why expand-then-reshape: `expand` creates a view (no memory copy), `reshape` materializes only when needed

**`CausalSelfAttention.__init__`:**
- Why separate `q_proj`/`k_proj`/`v_proj` instead of one fused projection: Q has `n_head` heads, K/V have `n_kv_head` heads - different output sizes

**`CausalSelfAttention.forward`:**
- QK-norm comment: normalizing Q and K before attention prevents attention logit explosion at large sequence lengths
- `is_causal=False` in single-token decode: the KV cache already encodes all prior context - no future tokens to mask

**`GPT.__init__`:**
- Weight tying comment: embedding rows and unembedding columns encode the same "token similarity" geometry - tying them halves parameters with no quality loss

### `train.py` inline comments

- `autocast_dtype`: why bfloat16 only on CUDA (Metal/CPU lack hardware support)
- `get_lr`: label the two phases (linear warmup / cosine decay) with a note on why cosine (smooth approach to min_lr, avoids abrupt LR drops)
- `clip_grad_norm_`: why 1.0 (prevents gradient spikes from destabilizing early training)

---

## 3. README Rewrite

**Structure:**

```
# nanochat

One-sentence description.

## What this is
2-3 sentences. Minimal modern GPT. ~350 lines. Character-level LM on Shakespeare.
Implements the key ideas behind LLaMA/GPT-2 without the production complexity.

## Architecture
Clean table:
| Component      | Choice         | Why                                              |
|----------------|----------------|--------------------------------------------------|
| Normalization  | RMSNorm        | No learned params, cheaper, same quality         |
| Position enc.  | RoPE           | Relative position in dot product, no extra params|
| Attention      | GQA (8Q / 2KV) | K/V are the memory bottleneck; share across heads|
| Inference      | KV cache       | Avoid recomputing past keys/values each step     |
| Embeddings     | Weight-tied    | Embedding and unembedding share weights          |

## Quick start
pip install torch
python train.py
# trains ~20k steps, generates Shakespeare every epoch

## Files
gpt.py   - model: config, attention, MLP, transformer blocks, generation
train.py - data loading, LR schedule, training loop

## Training
Short clean log (first 5 entries, last 5 entries - not the full raw dump)
Final loss: ~0.86
```

No wall of text. A learner reads this in 3 minutes and understands every design choice.

---

## 4. Project Hygiene

### `.gitignore`
```
__pycache__/
*.pyc
*.pyo
*.pth        # model checkpoints
*.pt
.DS_Store
```

### `requirements.txt`
```
torch>=2.1.0
numpy
```

`torch>=2.1.0` because `F.rms_norm` and `F.scaled_dot_product_attention` require it.

---

## What Does Not Change

- Model architecture (layer count, dims, GQA ratios)
- Training hyperparameters
- `input.txt` stays in repo
- No CLI args, no `generate.py`, no extra scripts
- No type annotations added to code that didn't have them
