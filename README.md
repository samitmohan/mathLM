# nanochat

From-scratch GPT in ~500 lines of PyTorch. Byte-level BPE tokenizer, SwiGLU MLP,
Sparse MoE (optional), top-p sampling, GQA, RoPE, KV cache. Train on any text corpus.

## What this is

A transformer implementation that covers the core architecture of modern LLMs
(LLaMA, GPT-2) without the production complexity. Every design decision has a
comment explaining the "why" - written to be read alongside any modern LLM explainer.

Default config is ~124M parameters (GPT-2 scale), tuned for a 24GB GPU with a
small corpus. Character-to-subword: BPE tokenizer trained from scratch on your corpus.

## Architecture

| Component     | Choice             | Why                                                              |
|---------------|--------------------|------------------------------------------------------------------|
| Tokenizer     | Byte-level BPE     | Byte alphabet eliminates unknown tokens; merges compress common patterns |
| MLP           | SwiGLU             | Gated activation: `fc2(silu(gate(x)) * fc1(x))` - better than GELU at same param count |
| MoE           | Sparse top-2       | Routes each token to 2 of N experts; scales capacity without scaling compute |
| Normalization | RMSNorm            | No learned scale/bias - cheaper, matches LayerNorm quality       |
| Position enc. | RoPE               | Relative position appears directly in the attention dot product  |
| Attention     | GQA (12Q / 3KV)    | K/V heads are the memory bottleneck at inference; share across query groups |
| Inference     | KV cache           | Skip recomputing past keys/values at each generation step        |
| Embeddings    | Weight-tied        | Input and output embeddings share the same token geometry        |
| Init          | Scaled residual    | Output projections initialized with `0.02/sqrt(2*n_layer)` std to prevent gradient explosion |

## Quick start

```bash
pip install torch numpy
```

Put your training corpus in `input.txt`, then:

```bash
python train.py
```

The tokenizer trains on first run and is cached to `tokenizer.json`. Resumes from
`checkpoint.pt` automatically if it exists. Validation loss and perplexity are logged
every 500 steps.

## Generating text

```bash
python generate.py
python generate.py --prompt "To be or not"
python generate.py --prompt "CHAPTER I" --tokens 500 --top_p 0.9 --temperature 0.8
```

Output:

```
model: 124.4M parameters
tokenizer: 4096 tokens

--- prompt ---
CHAPTER I
--- generated ---
CHAPTER I

It was a dark and stormy night...
```

All sampling parameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | `""` | Text to condition on |
| `--tokens` | `200` | Number of new tokens to generate |
| `--top_k` | `50` | Keep top-k logits before sampling |
| `--top_p` | `0.9` | Nucleus sampling: keep tokens covering top_p probability mass |
| `--temperature` | `1.0` | Scale logits; < 1.0 sharpens distribution, > 1.0 flattens |

## Using MoE

Enable Sparse Mixture-of-Experts (off by default) in `train.py`:

```python
config = GPTConfig(
    vocab_size=len(tok),
    seq_len=1024,
    n_layer=12,
    n_head=12,
    n_kv_head=3,
    n_embd=768,
    use_moe=True,   # enable sparse MoE
    n_experts=8,    # 8 experts, top-2 routing per token
)
```

Each token is routed to its top-2 experts. A load-balancing auxiliary loss
(`0.01 * aux_loss`) penalizes expert collapse. Disabled by default - adds
significant compute without benefit on small corpora.

## Training

Default config: 12 layers, 12 heads, 3 KV heads, 768-dim embeddings (~124M parameters).
Effective batch: 8 sequences x 8 gradient accumulation steps = 64 sequences (~65k tokens/step).

```
train tokens: 981,986 | val tokens: 109,109
124.4M parameters
step     0 | loss 8.3241 | lr 0.000000
step   100 | loss 5.1823 | lr 0.000030
step  1000 | loss 2.8104 | lr 0.000300
step   500 | val_loss 3.1240 | val_ppl 22.74
step  1000 | val_loss 2.7891 | val_ppl 16.26
...
checkpoint saved
```

Trains for up to 20k steps. Checkpoint saved every 500 steps alongside validation.
LR schedule: 1000-step linear warmup, then cosine decay to 3e-5.

## Files

```
gpt.py         model: config, RMSNorm, RoPE, GQA, SwiGLU, SparseMoE, KVCache, generation
tokenizer.py   byte-level BPE: train, encode, decode, save/load
train.py       BPE integration, step-based loop, grad accum, val split, checkpointing
generate.py    demo inference: load checkpoint, generate text, argparse CLI
```

## References

- Vaswani et al. (2017) - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Su et al. (2021) - [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Ainslie et al. (2023) - [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245)
- Shazeer (2020) - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Fedus et al. (2022) - [Switch Transformers](https://arxiv.org/abs/2101.03961) (MoE)
- Karpathy - [nanoGPT](https://github.com/karpathy/nanoGPT) (inspiration)
