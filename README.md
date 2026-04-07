# nanochat

Minimal GPT trained on Shakespeare. Implements the key ideas behind modern LLMs
(GQA, RoPE, KV cache) in ~350 lines of PyTorch.

## What this is

A from-scratch transformer that covers the core architecture of LLaMA and GPT-2
without the production complexity. Character-level language model: learns to generate
Shakespeare-like text, one character at a time.

Designed to be read alongside any modern LLM explainer. Each design choice maps
directly to what you will find in production models.

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
generated sample after each epoch. Runs on CPU; a CUDA GPU is significantly faster.

## Files

```
gpt.py    model definition: config, attention, MLP, transformer blocks, generation
train.py  data loading, learning rate schedule, training loop
```

## Training

Default config: 6 layers, 6 heads, 2 KV heads, 384-dim embeddings (~10M parameters).

```
step 1000  | loss 1.5823 | lr 0.000300
step 5000  | loss 1.0624 | lr 0.000272
step 5800  | loss 0.9066 | lr 0.000260
step 19600 | loss 0.1305 | lr 0.000030
step 20000 | loss 0.1480 | lr 0.000030
```

## References

- Vaswani et al. (2017) - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Su et al. (2021) - [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Ainslie et al. (2023) - [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245)
- Karpathy - [nanoGPT](https://github.com/karpathy/nanoGPT) (inspiration)
