# mathLM

A 40M-parameter GPT trained from scratch on math: custom BPE tokenizer →
pretraining on OpenWebMath + curated math datasets → unified SFT with
loss masking → GRPO reasoning RL → multi-domain evaluation. Builds on
Karpathy's nanoGPT and goes further into the modern decoder-only stack.

## What's inside

- **Custom BPE tokenizer** — 32k vocab trained on math corpora. Symbols like `∇`, `∫`, `x^2` become single tokens.
- **GPT architecture** — pre-norm RMSNorm, **GQA** (8 query heads / 4 KV heads), **RoPE** with QK-norm, **SwiGLU** MLPs, optional **Sparse MoE** (top-2 routing), weight-tied embedding/unembedding, scaled residual init.
- **Inference** — KV cache, streaming generation, top-k / top-p / temperature sampling.
- **Training pipeline** — bf16 autocast, cosine LR with linear warmup, gradient accumulation, checkpoint resume, WandB logging, `torch.compile`.
- **Fine-tuning** — SFT with loss masked to assistant tokens only; GRPO (Group Relative Policy Optimization) with binary correctness rewards, no critic, no reward model.
- **Evaluation** — GSM8K with 2-shot CoT prompting; held-out calculus and ML-math splits.
- **Demo** — Gradio Q/A and chat UI; HF Spaces entry point.

## Layout

```
mathlm/
  model/        gpt.py          # GPT, attention, RoPE, MoE, KV cache
                tokenizer.py    # MathTokenizer (HF tokenizers backend)
  data/         build_pretrain.py        # openwebmath.txt → train.bin
                download_{gsm8k,math,numina,openr1}.py
                generate_{calculus,ml_math}.py   # synthetic Q&A
  train/        pretrain.py     # 100k-step pretraining loop
                sft.py          # unified multi-domain SFT
                grpo.py         # GRPO reasoning RL
  eval/         gsm8k.py        # GSM8K accuracy harness
                math.py         # multi-domain accuracy + checkpoint compare
  infer/        inference.py    # load_model, generate_text, chat_response, CLI
                demo.py         # local Gradio UI
                app.py          # HF Spaces entry point
scripts/
  train_tokenizer.py            # train BPE on openwebmath.txt
  upload_hf.py                  # publish to HF Hub
  smoke_test.py                 # tiny forward + generate, ~3s on CPU
docs/
  HISTORY.md                    # early-experiment notes
  bpe.py, sampling.py           # learning-notes scratch files
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test builds a tiny GPT, runs a forward pass and a 4-token
generation, and exits 0. If it passes, the package wiring is correct.

If you have a checkpoint on disk:

```bash
python -m mathlm.infer.demo                        # localhost:7860
python -m mathlm.infer.inference --prompt "Q: derivative of x^5? A:"
python -m mathlm.eval.gsm8k --n 100
```

## Full pipeline

Each step is one command, runnable in order. Every stage writes a file
the next stage reads. Checkpoints save every eval interval, so any step
can be interrupted and resumed.

### 1. Train the tokenizer

```bash
python scripts/train_tokenizer.py
```

Trains a 32k byte-level BPE on `openwebmath.txt`. Output:
`math_tokenizer/{vocab.json, merges.txt}`. **What you'll learn:** how a
BPE tokenizer is built (frequency-based merges of adjacent byte pairs)
and how special tokens get added on top.

### 2. Build the pretraining bin file

```bash
python -m mathlm.data.build_pretrain          # → train.bin
```

Tokenizes `openwebmath.txt` in 5 MB chunks and writes int32 token ids
to `train.bin`. **What you'll learn:** why we pre-tokenize once
(keeps training I/O bounded) and how memmapped `.bin` files let the OS
handle 500 MB of tokens without loading them into RAM.

### 3. Download and tokenize SFT datasets

```bash
python -m mathlm.data.download_gsm8k          # → gsm8k_train.bin
python -m mathlm.data.download_math           # → math_train.bin
python -m mathlm.data.download_numina         # → numina_math.bin (859k problems)
python -m mathlm.data.download_openr1         # → openr1_math.bin
```

Each script pulls its dataset from HuggingFace, formats every example
as `Q: ...\nA: ...\n\n`, tokenizes with the math tokenizer, and writes
a `.bin` file. **What you'll learn:** the value of a single canonical
prompt format across heterogeneous sources — it lets one model learn
from competition math, word problems, and synthetic Q/A using the same
tokens and the same loss.

### 4. Generate synthetic data

```bash
python -m mathlm.data.generate_calculus       # → math_qa.bin
python -m mathlm.data.generate_ml_math        # → ml_math.bin
```

Programmatically generates ~12k calculus pairs (power rule, chain rule,
trig, exp/log, integrals) and ~5k ML-math pairs (matrix gradients, KL
divergence, attention, optimisers). **What you'll learn:** synthetic
data is exact and controllable — when the pretraining corpus is noisy
(forum threads, scraped LaTeX), a programmatic generator gives you
clean ground truth on the topics you care about.

### 5. Pretrain

```bash
python -m mathlm.train.pretrain
```

100k steps, batch=128, grad-accum=8 (524k tokens/step), LR cosine
3e-4 → 3e-5 with 2k-step warmup, bf16 autocast, `torch.compile`.
Saves `checkpoint.pt` every 500 steps. **What you'll learn:** how the
moving parts of a real pretraining loop fit together — gradient
accumulation as a bigger-batch shim, why warmup matters, why bf16 is
the default for LLMs, why memmap + ConcatDataset beats `np.concatenate`
when you have 1+ GB of tokens.

### 6. Unified SFT

```bash
python -m mathlm.train.sft                    # → checkpoint_mathlm.pt
```

Fine-tunes the pretrained model on calculus + ML-math + GSM8K + MATH
*together*, with the loss masked to assistant tokens (`-100` on the
prompt). Per-domain validation loss is tracked every epoch; a Δ > 0.3
fires a "catastrophic forgetting" warning. **What you'll learn:** loss
masking is what makes SFT actually about answers and not about
predicting questions; mixing domains at low LR is a cheap, effective
fix to single-domain forgetting.

### 7. GRPO reasoning RL

```bash
python -m mathlm.train.grpo                   # → checkpoint_mathlm_grpo.pt
```

Generates N candidates per question, scores each with a binary reward
(does the extracted number match? does the calculus expression
normalize-equal?), normalises the rewards within each question's group
to get advantages, and runs policy gradient with a KL penalty against
a frozen reference (the SFT checkpoint). **What you'll learn:** GRPO
replaces PPO's value critic with group-relative normalisation — when
rewards are *verifiable* (like math) you don't need a learned reward
model or a critic. KL keeps the policy from drifting away from the SFT
format.

### 8. Evaluate

```bash
python -m mathlm.eval.gsm8k --n 200                              # GSM8K accuracy
python -m mathlm.eval.math --checkpoint checkpoint_mathlm.pt --n 200
python -m mathlm.eval.math --compare checkpoint_mathlm.pt checkpoint_mathlm_grpo.pt --n 200
```

GSM8K eval uses 2-shot CoT prompting and number extraction. The
multi-domain harness reports per-domain accuracy and supports
side-by-side checkpoint comparison — useful for verifying GRPO improved
GSM8K without regressing calculus. **What you'll learn:** how to design
a forgiving but unambiguous eval (number extraction with `####`
fallback, normalised expression match) and why per-domain breakdowns
are the right way to spot forgetting.

### 9. Demo / serve

```bash
python -m mathlm.infer.demo                   # local Gradio
python -m mathlm.infer.app --share            # HF Spaces / public link
python scripts/upload_hf.py                   # publish weights + code to HF Hub
```

## Training recipe (pretraining defaults)

| | |
|---|---|
| Params | 40.4M |
| Layers / dim / heads | 8 / 512 / 8Q + 4KV (GQA) |
| Sequence length | 512 |
| Vocab | 32k (BPE) |
| Optimiser | AdamW, β=(0.9, 0.95), wd=0.1 (decay only on 2D weights) |
| LR schedule | linear warmup 2000 steps → cosine 3e-4 → 3e-5 |
| Batch | 128 × seq_len 512, grad-accum 8 → ~524k tokens/step |
| Steps | 100,000 |
| Mixed precision | bf16 autocast |
| Compile | `torch.compile(mode="default")` |
| Gradient clip | 1.0 |

## Reproducing without the OpenWebMath corpus

`openwebmath.txt` (~535 MB raw text) is not committed. The pipeline
still produces a working model from the four downloadable HF datasets
plus the two synthetic generators — start at step 3, skipping
steps 1-2, and adjust `mathlm/train/pretrain.py` to skip the missing
`train.bin` source. The `_add_source` helper silently skips any
`.bin` file that isn't on disk.

## Known limitations / future work

- No speculative decoding yet — could give 2-3× inference speedup.
- No process reward model — GRPO uses outcome-only binary rewards.
- No tool use (calculator) — the model has to do arithmetic itself.
- 40M params is small; expect plenty of wrong answers on harder MATH problems.

## Acknowledgements

- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — starting point.
- [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [Hendrycks MATH](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval), [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) — training data.
- DeepSeek-R1 — GRPO formulation.
