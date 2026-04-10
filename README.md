# nanochat

GPT trained from scratch — pretrain on OpenWebMath + synthetic math Q&A, SFT into a math chat model.  
Architecture: SwiGLU MLP, Sparse MoE (optional), GQA, RoPE, KV cache, QK-norm, weight tying, scaled residual init.

## Architecture

| Component     | Choice              | Why                                                                  |
|---------------|---------------------|----------------------------------------------------------------------|
| Tokenizer     | GPT-2 tiktoken BPE  | 50k vocab, extended with 5 chat special tokens for SFT               |
| MLP           | SwiGLU              | `fc2(silu(gate(x)) * fc1(x))` — gated activation, better gradient flow than GELU |
| MoE           | Sparse top-2        | Each token routes to 2 of N experts; scales capacity without scaling compute |
| Normalization | RMSNorm (pre-norm)  | Applied before attention and MLP; cheaper than LayerNorm              |
| Position enc. | RoPE                | Relative position appears directly in the attention dot product       |
| Attention     | GQA (8Q / 4KV)      | Fewer KV heads reduces KV cache memory at inference                  |
| QK-norm       | RMSNorm on Q and K  | Prevents attention logit explosion at long sequences                 |
| Inference     | KV cache            | Past keys/values cached; one token processed per generation step     |

## Current training run

| Hyperparameter | Value |
|---|---|
| Parameters | 49.3M |
| Layers / dim / heads | 8 / 512 / 8 (4 KV heads, GQA) |
| Sequence length | 256 |
| Batch size | 128 × grad\_accum 4 → ~131k tokens/step |
| Max steps | 100,000 |
| LR schedule | Cosine decay 3e-4 → 3e-5, 2k warmup |
| Training data | OpenWebMath (141M tokens) + synthetic math Q&A (×15 repetition) |
| Hardware | RTX 3090 24 GB (~16.5 GB peak VRAM) |

Previous run: 7M params (4L/128d), 20k steps — loss plateaued at ~4.1–4.3 (model too small).  
Current run: 49.3M params (8L/512d), 100k steps — loss dropped from 10.9 → 6.9 in 300 steps.

## Synthetic math data

`gen_math_data.py` generates ~12k structured Q&A pairs covering:

- Power rule `d/dx x^n` for n = 0..100, 8 question templates each
- Polynomials with coefficients (`d/dx 3x^5 = 15x^4`)
- Two-term polynomial derivatives
- Chain rule: `d/dx (ax+b)^n`, `d/dx sin(ax)`, `d/dx e^(ax)`
- Product rule
- Second derivatives
- Indefinite integrals (reverse power rule)
- Trig: sin, cos, tan, sec, csc, cot
- Exponentials and log
- Negative and fractional exponents
- Arithmetic

Format:
```
Q: What is the derivative of x^5?
A: 5x^4

Q: Differentiate (3x+1)^4 with respect to x.
A: 12(3x+1)^3
```

Regenerate with:
```bash
python gen_math_data.py          # writes math_qa.bin
python gen_math_data.py --check  # print 20 samples + token count
```

## Install

```bash
pip install torch numpy tiktoken datasets rich wandb
```

## Running

**Step 1 — download training data**

```bash
python dataset.py
```

Downloads ~70k examples from `open-web-math/open-web-math` → `openwebmath.txt`.

**Step 2 — tokenise**

```bash
python build_tiktoken_ds.py
```

Encodes `openwebmath.txt` with GPT-2 tiktoken → `train.bin`.

**Step 3 — generate synthetic math data**

```bash
python gen_math_data.py
```

Writes `math_qa.bin`. Training loads this automatically if present and repeats it 15× alongside the main corpus.

**Step 4 — pretrain**

```bash
python training.py
```

Logs every 100 steps. Validation loss, generation sample, and checkpoint saved every 5k steps. Resumes automatically from `checkpoint.pt` if it exists.

**Step 5 — inference**

```bash
python inference.py
```

Loads `checkpoint.pt`. Default prompt: `Q: What is the derivative of x^5?\nA:`.  
Set `use_chat = True` for interactive chat (requires a finetuned checkpoint).

**Step 6 — fine-tune on chat data (optional)**

Create `chat_data.json`:

```json
[
    {"system": "You are helpful.", "user": "What is 2+2?", "assistant": "4"},
    {"user": "What is the derivative of sin(x)?", "assistant": "cos(x)"}
]
```

Then:

```bash
python finetuning.py
```

Loss is computed only on assistant tokens (user/system tokens masked with -100). Saves to `checkpoint_chat.pt`.

Chat format used internally:
```
<|system|>You are helpful.<|end|><|user|>What is 2+2?<|end|><|assistant|>4<|end|>
```

## MoE

Sparse MoE is off by default. Enable in `training.py`:

```python
config = GPTConfig(
    ...
    use_moe=True,
    number_experts=8,   # top-2 routing per token
)
```

Adds a load-balancing auxiliary loss to prevent routing collapse. Slows training without much benefit below ~100M params.

---

## Roadmap

Roughly in priority order.

### 1. GSM8K evaluation harness
Run the model on GSM8K grade-school math problems, extract the final numerical answer with regex, report accuracy. GPT-2 baseline is ~2%. Even a low number is a real signal.

### 2. GRPO — RL for math reasoning (mini DeepSeek-R1)
Generate N candidate solutions per problem, reward = 1 if correct, group-normalize rewards into advantages, update with policy-gradient loss. No critic, no value function — ~150 lines on top of the existing training loop.

### 3. Process Reward Model (PRM)
A second small model that scores intermediate reasoning steps rather than just the final answer. Same GPT backbone, binary classifier head per token position. Feeds into GRPO as a richer reward signal.

### 4. Speculative decoding
Draft/verify inference: a small fast model proposes K tokens, the main model verifies in one forward pass, accept where they agree. ~80 lines, 2–3× inference speedup.

### 5. Tool use — calculator
Model emits `<|calc|>2+2<|/calc|>` mid-generation. Intercept, run `eval()`, inject the result, continue. Pairs naturally with chain-of-thought reasoning.
