# mathLM

GPT trained from scratch — pretrain on OpenWebMath, SFT into a math chat model.  
Architecture: tiktoken BPE, SwiGLU MLP, Sparse MoE (optional), GQA, RoPE, KV cache, weight tying, scaled residual init.

## Architecture

| Component     | Choice             | Why                                                              |
|---------------|--------------------|------------------------------------------------------------------|
| Tokenizer     | Byte-level BPE     | Byte alphabet eliminates unknown tokens; merges compress common patterns |
| MLP           | SwiGLU             | `fc2(silu(gate(x)) * fc1(x))` — gated activation, better gradient flow than GELU |
| MoE           | Sparse top-2       | Each token routes to 2 of N experts; scales capacity without scaling compute |
| Normalization | RMSNorm (pre-norm) | Applied before attention and MLP; no learned scale/bias, cheaper than LayerNorm |
| Position enc. | RoPE               | Relative position appears directly in the attention dot product  |
| Attention     | GQA (12Q / 3KV)    | Fewer KV heads reduces memory at inference; shared across query groups |
| Inference     | KV cache           | Past keys/values cached so only one token is processed per generation step |

## Install

```bash
pip install torch numpy tiktoken datasets rich wandb
```

## Running

**Step 1 — download training data (OpenWebMath)**

```bash
python dataset.py
```

Downloads ~70k examples from `open-web-math/open-web-math` and writes `openwebmath.txt`. Requires a network connection and takes a few minutes.

**Step 2 — pretrain**

```bash
python training.py
```

On first run, trains a BPE tokenizer on `openwebmath.txt` and caches it to `tokenizer.json`, then encodes the corpus to `train.bin`. Training resumes automatically from `checkpoint.pt` if it exists.

Logs every 100 steps. Validation loss and checkpoint saved every 1000 steps.

Default config: 4 layers, 128-dim (small, for testing). Edit `training.py` to scale up:

```python
config = GPTConfig(
    vocab_size=tokeniser.vocab_size,
    sequence_length=512,
    number_layers=12,
    number_heads=12,
    number_kv_heads=3,   # must divide number_heads; 12=MHA, 3=GQA, 1=MQA
    embedding_dim=768,
)
```

**Step 3 — inference**

```bash
python inference.py
```

Loads `checkpoint.pt` and `tokenizer.json`. Edit `inference.py` to set `chat = False` for single-prompt generation, or `True` for interactive chat (requires a finetuned checkpoint).

**Step 4 — fine-tune on chat data (optional)**

Create `chat_data.json`:

```json
[
    {"system": "You are helpful.", "user": "What is 2+2?", "assistant": "4"},
    {"user": "What is the derivative of sin(x)?", "assistant": "cos(x)"}
]
```

The `"system"` field is optional. Then:

```bash
python finetuning.py
```

Loss is computed only on assistant tokens (user/system tokens masked with -100).

Chat format used internally:

```
<|system|>You are helpful.<|end|><|user|>What is 2+2?<|end|><|assistant|>4<|end|>
```


## MoE

Sparse MoE is off by default. Enable in `training.py`:

```python
config = GPTConfig(
    ...
    use_moe=True,          # enable sparse MoE
    number_experts=8,      # 8 experts, top-2 routing per token
)
```

Each token routes to 2 experts. Slows training without benefit on small corpora.

---

## Roadmap

Roughly in priority order — each item is a meaningful step up in capability or credibility.

### 1. GSM8K evaluation harness
Measure whether training is actually working. Run the model on GSM8K grade-school math problems, extract the final numerical answer with regex, report accuracy. Even a low number is a real result; GPT-2 baseline is ~2%.

### 2. GRPO — RL for math reasoning (mini DeepSeek-R1)
The algorithm behind DeepSeek-R1. Generate N candidate solutions per problem, reward = 1 if the final answer is correct, group-normalize the rewards into advantages, update the policy with a policy-gradient loss. No critic model, no value function — ~150 lines on top of the existing training loop. Transforms the project from "I trained a small GPT" to "I built a reasoning model with RL."

### 3. Process Reward Model (PRM)
Train a second small model to score intermediate reasoning steps rather than just the final answer. Same GPT backbone, binary classifier head at each token position. Feeds back into GRPO as a richer reward signal. Same idea as OpenAI's Math-Shepherd and DeepMind's AlphaCode.

### 4. Speculative decoding
Draft/verify inference loop: a small fast model proposes K tokens, the main model verifies them in one forward pass, accept where they agree and resample where they diverge. ~80 lines, 2–3x inference speedup, shows understanding of inference-time compute.

### 5. Tool use — calculator
Model learns to emit `<|calc|>2+2<|/calc|>` tags mid-generation. Intercept the tag, run `eval()`, inject the result back into context, continue. Pairs naturally with CoT: the model reasons step by step and calls the calculator for intermediate arithmetic.