from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 8
    n_kv_head: int = 2   # for MQA/GQA
    n_embd: int = 768


def norm(x):
    # normalize across embedding dim
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary(x, cos, sin):
    # x: (B, T, H, D)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos

    return torch.cat([y1, y2], dim=-1)


def precompute_rotary(seq_len, head_dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    t = torch.arange(seq_len, device=device)

    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]

    return cos, sin


class KVCache:
    def __init__(self, n_layer):
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    def update(self, layer_idx, k, v):
        # k,v: (B, H, T, D)
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=2)

        return self.k[layer_idx], self.v[layer_idx]


def repeat_kv(x, n_rep):
    # x: (B, H_kv, T, D) → (B, H, T, D)
    if n_rep == 1:
        return x

    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :]                  # (B, H_kv, 1, T, D)
    x = x.expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd

        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0

        # separate projections (needed for MQA/GQA)
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos, sin, kv_cache=None, layer_idx=None):
        B, T, C = x.size()

        # project
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # rotary embeddings applied to q,k
        q = apply_rotary(q, cos[:, :T], sin[:, :T])
        k = apply_rotary(k, cos[:, :T], sin[:, :T])

        q, k = norm(q), norm(k)

        # (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV cache (inference)
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # expand KV heads → match Q heads
        n_rep = self.n_head // self.n_kv_head
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        # attention
        if kv_cache is None or T > 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # single-token decoding (already causal via cache)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(x, cos, sin, kv_cache, self.layer_idx)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.ModuleList([
            Block(config, i) for i in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.wte.weight

        # rotary cache
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rotary(config.seq_len, head_dim, device="cpu")
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape
        assert T <= self.config.seq_len

        x = self.wte(idx)

        for block in self.blocks:
            x = block(x, self.cos, self.sin, kv_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        kv_cache = KVCache(len(self.blocks))

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -1:]  # only last token
            logits, _ = self(idx_cond, kv_cache=kv_cache)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx