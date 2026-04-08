from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Minimal GPT with modern transformer features: RMSNorm, RoPE positional embeddings,
Grouped Query Attention (GQA), QK-norm, and KV cache for efficient autoregressive
generation. ~225 lines. No custom CUDA, no external dependencies beyond PyTorch.
"""


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 3
    n_embd: int = 768
    use_moe: bool = False   # enable sparse mixture-of-experts in feed-forward blocks
    n_experts: int = 8      # number of experts when use_moe=True


def norm(x):
    # normalize across embedding dim
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary(x, cos, sin):
    # x: (B, T, H, D). Split last dim into pairs: each pair (x1_i, x2_i) is a 2D
    # point rotated by angle theta_i. Standard 2D rotation: x*cos - y*sin, x*sin + y*cos.
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos

    return torch.cat([y1, y2], dim=-1)


def precompute_rotary(seq_len, head_dim, device):
    # Base 10000 from original RoPE paper (Su et al. 2021). Larger base = slower
    # rotation per dimension = longer effective context range.
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    t = torch.arange(seq_len, device=device)

    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]

    return cos, sin


class KVCache:
    """Stores past keys and values per layer to skip recomputing them each decode step."""

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
    # expand() creates a view (no memory copy); reshape materializes only when needed.
    # Gives each query head a matching key/value without duplicating data.
    if n_rep == 1:
        return x

    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :]                  # (B, H_kv, 1, T, D)
    x = x.expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)


class CausalSelfAttention(nn.Module):
    """Multi-head attention with GQA, RoPE, and QK-norm. Supports KV cache at inference."""

    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd

        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0

        # Q has n_head outputs; K/V have n_kv_head outputs - different sizes require
        # separate projections. Fusing them would need awkward slicing.
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.proj._is_residual = True  # output projection sits on the residual stream

    def forward(self, x, cos, sin, kv_cache=None, layer_idx=None):
        B, T, C = x.size()

        # project
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # rotary embeddings applied to q,k
        q = apply_rotary(q, cos[:, :T], sin[:, :T])
        k = apply_rotary(k, cos[:, :T], sin[:, :T])

        # QK-norm: normalizing Q and K prevents attention logit explosion at long sequences.
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
            # Single-token decode: KV cache holds all prior context, no future tokens
            # exist to mask, so causal masking is both unnecessary and incorrect here.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        return self.proj(y)


def _round_to_multiple(n, m):
    return ((n + m - 1) // m) * m


class MLP(nn.Module):
    """Gated feed-forward block (SwiGLU): two parallel projections, one gates the other."""

    def __init__(self, config):
        super().__init__()
        # (8/3)*n_embd rounded to nearest 64 keeps param count ~equal to 4x GELU expansion.
        hidden_dim = _round_to_multiple(int(8 * config.n_embd / 3), 64)
        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.fc1  = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.fc2  = nn.Linear(hidden_dim, config.n_embd, bias=False)
        # Mark for scaled residual init: output projection sits on the residual path.
        self.fc2._is_residual = True

    def forward(self, x):
        # SwiGLU: gate with SiLU activation controls information flow through fc1.
        return self.fc2(F.silu(self.gate(x)) * self.fc1(x))


class SparseMoE(nn.Module):
    """Sparse mixture-of-experts: routes each token to top-2 of n_experts SwiGLU experts."""

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_experts)])
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(B * T, C)                                      # (N, C)

        logits = self.router(x_flat)                                    # (N, n_experts)
        weights, indices = torch.topk(logits, 2, dim=-1)               # top-2
        weights = F.softmax(weights, dim=-1)                            # (N, 2)

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # find tokens routed to expert i in either top-2 slot
            mask = (indices == i).any(dim=-1)                           # (N,)
            if not mask.any():
                continue
            # sum weights for this expert across both slots
            expert_w = (weights * (indices == i).float()).sum(dim=-1, keepdim=True)
            out[mask] += expert_w[mask] * expert(x_flat[mask])

        # Load-balancing auxiliary loss: penalize unequal expert usage.
        # Minimized when all experts receive equal token fractions.
        router_probs = F.softmax(logits, dim=-1)                       # (N, n_experts)
        expert_frac = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.n_experts):
            # fraction of tokens routed to expert i (token visits at least one slot)
            expert_frac[i] = (indices == i).any(dim=-1).float().mean()
        aux_loss = (router_probs.mean(0) * expert_frac).sum() * self.n_experts

        return out.view(B, T, C), aux_loss


class Block(nn.Module):
    """Single transformer layer: Pre-LN attention + Pre-LN MLP/MoE, both with residual."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config)
        # use_moe=True swaps the dense MLP for a sparse mixture-of-experts block
        self.ff = SparseMoE(config) if config.use_moe else MLP(config)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(norm(x), cos, sin, kv_cache, self.layer_idx)
        ff_out = self.ff(norm(x))
        if isinstance(ff_out, tuple):
            ff_out, aux_loss = ff_out
        else:
            aux_loss = 0.0
        x = x + ff_out
        return x, aux_loss


class GPT(nn.Module):
    """Full transformer language model with weight-tied embeddings and precomputed RoPE."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.ModuleList([
            Block(config, i) for i in range(config.n_layer)
        ])

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding rows and unembedding columns encode the same token
        # similarity geometry. Sharing weights halves parameters with no quality loss.
        self.lm_head.weight = self.wte.weight

        # rotary cache
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rotary(config.seq_len, head_dim, device="cpu")
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Residual output projections use smaller std to prevent gradient explosion
            # at depth. Scaled by 1/sqrt(2*n_layer) following GPT-2's implementation.
            std = 0.02
            if getattr(module, "_is_residual", False):
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape
        assert T <= self.config.seq_len

        x = self.wte(idx)

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x, self.cos, self.sin, kv_cache)
            total_aux_loss += aux_loss

        x = norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            # Add auxiliary load-balancing loss when MoE is active (0.0 otherwise)
            loss = loss + 0.01 * total_aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        kv_cache = KVCache(len(self.blocks))

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:]
            logits, _ = self(idx_cond, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # top_k first, then top_p on the remaining candidates (HuggingFace convention)
            if top_p is not None:
                # Nucleus sampling: keep the smallest set of tokens whose
                # cumulative probability exceeds top_p.
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                # Shift right so the token that crosses top_p is kept, not removed.
                remove = (cumprobs - sorted_probs) > top_p
                sorted_logits[remove] = -float("inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx