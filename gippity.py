# custom GPT; building on top of nanoGPT
from dataclasses import dataclass
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# round n up to the nearest multiple of `multiple` for better GPU efficiency (tensor cores prefer multiples of 64)
def _round_to_multiple(number, multiple):
    return ((number + multiple - 1) // multiple) * multiple

@dataclass
class GPTConfig:
    vocab_size: int = 50304 # 50257 for standard GPT-2 vocab, +1 for padding token, +46 for special token
    sequence_length: int = 256 # max sequence length for positional embeddings and rotary cache
    number_layers: int = 6 # number of transformer blocks
    number_heads: int = 8 # number of attention heads
    number_kv_heads: int = 4 # number of key-value heads (for GQA)
    embedding_dim: int = 256 # token embedding dimension (must be divisible by number_heads)
    use_gqa: bool = True # whether to use Grouped Query Attention (GQA)
    use_moe: bool = False # whether to use Mixture of Experts (MoE) in the feedforward layers
    number_experts: int = 8 # number of experts for MoE (if use_moe is True)

# normalise across embedding dimension
# x is (batch_size, sequence_length, embedding_dim) and we want to norm over C (embedding dimension)
def norm(hidden_states): return F.rms_norm(hidden_states, (hidden_states.size(-1),), eps=1e-8)

def apply_rotary(hidden_states, cos, sin):
    # x: (batch_size, num_heads, sequence_length, head_dim)
    # cos, sin: (sequence_length, head_dim//2) — already the correct half size from precompute_rotary
    half_dim = hidden_states.shape[-1] // 2 # head_dim must be even for RoPE
    first_half, second_half = hidden_states[..., :half_dim], hidden_states[..., half_dim:] # split head_dim into two halves for RoPE
    # cos/sin broadcast against (batch, heads, seq, half_dim) — no slicing needed since they're already half size
    rotated_first = first_half * cos + second_half * sin
    rotated_second = -first_half * sin + second_half * cos
    return torch.cat([rotated_first, rotated_second], dim=-1)

def precompute_rotary(seq_len, head_dim, device):
    # precompute cos and sin for RoPE
    # step=2 gives frequencies 1/10000^(2i/d) for i=0..d/2-1, matching the original RoPE paper
    # returns (seq_len, head_dim//2) tensors for cos and sin
    position = torch.arange(seq_len, device=device).unsqueeze(1) # (seq_len, 1)
    dimension_indices = torch.arange(0, head_dim, 2, device=device).unsqueeze(0) # (1, head_dim//2)
    inverse_frequencies = 1.0 / (10000 ** (dimension_indices / head_dim)) # (1, head_dim//2)
    frequencies = position * inverse_frequencies # (seq_len, head_dim//2)
    cos = frequencies.cos() # (seq_len, head_dim//2)
    sin = frequencies.sin() # (seq_len, head_dim//2)
    return cos, sin

class KVCache:
    # simple key-value cache for autoregressive decoding
    def __init__(self, number_layers):
        self.cached_keys, self.cached_values = [None] * number_layers, [None] * number_layers # list of key and value caches for each layer

    def update(self, layer_idx, new_keys, new_values, max_seq_len=None):
        # update the cache for a given layer with new keys and values
        if self.cached_keys[layer_idx] is None:
            # first time, just set the cache to the new keys and values
            self.cached_keys[layer_idx] = new_keys
            self.cached_values[layer_idx] = new_values
        else:
            # append new keys and values to the existing cache along the sequence dimension
            self.cached_keys[layer_idx] = torch.cat([self.cached_keys[layer_idx], new_keys], dim=2)
            self.cached_values[layer_idx] = torch.cat([self.cached_values[layer_idx], new_values], dim=2)

            # prevent KV cache from growing beyond sequence_length
            if max_seq_len is not None:
                self.cached_keys[layer_idx] = self.cached_keys[layer_idx][:, :, -max_seq_len:]
                self.cached_values[layer_idx] = self.cached_values[layer_idx][:, :, -max_seq_len:]

        return self.cached_keys[layer_idx], self.cached_values[layer_idx]


def repeat_kv(key_value_states, number_repetitions):
    '''
    For GQA, we have fewer key-value heads than query heads, so we need to repeat the keys and values to match the number of query heads.
    For example, if we have 12 query heads and 3 key-value heads, we need to repeat each key and value head 4 times to get 12 key and value heads.
    x is (batch_size, num_kv_heads, sequence_length, head_dim) and we want to return (batch_size, num_heads, sequence_length, head_dim) where num_heads = num_kv_heads * n_rep
    '''
    batch_size, number_kv_heads, token_count, head_dim = key_value_states.size()
    assert number_repetitions >= 1, "number_repetitions must be at least 1"
    key_value_states = key_value_states.unsqueeze(2) # (batch_size, num_kv_heads, 1, sequence_length, head_dim)
    key_value_states = key_value_states.expand(-1, -1, number_repetitions, -1, -1) # (batch_size, num_kv_heads, n_rep, sequence_length, head_dim)
    key_value_states = key_value_states.contiguous().view(batch_size, number_kv_heads * number_repetitions, token_count, head_dim) # (batch_size, num_heads, sequence_length, head_dim)
    return key_value_states

class CausalSelfAttention(nn.Module):
    # multi head attention with gqa, rope, qk norm, and kv cache (at inference)
    def __init__(self, config):
        super().__init__()
        self.number_heads = config.number_heads
        self.max_seq_len = config.sequence_length
        self.number_kv_heads = config.number_kv_heads if config.use_gqa else config.number_heads
        self.head_dim = config.embedding_dim // config.number_heads
        assert self.head_dim * self.number_heads == config.embedding_dim, "embedding_dim must be divisible by number_heads"
        assert self.number_heads % self.number_kv_heads == 0, "number_heads must be divisible by number_kv_heads"

        # separate projections for queries and keys/values for GQA
        self.query_proj = nn.Linear(config.embedding_dim, self.number_heads * self.head_dim, bias=False)
        self.key_proj = nn.Linear(config.embedding_dim, self.number_kv_heads * self.head_dim, bias=False)
        self.value_proj = nn.Linear(config.embedding_dim, self.number_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.out_proj._is_residual = True

    def forward(self, hidden_states, cos, sin, layer_idx, kv_cache=None):
        # x is (batch_size, sequence_length, embedding_dim)
        batch_size, token_count, channel_dim = hidden_states.size()

        # project
        query = self.query_proj(hidden_states).view(batch_size, token_count, self.number_heads, self.head_dim)
        key = self.key_proj(hidden_states).view(batch_size, token_count, self.number_kv_heads, self.head_dim)
        value = self.value_proj(hidden_states).view(batch_size, token_count, self.number_kv_heads, self.head_dim)

        # (B, H, T, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # QK-norm: normalizing Q and K prevents attention logit explosion at long sequences.
        query, key = norm(query), norm(key)

        # RoPE: cos/sin are already sliced to the correct position range by GPT.forward —
        # do NOT re-offset here, that would double-count the position and produce empty tensors
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)

        # expand KV heads to match Q heads
        number_repetitions = self.number_heads // self.number_kv_heads
        key = repeat_kv(key, number_repetitions)
        value = repeat_kv(value, number_repetitions)

        # KV cache (inference)
        if kv_cache is not None:
            key, value = kv_cache.update(layer_idx, key, value, max_seq_len=self.max_seq_len)

        # attention
        if kv_cache is None or token_count > 1:
            # training or prefill: q_len == kv_len, Flash Attention causal mask is valid
            attention_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            # single-token decode: q_len=1, kv_len=T+k
            # Flash Attention requires q_len == kv_len even with is_causal=False on older PyTorch/CUDA;
            # use explicit matmul which has no such constraint
            scale = query.shape[-1] ** -0.5
            attn_weights = F.softmax(query * scale @ key.transpose(-2, -1), dim=-1)
            attention_output = attn_weights @ value

        # merge heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, token_count, -1)

        return self.out_proj(attention_output)

class MLP(nn.Module):
    # ffn with optional MOE
    def __init__(self, config):
        super().__init__()
        hidden_dim = _round_to_multiple((8 * config.embedding_dim) // 3, 64) # for GQA, we want to increase the hidden dimension of the MLP to compensate for the reduced capacity of the attention, but we round it to a multiple of 64 for better GPU efficiency
        self.gate = nn.Linear(config.embedding_dim, hidden_dim, bias=False) # gating mechanism for GQA: we will use the output of this linear layer to gate the output of the first linear layer, which allows the model to learn to route different tokens to different experts in the MoE setting (if use_moe is True) or to modulate the MLP output in the standard setting (if use_moe is False)
        self.fc1  = nn.Linear(config.embedding_dim, hidden_dim, bias=False) # the first linear layer of the MLP; we set bias=False because the gating mechanism can learn to shift the output as needed, and this can save some memory and computation
        self.fc2  = nn.Linear(hidden_dim, config.embedding_dim, bias=False)
        self.fc2._is_residual = True

    def forward(self, hidden_states):
        return self.fc2(F.silu(self.gate(hidden_states)) * self.fc1(hidden_states))

class SparseMOE(nn.Module):
    '''
    Sparse Mixture of Experts for GQA
    Top-1 MoE where each token is routed to a single expert based on the gating mechanism.
    The gate will output a hidden_dim vector which we will use to compute a routing score for each expert and then we will select the expert with the highest score for each token
    The selected expert will then process the token and produce the output
    '''
    def __init__(self, config):
        super().__init__()
        self.number_experts = config.number_experts
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.number_experts)])
        self.gate = nn.Linear(config.embedding_dim, config.number_experts, bias=False)
        self.top_k = 2 # number of experts each token is routed to (top-2 routing like Mixtral)

    def forward(self, hidden_states):
        batch_size, token_count, channel_dim = hidden_states.shape
        # flatten tokens for routing: each token independently selects its experts
        flat_tokens = hidden_states.view(batch_size * token_count, channel_dim) # (batch_size * token_count, channel_dim)
        num_tokens = flat_tokens.shape[0]

        # gate scores: which experts should handle each token
        gate_logits = self.gate(flat_tokens) # (batch_size * token_count, number_experts)
        top_scores, top_expert_indices = torch.topk(gate_logits, self.top_k, dim=-1) # (batch_size * token_count, top_k)
        # normalize top-k scores to sum to 1 for each token so outputs are a weighted average
        routing_weights = F.softmax(top_scores, dim=-1) # (batch_size * token_count, top_k)

        # load balancing aux loss: penalizes routing collapse where few experts get all tokens
        router_probs = F.softmax(gate_logits, dim=-1)  # (num_tokens, num_experts) — full softmax for gradient
        dispatch_counts = torch.zeros(self.number_experts, device=hidden_states.device)
        dispatch_counts.scatter_add_(0, top_expert_indices.view(-1), torch.ones(num_tokens * self.top_k, device=hidden_states.device))
        dispatch_fraction = dispatch_counts / (num_tokens * self.top_k)  # fraction of assignments per expert
        self.last_aux_loss = 0.01 * self.number_experts * (dispatch_fraction * router_probs.mean(0)).sum()

        # accumulate expert outputs: only dispatch each token to its assigned experts (sparse, not all experts run)
        expert_output = torch.zeros_like(flat_tokens) # (batch_size * token_count, channel_dim)
        for k_position in range(self.top_k):
            assigned_expert_ids = top_expert_indices[:, k_position] # (batch_size * token_count,) which expert handles each token at this k position
            expert_weight = routing_weights[:, k_position:k_position + 1] # (batch_size * token_count, 1)
            for expert_id, expert in enumerate(self.experts):
                token_mask = (assigned_expert_ids == expert_id) # boolean mask: which tokens go to this expert
                if token_mask.any():
                    # only run the expert on the tokens routed to it, then scale by the routing weight
                    expert_output[token_mask] = expert_output[token_mask] + expert_weight[token_mask] * expert(flat_tokens[token_mask])

        return expert_output.view(batch_size, token_count, channel_dim)

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config) if not config.use_moe else SparseMOE(config)
        self.layer_idx = layer_idx

    def forward(self, hidden_states, cos, sin, kv_cache=None):
        # pre-norm: normalise before attention and MLP for training stability (standard in modern LLMs like LLaMA/Mistral)
        hidden_states = hidden_states + self.attn(norm(hidden_states), cos, sin, self.layer_idx, kv_cache)
        hidden_states = hidden_states + self.mlp(norm(hidden_states))
        return hidden_states

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_dim = config.embedding_dim // config.number_heads
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.number_layers)])
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight  # weight tying: shares embedding and unembedding matrices, standard in GPT-2
        # precompute rotary position embeddings for the full sequence length; registered as buffers so they move with the model to the correct device
        rotary_cos, rotary_sin = precompute_rotary(config.sequence_length, head_dim, torch.device("cpu"))
        self.register_buffer("rotary_cos", rotary_cos) # (sequence_length, head_dim)
        self.register_buffer("rotary_sin", rotary_sin) # (sequence_length, head_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # residual output projections use std = 0.02 / sqrt(2 * n_layers) to prevent gradient explosion at depth
            std = 0.02 / math.sqrt(2 * self.config.number_layers) if getattr(module, '_is_residual', False) else 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, target_ids=None, kv_cache=None):
        _, token_count = input_ids.size()
        hidden_states = self.token_embedding(input_ids)
        # during KV-cache inference, offset the rotary embeddings by the number of already-cached tokens so each new token gets its correct position
        if kv_cache is not None and kv_cache.cached_keys[0] is not None:
            offset = kv_cache.cached_keys[0].shape[2]
        else:
            offset = 0

        # slice cos and sin to the current token count plus any offset from cached tokens; this allows us to handle variable sequence lengths and ensures that during autoregressive decoding with KV cache, each new token gets the correct positional embedding based on its position in the overall sequence (including cached tokens)
        end = offset + token_count
        if end > self.config.sequence_length:
            offset = max(0, self.config.sequence_length - token_count)
            end = offset + token_count

        cos = self.rotary_cos[offset:end]
        sin = self.rotary_sin[offset:end]
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin, kv_cache)
        hidden_states = norm(hidden_states) # final RMSNorm before the language model head
        logits = self.head(hidden_states)
        # compute cross-entropy loss when target_ids are provided (training); return None for loss during inference
        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            if self.config.use_moe:
                aux_loss = sum(block.mlp.last_aux_loss for block in self.blocks)
                loss = loss + aux_loss
        return logits, loss

    def _sample_next_token(self, logits, temperature, top_k, top_p):
        """Apply temperature, top-k, and top-p filtering then sample one token."""
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[:, [-1]]] = -float("inf")
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probabilities = torch.cumsum(sorted_probs, dim=-1)
            tokens_to_remove = (cumulative_probabilities - sorted_probs) > top_p
            sorted_logits[tokens_to_remove] = -float("inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        kv_cache = KVCache(len(self.blocks))

        # Prefill: process the full prompt to populate the KV cache and get logits for the last position
        logits, _ = self(token_ids, kv_cache=kv_cache)

        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(logits, temperature, top_k, top_p)
            token_ids = torch.cat((token_ids, next_token), dim=1)

            # Decode step: feed only the new token; KV cache holds all prior context
            logits, _ = self(next_token, kv_cache=kv_cache)

        return token_ids

    @torch.no_grad()
    def stream_generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Same as generate() but yields each new token id as it is sampled, enabling streaming output."""
        kv_cache = KVCache(len(self.blocks))
        logits, _ = self(token_ids, kv_cache=kv_cache)

        for _ in range(max_new_tokens):
            if kv_cache.cached_keys[0] is not None:
                if kv_cache.cached_keys[0].shape[2] >= self.config.sequence_length:
                    kv_cache = KVCache(len(self.blocks))
                    logits, _ = self(token_ids, kv_cache=kv_cache)
            next_token = self._sample_next_token(logits, temperature, top_k, top_p)
            token_ids = torch.cat((token_ids, next_token), dim=1)

            # FIX: keep sequence within max length
            if token_ids.size(1) > self.config.sequence_length:
                token_ids = token_ids[:, -self.config.sequence_length:]
            yield next_token.item()

            logits, _ = self(next_token, kv_cache=kv_cache)
