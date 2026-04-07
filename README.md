Attempting to understand and code karpathy's nanochat from scratch


# Configuration

- sequence length = 1024 (max seq)
 - GPT5 has context length of 400k   

- vocab size = 50304
  - size of model's input / output space (every token maps to an ID between 0
    and 50304 and output is predicted prob over 50304 possible tokens)

- q, k, v heads are different so we can have standard MHA or MQA (k, v are
  hashed)

- number_embeddings = 768 (every token is represented as a vector of 768 numbers
  inside the model)

Hence embedding layer will be (vocab_size, number_embeddings) = matrix of
(50304, 768) (EACH ROW = REPRESENTATION OF ONE TOKEN)

If sentence has 10 tokens: (10, 768)

# Components

- Normalisation: RMSNorm (no learned scale gamme or beta parameters) across
  embedding dimension of the vector

- For positional encoding we use ROPE.

- traditional absolute positonal encoding has the disadvantage that a model has to figure out that tokens at position 1 and 6 have the same distance between them as tokens at position 22 and 27. While they are quite capable of doing a good job at it (with enough data), they do not learn that equivariance perfectly and have to spend considerable model capacity on this simple task. So the modern way of providing positional information to a Transformer is to rotate token vectors according to their position (and rotate each position by a different frequency, similar to positional encoding). This has the advantage that the dot product between two tokens (as performed for key-query matching) is a function of the values of the two tokens and the difference of their positions:

- So what does ROPE do? It injects positional info into attention vectors by rotating pairs of dimensions

- Instead of adding position; we rotate vector based on position

  - We use cos and sin (precomputed rotation values) these encode where the token is in the sequence
  - Take half the head dimension and split last dimension into two halves (a1-32 and b1-32) so x1 is first half, x2 is second half
  - (a_i, b_i) is a 2d point; then rotate using 2d rotation formula (x,y) = xcos(theta) + ysin(theta), -xsin(theta) + ycos(theta) 
  - Each pair (a_i, b_i) is rotated by an angle
    - token at position 1 → small rotation and token at position 50 → larger rotation

- merge rotated half back together and ensure datatype consistency
  - y1 and y2 are combined back to have the same shape as the input x and their datatype is set to be consistent.

Why does this work? Dot products preserve relative angles
Attention uses Q dot K, with rotation the relative positions are encoded in
angle differences and attention can naturally detect distance


angle differences and attention can naturally detect distance

RoPE works by splitting the embedding into pairs and rotating each pair based on position, so position is encoded as geometry instead of addition.


number_embedddinge = 758, number_heads = 12 so head_dim =
number_embeddding/number_heads = 64 which is the size of vector each attention head works on.

To make it more clear:
Embedding Dimension which is 768 = All the information about a token; instead of
giving the whole thing to one attention mechansim; we split into 12 smaller
views (num_attention_heads) and each head then gets 64 dimensional slice

Head 1 → first 64 dims  
Head 2 → next 64 dims  
...  
Head 12 → last 64 dims

It does this so each head learns different relationships

Coming up to next code bit:
GQA

In standard attention: 12 heads → 12 Q, 12 K, 12 V
Keys and Values are expensive to store and compute, especially during inference
(KV cache grows with sequence length)

We can share K and V across group of heads and keep Q seperate
num_heads = 12
num_kv_heads = 4

Then:

Queries → still 12 (one per head)
Keys/Values → only 4

Each group of 3 (Q) share same K and V  
Heads:     [1  2  3]  [4  5  6]  [7  8  9]  [10 11 12]
KV group:   K1 V1      K2 V2      K3 V3        K4 V4

Each head = a “question asker” (Query)
K/V = “memory”
Standard attention:

Every head has its own private memory

GQA:

Multiple heads ask different questions, but look into the same shared memory

Taking example:
n_heads = 12
n_kv_heads = 4
n_rep = 3

input: x.shape = (batch, 4, seq_len, head_dim)

Now KV heads = [K1, K2, K3, K4]

Step1: add dimension: x[:, :, None, :, :]
So shape becomes (batch, 4, 1, seq_len, head_dim)

Step2: Expland: .expand(batch_size, 4, 3, seq_len, head_dim)
So shape becomes (batch, 4, 3, seq_len, head_dim)

K1 → [K1, K1, K1]
K2 → [K2, K2, K2]
...

This just creates a view^ not copy 

Step 3: Reshape .reshape(batch_size, 12, seq_len, head_dim)
So shape becomes (batch, 12, seq_len, head_dim)

So before: [ K1  K2  K3  K4 ]

Now its grouped: [ K1 K1 K1  K2 K2 K2  K3 K3 K3  K4 K4 K4 ]
We are reusing the same K/V across multiple heads

This function expands KV heads so each query head gets a matching key/value, without actually duplicating data in memory.

- Next: Causal Self Attention

Based on the values for key/value and query heads, we set up the projection matrices that will transform the inputs into keys, values and queries (c_q, c_k, c_v). All three of them have the same input dimension n_embed, since they all project the same input. Their second dimension depends on the values of n_head (for queries) and n_kv_heads (for keys and values).

### Step 1: Start
token → 768-d vector

### Step 2: Split into heads
→ 12 heads × 64 dims

### Step 3: Create Q, K, V
Q → 12 heads  
K/V → 4 heads  

### Step 4: Expand K/V (your earlier function!)
4 → 12 heads  

### Step 5: Attention happens
Each head:  
Q_i attends to K_i → produces output_i  

### Step 6: Merge heads
12 × 64 → 768  

### Step 7: Final projection
Mix everything together


## Forward Pass



The computation of keys and values is expensive and has to be repeated many times due to the autoregressive nature of LLMs, so it's common to store those values in a kv_cache. While that significantly increases our memory consumption it simultaneously decreases the time of new token generation by much

Before we come to the actual attention operation, we store the numbers of queries Tq and the number of key/values Tk, as we need them in a second. Furthermore, we need to call repeat_kv to change the dimension of keys and values such that they match those of the queries.

Applying the attention mechanism differs slightly depending on the availability of the kv_cache (only during inference) and the number of queries in case of an available kv-cache. Let's look at all three cases and see, how and why they differ:

Case 1 No kv-cache (training phase) or during prefill of inference (before autoregressive generation): During training we have the benefit of knowing already the full sequence we want to predict and we can use this fact to perform a prediction for the next token at every position simultaneously. This greatly increases efficiency and is possible by restricting the attention mechanism to only look back. E.g., for predicting the fifth token we want the attention to only use the first four tokens, while for the sixth token we want to look also at the fifth. This restriction of not looking back is implemented with a causal masking matrix that is a binary lower triangular matrix that is multiplied with the attention scores. We enable this causal masking by setting is_causal=True.
Important: Each next token prediction is based on the correct context, i.e., the model is not yet in an autoregressive mode. If we would sequentialize the process for illustration purposes, we would let the model predict the n-th token based on all tokens up to n, calculate the loss, then swap the prediction with the correct next token and let the model predict n+1.
Case 2 There is just a single query (simple next token prediction during inference). We call the same function as in the first case, but this time disable the causal masking with is_causal=False.
Case 3 Multiple queries during inference: This is a slightly advanced case, which could be used for speculative decoding. We basically provide a potential next token sequence and validate that. The implementation of it is a combination of case 1 and 2, since we need full attention for all previous kv values (like in case 2), but a causal masking for the newly predicted tokens (like in case 1). For that reason we have to manually construct the attention mask with True for the context (up until prefix_len) and a triangular lower part (tril) afterwards. This matrix is then passed to the dot product calculation.
Finally as a last step we just need reverse the transposing we did earlier to then combine the heads back into one dimension (.view(B, T, -1)). That tensor is then passed through the final output projection c_proj.


Results

ss 1.6450 | lr 0.000210 step 800 | loss 1.5396 | lr 0.000240 step 900 | loss 1.4526 | lr 0.000270 step 1000 | loss 1.5823 | lr 0.000300 step 1100 | loss 1.4367 | lr 0.000300 step 1200 | loss 1.3718 | lr 0.000300 step 1300 | loss 1.4295 | lr 0.000300 step 1400 | loss 1.3907 | lr 0.000300 step 1500 | loss 1.3534 | lr 0.000300 step 1600 | loss 1.3832 | lr 0.000299 step 1700 | loss 1.3367 | lr 0.000299 step 1800 | loss 1.2612 | lr 0.000299 step 1900 | loss 1.3259 | lr 0.000299 step 2000 | loss 1.2646 | lr 0.000298 step 2100 | loss 1.2539 | lr 0.000298 step 2200 | loss 1.2739 | lr 0.000297 step 2300 | loss 1.2425 | lr 0.000297 step 2400 | loss 1.2826 | lr 0.000296 step 2500 | loss 1.2589 | lr 0.000296 step 2600 | loss 1.2213 | lr 0.000295 step 2700 | loss 1.2713 | lr 0.000295 step 2800 | loss 1.2234 | lr 0.000294 step 2900 | loss 1.1928 | lr 0.000293 step 3000 | loss 1.2068 | lr 0.000293 step 3100 | loss 1.1849 | lr 0.000292 step 3200 | loss 1.1831 | lr 0.000291 step 3300 | loss 1.1471 | lr 0.000290 step 3400 | loss 1.1640 | lr 0.000290 step 3500 | loss 1.1237 | lr 0.000289 step 3600 | loss 1.1159 | lr 0.000288 step 3700 | loss 1.1362 | lr 0.000287 step 3800 | loss 1.1571 | lr 0.000286 step 3900 | loss 1.1171 | lr 0.000285 step 4000 | loss 1.1353 | lr 0.000284 step 4100 | loss 1.1304 | lr 0.000283 step 4200 | loss 1.0493 | lr 0.000282 step 4300 | loss 1.0865 | lr 0.000280 step 4400 | loss 1.0544 | lr 0.000279 step 4500 | loss 1.0677 | lr 0.000278 step 4600 | loss 1.0525 | lr 0.000277 step 4700 | loss 1.0005 | lr 0.000276 step 4800 | loss 1.0231 | lr 0.000274 step 4900 | loss 0.9781 | lr 0.000273 step 5000 | loss 1.0624 | lr 0.000272 step 5100 | loss 0.9813 | lr 0.000270 step 5200 | loss 1.0017 | lr 0.000269 step 5300 | loss 0.9361 | lr 0.000267 step 5400 | loss 0.9230 | lr 0.000266 step 5500 | loss 0.9175 | lr 0.000264 step 5600 | loss 0.9175 | lr 0.000263 step 5700 | loss 0.8594 | lr 0.000261 step 5800 | loss 0.9066 | lr 0.000260