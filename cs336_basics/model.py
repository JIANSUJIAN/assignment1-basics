from einops import einsum, rearrange
from numpy import dtype
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # We store weight matrix W of shape (out_features, in_features)
        # following the column vector notation of y = Wx
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        # Initialization
        sigma = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """
    Maps integer token IDs to dense vectors.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Shape: (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        # Returns: (batch_size, seq_len, d_model)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # Learnable gain parameter g_i, initialzied to 1s
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        d_model = x_f32.shape[-1]
        mean_sq = einsum(x_f32, x_f32, "... d, ... d -> ...") / d_model

        rms = torch.sqrt(mean_sq + self.eps)
        inv_rms = 1.0 / rms
        result = einsum(x_f32, inv_rms, self.weight, "... d, ..., d -> ... d")

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        x1 = self.w1(x)
        x3 = self.w3(x)

        silu_x1 = x1 * torch.sigmoid(x1)

        gated = einsum(silu_x1, x3, "... f, ... f -> ... f")

        return self.w2(gated)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k

        # Precompute frequencies: theta_k = theta^(-2(k-1)/d_k) for k in 1...d_k/2
        # We compute in float64 for numerical precision, then convert to float32.
        # Always compute on CPU for device compatibility (MPS doesn't support float64,
        # and this is only done once during init so there's no performance impact).
        # The buffers will be moved to the target device via model.to(device).
        powers = torch.arange(0, d_k, 2, dtype=torch.float64) / d_k
        freqs = 1.0 / (theta**powers)  # Shape: (d_k/2, )

        # Precompute angles for all positions
        # t: (max_seq_len, ), freqs: (d_k/2, ) -> angles: (max_seq_len, d_k/2)
        t = torch.arange(max_seq_len, dtype=torch.float64)
        angles = einsum(t, freqs, "t, f -> t f")  # Outer product

        # Convert to float32 and register as buffers
        self.register_buffer("cos", torch.cos(angles).float(), persistent=False)
        self.register_buffer("sin", torch.sin(angles).float(), persistent=False)

    def forward(self, x, token_positions):
        # x shape: (batch, num_heads, seq_len, head_dim) from multi-head attention
        # token_positions shape: (batch, seq_len)

        # Lookup cos/sin for the given positions
        # self.cos: (max_seq_len, head_dim/2), token_positions: (batch, seq_len)
        # Result: (batch, seq_len, head_dim/2) -> unsqueeze to (batch, 1, seq_len, head_dim/2)
        # The unsqueeze(1) adds a dimension for num_heads to broadcast correctly
        batch_cos = self.cos[token_positions].unsqueeze(1)
        batch_sin = self.sin[token_positions].unsqueeze(1)

        # Split head_dim into pairs (f=head_dim/2, c=2) for rotation.
        # x: (batch, heads, seq_len, head_dim) -> (batch, heads, seq_len, head_dim/2, 2)
        x_pairs = rearrange(x, "... s (f c) -> ... s f c", c=2)
        x_even = x_pairs[..., 0]  # x_{2k-1}, shape: (batch, heads, seq_len, head_dim/2)
        x_odd = x_pairs[..., 1]  # x_{2k}, shape: (batch, heads, seq_len, head_dim/2)

        # Apply the pair-wise rotation
        # batch_cos/sin broadcast from (batch, 1, seq_len, head_dim/2) to match x_even/x_odd
        x_even_new = (x_even * batch_cos) - (x_odd * batch_sin)
        x_odd_new = (x_even * batch_sin) + (x_odd * batch_cos)

        return rearrange([x_even_new, x_odd_new], "c ... s f -> ... s (f c)")


def softmax(x, dim: int = -1):
    x_max = x.max(dim=dim, keepdim=True).values

    exp_x = torch.exp(x - x_max)

    sum_exp = exp_x.sum(dim=dim, keepdim=True)

    return exp_x / sum_exp


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Math (Column-major): Attention(Q, K, V) = softmax( (Q^T K) / sqrt(d_k) ) V
    Code (Row-major):    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V
    """
    # q: (..., seq_q, d_k)
    # k: (..., seq_k, d_k)
    # v: (..., seq_k, d_v)
    d_k = q.shape[-1]

    # Compute attention scores: (Q K^T) / sqrt(d_k)
    scores = einsum(q, k, "... q d, ... k d -> ... q k") / (d_k**0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, dim=-1)

    return einsum(weights, v, "... q k, ... k v -> ... q v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, rope_theta: float | None, max_seq_len: int, device=None, dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Fused QKV projection
        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)

        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if rope_theta:
            self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)

        qkv_heads = rearrange(qkv, "b s (qkv h d) -> qkv b h s d", qkv=3, h=self.num_heads)
        q, k, v = qkv_heads[0], qkv_heads[1], qkv_heads[2]

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        context = scaled_dot_product_attention(q, k, v, mask=mask)

        context = rearrange(context, "b h s d -> b s (h d)")

        return self.o_proj(context)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, rope_theta: float, max_seq_len: int, device=None, dtype=None
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_theta, max_seq_len, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope_theta, max_seq_len, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x):
        # x shape (batch, seq_len)
        x = self.token_embeddings(x)
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], seq_len)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)


def cross_entropy(logits, targets):
    logits_flat = logits.view(-1, logits.shape[-1])
    targets_flat = targets.view(-1)

    logits_max = logits_flat.max(dim=-1, keepdim=True).values
    logits_stable = logits_flat - logits_max

    exp_logits = torch.exp(logits_stable)
    log_sum_exp = torch.log(exp_logits.sum(dim=-1))

    # Get logits of target classes
    # We select row 'i' and column 'targets[i]'
    target_logits = logits_stable[torch.arange(logits_flat.shape[0]), targets_flat]

    loss = log_sum_exp - target_logits

    return loss.mean()
