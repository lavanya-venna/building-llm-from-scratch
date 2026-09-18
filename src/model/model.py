"""A from-scratch decoder-only (GPT-style) Transformer language model.

Built entirely from primitive PyTorch layers -- `nn.Linear`, `nn.Embedding`,
`nn.LayerNorm`, `nn.Dropout` -- with no `nn.Transformer*`, no
`nn.MultiheadAttention`, and no pretrained/HuggingFace model classes. This
module defines architecture only: every class implements just `__init__`
and `forward`. There is no loss computation, no weight initialization, no
generation/sampling method, and no training loop here -- those are later
phases.

Classes, in the order the data flows through them:

- `TokenEmbedding` -- looks up a learned vector per token id and scales it
  by `sqrt(d_model)`.
- `PositionalEmbedding` -- adds a per-position vector so the otherwise
  order-agnostic token embeddings carry sequence position. Supports
  `"learned"` (a trainable `nn.Embedding` per position) and `"sinusoidal"`
  (a fixed, non-trainable sin/cos pattern) via a `type` switch.
- `MultiHeadCausalAttention` -- lets each position mix information from
  itself and all earlier positions (never later ones), across several
  attention heads in parallel.
- `FeedForward` -- a per-position two-layer MLP (`d_model -> d_ff ->
  d_model`) with a GELU nonlinearity, applied identically at every
  position.
- `Block` -- one decoder layer: pre-norm self-attention with a residual
  connection, then pre-norm feed-forward with a residual connection.
- `DecoderOnlyTransformer` -- stacks `n_layers` `Block`s between the
  embeddings and a final linear head that projects back to vocabulary
  logits.

Design decisions
-----------------
**Pre-norm, not post-norm.** Each sublayer normalizes its *input*
(`x = x + sublayer(norm(x))`) rather than normalizing the sum of the
residual and the sublayer's output. This keeps the residual stream itself a
clean, unnormalized additive path all the way from the embeddings to the
final `ln_f` -- gradients flow through the `+` unimpeded by any
normalization sitting *on* the residual path. This is what makes deep
pre-norm stacks (GPT-2/3 style) trainable without the fragile warmup
schedules the original post-norm Transformer needed.

**No weight tying between the token embedding and the output head.**
Standard GPT implementations tie `token_embedding.weight` and
`lm_head.weight` to save `vocab_size * d_model` parameters and for a
documented regularizing effect. This implementation deliberately does not
tie them -- `TokenEmbedding` and `lm_head` stay fully decoupled and
independently testable. This is a simplification made for this from-scratch
implementation, not a claim that untied weights perform better.

**`sqrt(d_model)` scaling on token embeddings.** `nn.Embedding`'s default
initialization gives embeddings roughly unit variance, while the additive
positional signal (sinusoidal values are bounded in [-1, 1]; learned
embeddings start near unit variance too) is comparable in scale on its own.
Scaling the token embedding by `sqrt(d_model)` keeps its magnitude from
being dominated by -- or dominating -- the positional term once the two are
summed, before either has been shaped by any training. There is no custom
weight initialization elsewhere in this file to control embedding scale any
other way, so this multiplicative scaling is what keeps the two terms
comparable at initialization.

**RoPE is not supported.** `PositionalEmbedding` only implements
`"learned"` and `"sinusoidal"`. Rotary Positional Embeddings are not an
additive `(seq_len, d_model)` term added to token embeddings before the
first `Block` -- real RoPE rotates the query/key vectors *inside* attention
using per-position angle tensors, computed independently for each head, on
its own `d_k`-dimensional pairs. Supporting it correctly would require
`MultiHeadCausalAttention.forward` to accept per-position rotation tensors,
changing its interface -- out of scope here. Requesting `type="rope"` (or
any other unrecognized value) raises `ValueError` rather than silently
returning something that isn't real RoPE.

**Bias-free attention projections.** `MultiHeadCausalAttention`'s Q/K/V/O
`nn.Linear` layers are constructed with `bias=False`. `FeedForward`'s two
linear layers and `DecoderOnlyTransformer.lm_head` keep PyTorch's default
`bias=True` -- the bias-free choice is specific to the attention
projections.

Full dimension trace
---------------------
```
idx                                  (B, T)              int64 token ids
  -> TokenEmbedding                  (B, T, d_model)
  -> + PositionalEmbedding           (T, d_model), broadcasts over B
  =                                  (B, T, d_model)
  -> Block x n_layers (shape-preserving)
  =                                  (B, T, d_model)
  -> final nn.LayerNorm              (B, T, d_model)
  -> lm_head (nn.Linear)             (B, T, vocab_size)
  = logits                           (B, T, vocab_size)
```
"""
import math
import os

import torch
import torch.nn as nn
import yaml

_MODEL_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "model_config.yaml"
)


def load_model_config(config_path=None):
    """Load config/model_config.yaml.

    Centralizing this avoids hardcoding architecture/training hyperparameters
    in code, and matches load_tokenizer_config/load_data_config's pattern.

    Example: load_model_config()["d_model"] -> 384
    """
    path = config_path or _MODEL_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TokenEmbedding(nn.Module):
    """Looks up a learned vector per token id, scaled by sqrt(d_model).

    The scaling (see module docstring's "Design decisions") keeps embedding
    magnitude comparable to the additive positional term at initialization.
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # (B, T) -> (B, T, d_model)


class PositionalEmbedding(nn.Module):
    """Produces a per-position vector to add to token embeddings.

    `type="learned"` (default) uses a trainable `nn.Embedding` indexed by
    position. `type="sinusoidal"` uses a fixed, non-trainable sin/cos
    pattern (registered as a buffer, not a parameter). No dropout is applied
    in this class. RoPE is not supported -- see the module docstring's
    "Design decisions" for why.
    """

    def __init__(self, d_model, seq_len, type="learned"):
        super().__init__()
        self.type = type
        if type == "learned":
            self.pos_emb = nn.Embedding(seq_len, d_model)
        elif type == "sinusoidal":
            pe = torch.zeros(seq_len, d_model)
            position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )  # (d_model / 2,)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)
        else:
            raise ValueError(
                f"Unknown positional embedding type: {type!r} "
                "(supported: 'learned', 'sinusoidal')"
            )

    def forward(self, x):
        T = x.shape[1]  # (B, T)
        if self.type == "learned":
            positions = torch.arange(T, device=x.device)  # (T,)
            return self.pos_emb(positions)  # (T,) -> (T, d_model)
        return self.pe[:T]  # (seq_len, d_model) -> (T, d_model)


class MultiHeadCausalAttention(nn.Module):
    """Causal (masked) multi-head self-attention, built from scratch.

    Each position attends to itself and all earlier positions, never later
    ones, enforced by an additive upper-triangular -inf mask. Dropout (0.2)
    is applied to the post-softmax attention weights only, before the
    matmul with V -- nowhere else in this class.
    """

    def __init__(self, d_model, n_heads, seq_len, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        B, T, _ = x.shape  # (B, T, d_model)
        q = self.w_q(x)  # (B, T, d_model) -> (B, T, d_model)
        k = self.w_k(x)  # (B, T, d_model) -> (B, T, d_model)
        v = self.w_v(x)  # (B, T, d_model) -> (B, T, d_model)

        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, T, d_model) -> (B, n_heads, T, d_k)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, T, d_model) -> (B, n_heads, T, d_k)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, T, d_model) -> (B, n_heads, T, d_k)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (B, n_heads, T, d_k) @ (B, n_heads, d_k, T) -> (B, n_heads, T, T)
        attn_scores = attn_scores + self.causal_mask[:T, :T]
        # (B, n_heads, T, T) + (T, T) -> (B, n_heads, T, T)
        attn_weights = attn_scores.softmax(dim=-1)  # (B, n_heads, T, T) -> (B, n_heads, T, T)
        attn_weights = self.dropout(attn_weights)  # dropout(p=0.2), post-softmax, pre-matmul-with-V

        out = attn_weights @ v  # (B, n_heads, T, T) @ (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_k)
        # (B, n_heads, T, d_k) -> (B, T, n_heads, d_k) -> (B, T, d_model)
        return self.w_o(out)  # (B, T, d_model) -> (B, T, d_model)


class FeedForward(nn.Module):
    """Per-position two-layer MLP with a GELU nonlinearity. No dropout."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)  # (B, T, d_model) -> (B, T, d_ff)
        x = self.gelu(x)  # (B, T, d_ff) -> (B, T, d_ff)
        return self.fc2(x)  # (B, T, d_ff) -> (B, T, d_model)


class Block(nn.Module):
    """One decoder block: pre-norm self-attention, then pre-norm feed-forward.

    Both sublayers are wrapped in a residual connection
    (`x = x + sublayer(norm(x))`), keeping the residual stream unnormalized
    -- see the module docstring's "Design decisions" for why this matters.
    """

    def __init__(self, d_model, n_heads, d_ff, seq_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadCausalAttention(d_model, n_heads, seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # (B, T, d_model) -> (B, T, d_model)
        x = x + self.ffwd(self.ln2(x))  # (B, T, d_model) -> (B, T, d_model)
        return x


class DecoderOnlyTransformer(nn.Module):
    """The full decoder-only Transformer LM.

    Stacks `n_layers` `Block`s between the embeddings and a final linear
    head that projects back to vocabulary logits. `lm_head` does not share
    weights with `token_embedding` -- see the module docstring's "Design
    decisions" for why.
    """

    def __init__(self, config):
        super().__init__()
        assert config["vocab_size"] is not None, (
            "config['vocab_size'] is null in model_config.yaml -- set it to "
            "the trained SentencePiece model's vocab size before building "
            "the model"
        )
        d_model = config["d_model"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        d_ff = config["d_ff"]
        seq_len = config["seq_len"]
        dropout = config["dropout"]
        vocab_size = config["vocab_size"]

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, seq_len)  # default type="learned"
        self.blocks = nn.Sequential(
            *[Block(d_model, n_heads, d_ff, seq_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)  # separate weights -- no tying with token_embedding

    def forward(self, idx):
        tok_emb = self.token_embedding(idx)  # (B, T) -> (B, T, d_model)
        pos_emb = self.positional_embedding(idx)  # (B, T) -> (T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model) + (T, d_model) -> (B, T, d_model)
        x = self.blocks(x)  # (B, T, d_model) -> (B, T, d_model)
        x = self.ln_f(x)  # (B, T, d_model) -> (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, d_model) -> (B, T, vocab_size)
        return logits
