"""Tests for src/model/model.py.

Uses small fixed dimensions throughout (d_model=16, n_heads=4, d_ff=32,
seq_len=8) so every test runs in milliseconds on CPU -- no mocking needed,
this is pure tensor-shape and numerical-contract verification.
"""
import math

import torch
import torch.nn as nn

from src.model.model import (
    Block,
    DecoderOnlyTransformer,
    FeedForward,
    MultiHeadCausalAttention,
    PositionalEmbedding,
    TokenEmbedding,
    load_model_config,
)

_D_MODEL = 16
_N_HEADS = 4
_D_FF = 32
_SEQ_LEN = 8
_VOCAB_SIZE = 50
_DROPOUT = 0.2


def _small_config(vocab_size=_VOCAB_SIZE):
    return {
        "d_model": _D_MODEL,
        "n_layers": 2,
        "n_heads": _N_HEADS,
        "d_ff": _D_FF,
        "seq_len": _SEQ_LEN,
        "dropout": _DROPOUT,
        "vocab_size": vocab_size,
    }


# --- TokenEmbedding ---------------------------------------------------------


def test_token_embedding_output_shape():
    module = TokenEmbedding(_VOCAB_SIZE, _D_MODEL)
    idx = torch.randint(0, _VOCAB_SIZE, (2, _SEQ_LEN))
    out = module(idx)
    assert out.shape == (2, _SEQ_LEN, _D_MODEL)


def test_token_embedding_scales_by_sqrt_d_model():
    module = TokenEmbedding(_VOCAB_SIZE, _D_MODEL)
    idx = torch.randint(0, _VOCAB_SIZE, (2, _SEQ_LEN))
    raw = module.embedding(idx)
    scaled = module(idx)
    assert torch.allclose(scaled, raw * math.sqrt(_D_MODEL))


# --- PositionalEmbedding ----------------------------------------------------


def test_positional_embedding_learned_shape():
    module = PositionalEmbedding(_D_MODEL, _SEQ_LEN, type="learned")
    idx = torch.randint(0, _VOCAB_SIZE, (2, _SEQ_LEN))
    out = module(idx)
    assert out.shape == (_SEQ_LEN, _D_MODEL)
    assert isinstance(module.pos_emb, nn.Embedding)


def test_positional_embedding_sinusoidal_known_values():
    module = PositionalEmbedding(_D_MODEL, _SEQ_LEN, type="sinusoidal")
    idx = torch.randint(0, _VOCAB_SIZE, (2, _SEQ_LEN))
    out = module(idx)
    assert out.shape == (_SEQ_LEN, _D_MODEL)
    # position 0: sin(0) == 0 at even indices, cos(0) == 1 at odd indices
    assert torch.allclose(out[0, 0], torch.tensor(0.0))
    assert torch.allclose(out[0, 1], torch.tensor(1.0))


def test_positional_embedding_invalid_type_raises():
    for bad_type in ("rope", "bogus"):
        try:
            PositionalEmbedding(_D_MODEL, _SEQ_LEN, type=bad_type)
            assert False, f"expected ValueError for type={bad_type!r}"
        except ValueError:
            pass


def test_positional_embedding_has_no_dropout():
    for type_ in ("learned", "sinusoidal"):
        module = PositionalEmbedding(_D_MODEL, _SEQ_LEN, type=type_)
        assert not any(isinstance(m, nn.Dropout) for m in module.modules())


# --- MultiHeadCausalAttention ------------------------------------------------


def test_multihead_causal_attention_output_shape():
    module = MultiHeadCausalAttention(_D_MODEL, _N_HEADS, _SEQ_LEN, _DROPOUT)
    x = torch.randn(2, _SEQ_LEN, _D_MODEL)
    out = module(x)
    assert out.shape == (2, _SEQ_LEN, _D_MODEL)


def test_multihead_causal_attention_qkvo_are_bias_free_linear():
    module = MultiHeadCausalAttention(_D_MODEL, _N_HEADS, _SEQ_LEN, _DROPOUT)
    for proj in (module.w_q, module.w_k, module.w_v, module.w_o):
        assert isinstance(proj, nn.Linear)
        assert proj.bias is None


def test_multihead_causal_attention_has_single_dropout_at_p():
    module = MultiHeadCausalAttention(_D_MODEL, _N_HEADS, _SEQ_LEN, _DROPOUT)
    dropouts = [m for m in module.modules() if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 1
    assert dropouts[0].p == _DROPOUT


def test_multihead_causal_attention_causal_mask_buffer_values():
    module = MultiHeadCausalAttention(_D_MODEL, _N_HEADS, _SEQ_LEN, _DROPOUT)
    mask = module.causal_mask
    assert mask.shape == (_SEQ_LEN, _SEQ_LEN)
    for i in range(_SEQ_LEN):
        for j in range(_SEQ_LEN):
            if j > i:
                assert mask[i, j] == float("-inf")
            else:
                assert mask[i, j] == 0.0


def test_multihead_causal_attention_blocks_future_positions():
    module = MultiHeadCausalAttention(_D_MODEL, _N_HEADS, _SEQ_LEN, dropout=0.0)
    module.eval()
    x_a = torch.randn(1, _SEQ_LEN, _D_MODEL)
    x_b = x_a.clone()
    x_b[0, -1] = torch.randn(_D_MODEL)  # perturb only the last timestep

    with torch.no_grad():
        out_a = module(x_a)
        out_b = module(x_b)

    # every position before the last must be unaffected by a future change
    assert torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-6)


# --- FeedForward -------------------------------------------------------------


def test_feedforward_output_shape():
    module = FeedForward(_D_MODEL, _D_FF)
    x = torch.randn(2, _SEQ_LEN, _D_MODEL)
    out = module(x)
    assert out.shape == (2, _SEQ_LEN, _D_MODEL)


def test_feedforward_uses_gelu():
    module = FeedForward(_D_MODEL, _D_FF)
    assert any(isinstance(m, nn.GELU) for m in module.modules())


def test_feedforward_has_no_dropout():
    module = FeedForward(_D_MODEL, _D_FF)
    assert not any(isinstance(m, nn.Dropout) for m in module.modules())


# --- Block -------------------------------------------------------------------


def test_block_output_shape_preserved():
    block = Block(_D_MODEL, _N_HEADS, _D_FF, _SEQ_LEN, _DROPOUT)
    x = torch.randn(2, _SEQ_LEN, _D_MODEL)
    out = block(x)
    assert out.shape == (2, _SEQ_LEN, _D_MODEL)


def test_block_zero_sublayer_weights_gives_identity():
    block = Block(_D_MODEL, _N_HEADS, _D_FF, _SEQ_LEN, dropout=0.0)
    block.eval()
    with torch.no_grad():
        block.sa.w_o.weight.zero_()
        block.ffwd.fc2.weight.zero_()
        block.ffwd.fc2.bias.zero_()

    x = torch.randn(2, _SEQ_LEN, _D_MODEL)
    with torch.no_grad():
        out = block(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_block_has_exactly_one_dropout_module():
    block = Block(_D_MODEL, _N_HEADS, _D_FF, _SEQ_LEN, _DROPOUT)
    dropouts = [m for m in block.modules() if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 1
    assert dropouts[0] is block.sa.dropout


# --- DecoderOnlyTransformer ---------------------------------------------------


def test_decoder_only_transformer_output_shape():
    model = DecoderOnlyTransformer(_small_config())
    idx = torch.randint(0, _VOCAB_SIZE, (2, _SEQ_LEN))
    logits = model(idx)
    assert logits.shape == (2, _SEQ_LEN, _VOCAB_SIZE)


def test_decoder_only_transformer_no_weight_tying():
    model = DecoderOnlyTransformer(_small_config())
    assert model.token_embedding.embedding.weight is not model.lm_head.weight


def test_decoder_only_transformer_raises_on_null_vocab_size():
    try:
        DecoderOnlyTransformer(_small_config(vocab_size=None))
        assert False, "expected AssertionError for null vocab_size"
    except AssertionError:
        pass


def test_decoder_only_transformer_block_count_matches_n_layers():
    config = _small_config()
    model = DecoderOnlyTransformer(config)
    assert len(model.blocks) == config["n_layers"]


def test_decoder_only_transformer_default_positional_embedding_is_learned():
    model = DecoderOnlyTransformer(_small_config())
    assert isinstance(model.positional_embedding.pos_emb, nn.Embedding)


# --- load_model_config ---------------------------------------------------


def test_load_model_config_returns_expected_keys():
    config = load_model_config()
    expected_keys = {
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "seq_len",
        "dropout",
        "vocab_size",
        "batch_size",
        "learning_rate",
        "max_epochs",
        "warmup_steps",
    }
    assert expected_keys.issubset(config.keys())
    assert config["vocab_size"] is None
