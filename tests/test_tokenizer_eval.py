"""Tests for src/evaluation/tokenizer_eval.py: fertility/UNK-rate measurement,
vocab selection, and the tokenizer_eval_metrics.md report renderer.
"""
import sentencepiece as spm

from src.evaluation.tokenizer_eval import (
    fertility,
    unk_rate,
    select_best_vocab,
    tokenizer_sweep_table,
    render_tokenizer_eval_report,
)

_SAMPLE_TEXTS = ["यह एक हिन्दी वाक्य है।", "भारत एक विशाल देश है।", "मुझे किताबें पढ़ना पसंद है।"] * 30


def _train_tiny_model(tmp_path, vocab_size, byte_fallback=True):
    input_path = tmp_path / f"sample_{vocab_size}.txt"
    input_path.write_text("\n".join(_SAMPLE_TEXTS), encoding="utf-8")
    prefix = str(tmp_path / f"model_{vocab_size}")
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        byte_fallback=byte_fallback,
    )
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


def test_fertility_reports_avg_chars_per_token(tmp_path):
    sp = _train_tiny_model(tmp_path, vocab_size=290)
    result = fertility(sp, _SAMPLE_TEXTS)
    assert result["avg_chars_per_token"] > 0
    assert result["avg_tokens_per_word"] > 0


def test_fertility_larger_vocab_has_higher_or_equal_chars_per_token(tmp_path):
    sp_small = _train_tiny_model(tmp_path, vocab_size=289)
    sp_large = _train_tiny_model(tmp_path, vocab_size=292)
    small_result = fertility(sp_small, _SAMPLE_TEXTS)
    large_result = fertility(sp_large, _SAMPLE_TEXTS)
    # Larger vocab -> longer average pieces -> more chars per token (or equal).
    assert large_result["avg_chars_per_token"] >= small_result["avg_chars_per_token"] - 1e-6


def test_unk_rate_is_zero_with_byte_fallback(tmp_path):
    sp = _train_tiny_model(tmp_path, vocab_size=290, byte_fallback=True)
    rate = unk_rate(sp, _SAMPLE_TEXTS + ["Some English text 🎉 with emoji and digits 12345"])
    assert rate == 0.0


def test_select_best_vocab_picks_smallest_near_optimal_fertility():
    sweep = {
        32000: {"avg_chars_per_token": 3.5, "unk_rate": 0.0},
        48000: {"avg_chars_per_token": 3.6, "unk_rate": 0.0},
        64000: {"avg_chars_per_token": 3.61, "unk_rate": 0.0},
    }
    # 48000 and 64000 are both near-optimal (within 5% of the best); pick the smallest.
    best = select_best_vocab(sweep, tolerance=0.05)
    assert best == 48000


def test_select_best_vocab_rejects_high_unk_rate_candidates():
    sweep = {
        32000: {"avg_chars_per_token": 3.9, "unk_rate": 0.2},
        48000: {"avg_chars_per_token": 3.6, "unk_rate": 0.0},
    }
    best = select_best_vocab(sweep, tolerance=0.05, max_unk_rate=0.01)
    assert best == 48000


def test_tokenizer_sweep_table_flags_selected_vocab():
    sweep = {
        32000: {"avg_chars_per_token": 3.5, "avg_tokens_per_word": 1.4, "unk_rate": 0.0},
        48000: {"avg_chars_per_token": 3.6, "avg_tokens_per_word": 1.3, "unk_rate": 0.0},
    }
    table = tokenizer_sweep_table(sweep, selected_vocab_size=48000)
    assert table[0]["vocab_size"] == 32000
    assert table[0]["selected"] is False
    assert table[1]["vocab_size"] == 48000
    assert table[1]["selected"] is True


def test_render_tokenizer_eval_report_produces_markdown_with_key_sections():
    report_data = {
        "tokenizer_sweep": [
            {"vocab_size": 32000, "avg_chars_per_token": 3.5, "avg_tokens_per_word": 1.4, "unk_rate": 0.0, "selected": True},
        ],
        "total_exact_tokens": 495_000_000,
        "examples": [{"text": "यह एक वाक्य है।", "pieces": ["यह", "एक", "वाक्य", "है", "।"]}],
    }
    markdown = render_tokenizer_eval_report(report_data)
    assert "# Tokenizer Evaluation Metrics" in markdown
    assert "32000" in markdown
    assert "495,000,000" in markdown or "495000000" in markdown
