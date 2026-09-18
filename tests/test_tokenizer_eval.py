"""Tests for src/evaluation/tokenizer_eval.py: fertility/UNK-rate measurement
and the JSON sweep report.
"""
import json

import sentencepiece as spm

from src.evaluation.tokenizer_eval import (
    evaluates_all_tokenizers_with_diff_vocabs,
    fertility,
    unk_rate,
)

_SAMPLE_TEXTS = ["यह एक हिन्दी वाक्य है।", "भारत एक विशाल देश है।", "मुझे किताबें पढ़ना पसंद है।"] * 30

_SWEEP_SENTENCES = [
    "यह एक हिन्दी वाक्य है।",
    "भारत एक विशाल देश है।",
    "मुझे किताबें पढ़ना पसंद है।",
    "आज मौसम बहुत अच्छा है।",
    "वह स्कूल जा रहा है।",
    "मुझे संगीत सुनना अच्छा लगता है।",
    "हमारा देश भारत महान है।",
    "बच्चे पार्क में खेल रहे हैं।",
    "यह किताब बहुत रोचक है।",
    "सूरज पूर्व दिशा से उगता है।",
] * 30  # enough diversity for unigram/bpe to reliably reach vocab_size 60/65


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


def _write_val_split(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(json.dumps({"text": text, "source": "val", "doc_id": str(i)}, ensure_ascii=False) + "\n")


def _train_sweep_candidates(artifacts_dir, testing_tokenizer_type, vocab_size_candidates):
    for model_type in testing_tokenizer_type:
        for vocab_size in vocab_size_candidates:
            input_path = artifacts_dir / f"train_{model_type}_{vocab_size}.txt"
            input_path.write_text("\n".join(_SWEEP_SENTENCES), encoding="utf-8")
            spm.SentencePieceTrainer.train(
                input=str(input_path),
                model_prefix=str(artifacts_dir / f"hindi_{model_type}_{vocab_size}"),
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=0.9995,
                byte_fallback=False,
            )


def _sweep_config(tmp_path, testing_tokenizer_type, vocab_size_candidates):
    artifacts_dir = tmp_path / "sample_vocabs"
    artifacts_dir.mkdir()
    _train_sweep_candidates(artifacts_dir, testing_tokenizer_type, vocab_size_candidates)

    val_split_path = tmp_path / "val.jsonl"
    _write_val_split(val_split_path, _SWEEP_SENTENCES[:10])

    return {
        "testing_tokenizer_type": testing_tokenizer_type,
        "vocab_size_candidates": vocab_size_candidates,
        "artificates_dir": str(artifacts_dir),
        "val_split_path": str(val_split_path),
        "report_path": str(tmp_path / "report.json"),
    }


def test_evaluates_all_tokenizers_with_diff_vocabs_returns_sweep_keyed_by_vocab_and_type(tmp_path):
    config = _sweep_config(tmp_path, testing_tokenizer_type=["unigram", "bpe"], vocab_size_candidates=[60, 65])

    sweep = evaluates_all_tokenizers_with_diff_vocabs(config)

    assert set(sweep.keys()) == {(60, "unigram"), (65, "unigram"), (60, "bpe"), (65, "bpe")}
    for metrics in sweep.values():
        assert set(metrics.keys()) == {"avg_chars_per_token", "avg_tokens_per_word", "unk_rate"}


def test_evaluates_all_tokenizers_with_diff_vocabs_writes_matching_json_report(tmp_path):
    config = _sweep_config(tmp_path, testing_tokenizer_type=["unigram"], vocab_size_candidates=[60])

    sweep = evaluates_all_tokenizers_with_diff_vocabs(config)

    with open(config["report_path"], encoding="utf-8") as f:
        report = json.load(f)

    assert report == {f"{vocab_size}_{model_type}": metrics for (vocab_size, model_type), metrics in sweep.items()}
