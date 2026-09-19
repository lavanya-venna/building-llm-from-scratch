"""Tests for src/tokenizer/tokenizer_test.py.

Trains tiny real SentencePiece models on small fixture corpora (fast, no
mocking needed -- SentencePiece training itself is cheap at this scale).
"""
import os

from src.tokenizer.tokenizer_test import (
    build_training_sample,
    decode_tokenized_samples,
    tokenize_val_samples,
    train_sentencepiece_model,
)

_SAMPLE_HINDI_LINES = [
    "यह एक हिन्दी वाक्य है।",
    "भारत एक विशाल देश है।",
    "मुझे किताबें पढ़ना पसंद है।",
    "आज मौसम बहुत अच्छा है।",
    "वह स्कूल जा रहा है।",
] * 50  # repeat so SentencePiece has enough signal to train on


def _write_jsonl(path, texts, source="wiki"):
    import json

    with open(path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(json.dumps({"text": text, "source": source, "doc_id": str(i)}, ensure_ascii=False) + "\n")


def test_build_training_sample_writes_capped_deterministic_file(tmp_path):
    split_path = tmp_path / "train.jsonl"
    _write_jsonl(split_path, _SAMPLE_HINDI_LINES)

    sample_a = build_training_sample(str(split_path), max_lines=10, seed=42)
    sample_b = build_training_sample(str(split_path), max_lines=10, seed=42)

    assert len(sample_a) == 10
    assert sample_a == sample_b  # deterministic for the same seed


def test_train_sentencepiece_model_produces_usable_model(tmp_path):
    split_path = tmp_path / "train.jsonl"
    _write_jsonl(split_path, _SAMPLE_HINDI_LINES)
    sentences = build_training_sample(str(split_path), max_lines=250, seed=1)

    model_prefix = str(tmp_path / "hindi_unigram_1000")
    train_sentencepiece_model(
        sentences=sentences,
        model_prefix=model_prefix,
        vocab_size=300,
        model_type="unigram",
        character_coverage=0.9995,
        byte_fallback=True,
    )

    assert os.path.exists(model_prefix + ".model")
    assert os.path.exists(model_prefix + ".vocab")

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    pieces = sp.encode("यह एक हिन्दी वाक्य है।", out_type=str)
    assert len(pieces) > 0


_TEN_UNIQUE_HINDI_SENTENCES = [
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
]


def test_tokenize_val_samples_returns_raw_text_to_pieces_mapping(tmp_path):
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(val_path, _TEN_UNIQUE_HINDI_SENTENCES)

    model_prefix = tmp_path / "hindi_unigram_60"
    train_sentencepiece_model(
        sentences=_TEN_UNIQUE_HINDI_SENTENCES * 30,
        model_prefix=str(model_prefix),
        vocab_size=60,
        model_type="unigram",
        character_coverage=0.9995,
        byte_fallback=False,
    )

    config = {
        "artificates_dir": str(tmp_path),
        "model_type": "unigram",
        "vocab_size": 60,
        "val_split_path": str(val_path),
    }
    result = tokenize_val_samples(config, num_samples=10)

    assert len(result) == 10
    assert set(result) == set(_TEN_UNIQUE_HINDI_SENTENCES)
    for pieces in result.values():
        assert isinstance(pieces, list) and len(pieces) > 0


def test_decode_tokenized_samples_recovers_original_sentences(tmp_path):
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(val_path, _TEN_UNIQUE_HINDI_SENTENCES)

    model_prefix = tmp_path / "hindi_unigram_60"
    train_sentencepiece_model(
        sentences=_TEN_UNIQUE_HINDI_SENTENCES * 30,
        model_prefix=str(model_prefix),
        vocab_size=60,
        model_type="unigram",
        character_coverage=0.9995,
        byte_fallback=False,
    )

    config = {
        "artificates_dir": str(tmp_path),
        "model_type": "unigram",
        "vocab_size": 60,
        "val_split_path": str(val_path),
    }
    tokenized_outputs = tokenize_val_samples(config, num_samples=10)
    decoded = decode_tokenized_samples(config, tokenized_outputs)

    assert len(decoded) == 10
    assert set(decoded) == set(tokenized_outputs.keys())  # round-trips back to the original raw text
