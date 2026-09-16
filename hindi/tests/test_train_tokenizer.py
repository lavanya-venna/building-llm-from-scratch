"""Tests for hindi/tokenizer/train_tokenizer.py.

Trains tiny real SentencePiece models on small fixture corpora (fast, no
mocking needed -- SentencePiece training itself is cheap at this scale).
"""
import os

from hindi.tokenizer.train_tokenizer import build_training_sample, train_sentencepiece_model

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

    out_path_a = tmp_path / "sample_a.txt"
    out_path_b = tmp_path / "sample_b.txt"
    build_training_sample(str(split_path), str(out_path_a), max_lines=10, seed=42)
    build_training_sample(str(split_path), str(out_path_b), max_lines=10, seed=42)

    lines_a = out_path_a.read_text(encoding="utf-8").strip().split("\n")
    lines_b = out_path_b.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines_a) == 10
    assert lines_a == lines_b  # deterministic for the same seed


def test_train_sentencepiece_model_produces_usable_model(tmp_path):
    split_path = tmp_path / "train.jsonl"
    _write_jsonl(split_path, _SAMPLE_HINDI_LINES)
    sample_path = tmp_path / "sample.txt"
    build_training_sample(str(split_path), str(sample_path), max_lines=250, seed=1)

    model_prefix = str(tmp_path / "hindi_unigram_1000")
    train_sentencepiece_model(
        input_file=str(sample_path),
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
