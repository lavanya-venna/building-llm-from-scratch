"""Tests for src/model/train.py.

Every test builds a small, self-contained config pointing entirely at
tmp_path fixtures (its own tokenizer_path, dataset_paths, tiny seq_len/
batch_size) -- never load_model_config() or the real data/splits/*.jsonl.
Uses a tiny real SentencePiece tokenizer (same pattern as test_tokenizer.py/
test_dataset.py) rather than mocking, since SentencePiece training is cheap
at this scale.
"""
import json
import os

import pytest
import sentencepiece as spm

from src.model.model import DecoderOnlyTransformer
from src.model.train import (
    _read_jsonl_texts,
    _resolve_repo_path,
    get_ds,
    get_or_build_tokenizer,
    tokenize_and_concatenate,
    train_model,
)
from src.tokenizer.tokenizer_test import build_training_sample as real_build_training_sample
from src.tokenizer.tokenizer_test import train_sentencepiece_model

_SAMPLE_HINDI_LINES = [
    "यह एक हिन्दी वाक्य है।",
    "भारत एक विशाल देश है।",
    "मुझे किताबें पढ़ना पसंद है।",
    "आज मौसम बहुत अच्छा है।",
    "वह स्कूल जा रहा है।",
] * 50

_TINY_TOKENIZER_CONFIG = {
    "selected_vocab_size": 300,
    "model_type": "unigram",
    "character_coverage": 0.9995,
    "byte_fallback": True,
    "training_sample_size": 1000,
    "training_sample_seed": 42,
}


def _write_jsonl(path, texts, source="wiki"):
    with open(path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(
                json.dumps(
                    {"text": text, "source": source, "doc_id": str(i), "pipeline_stage": "clean"},
                    ensure_ascii=False,
                )
                + "\n"
            )


@pytest.fixture(scope="module")
def tiny_tokenizer_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("train_tokenizer")

    model_prefix = str(tmp_path / "tiny_unigram_300")
    train_sentencepiece_model(
        sentences=_SAMPLE_HINDI_LINES,
        model_prefix=model_prefix,
        vocab_size=300,
        model_type="unigram",
        character_coverage=0.9995,
        byte_fallback=True,
    )
    return model_prefix + ".model"


# --- _resolve_repo_path / _read_jsonl_texts ---------------------------------


def test_resolve_repo_path_returns_absolute_path_unchanged(tmp_path):
    abs_path = str(tmp_path / "some_file.jsonl")
    assert _resolve_repo_path(abs_path) == abs_path


def test_resolve_repo_path_joins_relative_path_against_repo_root():
    import src.model.train as train_module

    repo_root = os.path.join(os.path.dirname(train_module.__file__), "..", "..")
    expected = os.path.normpath(os.path.join(repo_root, "data/splits/train.jsonl"))
    actual = os.path.normpath(_resolve_repo_path("data/splits/train.jsonl"))
    assert actual == expected


def test_read_jsonl_texts_extracts_text_field_in_file_order(tmp_path):
    path = tmp_path / "fixture.jsonl"
    _write_jsonl(path, ["a", "b", "c"])
    assert _read_jsonl_texts(str(path)) == ["a", "b", "c"]


# --- get_or_build_tokenizer --------------------------------------------------


def test_get_or_build_tokenizer_trains_and_saves_when_missing(tmp_path, monkeypatch):
    tokenizer_path = str(tmp_path / "new_tokenizer.model")
    monkeypatch.setattr("src.model.train.get_config", lambda: _TINY_TOKENIZER_CONFIG)

    config = {"tokenizer_path": tokenizer_path}
    tokenizer = get_or_build_tokenizer(config, _SAMPLE_HINDI_LINES)

    assert os.path.exists(tokenizer_path)
    assert os.path.exists(tokenizer_path.replace(".model", ".vocab"))
    assert tokenizer.get_piece_size() == 300


def test_get_or_build_tokenizer_loads_without_retraining_when_present(tiny_tokenizer_path, monkeypatch):
    def _raise_if_called(*args, **kwargs):
        raise AssertionError("train_sentencepiece_model should not be called when tokenizer_path already exists")

    monkeypatch.setattr("src.model.train.train_sentencepiece_model", _raise_if_called)

    config = {"tokenizer_path": tiny_tokenizer_path}
    tokenizer = get_or_build_tokenizer(config, _SAMPLE_HINDI_LINES)
    assert tokenizer.get_piece_size() == 300


def test_get_or_build_tokenizer_ignores_all_texts_when_model_exists(tiny_tokenizer_path):
    config = {"tokenizer_path": tiny_tokenizer_path}
    tokenizer = get_or_build_tokenizer(config, all_texts=["completely unrelated garbage text"])
    assert tokenizer.get_piece_size() == 300


# --- tokenize_and_concatenate ------------------------------------------------


def test_tokenize_and_concatenate_appends_eos_after_each_text(tiny_tokenizer_path):
    tokenizer = spm.SentencePieceProcessor(model_file=tiny_tokenizer_path)
    texts = ["यह एक हिन्दी वाक्य है।", "भारत एक विशाल देश है।"]
    expected = (
        tokenizer.encode(texts[0], out_type=int)
        + [tokenizer.eos_id()]
        + tokenizer.encode(texts[1], out_type=int)
        + [tokenizer.eos_id()]
    )
    assert tokenize_and_concatenate(texts, tokenizer) == expected


def test_tokenize_and_concatenate_empty_list_returns_empty_list(tiny_tokenizer_path):
    tokenizer = spm.SentencePieceProcessor(model_file=tiny_tokenizer_path)
    assert tokenize_and_concatenate([], tokenizer) == []


def test_tokenize_and_concatenate_single_text_ends_with_eos_id(tiny_tokenizer_path):
    tokenizer = spm.SentencePieceProcessor(model_file=tiny_tokenizer_path)
    result = tokenize_and_concatenate(["यह एक हिन्दी वाक्य है।"], tokenizer)
    assert result[-1] == tokenizer.eos_id()


def test_tokenize_and_concatenate_length_matches_sum_of_per_doc_lengths_plus_one(tiny_tokenizer_path):
    tokenizer = spm.SentencePieceProcessor(model_file=tiny_tokenizer_path)
    texts = ["यह एक हिन्दी वाक्य है।", "भारत एक विशाल देश है।", "मुझे किताबें पढ़ना पसंद है।"]
    result = tokenize_and_concatenate(texts, tokenizer)
    expected_len = sum(len(tokenizer.encode(t, out_type=int)) + 1 for t in texts)
    assert len(result) == expected_len


# --- get_ds -------------------------------------------------------------


def _base_config(tmp_path, tiny_tokenizer_path, seq_len=5, batch_size=2):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(train_path, _SAMPLE_HINDI_LINES[:20])
    _write_jsonl(val_path, _SAMPLE_HINDI_LINES[:5])
    _write_jsonl(test_path, _SAMPLE_HINDI_LINES[:5])
    return {
        "tokenizer_path": tiny_tokenizer_path,
        "dataset_paths": {"train": str(train_path), "val": str(val_path), "test": str(test_path)},
        "seq_len": seq_len,
        "batch_size": batch_size,
    }


def test_get_ds_returns_three_dataloaders_and_a_tokenizer(tmp_path, tiny_tokenizer_path):
    config = _base_config(tmp_path, tiny_tokenizer_path)
    train_dl, val_dl, test_dl, tokenizer = get_ds(config)

    from torch.utils.data import DataLoader

    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)
    assert isinstance(test_dl, DataLoader)
    assert isinstance(tokenizer, spm.SentencePieceProcessor)


def test_get_ds_batches_have_correct_seq_len_and_batch_size(tmp_path, tiny_tokenizer_path):
    seq_len, batch_size = 5, 2
    config = _base_config(tmp_path, tiny_tokenizer_path, seq_len=seq_len, batch_size=batch_size)
    train_dl, _, _, _ = get_ds(config)
    batch = next(iter(train_dl))
    assert batch["encoded_input"].shape == (batch_size, seq_len)
    assert batch["encoded_label"].shape == (batch_size, seq_len)


def test_get_ds_trains_tokenizer_via_build_training_sample_when_missing(tmp_path, monkeypatch):
    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_build_training_sample(*args, **kwargs)

    monkeypatch.setattr("src.model.train.build_training_sample", _spy)
    monkeypatch.setattr("src.model.train.get_config", lambda: _TINY_TOKENIZER_CONFIG)

    new_tokenizer_path = str(tmp_path / "fresh_tokenizer.model")
    config = _base_config(tmp_path, new_tokenizer_path)
    get_ds(config)

    assert len(calls) == 1
    assert calls[0][0][0] == config["dataset_paths"]["train"]
    assert os.path.exists(new_tokenizer_path)


def test_get_ds_skips_build_training_sample_when_tokenizer_exists(tmp_path, tiny_tokenizer_path, monkeypatch):
    def _raise_if_called(*args, **kwargs):
        raise AssertionError("build_training_sample should not be called when tokenizer_path already exists")

    monkeypatch.setattr("src.model.train.build_training_sample", _raise_if_called)

    config = _base_config(tmp_path, tiny_tokenizer_path)
    get_ds(config)  # should not raise


def test_get_ds_tokenizer_training_never_sees_val_or_test_texts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.model.train.get_config", lambda: _TINY_TOKENIZER_CONFIG)

    distinctive_word = "ज़ेब्राज़ेब्राज़ेब्रा"  # not present in any train fixture doc
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(train_path, _SAMPLE_HINDI_LINES[:20])
    _write_jsonl(val_path, [distinctive_word] * 5)
    _write_jsonl(test_path, [distinctive_word] * 5)

    new_tokenizer_path = str(tmp_path / "leakage_check.model")
    config = {
        "tokenizer_path": new_tokenizer_path,
        "dataset_paths": {"train": str(train_path), "val": str(val_path), "test": str(test_path)},
        "seq_len": 5,
        "batch_size": 2,
    }
    _, _, _, tokenizer = get_ds(config)

    pieces = tokenizer.encode(distinctive_word, out_type=int)
    assert len(pieces) > 1  # never learned as a single piece -- proxy for "val/test never leaked in"


# --- train_model --------------------------------------------------------


def _model_config(tmp_path, tiny_tokenizer_path):
    config = _base_config(tmp_path, tiny_tokenizer_path)
    config.update(
        {
            "d_model": 16,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 32,
            "dropout": 0.1,
            "vocab_size": None,
        }
    )
    return config


def test_train_model_sets_vocab_size_from_tokenizer_get_piece_size(tmp_path, tiny_tokenizer_path):
    config = _model_config(tmp_path, tiny_tokenizer_path)
    model, *_ = train_model(config)
    assert config["vocab_size"] == 300


def test_train_model_constructs_decoder_only_transformer_with_matching_vocab_size(tmp_path, tiny_tokenizer_path):
    config = _model_config(tmp_path, tiny_tokenizer_path)
    model, *_ = train_model(config)
    assert isinstance(model, DecoderOnlyTransformer)
    assert model.lm_head.out_features == config["vocab_size"]


def test_train_model_returns_expected_five_objects(tmp_path, tiny_tokenizer_path):
    from torch.utils.data import DataLoader

    config = _model_config(tmp_path, tiny_tokenizer_path)
    model, train_dl, val_dl, test_dl, tokenizer = train_model(config)
    assert isinstance(model, DecoderOnlyTransformer)
    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)
    assert isinstance(test_dl, DataLoader)
    assert isinstance(tokenizer, spm.SentencePieceProcessor)


def test_train_model_does_not_run_any_forward_or_training_step(tmp_path, tiny_tokenizer_path, monkeypatch):
    def _raise_if_called(*args, **kwargs):
        raise AssertionError("forward should not be called -- no training loop is implemented yet")

    monkeypatch.setattr(DecoderOnlyTransformer, "forward", _raise_if_called)

    config = _model_config(tmp_path, tiny_tokenizer_path)
    train_model(config)  # should not raise
