"""Tests for src/data/dataset.py.

Uses a tiny real SentencePiece model trained on a small repeated-Hindi-lines
fixture corpus (same pattern as tests/test_tokenizer.py) instead of mocking
the tokenizer -- SentencePiece training/decoding is cheap at this scale, and
a real tokenizer means decode() calls are exercised for real.
"""
import pytest
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from src.data.dataset import HindiDataset
from src.tokenizer.tokenizer_test import train_sentencepiece_model

_SAMPLE_HINDI_LINES = [
    "यह एक हिन्दी वाक्य है।",
    "भारत एक विशाल देश है।",
    "मुझे किताबें पढ़ना पसंद है।",
    "आज मौसम बहुत अच्छा है।",
    "वह स्कूल जा रहा है।",
] * 50  # repeat so SentencePiece has enough signal to train on


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dataset_tokenizer")

    model_prefix = str(tmp_path / "tiny_unigram_300")
    train_sentencepiece_model(
        sentences=_SAMPLE_HINDI_LINES,
        model_prefix=model_prefix,
        vocab_size=300,
        model_type="unigram",
        character_coverage=0.9995,
        byte_fallback=True,
    )
    return spm.SentencePieceProcessor(model_file=model_prefix + ".model")


@pytest.fixture
def token_ids():
    # Sequential ids within [0, 300) -- valid for the tiny tokenizer's vocab
    # regardless of whether they correspond to "real" language, which keeps
    # slicing-math tests deterministic and easy to reason about.
    return list(range(250))


def test_len_matches_floor_division_formula(token_ids, tokenizer):
    seq_len = 10
    dataset = HindiDataset(token_ids, seq_len, tokenizer)
    assert len(dataset) == (len(token_ids) - 1) // seq_len


def test_len_drops_incomplete_tail_chunk(tokenizer):
    seq_len = 10
    token_ids = list(range(35))  # (35 - 1) // 10 == 3, tail of 5 ids dropped
    dataset = HindiDataset(token_ids, seq_len, tokenizer)
    assert len(dataset) == 3
    last_start = (len(dataset) - 1) * seq_len
    assert last_start + seq_len <= len(token_ids)


def test_len_zero_when_token_ids_shorter_than_seq_len_plus_one(tokenizer):
    dataset = HindiDataset(list(range(5)), seq_len=10, tokenizer=tokenizer)
    assert len(dataset) == 0


def test_getitem_returns_expected_dict_keys(token_ids, tokenizer):
    dataset = HindiDataset(token_ids, seq_len=10, tokenizer=tokenizer)
    item = dataset[0]
    assert set(item.keys()) == {"encoded_input", "encoded_label", "raw_input", "raw_label"}


def test_getitem_shapes_and_dtype(token_ids, tokenizer):
    seq_len = 10
    dataset = HindiDataset(token_ids, seq_len, tokenizer)
    item = dataset[0]
    assert item["encoded_input"].shape == (seq_len,)
    assert item["encoded_label"].shape == (seq_len,)
    assert item["encoded_input"].dtype == torch.long
    assert item["encoded_label"].dtype == torch.long


def test_getitem_input_matches_slice_at_idx_times_seq_len(token_ids, tokenizer):
    seq_len = 10
    dataset = HindiDataset(token_ids, seq_len, tokenizer)
    for idx in (0, 1, len(dataset) - 1):
        start = idx * seq_len
        item = dataset[idx]
        assert item["encoded_input"].tolist() == token_ids[start : start + seq_len]


def test_getitem_label_is_input_shifted_by_one(token_ids, tokenizer):
    seq_len = 10
    dataset = HindiDataset(token_ids, seq_len, tokenizer)
    item = dataset[0]
    input_ids = item["encoded_input"].tolist()
    label_ids = item["encoded_label"].tolist()
    assert label_ids[:-1] == input_ids[1:]
    assert label_ids[-1] == token_ids[seq_len]  # one past the input chunk


def test_getitem_consecutive_indices_are_non_overlapping(token_ids, tokenizer):
    seq_len = 10
    dataset = HindiDataset(token_ids, seq_len, tokenizer)
    first = dataset[0]["encoded_input"].tolist()
    second = dataset[1]["encoded_input"].tolist()
    assert first == token_ids[0:seq_len]
    assert second == token_ids[seq_len : 2 * seq_len]


def test_getitem_decode_roundtrip_matches_tokenizer_decode(token_ids, tokenizer):
    seq_len = 10
    dataset = HindiDataset(token_ids, seq_len, tokenizer)
    item = dataset[0]
    assert item["raw_input"] == tokenizer.decode(token_ids[0:seq_len])
    assert item["raw_label"] == tokenizer.decode(token_ids[1 : seq_len + 1])


def test_getitem_raw_fields_are_strings(token_ids, tokenizer):
    dataset = HindiDataset(token_ids, seq_len=10, tokenizer=tokenizer)
    item = dataset[0]
    assert isinstance(item["raw_input"], str)
    assert isinstance(item["raw_label"], str)


def test_getitem_accepts_plain_list_token_ids(tokenizer):
    dataset = HindiDataset(list(range(20)), seq_len=5, tokenizer=tokenizer)
    item = dataset[0]
    assert item["encoded_input"].tolist() == [0, 1, 2, 3, 4]


def test_dataloader_batches_stack_encoded_and_collect_raw_as_list(token_ids, tokenizer):
    seq_len = 10
    batch_size = 4
    dataset = HindiDataset(token_ids, seq_len, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    assert batch["encoded_input"].shape == (batch_size, seq_len)
    assert batch["encoded_label"].shape == (batch_size, seq_len)
    assert isinstance(batch["raw_input"], list)
    assert len(batch["raw_input"]) == batch_size
