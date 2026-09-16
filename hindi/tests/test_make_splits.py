"""Tests for hindi/data/make_splits.py: deterministic, stratified doc-level splits."""
from hindi.data.make_splits import split_records


def _make_records(n, source):
    return [{"doc_id": f"{source}-{i}", "source": source, "text": f"text {i}"} for i in range(n)]


def test_split_records_is_deterministic_for_fixed_seed():
    records = _make_records(200, "wiki")
    split_a = split_records(records, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    split_b = split_records(records, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    assert [r["doc_id"] for r in split_a["train"]] == [r["doc_id"] for r in split_b["train"]]
    assert [r["doc_id"] for r in split_a["val"]] == [r["doc_id"] for r in split_b["val"]]
    assert [r["doc_id"] for r in split_a["test"]] == [r["doc_id"] for r in split_b["test"]]


def test_split_records_no_document_in_more_than_one_split():
    records = _make_records(300, "wiki")
    result = split_records(records, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=1)
    train_ids = {r["doc_id"] for r in result["train"]}
    val_ids = {r["doc_id"] for r in result["val"]}
    test_ids = {r["doc_id"] for r in result["test"]}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert train_ids | val_ids | test_ids == {r["doc_id"] for r in records}


def test_split_records_preserves_ratios_approximately():
    records = _make_records(1000, "wiki")
    result = split_records(records, ratios={"train": 0.98, "val": 0.01, "test": 0.01}, seed=7)
    assert abs(len(result["train"]) - 980) <= 5
    assert abs(len(result["val"]) - 10) <= 5
    assert abs(len(result["test"]) - 10) <= 5


def test_split_records_stratifies_by_source():
    records = _make_records(500, "wiki") + _make_records(500, "indiccorp")
    result = split_records(records, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=3)
    for split_name in ("train", "val", "test"):
        sources = [r["source"] for r in result[split_name]]
        wiki_fraction = sources.count("wiki") / len(sources)
        # With 50/50 input, each split should stay close to 50/50 too.
        assert 0.4 < wiki_fraction < 0.6
