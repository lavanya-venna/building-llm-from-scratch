"""Tests for src/utils/download_public.py.

`datasets.load_dataset` is never called in tests -- a fake streaming iterable
with the same shape (dicts with a text field) is injected instead, so these
tests run with no network access.
"""
import json

from src.utils.download_public import word_count, collect_source_to_budget


def _fake_stream(n, word_length=10):
    for i in range(n):
        yield {"text": " ".join(["शब्द"] * word_length), "id": str(i)}


def test_word_count_counts_whitespace_tokens():
    assert word_count("यह एक वाक्य है") == 4
    assert word_count("") == 0


def test_collect_source_to_budget_stops_once_budget_reached(tmp_path):
    output_path = tmp_path / "hi_wikipedia.jsonl"
    source_cfg = {"name": "hi_wikipedia", "text_field": "text"}

    stats = collect_source_to_budget(
        source_cfg,
        word_budget=100,
        stream=_fake_stream(n=1000, word_length=10),
        output_path=str(output_path),
    )

    assert stats["source"] == "hi_wikipedia"
    assert stats["word_count"] >= 100
    # Should stop shortly after crossing the budget, not consume the whole stream.
    assert stats["doc_count"] <= 15

    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == stats["doc_count"]
    first_record = json.loads(lines[0])
    assert first_record["source"] == "hi_wikipedia"
    assert first_record["doc_id"] == "hi_wikipedia-0"
    assert "text" in first_record


def test_collect_source_to_budget_is_idempotent_when_file_already_meets_budget(tmp_path):
    output_path = tmp_path / "hi_wikipedia.jsonl"
    source_cfg = {"name": "hi_wikipedia", "text_field": "text"}

    # First run writes real data.
    collect_source_to_budget(
        source_cfg, word_budget=50, stream=_fake_stream(200, 10), output_path=str(output_path)
    )
    existing_content = output_path.read_text(encoding="utf-8")

    # Second run: stream would raise if actually consumed, proving we skip it.
    def _raising_stream():
        raise AssertionError("stream should not be consumed when budget already met")
        yield  # pragma: no cover

    stats = collect_source_to_budget(
        source_cfg, word_budget=50, stream=_raising_stream(), output_path=str(output_path)
    )
    assert stats["skipped"] is True
    assert output_path.read_text(encoding="utf-8") == existing_content
