"""Tests for the dataset-stats-report functions in src/data/data_preprocessor.py
(the corpus/cleaning half of the old compute_stats.py; tokenizer metrics live
in src/evaluation/tokenizer_eval.py and are tested in test_tokenizer_eval.py).
"""
from src.data.data_preprocessor import (
    count_source_stats,
    dedup_removal_rate,
    split_size_report,
    pipeline_stage_table,
    render_dataset_stats_report,
)


def test_count_source_stats_counts_docs_and_words_per_source():
    records = [
        {"text": "एक दो तीन चार पांच", "source": "wiki"},
        {"text": "छह सात आठ", "source": "wiki"},
        {"text": "एक दो", "source": "indiccorp"},
    ]
    stats = count_source_stats(records)
    assert stats["wiki"]["doc_count"] == 2
    assert stats["wiki"]["word_count"] == 8
    assert stats["indiccorp"]["doc_count"] == 1
    assert stats["indiccorp"]["word_count"] == 2


def test_dedup_removal_rate_computes_percentage_removed():
    rate = dedup_removal_rate(raw_doc_count=100, deduped_doc_count=80)
    assert rate == 20.0


def test_dedup_removal_rate_handles_zero_raw_count():
    assert dedup_removal_rate(raw_doc_count=0, deduped_doc_count=0) == 0.0


def test_split_size_report_summarizes_docs_and_words_per_split():
    splits = {
        "train": [{"text": "एक दो तीन", "source": "wiki"}, {"text": "चार पांच", "source": "indiccorp"}],
        "val": [{"text": "छह सात", "source": "wiki"}],
        "test": [{"text": "आठ", "source": "wiki"}],
    }
    report = split_size_report(splits)
    assert report["train"]["doc_count"] == 2
    assert report["train"]["word_count"] == 5
    assert report["val"]["doc_count"] == 1
    assert report["test"]["word_count"] == 1


def test_pipeline_stage_table_builds_per_source_rows():
    collection_stats = {
        "raw_doc_counts": {"wiki": 100},
        "clean_stage": {"wiki": {"total": 100, "kept": 80, "dropped_clean": 15, "dropped_lid": 5}},
    }
    rows = pipeline_stage_table(collection_stats)
    assert rows == [{"source": "wiki", "raw": 100, "kept": 80, "dropped_clean": 15, "dropped_lid": 5}]


def test_render_dataset_stats_report_produces_markdown_with_key_sections():
    report_data = {
        "source_stats": {"wiki": {"doc_count": 10, "word_count": 1000}},
        "dedup_removal_pct": 12.5,
        "split_sizes": {
            "train": {"doc_count": 8, "word_count": 800},
            "val": {"doc_count": 1, "word_count": 100},
            "test": {"doc_count": 1, "word_count": 100},
        },
    }
    markdown = render_dataset_stats_report(report_data)
    assert "# Hindi Dataset Statistics" in markdown
    assert "wiki" in markdown
    assert "12.5" in markdown
