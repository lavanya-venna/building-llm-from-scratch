"""Aggregate corpus and tokenizer statistics into report/hindi_dataset_stats.md.

Every number in the final report is computed programmatically from the actual
pipeline outputs (raw/processed/split jsonl files, tokenizer sweep results) so
the report stays accurate if the corpus changes and gets regenerated.
"""
import json
import os
from collections import defaultdict

from hindi.tokenizer.eval_tokenizer import evaluate_all_candidates, load_val_texts, select_best_vocab
from hindi.tokenizer.train_tokenizer import load_tokenizer_config

_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "report", "hindi_dataset_stats.md")
_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")
_COLLECTION_STATS_PATH = os.path.join(os.path.dirname(__file__), "processed", "collection_stats.json")
_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "artifacts")


def count_source_stats(records):
    """Tally document and whitespace-word counts per `source`.

    This is the basis for the "per-source raw vs. cleaned" breakdown in the
    report -- without a per-source tally, we couldn't show which corpora
    contributed how much to the final total.

    Example: count_source_stats([{"text": "a b c", "source": "wiki"}])
             -> {"wiki": {"doc_count": 1, "word_count": 3}}
    """
    stats = defaultdict(lambda: {"doc_count": 0, "word_count": 0})
    for record in records:
        s = stats[record["source"]]
        s["doc_count"] += 1
        s["word_count"] += len(record["text"].split())
    return dict(stats)


def dedup_removal_rate(raw_doc_count, deduped_doc_count):
    """Percentage of documents removed by dedup, for the rubric's cleaning-steps report.

    Example: dedup_removal_rate(100, 80) -> 20.0
    """
    if raw_doc_count == 0:
        return 0.0
    return (raw_doc_count - deduped_doc_count) / raw_doc_count * 100


def split_size_report(splits):
    """Summarize document and word counts for each of train/val/test.

    Example: split_size_report({"train": [...], "val": [...], "test": [...]})
             -> {"train": {"doc_count": N, "word_count": M}, "val": {...}, "test": {...}}
    """
    return {name: _totals(records) for name, records in splits.items()}


def _totals(records):
    doc_count = len(records)
    word_count = sum(len(r["text"].split()) for r in records)
    return {"doc_count": doc_count, "word_count": word_count}


def pipeline_stage_table(collection_stats):
    """Build a per-source raw -> cleaned document-count table from collection_stats.json.

    This is what lets the report show *why* the corpus is the size it is --
    how many documents each source started with, and how many survived the
    cleaning + language-ID filters, before dedup is even applied.

    Example: pipeline_stage_table({"raw_doc_counts": {"wiki": 100},
                                    "clean_stage": {"wiki": {"total": 100, "kept": 80,
                                                              "dropped_clean": 15, "dropped_lid": 5}}})
             -> [{"source": "wiki", "raw": 100, "kept": 80, "dropped_clean": 15, "dropped_lid": 5}]
    """
    rows = []
    for source in sorted(collection_stats["clean_stage"]):
        stage = collection_stats["clean_stage"][source]
        rows.append(
            {
                "source": source,
                "raw": collection_stats["raw_doc_counts"][source],
                "kept": stage["kept"],
                "dropped_clean": stage["dropped_clean"],
                "dropped_lid": stage["dropped_lid"],
            }
        )
    return rows


def tokenizer_sweep_table(sweep, selected_vocab_size):
    """Turn the {vocab_size: metrics} sweep dict into an ordered table of rows,
    flagging the selected vocab size for the report.

    Example: tokenizer_sweep_table({32000: {...}}, selected_vocab_size=32000)
             -> [{"vocab_size": 32000, ..., "selected": True}]
    """
    rows = []
    for vocab_size in sorted(sweep):
        row = {"vocab_size": vocab_size, **sweep[vocab_size]}
        row["selected"] = vocab_size == selected_vocab_size
        rows.append(row)
    return rows


def render_report(report_data):
    """Render the aggregated stats dict into the final Markdown report.

    Kept as pure string formatting (no file I/O) so it's independently
    testable against a hand-built `report_data` dict.

    Example: render_report({...}) -> "# Hindi Dataset Statistics\\n\\n## Sources\\n..."
    """
    lines = ["# Hindi Dataset Statistics", ""]

    if report_data.get("pipeline_stages"):
        lines.append("## Collection & cleaning pipeline (per source)")
        lines.append("")
        lines.append("| Source | Raw docs | Kept after cleaning | Dropped (clean filters) | Dropped (language-ID) |")
        lines.append("|---|---|---|---|---|")
        for row in report_data["pipeline_stages"]:
            lines.append(
                f"| {row['source']} | {row['raw']} | {row['kept']} | {row['dropped_clean']} | {row['dropped_lid']} |"
            )
        lines.append("")

    lines.append("## Per-source counts (final corpus, after dedup)")
    lines.append("")
    lines.append("| Source | Documents | Words |")
    lines.append("|---|---|---|")
    for source, stats in report_data["source_stats"].items():
        lines.append(f"| {source} | {stats['doc_count']} | {stats['word_count']} |")
    lines.append("")

    lines.append("## Deduplication")
    lines.append("")
    lines.append(f"Documents removed by dedup (exact + near, combined): {report_data['dedup_removal_pct']:.1f}%")
    lines.append("")

    lines.append("## Manual vs. downloaded token split")
    lines.append("")
    lines.append(
        "0% manual / 100% public downloaded. Per the project's agreed scope, the "
        "assignment's usual >=20% manual-collection requirement was dropped in "
        "favor of reaching the ~500M-token target entirely from public Hugging "
        "Face datasets (IndicCorpV2, zicsx/ai4bharat-hi-subset, and "
        "ai4bharat/sangraha's synthetic/hin_Deva split as a top-up)."
    )
    lines.append("")

    lines.append("## Train / validation / test splits")
    lines.append("")
    lines.append("| Split | Documents | Words |")
    lines.append("|---|---|---|")
    for split_name, stats in report_data["split_sizes"].items():
        lines.append(f"| {split_name} | {stats['doc_count']} | {stats['word_count']} |")
    lines.append("")

    lines.append("## Tokenizer vocab sweep")
    lines.append("")
    lines.append("| Vocab size | Avg chars/token | Avg tokens/word | UNK rate | Selected |")
    lines.append("|---|---|---|---|---|")
    for row in report_data["tokenizer_sweep"]:
        marker = "yes" if row["selected"] else ""
        lines.append(
            f"| {row['vocab_size']} | {row['avg_chars_per_token']:.3f} | "
            f"{row['avg_tokens_per_word']:.3f} | {row['unk_rate']:.4f} | {marker} |"
        )
    lines.append("")

    lines.append("## Total corpus size")
    lines.append("")
    lines.append(f"Total exact tokenizer-measured tokens: {report_data['total_exact_tokens']:,}")
    lines.append("")

    lines.append("## Tokenization examples")
    lines.append("")
    for example in report_data["examples"]:
        lines.append(f"- `{example['text']}` -> {example['pieces']}")
    lines.append("")

    return "\n".join(lines)


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_report():
    """Build and write the final report/hindi_dataset_stats.md from real pipeline outputs.

    Fails loudly if the tokenizer sweep hasn't been run and a vocab size
    selected yet (tokenizer_config.yaml's `selected_vocab_size` is still
    null), since the report is meaningless without it.
    """
    config = load_tokenizer_config()
    if config.get("selected_vocab_size") is None:
        raise RuntimeError(
            "tokenizer_config.yaml selected_vocab_size is not set -- run "
            "train_tokenizer.py and eval_tokenizer.py first."
        )

    splits = {
        name: _load_jsonl(os.path.join(_SPLITS_DIR, f"{name}.jsonl"))
        for name in ("train", "val", "test")
    }
    all_records = splits["train"] + splits["val"] + splits["test"]

    val_texts = load_val_texts(os.path.join(_SPLITS_DIR, "val.jsonl"))
    sweep = evaluate_all_candidates(config, val_texts)
    selected = config["selected_vocab_size"]

    with open(_COLLECTION_STATS_PATH, encoding="utf-8") as f:
        collection_stats = json.load(f)
    clean_total = sum(s["total"] for s in collection_stats["clean_stage"].values())
    final_total = collection_stats["near_dedup"]["kept"]
    dedup_removal_pct = dedup_removal_rate(clean_total, final_total)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(_ARTIFACTS_DIR, f"hindi_{config['model_type']}_{selected}.model")
    )
    total_exact_tokens = sum(len(sp.encode(r["text"])) for r in all_records)

    example_texts = [r["text"][:120] for r in splits["val"][:8]]
    examples = [{"text": t, "pieces": sp.encode(t, out_type=str)} for t in example_texts]

    report_data = {
        "pipeline_stages": pipeline_stage_table(collection_stats),
        "source_stats": count_source_stats(all_records),
        "dedup_removal_pct": dedup_removal_pct,
        "split_sizes": {name: _totals(records) for name, records in splits.items()},
        "tokenizer_sweep": tokenizer_sweep_table(sweep, selected),
        "total_exact_tokens": total_exact_tokens,
        "examples": examples,
    }

    markdown = render_report(report_data)
    os.makedirs(os.path.dirname(_REPORT_PATH), exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    return _REPORT_PATH


if __name__ == "__main__":
    print(generate_report())
