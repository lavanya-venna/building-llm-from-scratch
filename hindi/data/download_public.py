"""Stream and sample public Hindi corpora from Hugging Face down to a token budget.

Datasets are opened with `streaming=True` so we never materialize a full
source to disk before sampling -- we pull only as many documents as needed to
hit each source's configured word-count budget (see hindi/configs/data_config.yaml).
"""
import json
import os

import yaml
from datasets import load_dataset

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "data_config.yaml")
_RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def load_data_config(config_path=None):
    """Load hindi/configs/data_config.yaml.

    Centralizing this avoids every script re-implementing its own YAML-loading
    boilerplate and guarantees they all read the same seed/budgets.

    Example: load_data_config()["target_total_tokens"] -> 500000000
    """
    path = config_path or _CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def word_count(text):
    """Count whitespace-separated words as a cheap proxy for token count.

    The tokenizer doesn't exist yet at collection time (it's trained on this
    very corpus), so we can't measure exact tokens up front. Word count is a
    fast, tokenizer-free stand-in used only to decide when to stop pulling
    from a source; exact counts are re-measured later in eval_tokenizer.py.

    Example: word_count("यह एक वाक्य है") -> 4
    """
    if not text:
        return 0
    return len(text.split())


def open_source_stream(source_cfg):
    """Open a streaming HF dataset iterator for one configured source.

    `streaming=True` means Hugging Face fetches shards on demand instead of
    downloading the entire dataset up front -- essential here since some
    sources (e.g. mC4) are far larger than what we'll actually sample.

    Example: open_source_stream({"hf_path": "wikimedia/wikipedia",
                                  "hf_config": "20231101.hi", "split": "train"})
             -> an iterable of dataset records
    """
    kwargs = {"split": source_cfg.get("split", "train"), "streaming": True}
    if source_cfg.get("hf_data_dir"):
        kwargs["data_dir"] = source_cfg["hf_data_dir"]
    return load_dataset(source_cfg["hf_path"], source_cfg.get("hf_config"), **kwargs)


def collect_source_to_budget(source_cfg, word_budget, stream, output_path):
    """Pull documents from `stream` until `word_budget` whitespace-words is reached.

    Skips entirely (without touching `stream`) if `output_path` already has
    enough content, so re-running the collection script after an interruption
    doesn't re-fetch data or duplicate it.

    Example: collect_source_to_budget({"name": "hi_wikipedia", "text_field": "text"},
                                       word_budget=120_000_000, stream=..., output_path=...)
             -> {"source": "hi_wikipedia", "doc_count": N, "word_count": M, "skipped": False}
    """
    text_field = source_cfg["text_field"]
    source_name = source_cfg["name"]

    if os.path.exists(output_path):
        existing_words = 0
        doc_count = 0
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                existing_words += word_count(json.loads(line)["text"])
                doc_count += 1
        if existing_words >= word_budget:
            return {
                "source": source_name,
                "doc_count": doc_count,
                "word_count": existing_words,
                "skipped": True,
            }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_words = 0
    doc_count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for record in stream:
            text = record[text_field]
            out.write(
                json.dumps(
                    {"text": text, "source": source_name, "doc_id": f"{source_name}-{doc_count}"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            total_words += word_count(text)
            doc_count += 1
            if total_words >= word_budget:
                break

    return {
        "source": source_name,
        "doc_count": doc_count,
        "word_count": total_words,
        "skipped": False,
    }


def collect_all_sources(config=None):
    """Run collect_source_to_budget for every configured source, stopping early
    once the overall proxy-estimated token target is met.

    Example: collect_all_sources() -> [stats_dict_per_source, ...], and writes
    hindi/data/raw/<source_name>.jsonl for each source actually pulled from.
    """
    config = config or load_data_config()
    oversample_factor = config["oversample_factor"]
    target_words = config["target_total_tokens"] / config["proxy_tokens_per_word"]

    results = []
    running_words = 0
    for source_cfg in config["sources"]:
        if running_words >= target_words:
            break
        remaining_words = target_words - running_words
        budget = min(source_cfg["word_budget"], remaining_words) * oversample_factor

        output_path = os.path.join(_RAW_DIR, f"{source_cfg['name']}.jsonl")
        stream = open_source_stream(source_cfg)
        stats = collect_source_to_budget(source_cfg, budget, stream, output_path)
        results.append(stats)
        running_words += stats["word_count"]

    return results


if __name__ == "__main__":
    for stats in collect_all_sources():
        print(stats)
