"""Deterministic, source-stratified document-level train/val/test splitting.

Splitting happens at the document level (not line level) so that near-identical
lines from the same document can't leak across splits, and is stratified by
`source` so val/test mirror the same public-corpus mix as train.
"""
import json
import os
import random
from collections import defaultdict

from hindi.data.download_public import load_data_config

_INPUT_PATH = os.path.join(os.path.dirname(__file__), "processed", "near_deduped.jsonl")
_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")


def _split_one_bucket(records, ratios, rng):
    """Shuffle and cut a single list of records into train/val/test slices."""
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def split_records(records, ratios, seed):
    """Split records into train/val/test, stratified by `source`, deterministically.

    Splitting per-source bucket (rather than shuffling the whole corpus at
    once) keeps each split's source mix representative of the full corpus --
    without this, a small source could end up entirely in one split by chance.

    Example: split_records(records, {"train": 0.98, "val": 0.01, "test": 0.01}, seed=42)
             -> {"train": [...], "val": [...], "test": [...]}, same output every
             call with the same `records`/`ratios`/`seed`.
    """
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)

    result = {"train": [], "val": [], "test": []}
    for source in sorted(by_source):
        rng = random.Random(f"{seed}:{source}")
        bucket_split = _split_one_bucket(by_source[source], ratios, rng)
        for split_name in result:
            result[split_name].extend(bucket_split[split_name])

    return result


def run_make_splits(config=None):
    """Read the final cleaned/deduped corpus, split it, and write splits/*.jsonl."""
    config = config or load_data_config()

    with open(_INPUT_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    result = split_records(records, ratios=config["splits"], seed=config["seed"])

    os.makedirs(_SPLITS_DIR, exist_ok=True)
    stats = {}
    for split_name, split_records_list in result.items():
        path = os.path.join(_SPLITS_DIR, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for record in split_records_list:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats[split_name] = {
            "doc_count": len(split_records_list),
            "word_count": sum(len(r["text"].split()) for r in split_records_list),
        }

    return stats


if __name__ == "__main__":
    print(json.dumps(run_make_splits(), ensure_ascii=False, indent=2))
