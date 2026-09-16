"""Driver script: run the full cleaning pipeline over the real raw corpus.

Not unit-tested itself (it's a thin orchestration layer over already-tested
functions in clean_normalize.py, lid_filter.py, and dedup.py); its job is
just to wire them together against the actual files on disk and report
progress, since this runs over millions of real documents.
"""
import glob
import json
import os
import sys
import time

from hindi.data.clean_normalize import clean_document
from hindi.data.dedup import exact_hash, shingle
from hindi.data.download_public import load_data_config
from hindi.data.lid_filter import keep_by_language_id, load_fasttext_lid_model
from datasketch import MinHash, MinHashLSH

_RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def clean_and_filter_source(input_path, output_path, cleaning_cfg, lid_model):
    """Apply clean_document + language-ID filtering to one raw source file.

    Streams line-by-line rather than loading the whole file into memory,
    since some raw sources here are multiple GB.
    """
    kept = 0
    dropped_clean = 0
    dropped_lid = 0
    total = 0

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            record = json.loads(line)
            cleaned = clean_document(
                record,
                min_words=cleaning_cfg["min_words_per_doc"],
                min_devanagari_ratio=cleaning_cfg["devanagari_ratio_min"],
            )
            if cleaned is None:
                dropped_clean += 1
                continue

            if not keep_by_language_id(
                cleaned["text"],
                lid_model,
                target_label=cleaning_cfg["lid_label"],
                min_confidence=cleaning_cfg["lid_confidence_min"],
            ):
                dropped_lid += 1
                continue

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1

            if total % 500000 == 0:
                _log(f"{os.path.basename(input_path)}: {total} read, {kept} kept so far")

    return {
        "source": cleaning_cfg.get("source_name", os.path.basename(input_path)),
        "total": total,
        "kept": kept,
        "dropped_clean": dropped_clean,
        "dropped_lid": dropped_lid,
    }


def run_clean_and_filter_stage(config):
    """Stage 1: clean_document + LID filter every raw source file."""
    os.makedirs(_PROCESSED_DIR, exist_ok=True)
    lid_model = load_fasttext_lid_model()

    stats = []
    for raw_path in sorted(glob.glob(os.path.join(_RAW_DIR, "*.jsonl"))):
        source_name = os.path.splitext(os.path.basename(raw_path))[0]
        output_path = os.path.join(_PROCESSED_DIR, f"{source_name}.stage1.jsonl")
        _log(f"cleaning {source_name}...")
        cfg = dict(config["cleaning"])
        cfg["source_name"] = source_name
        result = clean_and_filter_source(raw_path, output_path, cfg, lid_model)
        _log(f"{source_name}: {result}")
        stats.append(result)

    return stats


def run_dedup_stage():
    """Stage 2: exact dedup across all stage-1 files, streaming to control memory."""
    stage1_paths = sorted(glob.glob(os.path.join(_PROCESSED_DIR, "*.stage1.jsonl")))
    output_path = os.path.join(_PROCESSED_DIR, "deduped.jsonl")

    seen_hashes = set()
    total = 0
    kept = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for path in stage1_paths:
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    total += 1
                    record = json.loads(line)
                    h = exact_hash(record["text"])
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    fout.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
                    if total % 1000000 == 0:
                        _log(f"dedup: {total} read, {kept} kept so far")

    _log(f"exact dedup done: {total} total, {kept} kept, {total - kept} removed")
    return {"total": total, "kept": kept}


def run_near_dedup_stage(dedup_cfg):
    """Stage 3: MinHash/LSH near-dedup over the exact-deduped corpus, streaming.

    Streams input rather than loading the whole 5GB+ corpus into memory --
    only the LSH index (one MinHash signature per surviving document) and
    the current line are held at once.
    """
    input_path = os.path.join(_PROCESSED_DIR, "deduped.jsonl")
    output_path = os.path.join(_PROCESSED_DIR, "near_deduped.jsonl")

    lsh = MinHashLSH(threshold=dedup_cfg["jaccard_threshold"], num_perm=dedup_cfg["minhash_num_perm"])
    total = 0
    kept = 0
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            record = json.loads(line)
            m = MinHash(num_perm=dedup_cfg["minhash_num_perm"])
            for s in shingle(record["text"], size=dedup_cfg["shingle_size"]):
                m.update(s.encode("utf-8"))

            if lsh.query(m):
                continue

            lsh.insert(record["doc_id"], m)
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1

            if total % 200000 == 0:
                _log(f"near-dedup: {total} read, {kept} kept so far")

    _log(f"near dedup done: {total} total, {kept} kept, {total - kept} removed")
    return {"total": total, "kept": kept}


if __name__ == "__main__":
    config = load_data_config()
    clean_stats = run_clean_and_filter_stage(config)
    dedup_stats = run_dedup_stage()
    near_dedup_stats = run_near_dedup_stage(config["dedup"])
    print(
        json.dumps(
            {"clean_stats": clean_stats, "dedup_stats": dedup_stats, "near_dedup_stats": near_dedup_stats},
            ensure_ascii=False,
            indent=2,
        )
    )
