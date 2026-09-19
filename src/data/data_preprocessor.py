"""Cleaning, normalization, language-ID filtering, deduplication orchestration,
and dataset-stats reporting for the Hindi corpus.

This merges what used to be clean_normalize.py, lid_filter.py, and
run_pipeline.py's orchestration into one module, plus the corpus/cleaning
half of the old compute_stats.py (the tokenizer-metrics half lives in
src/evaluation/tokenizer_eval.py instead, writing a separate report).
"""
import glob
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict

import fasttext
import ftfy
import unicodedata
import re
from datasketch import MinHash, MinHashLSH
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

from src.utils.dedup import exact_hash, shingle

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_RAW_DIR = os.path.join(_DATA_DIR, "raw")
_PROCESSED_DIR = os.path.join(_DATA_DIR, "processed")
_SPLITS_DIR = os.path.join(_DATA_DIR, "splits")
_ARTIFACTS_DIR = os.path.join(_DATA_DIR, "artifacts")
_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "report", "dataset_stats.md")
_COLLECTION_STATS_PATH = os.path.join(_PROCESSED_DIR, "collection_stats.json")

_LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_DEFAULT_LID_MODEL_PATH = os.path.join(_ARTIFACTS_DIR, "lid.176.bin")

_INDIC_NORMALIZER = IndicNormalizerFactory().get_normalizer("hi")

_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")
_ALLOWED_NON_DEVANAGARI = re.compile(r"[\s\d.,!?;:()\-।॥]")

_REF_TAG_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WIKI_BRACKET_RE = re.compile(r"\[\[|\]\]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PAREN_RE = re.compile(r" ?\([^()]*\) ?")
_TRAILING_PUNCTUATION = ".,!?;:।॥"


# ---------------------------------------------------------------------------
# Cleaning & normalization (formerly clean_normalize.py)
# ---------------------------------------------------------------------------


def fix_mojibake(text):
    """Repair mangled encodings (e.g. UTF-8 text mis-decoded as Latin-1).

    Web-scraped and OCR'd text frequently arrives double-encoded. Without this
    step, a string like "नमस्ते" mis-decoded upstream turns into garbage bytes
    (e.g. "à¤¨à¤®à¤¸à¥\x8dà¤¤à¥\x87") that survive every later filter as "valid" text
    but are unusable for training.

    Example: fix_mojibake("à¤¨à¤®à¤¸à¥\x8dà¤¤à¥\x87") -> "नमस्ते"
    """
    return ftfy.fix_text(text)


def normalize_unicode(text):
    """Canonicalize Unicode form and Devanagari-specific glyph composition.

    Plain NFC normalization alone is not enough for Devanagari: the same visual
    word can be represented with different underlying codepoint orderings for
    nukta/matra marks. Without this, two byte-different-but-visually-identical
    strings would be treated as different tokens by the tokenizer, hurting
    vocabulary efficiency and inflating "duplicate" counts inconsistently.

    Example: normalize_unicode("क" + "़") -> "क़" (NFC-composed qa)
    """
    nfc = unicodedata.normalize("NFC", text)
    return _INDIC_NORMALIZER.normalize(nfc)


def strip_markup(text):
    """Remove HTML tags and MediaWiki markup, keeping only prose content.

    Source text (Wikipedia dumps, web-crawled HTML) is full of structural
    noise that isn't natural language: citation refs, infobox templates, link
    brackets. Left in, these add non-linguistic tokens that waste tokenizer
    vocabulary and confuse a language model trained on the result.

    Example: strip_markup("<p>यह [[लेख]] है {{इन्फो}}</p>") -> "यह लेख है "
    (the <p> tags are dropped but their text kept, the [[ ]] link brackets are
    dropped but the link text kept, and the {{ }} template is dropped
    entirely including its content since templates are typically metadata,
    not prose.)
    """
    text = _REF_TAG_RE.sub(" ", text)
    text = _TEMPLATE_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _WIKI_BRACKET_RE.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def strip_urls(text):
    """Remove raw URLs (web-crawled text is full of embedded links/image URLs).

    URLs are not natural-language content -- left in, they burn tokenizer
    vocabulary on domain names/paths and add noise a language model shouldn't
    learn to imitate. Common in web-crawled sources like mC4.

    Example: strip_urls("यह लेख है https://example.com/page और आगे पढ़ें")
             -> "यह लेख है  और आगे पढ़ें"
    """
    return _URL_RE.sub(" ", text)


def strip_non_devanagari_parens(text):
    """Remove parenthetical asides that contain no Devanagari script.

    Hindi news text frequently glosses a transliterated proper noun or
    acronym with its English original in parentheses (e.g. "बिहार (Bihar)
    के मुजफ्फरपुर (Muzaffarpur) में"), or carries a bare numeric aside (a
    date, a list number). None of that is useful signal for a Hindi tokenizer
    and it burns vocabulary on noise. A parenthetical is kept only if it has
    at least one Devanagari character -- covers genuine Hindi asides (e.g.
    wire-service tags like "(भाषा)") and mixed asides where an English
    acronym sits inside an otherwise-Hindi remark, without ever dropping a
    parenthetical solely because it's English.

    Example: strip_non_devanagari_parens("भारत निर्वाचन आयोग (ECI) को सूचना दी")
             -> "भारत निर्वाचन आयोग को सूचना दी"
    Example: strip_non_devanagari_parens("सिडनी, (भाषा)। ग्लेन मैक्सवेल ने कहा")
             -> "सिडनी, (भाषा)। ग्लेन मैक्सवेल ने कहा"
    Example: strip_non_devanagari_parens("आबकारी अधिनियम की धारा 34(1) व (2) के अपराध")
             -> "आबकारी अधिनियम की धारा 34 व के अपराध"
    """

    def _replace(match):
        span = match.group(0)
        content = span.strip()[1:-1]
        if _DEVANAGARI_RANGE.search(content):
            return span  # has Devanagari -- keep exactly as-is, spaces included
        if span[0] != " " and span[-1] != " ":
            return ""  # glued to neighbors on both sides -- nothing to reconcile
        next_char = match.string[match.end() : match.end() + 1]
        return "" if next_char and next_char in _TRAILING_PUNCTUATION else " "

    return _PAREN_RE.sub(_replace, text).strip()


def devanagari_ratio(text):
    """Return the fraction of characters that are Devanagari script.

    This is the cheapest, most reliable first-pass signal for "is this
    actually Hindi text" before running a heavier language-ID model — a page
    that's mostly English with a few Hindi words (or vice versa) should not
    be counted as Hindi training data.

    Example: devanagari_ratio("यह हिन्दी है") -> ~0.85 (high, keep)
             devanagari_ratio("This is English") -> ~0.0 (low, drop)
    """
    if not text:
        return 0.0
    devanagari_chars = len(_DEVANAGARI_RANGE.findall(text))
    countable_chars = len(text) - len(re.findall(r"\s", text))
    if countable_chars == 0:
        return 0.0
    return devanagari_chars / countable_chars


def keep_by_devanagari_ratio(text, min_ratio):
    """Decide whether a document has enough Devanagari script to keep.

    Wraps `devanagari_ratio` with the corpus's configured threshold
    (see config/data_config.yaml: cleaning.devanagari_ratio_min) so the
    cutoff lives in one place instead of being re-implemented at each call site.

    Example: keep_by_devanagari_ratio("यह हिन्दी वाक्य है", 0.6) -> True
             keep_by_devanagari_ratio("This is English", 0.6) -> False
    """
    return devanagari_ratio(text) >= min_ratio


def keep_by_length(text, min_words):
    """Decide whether a document has enough content to be useful training data.

    Very short documents (nav-menu fragments, stub pages, disclaimers) add
    noise without much signal, and inflate document counts without
    contributing meaningful token volume.

    Example: keep_by_length("यह छोटा है।", min_words=20) -> False
    """
    return len(text.split()) >= min_words


def clean_document(record, min_words, min_devanagari_ratio):
    """Run the full per-document cleaning pipeline.

    Applies mojibake repair, markup stripping, URL stripping, non-Devanagari
    parenthetical stripping, then Unicode normalization, then the
    Devanagari-ratio and length filters, in that order (markup must be
    stripped before length/ratio checks so tag noise doesn't skew them).
    Returns a new record with `pipeline_stage="clean"` set, or None if the
    document should be dropped.

    Example: clean_document({"text": "<p>...</p>", "source": "wiki", "doc_id": "1"},
                             min_words=20, min_devanagari_ratio=0.6)
             -> {"text": "...", "source": "wiki", "doc_id": "1", "pipeline_stage": "clean"}
             or None if the cleaned text fails the length/script filters.
    """
    text = fix_mojibake(record["text"])
    text = strip_markup(text)
    text = strip_urls(text)
    text = strip_non_devanagari_parens(text)
    text = normalize_unicode(text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if not keep_by_length(text, min_words):
        return None
    if not keep_by_devanagari_ratio(text, min_devanagari_ratio):
        return None

    cleaned = dict(record)
    cleaned["text"] = text
    cleaned["pipeline_stage"] = "clean"
    return cleaned


# ---------------------------------------------------------------------------
# Language-ID filtering (formerly lid_filter.py)
# ---------------------------------------------------------------------------


def load_fasttext_lid_model(model_path=None):
    """Load (downloading if needed) the fasttext language-ID model.

    Kept separate from the decision logic below so tests can exercise
    `predict_language`/`keep_by_language_id` with a fake model and never need
    network access or the ~130MB binary.

    Note: lid.176 is a pretrained *classifier* used only to filter which
    documents enter the corpus. It is not part of the LLM or its tokenizer,
    so it does not violate the "no pretrained model/tokenizer" constraint on
    the language model itself -- it plays the same role a hand-written
    heuristic filter would, just with higher accuracy.

    Example: load_fasttext_lid_model() -> fasttext model object usable with
    predict_language(text, model)
    """
    path = model_path or _DEFAULT_LID_MODEL_PATH
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(_LID_MODEL_URL, path)
    return fasttext.load_model(path)


def predict_language(text, model):
    """Return (label, confidence) for the most likely language of `text`.

    fasttext's predict() raises on embedded newlines, so this wrapper collapses
    them first -- callers shouldn't have to know that quirk.

    Example: predict_language("यह हिन्दी वाक्य है", model) -> ("__label__hi", 0.97)
    """
    single_line = text.replace("\n", " ")
    labels, confidences = model.predict(single_line)
    return labels[0], float(confidences[0])


def keep_by_language_id(text, model, target_label, min_confidence):
    """Decide whether a document is confidently in the target language.

    Used as a secondary check after the cheaper Devanagari-ratio filter above,
    to catch cases like transliterated or code-switched text that has enough
    Devanagari characters to pass the ratio filter but isn't actually Hindi
    prose.

    Example: keep_by_language_id("यह हिन्दी वाक्य है", model, "__label__hi", 0.5) -> True
             keep_by_language_id("This is English", model, "__label__hi", 0.5) -> False
    """
    label, confidence = predict_language(text, model)
    return label == target_label and confidence >= min_confidence


# ---------------------------------------------------------------------------
# Pipeline orchestration (formerly run_pipeline.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dataset stats reporting (dataset-level half of the old compute_stats.py)
# ---------------------------------------------------------------------------


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


def _totals(records):
    doc_count = len(records)
    word_count = sum(len(r["text"].split()) for r in records)
    return {"doc_count": doc_count, "word_count": word_count}


def split_size_report(splits):
    """Summarize document and word counts for each of train/val/test.

    Example: split_size_report({"train": [...], "val": [...], "test": [...]})
             -> {"train": {"doc_count": N, "word_count": M}, "val": {...}, "test": {...}}
    """
    return {name: _totals(records) for name, records in splits.items()}


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


def render_dataset_stats_report(report_data):
    """Render the aggregated corpus/cleaning stats dict into the dataset_stats.md Markdown.

    Kept as pure string formatting (no file I/O) so it's independently
    testable against a hand-built `report_data` dict. Tokenizer-specific
    metrics (vocab sweep, exact token counts, tokenization examples) are
    NOT part of this report -- see src/evaluation/tokenizer_eval.py for those.

    Example: render_dataset_stats_report({...}) -> "# Hindi Dataset Statistics\\n\\n..."
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

    return "\n".join(lines)


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_dataset_stats_report():
    """Build and write report/dataset_stats.md from real pipeline outputs.

    See src/evaluation/tokenizer_eval.py's generate_tokenizer_eval_report for
    the companion report covering tokenizer-specific metrics.
    """
    splits = {
        name: _load_jsonl(os.path.join(_SPLITS_DIR, f"{name}.jsonl"))
        for name in ("train", "val", "test")
    }
    all_records = splits["train"] + splits["val"] + splits["test"]

    with open(_COLLECTION_STATS_PATH, encoding="utf-8") as f:
        collection_stats = json.load(f)
    clean_total = sum(s["total"] for s in collection_stats["clean_stage"].values())
    final_total = collection_stats["near_dedup"]["kept"]
    dedup_removal_pct = dedup_removal_rate(clean_total, final_total)

    report_data = {
        "pipeline_stages": pipeline_stage_table(collection_stats),
        "source_stats": count_source_stats(all_records),
        "dedup_removal_pct": dedup_removal_pct,
        "split_sizes": {name: _totals(records) for name, records in splits.items()},
    }

    markdown = render_dataset_stats_report(report_data)
    os.makedirs(os.path.dirname(_REPORT_PATH), exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    return _REPORT_PATH


if __name__ == "__main__":
    from src.utils.download_public import load_data_config

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
