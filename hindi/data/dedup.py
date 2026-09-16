"""Exact and near-duplicate removal for the Hindi corpus.

Exact dedup catches byte-identical (post-normalization) repeats, cheap to run
first. Near-dedup catches boilerplate-heavy near-duplicates (e.g. the same
article re-published with a different footer) that exact hashing misses,
using MinHash/LSH so we don't need an O(n^2) pairwise comparison.
"""
import hashlib
import re

from datasketch import MinHash, MinHashLSH


def normalize_for_hash(text):
    """Collapse whitespace so hashing is robust to formatting-only differences.

    Two documents that differ only in spacing/line breaks are the same content
    for training purposes; without this, exact_hash would treat them as
    distinct and both would survive dedup.

    Example: normalize_for_hash("यह   एक\\nवाक्य है।") -> "यह एक वाक्य है।"
    """
    return re.sub(r"\s+", " ", text).strip()


def exact_hash(text):
    """Compute a stable hash of the normalized text for exact-dedup lookups.

    SHA256 is used (over e.g. Python's built-in hash()) because it's stable
    across process runs/seeds, which id-based dedup needs to be reproducible.

    Example: exact_hash("यह एक वाक्य है।") == exact_hash("यह   एक वाक्य है।")
    """
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedup_exact(records):
    """Drop records whose text hash has already been seen, keeping the first.

    This is the cheap first pass before the more expensive near-dedup step --
    removing verbatim repeats (common when the same page is pulled from two
    different public sources) shrinks the input before MinHash has to run.

    Example: dedup_exact([{"text": "क"}, {"text": "क"}, {"text": "ख"}])
             -> [{"text": "क"}, {"text": "ख"}]
    """
    seen_hashes = set()
    kept = []
    for record in records:
        h = exact_hash(record["text"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        kept.append(record)
    return kept


def shingle(text, size):
    """Break text into a set of overlapping word n-grams ("shingles").

    MinHash operates on sets of shingles rather than raw text so that
    near-identical documents (one word added/removed) still share most of
    their shingles and register as similar.

    Example: shingle("एक दो तीन चार पांच", size=3)
             -> {"एक दो तीन", "दो तीन चार", "तीन चार पांच"}
    """
    words = text.split()
    if len(words) < size:
        return {text}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def minhash_signature(text, num_perm):
    """Compute a MinHash signature summarizing a document's shingle set.

    A fixed-size signature lets us estimate Jaccard similarity between two
    documents in O(num_perm) instead of comparing their full shingle sets,
    which is what makes near-dedup tractable at corpus scale.

    Example: minhash_signature(text, 128) -> a datasketch.MinHash object;
    two documents with mostly-overlapping shingles get near-identical signatures.
    """
    m = MinHash(num_perm=num_perm)
    for s in shingle(text, size=5):
        m.update(s.encode("utf-8"))
    return m


def dedup_near(records, shingle_size, num_perm, jaccard_threshold):
    """Remove near-duplicate documents using MinHash LSH, keeping the first seen.

    Run after dedup_exact. Catches cases exact hashing misses, e.g. the same
    article with a different trailing footer/timestamp appended -- common in
    web-crawled sources like OSCAR/mC4.

    Example: dedup_near([{"text": A}, {"text": A + " आज"}, {"text": B}], 5, 128, 0.8)
             -> keeps one of the two near-identical A records plus B.
    """
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    kept = []
    for record in records:
        shingles = shingle(record["text"], size=shingle_size)
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))

        if lsh.query(m):
            continue  # a near-duplicate is already in the LSH index

        lsh.insert(record["doc_id"], m)
        kept.append(record)
    return kept
