"""Tests for hindi/data/dedup.py: exact-hash and near-duplicate (MinHash) dedup."""
from hindi.data.dedup import (
    normalize_for_hash,
    exact_hash,
    dedup_exact,
    shingle,
    minhash_signature,
    dedup_near,
)


def test_normalize_for_hash_collapses_whitespace():
    a = "यह   एक\n\tवाक्य है।"
    b = "यह एक वाक्य है।"
    assert normalize_for_hash(a) == normalize_for_hash(b)


def test_exact_hash_matches_for_identical_normalized_text():
    a = "यह एक वाक्य है।"
    b = "यह   एक वाक्य   है।"
    assert exact_hash(a) == exact_hash(b)


def test_exact_hash_differs_for_different_text():
    assert exact_hash("पहला वाक्य") != exact_hash("दूसरा वाक्य")


def test_dedup_exact_removes_duplicate_records_keeps_first():
    records = [
        {"text": "यह एक वाक्य है।", "doc_id": "1"},
        {"text": "यह   एक वाक्य है।", "doc_id": "2"},  # exact dup of doc 1 after normalization
        {"text": "यह दूसरा वाक्य है।", "doc_id": "3"},
    ]
    result = dedup_exact(records)
    assert [r["doc_id"] for r in result] == ["1", "3"]


def test_shingle_produces_expected_word_ngrams():
    text = "एक दो तीन चार पांच"
    shingles = shingle(text, size=3)
    assert shingles == {"एक दो तीन", "दो तीन चार", "तीन चार पांच"}


def test_minhash_signature_is_deterministic():
    text = "यह एक वाक्य है जिसका मिनहैश निकाला जाएगा"
    sig1 = minhash_signature(text, num_perm=64)
    sig2 = minhash_signature(text, num_perm=64)
    assert list(sig1.hashvalues) == list(sig2.hashvalues)


def test_dedup_near_removes_near_duplicate_but_keeps_distinct():
    base = "भारत एक विशाल और विविधतापूर्ण देश है जिसकी संस्कृति हजारों वर्ष पुरानी है"
    near_dup = base + " आज"  # tiny edit -> should collapse into a near-duplicate
    distinct = "क्रिकेट भारत में सबसे लोकप्रिय खेलों में से एक है और इसे करोड़ों लोग देखते हैं"

    records = [
        {"text": base, "doc_id": "1"},
        {"text": near_dup, "doc_id": "2"},
        {"text": distinct, "doc_id": "3"},
    ]
    result = dedup_near(records, shingle_size=5, num_perm=128, jaccard_threshold=0.8)
    kept_ids = {r["doc_id"] for r in result}
    assert "1" in kept_ids or "2" in kept_ids
    assert not ({"1", "2"} <= kept_ids)  # not both survive
    assert "3" in kept_ids
