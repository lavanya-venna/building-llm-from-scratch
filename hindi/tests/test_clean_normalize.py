"""Tests for hindi/data/clean_normalize.py, written before the implementation.

Each test targets exactly one standalone cleaning function and checks it against
a small, hand-crafted example so behavior is pinned down before the module exists.
"""
import unicodedata

from hindi.data.clean_normalize import (
    fix_mojibake,
    normalize_unicode,
    strip_markup,
    strip_urls,
    devanagari_ratio,
    keep_by_devanagari_ratio,
    keep_by_length,
    clean_document,
)


def test_fix_mojibake_repairs_broken_encoding():
    # "नमस्ते" mis-decoded as Latin-1-over-UTF-8 mojibake, as ftfy would encounter
    # from a badly-decoded web page.
    broken = "नमस्ते".encode("utf-8").decode("latin-1")
    fixed = fix_mojibake(broken)
    assert fixed == "नमस्ते"


def test_fix_mojibake_leaves_clean_text_untouched():
    clean = "यह एक स्वच्छ वाक्य है।"
    assert fix_mojibake(clean) == clean


def test_normalize_unicode_applies_nfc():
    # Devanagari "क" + combining nukta, decomposed form, should normalize to
    # the single precomposed/canonical NFC form used consistently downstream.
    decomposed = "क" + "़"  # क + nukta -> क़ (qa)
    normalized = normalize_unicode(decomposed)
    assert normalized == unicodedata.normalize("NFC", decomposed)
    # Re-normalizing should be a no-op (idempotent).
    assert normalize_unicode(normalized) == normalized


def test_strip_markup_removes_html_tags():
    html = "<p>यह एक <b>परीक्षण</b> वाक्य है।</p>"
    assert strip_markup(html) == "यह एक परीक्षण वाक्य है।"


def test_strip_markup_removes_wiki_markup():
    wiki = "यह [[विकिपीडिया]] का {{इन्फोबॉक्स}} है। <ref>स्रोत</ref>"
    result = strip_markup(wiki)
    assert "[[" not in result and "]]" not in result
    assert "{{" not in result and "}}" not in result
    assert "<ref>" not in result
    assert "विकिपीडिया" in result
    assert "इन्फोबॉक्स" not in result  # template content is dropped, not just the braces


def test_strip_urls_removes_http_links():
    text = "यह एक लेख है https://example.com/page?x=1 और यहाँ और जानकारी है।"
    result = strip_urls(text)
    assert "http" not in result
    assert "यह एक लेख है" in result
    assert "और यहाँ और जानकारी है" in result


def test_devanagari_ratio_pure_hindi_is_high():
    text = "यह पूरी तरह हिन्दी वाक्य है।"
    assert devanagari_ratio(text) > 0.6


def test_devanagari_ratio_pure_english_is_low():
    text = "This is a fully English sentence with no Devanagari at all."
    assert devanagari_ratio(text) < 0.1


def test_keep_by_devanagari_ratio_thresholds_correctly():
    hindi_text = "यह हिन्दी पाठ है और इसे रखा जाना चाहिए।"
    english_text = "This is English text and should be dropped."
    assert keep_by_devanagari_ratio(hindi_text, min_ratio=0.6) is True
    assert keep_by_devanagari_ratio(english_text, min_ratio=0.6) is False


def test_keep_by_length_drops_short_docs():
    short_doc = "यह छोटा है।"  # ~3 words
    long_doc = " ".join(["शब्द"] * 25)  # 25 words
    assert keep_by_length(short_doc, min_words=20) is False
    assert keep_by_length(long_doc, min_words=20) is True


def test_clean_document_pipeline_end_to_end():
    record = {
        "text": "<p>" + " ".join(["à¤¯à¤¹ à¤¹à¤¿à¤¨à¥\x8dà¤¦à¥€ à¤µà¤¾à¤•à¥\x8dà¤¯ à¤¹à¥ˆ"] * 5) + "</p>",
        "source": "unit_test",
        "doc_id": "doc-1",
    }
    cleaned = clean_document(record, min_words=5, min_devanagari_ratio=0.6)
    assert cleaned is not None
    assert "<p>" not in cleaned["text"]
    assert cleaned["source"] == "unit_test"
    assert cleaned["pipeline_stage"] == "clean"


def test_clean_document_drops_non_hindi_record():
    record = {"text": "This is a purely English document with more than twenty words in it to pass the length filter easily.", "source": "unit_test", "doc_id": "doc-2"}
    assert clean_document(record, min_words=5, min_devanagari_ratio=0.6) is None
