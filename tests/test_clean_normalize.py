"""Tests for the cleaning/normalization functions in src/data/data_preprocessor.py.

Each test targets exactly one standalone cleaning function and checks it against
a small, hand-crafted example so behavior is pinned down independently.
"""
import unicodedata

from src.data.data_preprocessor import (
    fix_mojibake,
    normalize_unicode,
    strip_markup,
    strip_urls,
    strip_non_devanagari_parens,
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


def test_strip_non_devanagari_parens_removes_english_acronym():
    text = "भारत निर्वाचन आयोग (ECI) को सूचना दी"
    assert strip_non_devanagari_parens(text) == "भारत निर्वाचन आयोग को सूचना दी"


def test_strip_non_devanagari_parens_removes_multiword_english_phrase():
    text = "आचार्य चाणक्य के विचार(Acharya Chanakya Thoughts) में घुली नीति"
    assert strip_non_devanagari_parens(text) == "आचार्य चाणक्य के विचार में घुली नीति"


def test_strip_non_devanagari_parens_keeps_hindi_aside():
    text = "सिडनी, (भाषा)। ग्लेन मैक्सवेल ने कहा"
    assert strip_non_devanagari_parens(text) == text


def test_strip_non_devanagari_parens_removes_numeric_aside():
    text = "आबकारी अधिनियम की धारा 34(1) व (2) के अपराध"
    assert strip_non_devanagari_parens(text) == "आबकारी अधिनियम की धारा 34 व के अपराध"


def test_strip_non_devanagari_parens_keeps_mixed_hindi_english_aside():
    text = "जॉन सीना (WWE यूनिवर्सल चैंपियनशिप के लिए मैच)"
    assert strip_non_devanagari_parens(text) == text


def test_strip_non_devanagari_parens_removes_empty_parens():
    text = "एमिरे मकुपसन (), का जन्म 30 सितंबर 1947 को हुआ"
    assert strip_non_devanagari_parens(text) == "एमिरे मकुपसन, का जन्म 30 सितंबर 1947 को हुआ"


def test_strip_non_devanagari_parens_removes_other_script_aside():
    text = "त्सेरिगो (τεριγο) प्रथम विश्व युद्ध के दौरान"
    assert strip_non_devanagari_parens(text) == "त्सेरिगो प्रथम विश्व युद्ध के दौरान"


def test_strip_non_devanagari_parens_removes_two_asides_in_one_sentence():
    text = "बिहार (Bihar) के मुजफ्फरपुर (Muzaffarpur) में मामला सामने आया"
    assert strip_non_devanagari_parens(text) == "बिहार के मुजफ्फरपुर में मामला सामने आया"


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


def test_clean_document_strips_non_devanagari_parens():
    record = {
        "text": " ".join(["भारत निर्वाचन आयोग (ECI) को सूचना दी गई थी"] * 3),
        "source": "unit_test",
        "doc_id": "doc-3",
    }
    cleaned = clean_document(record, min_words=5, min_devanagari_ratio=0.6)
    assert cleaned is not None
    assert "ECI" not in cleaned["text"]
