
"""Cleaning and normalization stages for the Hindi corpus.

Each transformation is its own standalone function so it can be unit-tested and
reasoned about independently. `clean_document` composes them into the full
per-document pipeline used by the collection scripts.
"""
import re
import unicodedata

import ftfy
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

_INDIC_NORMALIZER = IndicNormalizerFactory().get_normalizer("hi")

_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")
_ALLOWED_NON_DEVANAGARI = re.compile(r"[\s\d.,!?;:()\-।॥]")

_REF_TAG_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WIKI_BRACKET_RE = re.compile(r"\[\[|\]\]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


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
    (see hindi/configs/data_config.yaml: cleaning.devanagari_ratio_min) so the
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

    Applies mojibake repair, markup stripping, Unicode normalization, then the
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
