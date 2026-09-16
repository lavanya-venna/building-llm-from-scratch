"""Tests for hindi/data/lid_filter.py.

The real fasttext lid.176 model is a large binary downloaded at runtime, so unit
tests use a tiny fake model object with the same `.predict(text)` interface
fasttext models expose, instead of loading the real one.
"""
from hindi.data.lid_filter import predict_language, keep_by_language_id


class _FakeFastTextModel:
    """Stands in for a loaded fasttext model in tests."""

    def __init__(self, label, confidence):
        self._label = label
        self._confidence = confidence

    def predict(self, text):
        # Real fasttext returns (labels_tuple, confidences_array).
        return ((self._label,), (self._confidence,))


def test_predict_language_extracts_label_and_confidence():
    model = _FakeFastTextModel("__label__hi", 0.97)
    label, confidence = predict_language("यह हिन्दी वाक्य है", model)
    assert label == "__label__hi"
    assert confidence == 0.97


def test_keep_by_language_id_accepts_confident_hindi():
    model = _FakeFastTextModel("__label__hi", 0.9)
    assert keep_by_language_id(
        "यह हिन्दी वाक्य है", model, target_label="__label__hi", min_confidence=0.5
    ) is True


def test_keep_by_language_id_rejects_other_language():
    model = _FakeFastTextModel("__label__en", 0.95)
    assert keep_by_language_id(
        "This is English", model, target_label="__label__hi", min_confidence=0.5
    ) is False


def test_keep_by_language_id_rejects_low_confidence_hindi():
    model = _FakeFastTextModel("__label__hi", 0.2)
    assert keep_by_language_id(
        "यह अस्पष्ट है", model, target_label="__label__hi", min_confidence=0.5
    ) is False


def test_predict_language_strips_newlines_before_predicting():
    # fasttext's predict() raises on embedded newlines, so the wrapper must
    # sanitize input rather than passing multi-line text straight through.
    seen = {}

    class _RecordingModel:
        def predict(self, text):
            seen["text"] = text
            return (("__label__hi",), (0.8,))

    predict_language("पहली पंक्ति\nदूसरी पंक्ति", _RecordingModel())
    assert "\n" not in seen["text"]
