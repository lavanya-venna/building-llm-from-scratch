"""Language-identification filtering, using fasttext's lid.176 model.

Note: lid.176 is a pretrained *classifier* used only to filter which documents
enter the corpus. It is not part of the LLM or its tokenizer, so it does not
violate the "no pretrained model/tokenizer" constraint on the language model
itself -- it plays the same role a hand-written heuristic filter would, just
with higher accuracy. This is called out explicitly in the dataset stats report.
"""
import os
import urllib.request

import fasttext

_LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tokenizer", "artifacts", "lid.176.bin"
)


def load_fasttext_lid_model(model_path=None):
    """Load (downloading if needed) the fasttext language-ID model.

    Kept separate from the decision logic below so tests can exercise
    `predict_language`/`keep_by_language_id` with a fake model and never need
    network access or the ~130MB binary.

    Example: load_fasttext_lid_model() -> fasttext model object usable with
    predict_language(text, model)
    """
    path = model_path or _DEFAULT_MODEL_PATH
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

    Used as a secondary check after the cheaper Devanagari-ratio filter in
    clean_normalize.py, to catch cases like transliterated or code-switched
    text that has enough Devanagari characters to pass the ratio filter but
    isn't actually Hindi prose.

    Example: keep_by_language_id("यह हिन्दी वाक्य है", model, "__label__hi", 0.5) -> True
             keep_by_language_id("This is English", model, "__label__hi", 0.5) -> False
    """
    label, confidence = predict_language(text, model)
    return label == target_label and confidence >= min_confidence
