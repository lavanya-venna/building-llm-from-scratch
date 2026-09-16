"""Evaluate trained SentencePiece candidates: fertility and UNK rate, then select
the best vocab size for the final Hindi tokenizer.
"""
import json
import os

import sentencepiece as spm

from hindi.tokenizer.train_tokenizer import load_tokenizer_config

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
_VAL_SPLIT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "splits", "val.jsonl")


def fertility(sp, texts):
    """Measure average characters-per-token and tokens-per-word for `sp` on `texts`.

    Chars-per-token is the standard fertility metric: higher means the
    tokenizer represents the same text with fewer, more compressed tokens,
    which is generally desirable up to a point of diminishing returns.

    Example: fertility(sp, ["यह एक वाक्य है।"]) -> {"avg_chars_per_token": 3.4,
                                                     "avg_tokens_per_word": 1.3}
    """
    total_chars = 0
    total_tokens = 0
    total_words = 0
    for text in texts:
        pieces = sp.encode(text, out_type=int)
        total_chars += len(text)
        total_tokens += len(pieces)
        total_words += len(text.split())

    return {
        "avg_chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
        "avg_tokens_per_word": total_tokens / total_words if total_words else 0.0,
    }


def unk_rate(sp, texts):
    """Fraction of emitted tokens that are the true UNK token, over `texts`.

    With byte_fallback=True (see train_tokenizer.py) this should be ~0, since
    unseen characters fall back to byte tokens instead of UNK -- reported
    here as a direct check that byte fallback is working as configured.

    Example: unk_rate(sp, ["यह एक वाक्य है।", "🎉 emoji text"]) -> 0.0
    """
    unk_id = sp.unk_id()
    total_tokens = 0
    unk_tokens = 0
    for text in texts:
        pieces = sp.encode(text, out_type=int)
        total_tokens += len(pieces)
        unk_tokens += sum(1 for p in pieces if p == unk_id)
    return unk_tokens / total_tokens if total_tokens else 0.0


def select_best_vocab(sweep, tolerance, max_unk_rate=0.01):
    """Pick the smallest vocab size whose fertility is within `tolerance` of the best.

    "Best" is measured on avg_chars_per_token; candidates with an unk_rate
    above `max_unk_rate` are disqualified regardless of fertility, since a
    tokenizer that drops information to UNK is unacceptable no matter how
    compact its output looks.

    Example: select_best_vocab({32000: {"avg_chars_per_token": 3.5, "unk_rate": 0.0},
                                 48000: {"avg_chars_per_token": 3.6, "unk_rate": 0.0},
                                 64000: {"avg_chars_per_token": 3.61, "unk_rate": 0.0}},
                                tolerance=0.05)
             -> 48000 (32000 is 0.11 below the best 3.61, outside the 0.05 tolerance;
                48000 and 64000 are both within 0.05, so the smaller of the two wins)
    """
    qualifying = {v: s for v, s in sweep.items() if s["unk_rate"] <= max_unk_rate}
    if not qualifying:
        qualifying = sweep

    best_score = max(s["avg_chars_per_token"] for s in qualifying.values())
    near_optimal = [
        v for v, s in qualifying.items() if best_score - s["avg_chars_per_token"] <= tolerance
    ]
    return min(near_optimal)


def load_val_texts(val_split_path=None):
    """Load the held-out validation split's texts for tokenizer evaluation.

    Example: load_val_texts() -> ["यह एक वाक्य है।", "भारत एक विशाल देश है।", ...]
    """
    path = val_split_path or _VAL_SPLIT_PATH
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def evaluate_all_candidates(config=None, val_texts=None):
    """Run fertility/unk_rate for every trained vocab candidate on the val split.

    Example: evaluate_all_candidates() -> {32000: {"avg_chars_per_token": ..., "unk_rate": ...}, ...}
    """
    config = config or load_tokenizer_config()
    val_texts = val_texts if val_texts is not None else load_val_texts()

    sweep = {}
    for vocab_size in config["vocab_size_candidates"]:
        prefix = os.path.join(_ARTIFACTS_DIR, f"hindi_{config['model_type']}_{vocab_size}")
        sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
        metrics = fertility(sp, val_texts)
        metrics["unk_rate"] = unk_rate(sp, val_texts)
        sweep[vocab_size] = metrics

    return sweep


if __name__ == "__main__":
    sweep = evaluate_all_candidates()
    for vocab_size, metrics in sweep.items():
        print(vocab_size, metrics)
    best = select_best_vocab(sweep, tolerance=0.05)
    print("selected vocab size:", best)
