"""Evaluate trained SentencePiece candidates (fertility, UNK rate) and write
the sweep to a JSON report for manual vocab-size/type selection.
"""
import json
import os

import sentencepiece as spm

from src.tokenizer.tokenizer_config import get_config


def fertility(sp, texts):
    """Measure average characters-per-token and tokens-per-word for `sp` on `texts`.

    Args: sp (sentencepiece.SentencePieceProcessor), texts (list[str]).
    Returns: dict with "avg_chars_per_token" and "avg_tokens_per_word".
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

    Args: sp (sentencepiece.SentencePieceProcessor), texts (list[str]).
    Returns: float in [0, 1].
    """
    unk_id = sp.unk_id()
    total_tokens = 0
    unk_tokens = 0
    for text in texts:
        pieces = sp.encode(text, out_type=int)
        total_tokens += len(pieces)
        unk_tokens += sum(1 for p in pieces if p == unk_id)
    return unk_tokens / total_tokens if total_tokens else 0.0


def load_val_texts(val_split_path):
    """Load the held-out validation split's texts for tokenizer evaluation.

    Args: val_split_path (str, JSONL path with a "text" field per line).
    Returns: list[str].
    """
    with open(val_split_path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def evaluates_all_tokenizers_with_diff_vocabs(config):
    """Evaluate fertility/unk_rate for every (vocab size, tokenizer type) candidate and write the sweep to config["report_path"] as JSON.

    Args: config (dict, see tokenizer_config.get_config()).
    Returns: dict mapping (vocab_size, model_type) to a metrics dict.
    """
    val_texts = load_val_texts(config["val_split_path"])

    sweep = {}
    for model_type in config["testing_tokenizer_type"]:
        for vocab_size in config["vocab_size_candidates"]:
            prefix = os.path.join(config["artificates_dir"], f"hindi_{model_type}_{vocab_size}")
            sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
            metrics = fertility(sp, val_texts)
            metrics["unk_rate"] = unk_rate(sp, val_texts)
            sweep[(vocab_size, model_type)] = metrics

    with open(config["report_path"], "w", encoding="utf-8") as f:
        json.dump(
            {f"{vocab_size}_{model_type}": metrics for (vocab_size, model_type), metrics in sweep.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return sweep


if __name__ == "__main__":
    config = get_config()
    evaluates_all_tokenizers_with_diff_vocabs(config)
