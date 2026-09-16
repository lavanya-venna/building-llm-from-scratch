"""Train a from-scratch SentencePiece tokenizer for Hindi.

No pretrained tokenizer is used anywhere here -- SentencePieceTrainer.train()
fits a brand-new model purely from our own cleaned corpus sample.
"""
import json
import os
import random

import sentencepiece as spm
import yaml

_TOKENIZER_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "tokenizer_config.yaml"
)
_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
_TRAIN_SPLIT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "splits", "train.jsonl")


def load_tokenizer_config(config_path=None):
    """Load hindi/configs/tokenizer_config.yaml.

    Keeps vocab-size candidates, model type, and coverage settings in one
    reviewable place instead of hardcoded in the training script.

    Example: load_tokenizer_config()["vocab_size_candidates"] -> [32000, 48000, 64000]
    """
    path = config_path or _TOKENIZER_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_training_sample(train_split_path, output_path, max_lines, seed):
    """Sample up to `max_lines` documents from the train split for SentencePiece training.

    SentencePiece training cost grows with input size, so training on a
    representative sample (rather than the full ~500M-token train split) keeps
    training fast while still reflecting the corpus's real vocabulary. A fixed
    seed makes the sample reproducible across reruns.

    Example: build_training_sample("hindi/data/splits/train.jsonl", "sample.txt",
                                    max_lines=8_000_000, seed=42)
             -> writes one document's text per line to sample.txt
    """
    with open(train_split_path, encoding="utf-8") as f:
        lines = f.readlines()

    rng = random.Random(seed)
    rng.shuffle(lines)
    sample_lines = lines[:max_lines]

    with open(output_path, "w", encoding="utf-8") as out:
        for line in sample_lines:
            text = json.loads(line)["text"]
            out.write(text.replace("\n", " ") + "\n")


def train_sentencepiece_model(
    input_file, model_prefix, vocab_size, model_type, character_coverage, byte_fallback
):
    """Train one SentencePiece model and write `<model_prefix>.model/.vocab`.

    `byte_fallback=True` guarantees the tokenizer can represent any input
    (falling back to raw bytes for unseen characters) instead of ever emitting
    a true UNK token, which matters for reporting a meaningful UNK rate later.

    Example: train_sentencepiece_model("sample.txt", "artifacts/hindi_unigram_32000",
                                        32000, "unigram", 0.9995, True)
             -> writes artifacts/hindi_unigram_32000.model and .vocab
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
    )


def train_all_candidates(config=None):
    """Train one SentencePiece model per vocab size in `vocab_size_candidates`.

    Example: train_all_candidates() -> trains hindi_unigram_32000/48000/64000
    into hindi/tokenizer/artifacts/, ready for eval_tokenizer.py's sweep.
    """
    config = config or load_tokenizer_config()
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

    sample_path = os.path.join(_ARTIFACTS_DIR, "train_sample.txt")
    build_training_sample(
        _TRAIN_SPLIT_PATH,
        sample_path,
        max_lines=config["train_sample"]["max_lines"],
        seed=config["train_sample"]["seed"],
    )

    model_prefixes = []
    for vocab_size in config["vocab_size_candidates"]:
        prefix = os.path.join(_ARTIFACTS_DIR, f"hindi_{config['model_type']}_{vocab_size}")
        train_sentencepiece_model(
            input_file=sample_path,
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type=config["model_type"],
            character_coverage=config["character_coverage"],
            byte_fallback=config["byte_fallback"],
        )
        model_prefixes.append(prefix)

    return model_prefixes


if __name__ == "__main__":
    for prefix in train_all_candidates():
        print(f"trained: {prefix}.model")
