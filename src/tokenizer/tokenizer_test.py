"""Train a from-scratch SentencePiece tokenizer for Hindi.

No pretrained tokenizer is used anywhere here -- SentencePieceTrainer.train()
fits a brand-new model purely from our own cleaned corpus sample.
"""
import json
import os
import random

import sentencepiece as spm

from src.tokenizer.tokenizer_config import get_config


def build_training_sample(train_split_path, max_lines, seed):
    """Sample up to `max_lines` documents from the train split for SentencePiece training.

    Args: train_split_path (str, JSONL path with a "text" field per line),
    max_lines (int, cap on sampled documents), seed (int, RNG seed for
    reproducible sampling).
    Returns: list[str], one document's text per entry, newlines flattened to spaces.
    """
    with open(train_split_path, encoding="utf-8") as f:
        lines = f.readlines()

    rng = random.Random(seed)
    rng.shuffle(lines)
    sample_lines = lines[:max_lines]

    return [json.loads(line)["text"].replace("\n", " ") for line in sample_lines]


def train_sentencepiece_model(
    sentences, model_prefix, vocab_size, model_type, character_coverage, byte_fallback
):
    """Train one SentencePiece model in-memory and write `<model_prefix>.model/.vocab`.

    Args: sentences (list[str], training corpus), model_prefix (str, output path
    prefix), vocab_size (int), model_type (str, "unigram" or "bpe"),
    character_coverage (float), byte_fallback (bool).
    Returns: nothing; writes `<model_prefix>.model` and `.vocab` as a side effect.
    """
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
    )


def train_all_candidates(config):
    """Train one SentencePiece model per (tokenizer type, vocab size) combination in `config`.

    Args: config (dict, see tokenizer_config.get_config()).
    Returns: list[str], the trained model prefixes (without the `.model` extension).
    """
    sentences = build_training_sample(
        config["train_split_path"],
        config["training_sample_size"],
        config["training_sample_seed"],
    )

    model_prefixes = []
    for tokenizer_type in config["testing_tokenizer_type"]:
        for vocab_size in config["vocab_size_candidates"]:
            prefix = os.path.join(config["artificates_dir"], f"hindi_{tokenizer_type}_{vocab_size}")
            train_sentencepiece_model(
                sentences=sentences,
                model_prefix=prefix,
                vocab_size=vocab_size,
                model_type=tokenizer_type,
                character_coverage=config["character_coverage"],
                byte_fallback=config["byte_fallback"],
            )
            model_prefixes.append(prefix)

    return model_prefixes


def testing_tokenizers_on_samples(config):
    """Train every configured tokenizer-type/vocab-size candidate and print each trained model's path.

    Args: config (dict, see tokenizer_config.get_config()).
    Returns: nothing; prints one line per trained model.
    """
    for prefix in train_all_candidates(config):
        print(f"trained: {prefix}.model")


if __name__ == "__main__":
    config = get_config()
    testing_tokenizers_on_samples(config)
