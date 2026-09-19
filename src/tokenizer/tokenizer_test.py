"""Train a from-scratch SentencePiece tokenizer for Hindi.

No pretrained tokenizer is used anywhere here -- SentencePieceTrainer.train()
fits a brand-new model purely from our own cleaned corpus sample.
"""
import json
import multiprocessing
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
    """Train one SentencePiece model per (tokenizer type, vocab size) combination in `config`, in parallel.

    Args: config (dict, see tokenizer_config.get_config()).
    Returns: list[str], the trained model prefixes (without the `.model` extension).
    """
    sentences = build_training_sample(
        config["train_split_path"],
        config["training_sample_size"],
        config["training_sample_seed"],
    )

    tasks = [
        (
            sentences,
            os.path.join(config["artificates_dir"], f"hindi_{tokenizer_type}_{vocab_size}"),
            vocab_size,
            tokenizer_type,
            config["character_coverage"],
            config["byte_fallback"],
        )
        for tokenizer_type in config["testing_tokenizer_type"]
        for vocab_size in config["vocab_size_candidates"]
    ]

    with multiprocessing.Pool(len(tasks)) as pool:
        pool.starmap(train_sentencepiece_model, tasks)

    return [task[1] for task in tasks]


def testing_tokenizers_on_samples(config):
    """Train every configured tokenizer-type/vocab-size candidate and print each trained model's path.

    Args: config (dict, see tokenizer_config.get_config()).
    Returns: nothing; prints one line per trained model.
    """
    for prefix in train_all_candidates(config):
        print(f"trained: {prefix}.model")


def tokenize_val_samples(config, num_samples=10):
    """Tokenize the first `num_samples` val records with the selected (model_type, vocab_size) model.

    Args: config (dict, see tokenizer_config.get_config()), num_samples (int).
    Returns: dict mapping each record's raw text to its tokenized pieces (list[str]).
    """
    model_path = os.path.join(
        config["artificates_dir"], f"hindi_{config['model_type']}_{config['vocab_size']}.model"
    )
    sp = spm.SentencePieceProcessor(model_file=model_path)

    val_records = []
    with open(config["val_split_path"], encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            val_records.append(json.loads(line))

    return {record["text"]: sp.encode(record["text"], out_type=str) for record in val_records}


def decode_tokenized_samples(config, tokenized_outputs):
    """Decode each piece list in tokenized_outputs back to its original sentence.

    Args: config (dict, see tokenizer_config.get_config()), tokenized_outputs
    (dict mapping raw text to a list[str] of pieces, as returned by tokenize_val_samples).
    Returns: list[str], one decoded sentence per value in tokenized_outputs.
    """
    model_path = os.path.join(
        config["artificates_dir"], f"hindi_{config['model_type']}_{config['vocab_size']}.model"
    )
    sp = spm.SentencePieceProcessor(model_file=model_path)

    return [sp.decode(pieces) for pieces in tokenized_outputs.values()]


if __name__ == "__main__":
    config = get_config()

    if not config['model_type'] and not config['vocab_size']:
        testing_tokenizers_on_samples(config)

    else:

        # Encode validation samples and save it in src/tokenizer/sample_data/tokenized_sample_outputs.json
        tokenized_outputs = tokenize_val_samples(config)
        with open("src/tokenizer/sample_data/tokenized_sample_outputs.json", "w", encoding="utf-8") as out:
            json.dump(tokenized_outputs, out, ensure_ascii=False, indent=2)

        # Decode those same tokenized outputs back to sentences and save one per line
        decoded_samples = decode_tokenized_samples(config, tokenized_outputs)
        with open("src/tokenizer/sample_data/decoded_samples.txt", "w", encoding="utf-8") as out:
            out.write("\n".join(decoded_samples) + "\n")

        # Compare each original sentence to its decoded round-trip
        for original, decoded in zip(tokenized_outputs.keys(), decoded_samples):
            if original == decoded:
                print(f"MATCH: {original}")
            else:
                print(f"MISMATCH:\n  original: {original}\n  decoded:  {decoded}")
        
