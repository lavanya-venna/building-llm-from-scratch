def get_config():
    """Return tokenizer sweep settings plus the production candidate once selected.

    Returns: dict with sweep keys (testing_tokenizer_type, vocab_size_candidates,
    training_sample_size, training_sample_seed, train_split_path, val_split_path,
    artificates_dir, report_path, character_coverage, byte_fallback) and
    selection keys (model_type, selected_vocab_size), the latter left as None
    until a candidate is chosen from the eval report.
    """
    return {
        "testing_tokenizer_type": ["unigram", "bpe"],
        "training_sample_size": 1000000,
        "training_sample_seed": 42,
        "train_split_path": "data/splits/train.jsonl",
        "val_split_path": "data/splits/val.jsonl",
        "artificates_dir": "src/tokenizer/sample_vocabs",
        "report_path": "report/tokenizer_eval_metrics.json",
        "vocab_size_candidates": [32000, 48000, 64000],
        "character_coverage": 0.9995,
        "byte_fallback": False,
        "model_type": None,
        "selected_vocab_size": None,
    }
