def get_config():
    """Return tokenizer sweep settings plus the production candidate once selected.

    Returns: dict with sweep keys (testing_tokenizer_type, vocab_size_candidates,
    training_sample_size, training_sample_seed, train_split_path, val_split_path,
    test_split_path, artificates_dir, report_path, tokenizer_path,
    character_coverage, byte_fallback) and selection keys (model_type,
    vocab_size), the latter used both to pick a sweep candidate and as the
    training params for the full-corpus production tokenizer.
    """
    return {
        "testing_tokenizer_type": ["unigram", "bpe"],
        "training_sample_size": 1000000,
        "training_sample_seed": 42,
        "train_split_path": "data/splits/train.cleaned.jsonl",
        "val_split_path": "data/splits/val.cleaned.jsonl",
        "test_split_path": "data/splits/test.cleaned.jsonl",
        "artificates_dir": "sample_vocabs",
        "report_path": "report/tokenizer_eval_metrics.json",
        "tokenizer_path": "/fsxvision_new/lavanya.venna/llm_from_scratch/src/tokenizer/hindi_tokenizer.model",
        "vocab_size_candidates": [32000, 48000, 64000],
        "character_coverage": 0.9995,
        "byte_fallback": False,
        "model_type": "unigram",
        "vocab_size": 48000,
    }

