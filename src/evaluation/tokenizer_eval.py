"""Evaluate trained SentencePiece candidates (fertility, UNK rate), select the
best vocab size, and generate report/tokenizer_eval_metrics.md.

This merges what used to be eval_tokenizer.py with the tokenizer-metrics half
of the old compute_stats.py (the corpus/cleaning half lives in
src/data/data_preprocessor.py instead, writing a separate report).
"""
import json
import os

import sentencepiece as spm

from src.tokenizer.tokenizer import load_tokenizer_config

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "artifacts")
_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "splits")
_VAL_SPLIT_PATH = os.path.join(_SPLITS_DIR, "val.jsonl")
_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "report", "tokenizer_eval_metrics.md")


# ---------------------------------------------------------------------------
# Fertility / UNK-rate evaluation (formerly eval_tokenizer.py)
# ---------------------------------------------------------------------------


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

    With byte_fallback=True (see src/tokenizer/tokenizer.py) this should be
    ~0, since unseen characters fall back to byte tokens instead of UNK --
    reported here as a direct check that byte fallback is working as
    configured.

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


# ---------------------------------------------------------------------------
# Tokenizer metrics reporting (tokenizer-metrics half of the old compute_stats.py)
# ---------------------------------------------------------------------------


def tokenizer_sweep_table(sweep, selected_vocab_size):
    """Turn the {vocab_size: metrics} sweep dict into an ordered table of rows,
    flagging the selected vocab size for the report.

    Example: tokenizer_sweep_table({32000: {...}}, selected_vocab_size=32000)
             -> [{"vocab_size": 32000, ..., "selected": True}]
    """
    rows = []
    for vocab_size in sorted(sweep):
        row = {"vocab_size": vocab_size, **sweep[vocab_size]}
        row["selected"] = vocab_size == selected_vocab_size
        rows.append(row)
    return rows


def render_tokenizer_eval_report(report_data):
    """Render the tokenizer-metrics dict into the tokenizer_eval_metrics.md Markdown.

    Kept as pure string formatting (no file I/O) so it's independently
    testable against a hand-built `report_data` dict. Corpus/cleaning stats
    are NOT part of this report -- see src/data/data_preprocessor.py's
    render_dataset_stats_report for those.

    Example: render_tokenizer_eval_report({...}) -> "# Tokenizer Evaluation Metrics\\n\\n..."
    """
    lines = ["# Tokenizer Evaluation Metrics", ""]

    lines.append("## Tokenizer vocab sweep")
    lines.append("")
    lines.append("| Vocab size | Avg chars/token | Avg tokens/word | UNK rate | Selected |")
    lines.append("|---|---|---|---|---|")
    for row in report_data["tokenizer_sweep"]:
        marker = "yes" if row["selected"] else ""
        lines.append(
            f"| {row['vocab_size']} | {row['avg_chars_per_token']:.3f} | "
            f"{row['avg_tokens_per_word']:.3f} | {row['unk_rate']:.4f} | {marker} |"
        )
    lines.append("")

    lines.append("## Total corpus size")
    lines.append("")
    lines.append(f"Total exact tokenizer-measured tokens: {report_data['total_exact_tokens']:,}")
    lines.append("")

    lines.append("## Tokenization examples")
    lines.append("")
    for example in report_data["examples"]:
        lines.append(f"- `{example['text']}` -> {example['pieces']}")
    lines.append("")

    return "\n".join(lines)


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_tokenizer_eval_report():
    """Build and write report/tokenizer_eval_metrics.md from real pipeline outputs.

    Fails loudly if the tokenizer sweep hasn't been run and a vocab size
    selected yet (tokenizer_config.yaml's `selected_vocab_size` is still
    null), since the report is meaningless without it.
    """
    config = load_tokenizer_config()
    if config.get("selected_vocab_size") is None:
        raise RuntimeError(
            "tokenizer_config.yaml selected_vocab_size is not set -- run "
            "src.tokenizer.tokenizer and src.evaluation.tokenizer_eval's sweep first."
        )

    splits = {
        name: _load_jsonl(os.path.join(_SPLITS_DIR, f"{name}.jsonl"))
        for name in ("train", "val", "test")
    }
    all_records = splits["train"] + splits["val"] + splits["test"]

    val_texts = load_val_texts(os.path.join(_SPLITS_DIR, "val.jsonl"))
    sweep = evaluate_all_candidates(config, val_texts)
    selected = config["selected_vocab_size"]

    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(_ARTIFACTS_DIR, f"hindi_{config['model_type']}_{selected}.model")
    )
    total_exact_tokens = sum(len(sp.encode(r["text"])) for r in all_records)

    example_texts = [r["text"][:120] for r in splits["val"][:8]]
    examples = [{"text": t, "pieces": sp.encode(t, out_type=str)} for t in example_texts]

    report_data = {
        "tokenizer_sweep": tokenizer_sweep_table(sweep, selected),
        "total_exact_tokens": total_exact_tokens,
        "examples": examples,
    }

    markdown = render_tokenizer_eval_report(report_data)
    os.makedirs(os.path.dirname(_REPORT_PATH), exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    return _REPORT_PATH


if __name__ == "__main__":
    sweep = evaluate_all_candidates()
    for vocab_size, metrics in sweep.items():
        print(vocab_size, metrics)
    best = select_best_vocab(sweep, tolerance=0.05)
    print("selected vocab size:", best)
