"""Apply strip_non_devanagari_parens to every document in a JSONL file, in parallel."""
import json
import multiprocessing
import os

from src.data.data_preprocessor import strip_non_devanagari_parens


def _clean_line(line):
    """Parse one JSONL line, clean its "text" field, and re-serialize it.

    Args: line (str, one JSON record).
    Returns: str, the cleaned record serialized back to a JSON line (with trailing newline).
    """
    record = json.loads(line)
    record["text"] = strip_non_devanagari_parens(record["text"])
    return json.dumps(record, ensure_ascii=False) + "\n"


def clean_jsonl_file(input_path, output_path, num_workers=None):
    """Strip non-Devanagari parentheticals from every record's "text" field in parallel, writing the cleaned records to output_path.

    Args: input_path (str, JSONL path), output_path (str, JSONL path to write
    to; may be the same as input_path -- writes via a temp file and renames
    at the end so cleaning a file in place is safe), num_workers (int,
    defaults to os.cpu_count()).
    Returns: nothing; writes output_path as a side effect.
    """
    num_workers = num_workers or os.cpu_count() or 1
    tmp_path = output_path + ".tmp"
    with open(input_path, encoding="utf-8") as f:
        with multiprocessing.Pool(num_workers) as pool, open(tmp_path, "w", encoding="utf-8") as out:
            for cleaned_line in pool.imap(_clean_line, f, chunksize=2000):
                out.write(cleaned_line)
    os.replace(tmp_path, output_path)


if __name__ == "__main__":
    clean_jsonl_file("src/tokenizer/sample_data/data.jsonl", "src/tokenizer/sample_data/data.jsonl")
