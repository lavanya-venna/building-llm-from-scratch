"""Pull a small random sample of documents from a JSONL split for quick manual inspection."""
import os
import random


def sample_random_documents(input_path, output_path, num_samples, seed):
    """Reservoir-sample num_samples random lines from input_path and write them to output_path.

    Args: input_path (str, JSONL path), output_path (str, where to write the
    sample), num_samples (int), seed (int, RNG seed for reproducible sampling).
    Returns: nothing; writes output_path as a side effect.
    """
    rng = random.Random(seed)
    reservoir = []
    # Reservoir sampling: input_path (data/splits/train.jsonl) is multi-GB,
    # so we stream line by line instead of loading the whole file into memory.
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < num_samples:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < num_samples:
                    reservoir[j] = line

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        out.writelines(reservoir)


if __name__ == "__main__":
    sample_random_documents(
        input_path="data/splits/train.jsonl",
        output_path="src/tokenizer/sample_data/data.jsonl",
        num_samples=2000,
        seed=42,
    )
