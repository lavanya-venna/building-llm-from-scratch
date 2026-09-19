# Tokenizer module: what changed, why, and how

Covers `src/tokenizer/tokenizer_test.py`, `src/tokenizer/tokenizer_config.py`,
and `src/evaluation/tokenizer_eval.py` — the from-scratch SentencePiece
tokenizer training and evaluation pipeline. This doc is a design/change
record; `docs/worklog.md` has the plain running log, `report/tokenizer_eval_metrics.json`
has the live numbers.

## Where things stood before this round of changes

The original Phase 1 tokenizer had one config file, `config/tokenizer_config.yaml`,
loaded by a `load_tokenizer_config()` function inside `tokenizer.py` itself:

```yaml
model_type: unigram
character_coverage: 0.9995
byte_fallback: true
vocab_size_candidates: [32000, 48000, 64000]
selected_vocab_size: 48000
train_sample: {max_lines: 1000000, seed: 42}
```

`train_all_candidates()` would sample the train split down to a file
(`build_training_sample(train_split_path, output_path, max_lines, seed)`,
writing to disk), then train one **unigram-only** SentencePiece model per
`vocab_size_candidates` entry by pointing `SentencePieceTrainer.train(input=...)`
at that file. `src/evaluation/tokenizer_eval.py` read the same YAML (via the
same `load_tokenizer_config`) to sweep those three models' fertility/UNK
rate, pick a winner with `select_best_vocab`, and render a Markdown report
(`report/tokenizer_eval_metrics.md`) with a sweep table, total corpus token
count, and example tokenizations. `src/model/train.py` also depended on
`load_tokenizer_config`/`build_training_sample`/`train_sentencepiece_model`
to train a fallback tokenizer if `data/artifacts/hindi_unigram_48000.model`
wasn't present yet.

## What changed, and why

### 1. Python config instead of YAML: `tokenizer_config.py`

**Why**: wanted a single reviewable `get_config()` (styled like a typical
ML training config module) instead of a YAML file + loader function, and
wanted to widen the sweep to compare tokenizer *types*, not just vocab
sizes — `unigram` alone was never actually compared against `bpe`.

**What**: `config/tokenizer_config.yaml` is deleted. `src/tokenizer/tokenizer_config.py`'s
`get_config()` returns one flat dict that both the training sweep and the
evaluation step read (current values, after later changes below):

| Key | Value | Purpose |
|---|---|---|
| `testing_tokenizer_type` | `["unigram", "bpe"]` | types to sweep |
| `vocab_size_candidates` | `[32000, 48000, 64000]` | vocab sizes to sweep |
| `training_sample_size` | `1000000` | cap on sampled train docs |
| `training_sample_seed` | `42` | RNG seed for the sample |
| `train_split_path` | `"data/splits/train.cleaned.jsonl"` | where to sample training docs from |
| `val_split_path` | `"data/splits/val.cleaned.jsonl"` | where to read eval/tokenization-demo texts from |
| `test_split_path` | `"data/splits/test.cleaned.jsonl"` | added for completeness; not consumed by any code yet |
| `artificates_dir` | `"sample_vocabs"` | where trained `.model`/`.vocab` files go (repo root; spelling kept as originally specified) |
| `report_path` | `"report/tokenizer_eval_metrics.json"` | where the eval sweep gets written |
| `character_coverage` | `0.9995` | SentencePiece training param |
| `byte_fallback` | `False` | SentencePiece training param (was `true` in the old YAML) |
| `model_type` | `"unigram"` | the production candidate's type, chosen by hand from the eval report (see §9) |
| `vocab_size` | `48000` | the production candidate's vocab size, chosen the same way (renamed from `selected_vocab_size` — see §9) |

Every path that used to be a module-level constant computed from `__file__`
(`_TOKENIZER_CONFIG_PATH`, `_ARTIFACTS_DIR`, `_TRAIN_SPLIT_PATH`,
`_VAL_SPLIT_PATH`, `_REPORT_PATH`) was removed in favor of a config key —
no global paths, no directory creation, in either `tokenizer_test.py` or
`tokenizer_eval.py`.

### 2. In-memory sampling instead of scratch files

**Why**: the old flow wrote the sampled training text to a `.txt` file and
then pointed `SentencePieceTrainer.train(input=...)` at that file — a
round-trip through disk that wasn't needed once we're already holding the
sample in memory.

**How**: confirmed the installed `sentencepiece==0.2.2` supports training
directly from an in-memory iterable via `sentence_iterator=`. So:
- `build_training_sample(train_split_path, max_lines, seed)` now **returns**
  the sampled `list[str]` instead of writing it to an `output_path`.
- `train_sentencepiece_model(sentences, model_prefix, vocab_size, model_type, character_coverage, byte_fallback)`
  now calls `spm.SentencePieceTrainer.train(sentence_iterator=iter(sentences), ...)`
  instead of `input=input_file`.

This also let `src/model/train.py`'s `get_ds`/`get_or_build_tokenizer` drop
a double scratch-file dance they had (one file written and re-read in
`get_ds`, a second file written in `get_or_build_tokenizer`) — the sampled
texts now flow straight through as a list.

### 3. Sweep over type × vocab size: `train_all_candidates` / `testing_tokenizers_on_samples`

**Why**: compare `unigram` vs `bpe`, not just vocab sizes within `unigram`.

**How**: `train_all_candidates(config)` builds the training sample **once**,
then trains every `testing_tokenizer_type × vocab_size_candidates`
combination (2 × 3 = 6) into `{artificates_dir}/hindi_{tokenizer_type}_{vocab_size}.model`.
It returns the list of trained prefixes. `testing_tokenizers_on_samples(config)`
wraps that and just prints `trained: {prefix}.model` per model. (Originally
a sequential loop — see §8 for the later switch to parallel training.)

### 4. Simplified evaluation: a JSON sweep instead of a Markdown report

**Why**: the old Markdown report (sweep table + total corpus token count +
example tokenizations + an automatically-`select_best_vocab`-picked winner)
was more than needed — the decision of which candidate to use is being
made by a human reading the numbers, not by an automatic tie-break rule.

**How**: `src/evaluation/tokenizer_eval.py` was cut down to:
- `fertility(sp, texts)` / `unk_rate(sp, texts)` — unchanged metrics.
- `load_val_texts(val_split_path)` — now a required arg (no more
  module-constant fallback), returns a plain `list[str]`, writes nothing.
- `evaluates_all_tokenizers_with_diff_vocabs(config)` — the sole entry
  point. Loops the same `testing_tokenizer_type × vocab_size_candidates`
  as the training sweep, loads each `{artificates_dir}/hindi_{type}_{size}.model`,
  computes `fertility` + `unk_rate` against `load_val_texts(config["val_split_path"])`,
  and builds `sweep[(vocab_size, model_type)] = {avg_chars_per_token, avg_tokens_per_word, unk_rate}`.
  It then dumps that sweep straight to `config["report_path"]` as JSON
  (tuple keys stringified as `"{vocab_size}_{model_type}"`, since JSON
  object keys must be strings) and returns the sweep dict.
- Removed entirely: `select_best_vocab`, `tokenizer_sweep_table`,
  `render_tokenizer_eval_report`, `generate_tokenizer_eval_report`,
  `_load_jsonl`. `report/tokenizer_eval_metrics.md` (the old Markdown
  output) was deleted.
- `__main__` is `config = get_config(); evaluates_all_tokenizers_with_diff_vocabs(config)`.

### 5. `tokenizer.py` → `tokenizer_test.py`

The implementation file was renamed (via `git mv`, history preserved) from
`src/tokenizer/tokenizer.py` to `src/tokenizer/tokenizer_test.py`. Every
importer was updated to match: `src/model/train.py`, `tests/test_tokenizer.py`,
`tests/test_dataset.py`, `tests/test_train.py`, plus the `CLAUDE.md`
structure listing and its two `src/tokenizer/tokenizer.py` mentions.

Note the naming is deliberately unusual: `tokenizer_test.py` is the *real
implementation* (training functions), not a test file — the actual tests
for it still live in `tests/test_tokenizer.py`. This was an explicit,
repeated instruction, not an accident; flagging it here so it isn't
"corrected" back by mistake later. One practical consequence: pytest's
default `python_files` pattern (`test_*.py` / `*_test.py`) means a bare
`pytest` run from the repo root would attempt to collect this file too
(it would just find zero test functions in it, harmlessly) — this doesn't
affect the documented `pytest tests/` invocation, which only scans `tests/`.

**This naming/scope is also why a *separate* `tokenizer.py` is now planned**
(§11): `tokenizer_test.py` is (and stays) the small-sample sweep/comparison
tool; the real, full-corpus production training script is a distinct file.

### 6. Docstring style

Every function above uses the terse style written into `CLAUDE.md`'s
Conventions section: one sentence on what it does, an `Args:` line, a
`Returns:` line — no rationale paragraphs, no before/after examples. That's
the default everywhere except `src/data/data_preprocessor.py`'s
cleaning/normalization functions, which keep the fuller what/why/example
style they already had.

### 7. Corpus cleaning: strip non-Devanagari parentheticals, retrain on cleaned data

**Why**: eyeballing sample documents turned up parenthetical asides that
add noise — English glosses of transliterated proper nouns (e.g. "बिहार
(Bihar)"), bare numeric asides, even stray empty `()`. Decided (after
iterating on the rule) to keep a parenthetical only if it contains at
least one Devanagari character; everything else inside `(...)` gets
stripped, including English, numeric, empty, and other-script content.

**How**: `strip_non_devanagari_parens(text)` lives in `src/data/data_preprocessor.py`
(a proper cleaning function there, wired into `clean_document`'s pipeline)
and is reused by a thin script, `src/utils/strip_non_devanagari_parens.py`
(`clean_jsonl_file(input_path, output_path)`), which is what actually
produced `data/splits/{train,val,test}.cleaned.jsonl` alongside the
untouched originals. `tokenizer_config.py`'s `train_split_path`/
`val_split_path`/`test_split_path` were then repointed at the `.cleaned.jsonl`
files (§1's table already reflects this), and the full 6-model sweep was
retrained from scratch against the cleaned train sample — see §9 for the
refreshed numbers.

### 8. Parallelized training: `multiprocessing` instead of a sequential loop

**Why**: `sample_vocabs/`'s 6 models were being trained one after another;
with 192 cores available on this box, training all 6 simultaneously is
strictly faster since nothing serializes them.

**How**: `train_all_candidates(config)` now builds the list of 6
`(sentences, model_prefix, vocab_size, model_type, character_coverage, byte_fallback)`
argument tuples and runs them through `multiprocessing.Pool(len(tasks)).starmap(train_sentencepiece_model, tasks)`
instead of a `for` loop. Same for the corpus-cleaning script
(`clean_jsonl_file`), which parallelizes over *lines* instead: a
module-level `_clean_line(line)` worker function fed through
`pool.imap(_clean_line, f, chunksize=2000)`, streaming so the multi-GB
`train.jsonl` is never fully loaded into memory. Real numbers: the 6-model
sweep went from ~13 minutes (sequential, on uncleaned data) to ~5 minutes
(parallel, on cleaned data); cleaning `train.jsonl` (4.87M docs) went from
an estimated ~21 minutes (sequential) to 45 seconds (192-way parallel).

### 9. Production candidate selected: `unigram`, `vocab_size=48000`

`model_type`/`vocab_size` (renamed from `selected_vocab_size` — plain
`vocab_size` reads better once it's actually set) were filled in by hand
after reading the retrained-on-cleaned-data sweep. Current
`report/tokenizer_eval_metrics.json`:

| Type | Vocab | Avg chars/token | Avg tokens/word | UNK rate |
|---|---|---|---|---|
| unigram | 32000 | 4.287 | 1.204 | 0.223% |
| unigram | **48000** | **4.369** | **1.182** | **0.227%** |
| unigram | 64000 | 4.413 | 1.170 | 0.230% |
| bpe | 32000 | 4.262 | 1.211 | 0.222% |
| bpe | 48000 | 4.354 | 1.186 | 0.227% |
| bpe | 64000 | 4.405 | 1.172 | 0.229% |

Same pattern as the pre-cleaning sweep: unigram edges out bpe at every
vocab size, fertility rises with vocab size, UNK rate is flat and low.
`unigram`/`48000` chosen as a good fertility/size tradeoff (within 0.05
chars/token of the best `64000` result, smaller vocab).

### 10. Tokenize/decode a handful of val samples for a sanity check

**Why**: wanted to see real tokenizer output on real val documents, and
confirm the encode → decode round trip is lossless, before trusting the
selected model further.

**How**: two new functions in `tokenizer_test.py`:
- `tokenize_val_samples(config, num_samples=10)` — loads the first
  `num_samples` records from `config["val_split_path"]`, tokenizes each
  with `{artificates_dir}/hindi_{model_type}_{vocab_size}.model`, returns
  `{raw_text: pieces}` (pieces as strings, `out_type=str`, for readability).
- `decode_tokenized_samples(config, tokenized_outputs)` — loads the same
  model and calls `sp.decode(pieces)` on each value, returning the
  decoded sentences in the same order.

`__main__` now branches on whether a production candidate has been
selected:
```python
if not config['model_type'] and not config['vocab_size']:
    testing_tokenizers_on_samples(config)          # run the sweep
else:
    tokenized_outputs = tokenize_val_samples(config)
    # ... save to src/tokenizer/sample_data/tokenized_sample_outputs.json
    decoded_samples = decode_tokenized_samples(config, tokenized_outputs)
    # ... save to src/tokenizer/sample_data/decoded_samples.txt, one per line
    for original, decoded in zip(tokenized_outputs.keys(), decoded_samples):
        if original == decoded:
            print(f"MATCH: {original}")
        else:
            print(f"MISMATCH:\n  original: {original}\n  decoded:  {decoded}")
```
Real run against the selected `unigram`/`48000` model: all 10 val samples
came back `MATCH` — the encode/decode round trip is exact on real data.

## Current files under `src/tokenizer/sample_data/`

- `data.jsonl` — 2000 random train-split samples (for eyeballing the
  corpus; also the file the bracket-cleaning problem was first spotted in).
- `tokenized_sample_outputs.json` — `{raw_text: pieces}` for 10 val records,
  from the selected production model.
- `decoded_samples.txt` — those same 10 records' pieces decoded back to
  text, one per line; currently identical to their originals.

## Current end-to-end flow

1. `python3 -m src.tokenizer.tokenizer_test` → `config = get_config()` →
   since `model_type`/`vocab_size` are now set, this takes the tokenize/
   decode/compare branch above rather than re-running the sweep. To re-run
   the sweep itself, call `testing_tokenizers_on_samples(config)` directly
   (or temporarily clear `model_type`/`vocab_size`).
2. `python3 -m src.evaluation.tokenizer_eval` → `config = get_config()` →
   `evaluates_all_tokenizers_with_diff_vocabs(config)` → loads all 6 models,
   scores each against the cleaned ~49.7k-doc val split, writes
   `report/tokenizer_eval_metrics.json`.

Both are documented as heavy in `CLAUDE.md`'s "Known environment quirks"
(the sweep genuinely trains 6 models; eval genuinely scores all of them).

## TODO — not started yet, tracked here on purpose

Two outstanding items, explicitly deferred (not to be worked on until
asked):

1. **Train the tokenizer on the entire corpus**, not the current
   1,000,000-document sample (`training_sample_size` in `tokenizer_config.py`).
   The sweep above exists to compare types/vocab sizes cheaply; the actual
   production tokenizer should be fit on the full cleaned train split
   (`data/splits/train.cleaned.jsonl`, ~4.87M documents).
2. **Write a new `src/tokenizer/tokenizer.py`** dedicated to generating
   *the* production tokenizer from the full corpus — distinct from
   `tokenizer_test.py`, which stays the small-sample sweep/comparison tool
   (see §5's naming note). Not yet designed; open questions include
   whether it reuses `train_sentencepiece_model`/`strip_non_devanagari_parens`
   as-is, what its own config looks like, and whether multiprocessing
   (§8) is still the right approach at full-corpus scale (a single
   SentencePiece training run over ~4.87M documents is a much bigger job
   than 6 parallel runs over a 1M-document sample).
