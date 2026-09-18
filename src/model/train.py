"""Data pipeline + training scaffold for the Hindi decoder-only Transformer LM.

"Concatenate and chop" causal-LM pipeline: tokenize each doc, append <eos>,
concatenate per split into one flat token-id stream (tokenize_and_concatenate),
then wrap each stream in a HindiDataset (src/data/dataset.py) that yields
fixed-length, non-overlapping (input, shifted-target) chunks.

DEVIATION FROM THE LITERAL HF-`tokenizers`-FLAVORED SPEC (explicit,
user-approved decision -- not an oversight): this project has exactly one
tokenizer, the from-scratch SentencePiece Unigram model trained in Phase 1
(src/tokenizer/tokenizer_test.py; CLAUDE.md hard constraint: no pretrained
tokenizer classes). `get_or_build_tokenizer` below therefore loads/builds a
`sentencepiece.SentencePieceProcessor`, not a HuggingFace `tokenizers.Tokenizer`.
`get_ds` loads the existing, frozen data/splits/{train,val,test}.jsonl
produced by Phase 1 rather than combining raw per-source files and
re-deriving a fresh 90/5/5 split -- re-splitting here would silently
contradict Phase 1's real 98/1/1 source-stratified split and risk
train/val/test leakage. When the tokenizer needs to be trained from scratch
(fallback path only -- production runs already have
data/artifacts/hindi_unigram_48000.model on disk), it is trained on a
sample of the TRAIN split ONLY, never val/test -- this matches Phase 1's
actual practice (build_training_sample only ever reads train.jsonl) and is
a deliberate improvement over "train on all texts including val/test",
which would leak val/test docs into the tokenizer's vocabulary.

SentencePiece API notes (verified against the real installed sentencepiece
0.2.2 and the real hindi_unigram_48000.model):
  - Load:   spm.SentencePieceProcessor(model_file=path)   (not Tokenizer.from_file)
  - Encode: sp.encode(text, out_type=int) -> plain list[int] directly
            (no .ids attribute; no auto bos/eos)
  - Vocab size: sp.get_piece_size()  -- get_vocab_size() DOES NOT EXIST on
            SentencePieceProcessor and must not be called.
  - Decode: sp.decode(list[int]) -- NOT a torch.Tensor (raises TypeError);
            HindiDataset.__getitem__ decodes from the pre-tensor list.
  - eos_id() == 2 for this model; pad_id() == -1 (padding disabled/absent
    by default). "Concatenate and chop" needs no padding, so this is a
    non-issue -- but it IS an honest boundary vs. the literal spec's
    mention of a <pad> special token, which does not exist in this
    tokenizer.

*** WARNING: DO NOT RUN `python -m src.model.train` CASUALLY ***
This module's `__main__` block unconditionally calls train_model(config) ->
get_ds(config), which will tokenize the ENTIRE real train split
(data/splits/train.jsonl: 4,869,410 documents, ~386M words) end to end if
pointed at the real config/model_config.yaml. This is full-corpus
tokenization, not a quick smoke test -- expect a long run, mirroring the
existing heavy-`__main__` warnings in CLAUDE.md for
src/data/data_preprocessor.py and src/tokenizer/tokenizer_test.py. For a quick
check, import get_ds/train_model directly with a small tmp_path config
(see tests/test_train.py) instead of running this file as a script.
"""
import json
import os

import sentencepiece as spm
from torch.utils.data import DataLoader

from src.data.dataset import HindiDataset
from src.model.model import DecoderOnlyTransformer, load_model_config
from src.tokenizer.tokenizer_test import build_training_sample, train_sentencepiece_model
from src.tokenizer.tokenizer_config import get_config

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _resolve_repo_path(path):
    """Resolve a config-relative path (e.g. "data/splits/train.jsonl") against the repo root.

    Lets config/model_config.yaml store clean relative paths while this
    module works correctly regardless of the caller's current working
    directory.

    Example: _resolve_repo_path("data/splits/train.jsonl") ->
             "<repo_root>/data/splits/train.jsonl"
    """
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _read_jsonl_texts(path):
    """Read one split file's "text" field, one document per line, in file order.

    Matches data/splits/{train,val,test}.jsonl's schema:
    {"text": ..., "source": ..., "doc_id": ..., "pipeline_stage": "clean"}.

    Example: _read_jsonl_texts("data/splits/val.jsonl") ->
             ["यह एक वाक्य है।", "भारत एक विशाल देश है।", ...]
    """
    texts = []
    with open(_resolve_repo_path(path), encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def get_or_build_tokenizer(config, all_texts):
    """Load the existing from-scratch SentencePiece tokenizer, or train one if missing.

    config["tokenizer_path"] points at the real trained artifact
    (data/artifacts/hindi_unigram_48000.model) in production use, in which
    case all_texts is ignored entirely -- it only matters for the fallback
    path below (e.g. tests with a tmp_path tokenizer_path that doesn't
    exist yet).

    Fallback (tokenizer_path missing): trains a brand-new SentencePiece
    Unigram model by reusing src.tokenizer.tokenizer_test's
    train_sentencepiece_model (the exact function Phase 1 used) and
    src.tokenizer.tokenizer_config's get_config (for model_type/
    character_coverage/byte_fallback/selected_vocab_size), passing all_texts
    to it directly -- train_sentencepiece_model trains from an in-memory
    sentence iterator, so no scratch file is needed.

    Args:
        config: dict with at least "tokenizer_path".
        all_texts: list[str] of raw texts to train on if tokenizer_path
            doesn't exist yet. Callers (get_ds) must pass TRAIN-SPLIT-ONLY
            texts here -- never val/test -- per the no-leakage decision
            above.

    Returns:
        sentencepiece.SentencePieceProcessor, ready to
        .encode(text, out_type=int) / .decode(ids).
    """
    tokenizer_path = _resolve_repo_path(config["tokenizer_path"])

    if os.path.exists(tokenizer_path):
        return spm.SentencePieceProcessor(model_file=tokenizer_path)

    tc = get_config()  # reuse Phase 1's tokenizer hyperparameters
    os.makedirs(os.path.dirname(tokenizer_path) or ".", exist_ok=True)

    model_prefix = tokenizer_path[: -len(".model")] if tokenizer_path.endswith(".model") else tokenizer_path
    train_sentencepiece_model(
        sentences=all_texts,
        model_prefix=model_prefix,
        vocab_size=tc["selected_vocab_size"],
        model_type=tc["model_type"],
        character_coverage=tc["character_coverage"],
        byte_fallback=tc["byte_fallback"],
    )
    return spm.SentencePieceProcessor(model_file=model_prefix + ".model")


def tokenize_and_concatenate(texts, tokenizer):
    """Tokenize every text, append <eos>, and concatenate into one flat id stream.

    This is the "concatenate" half of "concatenate and chop": each document
    is encoded independently, an end-of-sequence id is appended after every
    document (the only signal HindiDataset's chunks retain that one
    document ended and another began -- chunks are not aligned to document
    boundaries), and the per-document id lists are concatenated into one
    flat list[int] as long as the whole split's worth of ids.

    Example: texts = ["यह एक वाक्य है।", "दूसरा वाक्य।"], eos_id == 2 ->
        tokenizer.encode(texts[0], out_type=int) + [2]
        + tokenizer.encode(texts[1], out_type=int) + [2]

    Args:
        texts: list[str], raw document texts for one split.
        tokenizer: sentencepiece.SentencePieceProcessor.

    Returns:
        list[int], length == sum(len(encode(t)) + 1 for t in texts).
    """
    eos_id = tokenizer.eos_id()
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, out_type=int)  # list[int], length varies per doc
        all_ids.extend(ids)
        all_ids.append(eos_id)
    return all_ids


def get_ds(config):
    """Build train/val/test DataLoaders and the shared tokenizer for one config.

    Does NOT combine raw per-source files and re-derive a fresh split (see
    module docstring's deviation note) -- reads Phase 1's frozen
    data/splits/{train,val,test}.jsonl directly via config["dataset_paths"].
    config["train_split"]/["val_split"]/["test_split"] are intentionally
    unused here (see config/model_config.yaml's comments).

    Steps:
      1. Read each split's "text" fields (_read_jsonl_texts).
      2. If config["tokenizer_path"] doesn't exist yet, sample the TRAIN
         split (only) down to a tractable size by reusing
         src.tokenizer.tokenizer_test.build_training_sample -- the exact
         function/sizing (src.tokenizer.tokenizer_config's
         training_sample_size/training_sample_seed) Phase 1 used -- then
         hand that sample to get_or_build_tokenizer. If it already exists,
         get_or_build_tokenizer just loads it and this sampling work is
         skipped entirely.
      3. tokenize_and_concatenate each of the three text lists into one
         flat token-id stream per split.
      4. Wrap each stream in a HindiDataset(token_ids, seq_len, tokenizer).
      5. Wrap each HindiDataset in a DataLoader.

    Returns:
        (train_dataloader, val_dataloader, test_dataloader, tokenizer)
    """
    train_texts = _read_jsonl_texts(config["dataset_paths"]["train"])
    val_texts = _read_jsonl_texts(config["dataset_paths"]["val"])
    test_texts = _read_jsonl_texts(config["dataset_paths"]["test"])

    tokenizer_path = _resolve_repo_path(config["tokenizer_path"])
    if os.path.exists(tokenizer_path):
        tokenizer_train_texts = []  # unused -- get_or_build_tokenizer just loads the existing model
    else:
        tc = get_config()
        tokenizer_train_texts = build_training_sample(
            _resolve_repo_path(config["dataset_paths"]["train"]),
            max_lines=tc["training_sample_size"],
            seed=tc["training_sample_seed"],
        )

    tokenizer = get_or_build_tokenizer(config, tokenizer_train_texts)

    train_ids = tokenize_and_concatenate(train_texts, tokenizer)
    val_ids = tokenize_and_concatenate(val_texts, tokenizer)
    test_ids = tokenize_and_concatenate(test_texts, tokenizer)

    train_ds = HindiDataset(train_ids, config["seq_len"], tokenizer)
    val_ds = HindiDataset(val_ids, config["seq_len"], tokenizer)
    test_ds = HindiDataset(test_ids, config["seq_len"], tokenizer)

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer


def train_model(config):
    """Build the DataLoaders, tokenizer, and DecoderOnlyTransformer for one run.

    Sets config["vocab_size"] from tokenizer.get_piece_size() -- NOT
    get_vocab_size(), which does not exist on SentencePieceProcessor --
    before constructing DecoderOnlyTransformer, satisfying its assertion
    that config["vocab_size"] is not None (src/model/model.py).

    NO TRAINING LOOP IS IMPLEMENTED HERE (see TODO below) -- this is a
    scaffold only. No weight initialization and no generation/inference
    happen here either, both explicitly out of scope.

    Args:
        config: dict, model_config.yaml contents (mutated in place: sets
            config["vocab_size"]).

    Returns:
        (model, train_dataloader, val_dataloader, test_dataloader, tokenizer)
    """
    train_dataloader, val_dataloader, test_dataloader, tokenizer = get_ds(config)

    config["vocab_size"] = tokenizer.get_piece_size()  # NOT get_vocab_size() -- see module docstring

    model = DecoderOnlyTransformer(config)

    # TODO: training loop (not implemented this phase).
    #   optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    #   scheduler = <linear warmup over config["warmup_steps"], then decay>
    #   for epoch in range(config["max_epochs"]):
    #       for batch in train_dataloader:
    #           logits = model(batch["encoded_input"])                      # (B, T, vocab_size)
    #           loss = F.cross_entropy(
    #               logits.view(-1, config["vocab_size"]),
    #               batch["encoded_label"].view(-1),
    #           )
    #           loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
    #       <evaluate on val_dataloader>
    #       <save a resumable checkpoint: model/optimizer/scheduler state,
    #        step count, config -- per CLAUDE.md's hard constraint>

    return model, train_dataloader, val_dataloader, test_dataloader, tokenizer


if __name__ == "__main__":
    # *** WARNING: see module docstring -- this triggers full-corpus
    # tokenization of the real 4.87M-doc train split. Not a smoke test. ***
    config = load_model_config()
    train_model(config)
