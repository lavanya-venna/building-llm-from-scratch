"""Production Hindi tokenizer wrapper for model training and inference.

`tokenizer_test.py` sweeps vocab sizes/types on a *sample* of the corpus to
compare candidates cheaply. This module is the other half: a small class
that either loads an already-trained SentencePiece model, or trains one
from scratch on the *entire* cleaned corpus, and exposes the handful of
operations (encode/decode/vocab size) the rest of the model code needs
without every caller having to know SentencePiece's own API.
"""
import json
import os

import sentencepiece as spm

from src.tokenizer.tokenizer_config import get_config
from src.tokenizer.tokenizer_test import train_sentencepiece_model


class HindiTokenizer:
    """Loads an existing SentencePiece model, or trains one on the full corpus if missing."""

    def __init__(self, model_path=None, config=None):
        """Load `model_path` if it already exists, else train a fresh model from `config`.

        Why a class instead of bare functions: every caller downstream
        (dataset building, the training loop, inference) just wants "a
        tokenizer object with encode/decode on it" and shouldn't have to
        care whether that meant loading a file or training one first --
        this constructor is where that decision gets made once.

        Args: model_path (str, path to a SentencePiece .model file; may not
        exist yet), config (dict, see tokenizer_config.get_config() --
        required if model_path doesn't exist yet, since training needs it).
        Returns: nothing (constructor). Raises ValueError if model_path is
        missing/unusable and no config was given to train a new one from.
        """
        if model_path and os.path.exists(model_path):
            # Common case in production: the tokenizer was already trained in
            # a previous run, so we just load the artifact -- no need to touch
            # the corpus or config at all.
            resolved_path = model_path
        elif model_path and config is not None:
            # model_path names where the tokenizer *should* live but doesn't
            # yet -- train it there now. train() is the source of truth for
            # where it actually ends up (see its docstring), so we load from
            # its return value rather than assuming it matches model_path.
            resolved_path = self.train(config)
        else:
            # Neither an existing model nor enough information to train one --
            # failing loudly here is much easier to debug than a confusing
            # error from deep inside SentencePiece later.
            raise ValueError(
                "HindiTokenizer needs either an existing model_path to load, "
                "or a model_path plus a config to train a new one from."
            )

        self.model_path = resolved_path
        self.sp = spm.SentencePieceProcessor(model_file=resolved_path)

    def train(self, config):
        """Train a new SentencePiece model on the entire corpus and save it.

        Args: config (dict, see tokenizer_config.get_config()).
        Returns: str, the path of the trained `.model` file (== config["tokenizer_path"]).
        """

        def _iter_corpus_texts(train_split_path):
            # Generator, not a list: avoids holding the whole (multi-GB) corpus
            # in memory just to hand it to SentencePieceTrainer once.
            with open(train_split_path, encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)["text"].replace("\n", " ")

        tokenizer_path = config["tokenizer_path"]
        # SentencePieceTrainer wants a model_prefix (no ".model" suffix) --
        # it appends ".model"/".vocab" itself.
        model_prefix = (
            tokenizer_path[: -len(".model")] if tokenizer_path.endswith(".model") else tokenizer_path
        )
        os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

        train_sentencepiece_model(
            sentences=_iter_corpus_texts(config["train_split_path"]),
            model_prefix=model_prefix,
            vocab_size=config["vocab_size"],
            model_type=config["model_type"],
            character_coverage=config["character_coverage"],
            byte_fallback=config["byte_fallback"],
        )

        return model_prefix + ".model"

    def encode(self, text):
        """Encode `text` into SentencePiece token ids.

        Args: text (str).
        Returns: list[int].
        """
        return self.sp.encode(text, out_type=int)

    def encode_as_pieces(self, text):
        """Encode `text` into SentencePiece string pieces (for debugging/inspection).
        Args: text (str).
        Returns: list[str].
        """
        return self.sp.encode(text, out_type=str)

    def decode(self, ids):
        """Decode a list of token ids back into the original string.

        Args: ids (list[int]).
        Returns: str.
        """
        return self.sp.decode(ids)

    @property
    def vocab_size(self):
        """The loaded model's actual vocabulary size.
        Returns: int.
        """
        return self.sp.get_piece_size()


if __name__ == "__main__":
    # Trains on the *entire* corpus if config["tokenizer_path"] doesn't exist
    # yet, or just loads it if it does -- see HindiTokenizer.__init__.
    config = get_config()
    tok = HindiTokenizer(model_path=config["tokenizer_path"], config=config)
    print(f"trained: {tok.model_path}")
    print(f"vocab_size: {tok.vocab_size}")
