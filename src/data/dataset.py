"""Fixed-length causal-LM dataset for the Hindi decoder-only Transformer LM.

Implements "concatenate and chop": src/model/train.py's
tokenize_and_concatenate tokenizes every document in a split, appends the
SentencePiece <eos> id after each doc, and concatenates all of that into one
flat 1-D list[int] of token ids per split. HindiDataset slices that flat
stream into fixed-length, non-overlapping seq_len chunks for the model's
input and a 1-position-shifted seq_len chunk for the causal-LM target. There
is no padding and no per-document boundary alignment -- a chunk may span the
end of one document and the start of the next; the <eos> id between docs is
the only signal a boundary occurred, and it decodes to nothing visible.

Raw text -> token ids -> chunks -> batches
-------------------------------------------
1. Each doc's text -> list[int] via the from-scratch SentencePiece model,
   plus a single <eos> id appended (tokenize_and_concatenate, not in this
   file).
2. All docs' id lists concatenated end-to-end -> one flat list[int],
   N tokens long, for the whole split.
3. HindiDataset exposes (N - 1) // seq_len fixed-length windows: index idx
   covers token_ids[idx*seq_len : idx*seq_len+seq_len] as input, the same
   window shifted right by one position as target.
4. DataLoader(dataset, batch_size=B) stacks B windows into (B, seq_len)
   tensors for encoded_input/encoded_label (PyTorch's default collate);
   raw_input/raw_label (str) collate into a list[str] of length B.
"""
import torch
from torch.utils.data import Dataset


class HindiDataset(Dataset):
    """One fixed-length causal-LM example per index, over a flat token-id stream.

    Given a single flat stream of token ids for one split (already
    tokenized and <eos>-joined by tokenize_and_concatenate) and a fixed
    sequence length, exposes seq_len-long (input, target) chunks where
    target is input shifted right by one token -- standard next-token-
    prediction framing for a decoder-only LM.
    """

    def __init__(self, token_ids, seq_len, tokenizer):
        """
        Args:
            token_ids: flat list[int] (or any sequence supporting slicing)
                of token ids for one split, already concatenated across all
                documents with <eos> ids inserted between them. Length N.
            seq_len: int, tokens per example (matches config["seq_len"] in
                config/model_config.yaml, e.g. 256).
            tokenizer: a loaded sentencepiece.SentencePieceProcessor, used
                only for decoding chunks back to raw text for inspection
                (.decode(list[int]) -> str). It requires a plain list[int],
                not a torch.Tensor -- see __getitem__.
        """
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        """Number of non-overlapping seq_len chunks available.

        Each example needs seq_len+1 ids (seq_len for input, +1 for the
        shifted target), so usable start indices run 0..len(token_ids)-seq_len-1
        in steps of seq_len. Any partial remainder at the tail is dropped,
        not zero-padded (this tokenizer's model has pad_id() == -1 -- padding
        is disabled -- so "concatenate and chop" deliberately avoids ever
        needing a pad token).

        Example: len(token_ids) == 1000, seq_len == 256 ->
                 (1000 - 1) // 256 == 3 usable chunks.
        """
        return (len(self.token_ids) - 1) // self.seq_len

    def __getitem__(self, idx):
        """Return one (input, target) pair plus their decoded raw text.

        Shapes: encoded_input, encoded_label are both 1-D LongTensors of
        shape (seq_len,). DataLoader's default collate stacks B of these
        into (B, seq_len) batches, matching DecoderOnlyTransformer.forward's
        expected (B, T) input (see src/model/model.py).

        Example (seq_len=4, token_ids=[10,11,12,13,14,15,...], idx=0):
            encoded_input = [10,11,12,13]
            encoded_label = [11,12,13,14]   (shifted right by 1)
        """
        start = idx * self.seq_len  # first token-id index for this chunk

        # Slice plain python ints first: SentencePiece's decode() requires a
        # list[int] (or similar), not a torch.Tensor -- calling .decode() on
        # a tensor raises TypeError.
        input_ids = self.token_ids[start : start + self.seq_len]  # list[int], len seq_len
        label_ids = self.token_ids[start + 1 : start + self.seq_len + 1]  # list[int], len seq_len

        encoded_input = torch.tensor(input_ids, dtype=torch.long)  # (seq_len,)
        encoded_label = torch.tensor(label_ids, dtype=torch.long)  # (seq_len,)

        raw_input = self.tokenizer.decode(input_ids)  # str; any embedded <eos> decodes to nothing visible
        raw_label = self.tokenizer.decode(label_ids)  # str

        return {
            "encoded_input": encoded_input,  # (seq_len,) LongTensor
            "encoded_label": encoded_label,  # (seq_len,) LongTensor
            "raw_input": raw_input,  # str
            "raw_label": raw_label,  # str
        }
