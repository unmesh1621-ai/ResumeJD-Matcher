"""
Shared tokenizer/vocab for the three from-scratch models (MLP, 1D-CNN,
BiLSTM). Built once on the training split only, then reused unchanged by
all three so they see identical integer-encoded input (same idea as the
sample report's shared `tokenizer.pkl`).

The SBERT model does NOT use this module; it uses its own pretrained
WordPiece/BPE tokenizer via sentence-transformers.
"""
import json
import re
from pathlib import Path

VOCAB_SIZE = 5000
MAX_LEN = 128
OOV_TOKEN = "<OOV>"
PAD_TOKEN = "<PAD>"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.pad_id = token_to_id[PAD_TOKEN]
        self.oov_id = token_to_id[OOV_TOKEN]

    @classmethod
    def build(cls, texts, vocab_size=VOCAB_SIZE):
        counts = {}
        for text in texts:
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
        most_common = sorted(counts.items(), key=lambda kv: -kv[1])[: vocab_size - 2]
        token_to_id = {PAD_TOKEN: 0, OOV_TOKEN: 1}
        for tok, _ in most_common:
            token_to_id[tok] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text, max_len=MAX_LEN):
        ids = [self.token_to_id.get(tok, self.oov_id) for tok in tokenize(text)]
        ids = ids[:max_len]
        ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def save(self, path):
        Path(path).write_text(json.dumps(self.token_to_id), encoding="utf-8")

    @classmethod
    def load(cls, path):
        token_to_id = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(token_to_id)

    def __len__(self):
        return len(self.token_to_id)
