"""
Load trained encoders from results/<data>/<model>/ and expose one uniform
interface for downstream scripts (bias audit, debiasing, prediction demo):

    enc = load_encoder("sbert", data="real")
    vecs = enc.embed(list_of_texts)          # (n, d) tensor on CPU
    scores = enc.score(resume_texts, jd_texts)  # cosine similarity, numpy
    enc.threshold                            # val-tuned decision threshold
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import preprocessing
from preprocessing import Vocab
from models import SiameseMLP, SiameseCNN, SiameseBiLSTM, SiameseSBERT

ROOT = Path(__file__).parent.parent
RESULTS_ROOT = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_KEYS = ["mlp", "cnn", "bilstm", "sbert"]
LABELS = {"mlp": "MLP", "cnn": "1D-CNN", "bilstm": "BiLSTM", "sbert": "SBERT"}
TOKEN_BUILDERS = {"mlp": SiameseMLP, "cnn": SiameseCNN, "bilstm": SiameseBiLSTM}


class Encoder:
    def __init__(self, key, model, metrics, vocab=None, max_len=None):
        self.key = key
        self.label = LABELS[key]
        self.model = model.eval()
        self.metrics = metrics
        self.threshold = metrics.get("decision_threshold", 0.5)
        self.vocab = vocab
        self.max_len = max_len

    @torch.no_grad()
    def embed(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            if self.key == "sbert":
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                    v = self.model.encode_texts(chunk, DEVICE)
            else:
                ids = torch.tensor([self.vocab.encode(t, self.max_len) for t in chunk], dtype=torch.long).to(DEVICE)
                v = self.model.encode(ids)
            out.append(v.float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def score_vectors(self, r, j):
        """Fit score from precomputed embeddings, using the model's own
        scorer (cosine or interaction head)."""
        return self.model.score_vectors(r.to(DEVICE), j.to(DEVICE)).float().cpu().numpy()

    def score(self, resume_texts, jd_texts):
        return self.score_vectors(self.embed(resume_texts), self.embed(jd_texts))


def load_metrics(model_dir):
    path = model_dir / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_encoder(key, data="real", model_dir=None):
    model_dir = Path(model_dir) if model_dir else RESULTS_ROOT / data / key
    metrics = load_metrics(model_dir)
    head = metrics.get("head", "cosine")  # runs before the head option were cosine
    if key == "sbert":
        model = SiameseSBERT(model_name=str(model_dir / "backbone"), max_len=metrics.get("max_len", 512), head=head)
        if model.head is not None:
            model.head.load_state_dict(torch.load(model_dir / "head.pt", map_location="cpu"))
        return Encoder(key, model.to(DEVICE), metrics)
    vocab = Vocab.load(RESULTS_ROOT / data / "vocab.json")
    model = TOKEN_BUILDERS[key](len(vocab), head=head)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    return Encoder(key, model.to(DEVICE), metrics, vocab=vocab, max_len=metrics.get("max_len", preprocessing.MAX_LEN))


def available_models(data="real"):
    return [k for k in MODEL_KEYS if (RESULTS_ROOT / data / k / "metrics.json").exists()]


def best_model_key(data="real"):
    """Best encoder by validation AUC (selection never looks at test)."""
    best, best_auc = None, -1.0
    for key in available_models(data):
        auc = load_metrics(RESULTS_ROOT / data / key).get("val_auc", -1.0)
        if auc > best_auc:
            best, best_auc = key, auc
    return best, best_auc
