"""
Bias-detection probing layer (synopsis Objective 4).

Takes the best-performing matching encoder (chosen by val AUC across
results/*/metrics.json), computes resume embeddings for the held-out test
split, and trains a logistic-regression probe to predict a demographic
PROXY group (derived at data-generation time from first name, stored in
gender_proxy.csv, and never shown to the matching models) purely from the
resume embedding. High probe accuracy (clearly above the majority-class
baseline) indicates the embedding space leaks demographic-proxy
information even though gender was never an explicit input to matching.

Adversarial debiasing (training the encoder further to reduce this
leakage) is intentionally NOT run here -- see report "Remaining Work".
"""
import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocessing import Vocab
from models import SiameseMLP, SiameseCNN, SiameseBiLSTM, SiameseSBERT

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

MODEL_BUILDERS = {
    "mlp": lambda vs: SiameseMLP(vs),
    "cnn": lambda vs: SiameseCNN(vs),
    "bilstm": lambda vs: SiameseBiLSTM(vs),
}


def pick_best_model():
    best_key, best_auc = None, -1.0
    for key in ["mlp", "cnn", "bilstm", "sbert"]:
        metrics_path = RESULTS_DIR / key / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        auc = metrics.get("val_auc", -1.0)
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_key = key
    return best_key, best_auc


def load_resume_lookup():
    rows = {}
    with open(DATA_DIR / "pairs.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["resume_id"]] = row["resume_text"]
    return rows


def load_proxy_labels():
    labels = {}
    with open(DATA_DIR / "gender_proxy.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["resume_id"]] = row["proxy_group"]
    return labels


def embed_token_model(model_key, resume_ids, resume_texts):
    vocab = Vocab.load(DATA_DIR / "vocab.json")
    model = MODEL_BUILDERS[model_key](len(vocab))
    model.load_state_dict(torch.load(RESULTS_DIR / model_key / "model.pt"))
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([vocab.encode(t) for t in resume_texts], dtype=torch.long)
        emb = model.encode(ids)
    return emb.numpy()


def embed_sbert_model(resume_texts):
    backbone_dir = RESULTS_DIR / "sbert" / "backbone"
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone_dir)
    backbone = AutoModel.from_pretrained(backbone_dir)
    backbone.eval()
    model = SiameseSBERT.__new__(SiameseSBERT)
    torch.nn.Module.__init__(model)
    model.tokenizer = tokenizer
    model.backbone = backbone
    with torch.no_grad():
        emb = model.encode_texts(resume_texts, torch.device("cpu"))
    return emb.numpy()


def main():
    best_key, best_auc = pick_best_model()
    if best_key is None:
        raise SystemExit("No trained model metrics found in results/. Run src/train.py first.")
    print(f"Best-performing encoder by val AUC: {best_key} (AUC={best_auc:.4f})")

    resume_text_by_id = load_resume_lookup()
    proxy_by_id = load_proxy_labels()
    resume_ids = sorted(resume_text_by_id.keys())
    resume_texts = [resume_text_by_id[rid] for rid in resume_ids]
    proxy_labels = [proxy_by_id[rid] for rid in resume_ids]

    if best_key == "sbert":
        embeddings = embed_sbert_model(resume_texts)
    else:
        embeddings = embed_token_model(best_key, resume_ids, resume_texts)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, proxy_labels, test_size=0.3, random_state=42, stratify=proxy_labels
    )

    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    probe_acc = accuracy_score(y_test, probe.predict(X_test))

    majority_class = max(set(y_train), key=y_train.count)
    majority_baseline_acc = accuracy_score(y_test, [majority_class] * len(y_test))

    report = {
        "encoder_used": best_key,
        "encoder_val_auc": best_auc,
        "n_resumes": len(resume_ids),
        "probe_test_accuracy": probe_acc,
        "majority_class_baseline_accuracy": majority_baseline_acc,
        "leakage_gap": probe_acc - majority_baseline_acc,
        "interpretation": (
            "Probe accuracy clearly above the majority-class baseline indicates "
            "the matching encoder's embedding space leaks demographic-proxy "
            "(name-derived) information, even though proxy group was never an "
            "explicit input to the matching model."
            if probe_acc - majority_baseline_acc > 0.05
            else "Probe accuracy is close to the majority-class baseline; no "
                 "strong demographic-proxy leakage detected in this run."
        ),
    }
    out_path = RESULTS_DIR / "bias_probe_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
