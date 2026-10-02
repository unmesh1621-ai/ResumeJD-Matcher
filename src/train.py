"""
Shared training/evaluation loop for all four Siamese encoders.

Usage: python src/train.py --model {mlp,cnn,bilstm,sbert}

For each model, saves into results/<model>/:
  - metrics.json            (val accuracy/precision/recall/f1/auc, best epoch, etc.)
  - training_curves.png     (loss + accuracy-proxy per epoch)
  - confusion_matrix.png
  - roc_curve.png
"""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

LABEL_THRESHOLD = 0.5  # binarize graded similarity label -> match / no-match
MAX_EPOCHS = 35
PATIENCE = 5
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pairs():
    rows = {"train": [], "val": [], "test": []}
    with open(DATA_DIR / "pairs.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["similarity_label"] = float(row["similarity_label"])
            rows[row["split"]].append(row)
    return rows


def batches(rows, batch_size, shuffle=True):
    idx = list(range(len(rows)))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield [rows[i] for i in batch_idx]


def encode_token_batch(batch, vocab):
    resume_ids = torch.tensor([vocab.encode(r["resume_text"]) for r in batch], dtype=torch.long)
    jd_ids = torch.tensor([vocab.encode(r["jd_text"]) for r in batch], dtype=torch.long)
    labels = torch.tensor([r["similarity_label"] for r in batch], dtype=torch.float32)
    return resume_ids.to(DEVICE), jd_ids.to(DEVICE), labels.to(DEVICE)


def run_epoch_token_model(model, rows, optimizer, vocab, train=True):
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    all_preds, all_labels = [], []
    loss_fn = nn.MSELoss()
    with torch.set_grad_enabled(train):
        for batch in batches(rows, BATCH_SIZE, shuffle=train):
            resume_ids, jd_ids, labels = encode_token_batch(batch, vocab)
            preds = model(resume_ids, jd_ids)
            loss = loss_fn(preds, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    return total_loss / max(n_batches, 1), np.array(all_preds), np.array(all_labels)


def run_epoch_sbert(model, rows, optimizer, train=True):
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    all_preds, all_labels = [], []
    loss_fn = nn.MSELoss()
    with torch.set_grad_enabled(train):
        for batch in batches(rows, 16, shuffle=train):  # smaller batch, transformer is heavier
            resume_texts = [r["resume_text"] for r in batch]
            jd_texts = [r["jd_text"] for r in batch]
            labels = torch.tensor([r["similarity_label"] for r in batch], dtype=torch.float32).to(DEVICE)
            preds = model(resume_texts, jd_texts, DEVICE)
            loss = loss_fn(preds, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    return total_loss / max(n_batches, 1), np.array(all_preds), np.array(all_labels)


def binary_metrics(preds, labels):
    bin_labels = (labels >= LABEL_THRESHOLD).astype(int)
    bin_preds = (preds >= LABEL_THRESHOLD).astype(int)
    acc = accuracy_score(bin_labels, bin_preds)
    prec = precision_score(bin_labels, bin_preds, zero_division=0)
    rec = recall_score(bin_labels, bin_preds, zero_division=0)
    f1 = f1_score(bin_labels, bin_preds, zero_division=0)
    try:
        auc = roc_auc_score(bin_labels, preds)
    except ValueError:
        auc = float("nan")
    return acc, prec, rec, f1, auc, bin_labels, bin_preds


def save_training_curves(out_dir, model_name, train_losses, val_losses, val_accs, best_epoch):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="Train loss", color="tab:blue")
    axes[0].plot(epochs, val_losses, label="Val loss", color="tab:red")
    axes[0].axvline(best_epoch, color="green", linestyle="--", label=f"Best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title(f"{model_name} — Loss per Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_accs, label="Val accuracy (binarized)", color="tab:red", marker="o")
    axes[1].axvline(best_epoch, color="green", linestyle="--", label=f"Best epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} — Accuracy per Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)


def save_confusion_and_roc(out_dir, model_name, bin_labels, bin_preds, preds, auc):
    cm = confusion_matrix(bin_labels, bin_preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred No-Match", "Pred Match"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No-Match", "Actual Match"])
    ax.set_title(f"{model_name} — Confusion Matrix (Val)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(bin_labels, preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve (Val)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=130)
    plt.close(fig)


def train_token_model(model_key, model_label, build_model, lr):
    from preprocessing import Vocab

    rows = load_pairs()
    vocab = Vocab.build([r["resume_text"] for r in rows["train"]] + [r["jd_text"] for r in rows["train"]])
    vocab.save(DATA_DIR / "vocab.json")

    model = build_model(len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improve = 0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _ = run_epoch_token_model(model, rows["train"], optimizer, vocab, train=True)
        val_loss, val_preds, val_labels = run_epoch_token_model(model, rows["val"], optimizer, vocab, train=False)
        acc, prec, rec, f1, auc, bin_labels, bin_preds = binary_metrics(val_preds, val_labels)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(acc)
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-5
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = (acc, prec, rec, f1, auc, bin_labels, bin_preds, val_preds)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        print(f"[{model_label}] epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={acc:.4f}")

        if epochs_without_improve >= PATIENCE:
            print(f"[{model_label}] early stopping at epoch {epoch}, best epoch {best_epoch}")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    acc, prec, rec, f1, auc, bin_labels, bin_preds, val_preds = best_metrics

    save_training_curves(out_dir, model_label, train_losses, val_losses, val_accs, best_epoch)
    save_confusion_and_roc(out_dir, model_label, bin_labels, bin_preds, val_preds, auc)

    metrics = {
        "model": model_label,
        "trainable_parameters": n_params,
        "epochs_run": len(train_losses),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_accuracy": acc,
        "val_precision": prec,
        "val_recall": rec,
        "val_f1": f1,
        "val_auc": auc,
        "train_seconds": elapsed,
        "device": str(DEVICE),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(json.dumps(metrics, indent=2))


def train_sbert_model():
    from models import SiameseSBERT

    rows = load_pairs()
    model = SiameseSBERT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    out_dir = RESULTS_DIR / "sbert"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improve = 0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    MAX_SBERT_EPOCHS = 8
    SBERT_PATIENCE = 2

    t0 = time.time()
    for epoch in range(1, MAX_SBERT_EPOCHS + 1):
        train_loss, _, _ = run_epoch_sbert(model, rows["train"], optimizer, train=True)
        val_loss, val_preds, val_labels = run_epoch_sbert(model, rows["val"], optimizer, train=False)
        acc, prec, rec, f1, auc, bin_labels, bin_preds = binary_metrics(val_preds, val_labels)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(acc)

        improved = val_loss < best_val_loss - 1e-5
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = (acc, prec, rec, f1, auc, bin_labels, bin_preds, val_preds)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        print(f"[SBERT] epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={acc:.4f}")

        if epochs_without_improve >= SBERT_PATIENCE:
            print(f"[SBERT] early stopping at epoch {epoch}, best epoch {best_epoch}")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    acc, prec, rec, f1, auc, bin_labels, bin_preds, val_preds = best_metrics

    save_training_curves(out_dir, "SBERT (MiniLM, fine-tuned)", train_losses, val_losses, val_accs, best_epoch)
    save_confusion_and_roc(out_dir, "SBERT (MiniLM, fine-tuned)", bin_labels, bin_preds, val_preds, auc)

    metrics = {
        "model": "SBERT (all-MiniLM-L6-v2, fine-tuned)",
        "trainable_parameters": n_params,
        "epochs_run": len(train_losses),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_accuracy": acc,
        "val_precision": prec,
        "val_recall": rec,
        "val_f1": f1,
        "val_auc": auc,
        "train_seconds": elapsed,
        "device": str(DEVICE),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.backbone.save_pretrained(out_dir / "backbone")
    model.tokenizer.save_pretrained(out_dir / "backbone")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp", "cnn", "bilstm", "sbert"])
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from models import SiameseMLP, SiameseCNN, SiameseBiLSTM

    if args.model == "mlp":
        train_token_model("mlp", "MLP", lambda vs: SiameseMLP(vs), lr=1e-3)
    elif args.model == "cnn":
        train_token_model("cnn", "1D-CNN", lambda vs: SiameseCNN(vs), lr=1e-3)
    elif args.model == "bilstm":
        train_token_model("bilstm", "BiLSTM", lambda vs: SiameseBiLSTM(vs), lr=1e-3)
    elif args.model == "sbert":
        train_sbert_model()


if __name__ == "__main__":
    main()
