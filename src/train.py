"""
Shared training/evaluation loop for all four Siamese encoders.

Usage:
  python src/train.py --model {mlp,cnn,bilstm,sbert} [--data real|synthetic] [--seed 42]

Common protocol for every model (guidelines Sec. 5):
  - same train / val / test split (data/pairs_real.csv or data/pairs.csv)
  - same objective: MSE between cosine(resume_vec, jd_vec) and the graded label
  - checkpoint selected on validation AUC
  - decision threshold chosen on validation (max accuracy), then applied
    unchanged to the held-out test split
  - same test metrics: accuracy, precision, recall, F1, ROC-AUC,
    Precision@k (ranking resumes per JD) and Spearman rho vs. graded label

Outputs, for the main seed (42), into results/<data>/<model>/:
  metrics.json, test_predictions.csv, training_curves.png,
  confusion_matrix.png, roc_curve.png, model.pt (or backbone/ for SBERT)
Extra seeds only write results/<data>/<model>/seeds/seed_<n>.json.
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
import preprocessing
from preprocessing import Vocab

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_ROOT = ROOT / "results"

LABEL_THRESHOLD = 0.5  # binarize graded similarity label -> match / no-match
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN_SEED = 42

# Per-dataset settings. Real resumes have a median of ~620 words, so the
# scratch models see 512 tokens and use a larger vocabulary.
DATA_CONFIG = {
    "synthetic": {"pairs": "pairs.csv", "max_len": 128, "vocab_size": 5000,
                  "max_epochs": 60, "patience": 15, "sbert_max_len": 128},
    "real": {"pairs": "pairs_real.csv", "max_len": 512, "vocab_size": 20000,
             "max_epochs": 40, "patience": 8, "sbert_max_len": 512},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pairs(data="real"):
    rows = {"train": [], "val": [], "test": []}
    with open(DATA_DIR / DATA_CONFIG[data]["pairs"], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["similarity_label"] = float(row["similarity_label"])
            rows[row["split"]].append(row)
    return rows


def batches(rows, batch_size, shuffle=True):
    idx = list(range(len(rows)))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start:start + batch_size]


def encode_split(rows, vocab):
    """Integer-encode a split once up front (re-tokenising ~600-word
    documents every epoch would dominate training time)."""
    resume = torch.tensor([vocab.encode(r["resume_text"]) for r in rows], dtype=torch.long)
    jd = torch.tensor([vocab.encode(r["jd_text"]) for r in rows], dtype=torch.long)
    labels = torch.tensor([r["similarity_label"] for r in rows], dtype=torch.float32)
    return resume, jd, labels


def run_epoch_token_model(model, encoded, optimizer, train=True):
    resume_all, jd_all, labels_all = encoded
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []
    loss_fn = nn.MSELoss()
    with torch.set_grad_enabled(train):
        for idx in batches(range(len(labels_all)), BATCH_SIZE, shuffle=train):
            resume_ids = resume_all[idx].to(DEVICE)
            jd_ids = jd_all[idx].to(DEVICE)
            labels = labels_all[idx].to(DEVICE)
            preds = model(resume_ids, jd_ids)
            loss = loss_fn(preds, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(preds.detach().float().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / max(n_batches, 1), np.array(all_preds), np.array(all_labels)


def run_epoch_sbert(model, rows, optimizer, scaler, train=True, batch_size=16):
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []
    loss_fn = nn.MSELoss()
    use_amp = DEVICE.type == "cuda"
    with torch.set_grad_enabled(train):
        for idx in batches(rows, batch_size, shuffle=train):
            batch = [rows[i] for i in idx]
            resume_texts = [r["resume_text"] for r in batch]
            jd_texts = [r["jd_text"] for r in batch]
            labels = torch.tensor([r["similarity_label"] for r in batch], dtype=torch.float32).to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                preds = model(resume_texts, jd_texts, DEVICE)
            preds = preds.float()
            loss = loss_fn(preds, labels)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / max(n_batches, 1), np.array(all_preds), np.array(all_labels)


def binary_metrics(preds, labels, threshold=LABEL_THRESHOLD):
    bin_labels = (labels >= LABEL_THRESHOLD).astype(int)
    bin_preds = (preds >= threshold).astype(int)
    acc = accuracy_score(bin_labels, bin_preds)
    prec = precision_score(bin_labels, bin_preds, zero_division=0)
    rec = recall_score(bin_labels, bin_preds, zero_division=0)
    f1 = f1_score(bin_labels, bin_preds, zero_division=0)
    try:
        auc = roc_auc_score(bin_labels, preds)
    except ValueError:
        auc = float("nan")
    return acc, prec, rec, f1, auc, bin_labels, bin_preds


def tune_threshold(preds, labels):
    """Pick the cosine threshold that maximises validation accuracy. The
    models are trained by regression, so their raw cosine scale differs
    (e.g. a model may score every pair in 0.3-0.8); a val-tuned cut-off
    makes the binary metrics comparable across architectures."""
    bin_labels = (labels >= LABEL_THRESHOLD).astype(int)
    best_t, best_acc = LABEL_THRESHOLD, -1.0
    for t in np.unique(np.round(preds, 3)):
        acc = accuracy_score(bin_labels, (preds >= t).astype(int))
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t


def precision_at_k(rows, preds, ks=(1, 3, 5)):
    """For each JD, rank its candidate resumes by predicted score and measure
    the fraction of the top-k that are true matches. Averaged over JDs that
    have at least k candidates."""
    by_jd = {}
    for row, p in zip(rows, preds):
        by_jd.setdefault(row["jd_id"], []).append((p, row["similarity_label"] >= LABEL_THRESHOLD))
    out = {}
    for k in ks:
        scores, random_baseline = [], []
        for cands in by_jd.values():
            if len(cands) < k:
                continue
            top = sorted(cands, key=lambda c: -c[0])[:k]
            scores.append(sum(m for _, m in top) / k)
            # expected P@k of a random ranking = this JD's share of true matches
            random_baseline.append(sum(m for _, m in cands) / len(cands))
        out[f"p_at_{k}"] = float(np.mean(scores)) if scores else float("nan")
        out[f"p_at_{k}_random"] = float(np.mean(random_baseline)) if scores else float("nan")
        out[f"p_at_{k}_n_jds"] = len(scores)
    return out


def within_resume_auc(rows, preds):
    """Mean ROC-AUC computed separately for each resume over the JDs it is
    paired with (only resumes that have both a matching and a non-matching
    JD). A model that scores resumes by a JD-independent 'resume quality'
    prior gets exactly 0.5 here, so this isolates genuine resume<->JD
    matching ability."""
    by_resume = {}
    for row, p in zip(rows, preds):
        by_resume.setdefault(row["resume_id"], ([], []))
        by_resume[row["resume_id"]][0].append(int(row["similarity_label"] >= LABEL_THRESHOLD))
        by_resume[row["resume_id"]][1].append(p)
    aucs = [roc_auc_score(y, s) for y, s in by_resume.values() if 0 < sum(y) < len(y)]
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def test_metrics(rows, preds, labels, threshold):
    acc, prec, rec, f1, auc, bin_labels, bin_preds = binary_metrics(preds, labels, threshold)
    rho = spearmanr(preds, labels).correlation
    wr_auc, wr_n = within_resume_auc(rows, preds)
    out = {
        "test_accuracy": acc, "test_precision": prec, "test_recall": rec,
        "test_f1": f1, "test_auc": auc, "test_spearman": float(rho),
        "test_within_resume_auc": wr_auc, "test_within_resume_auc_n_resumes": wr_n,
        "test_mse": float(np.mean((preds - labels) ** 2)),
    }
    out.update({f"test_{k}": v for k, v in precision_at_k(rows, preds).items()})
    return out, bin_labels, bin_preds


def save_training_curves(out_dir, model_name, train_losses, val_losses, val_aucs, best_epoch):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="Train loss", color="tab:blue")
    axes[0].plot(epochs, val_losses, label="Val loss", color="tab:red")
    axes[0].axvline(best_epoch, color="green", linestyle="--", label=f"Best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title(f"{model_name} — Loss per Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_aucs, label="Val ROC-AUC", color="tab:red", marker="o")
    axes[1].axvline(best_epoch, color="green", linestyle="--", label=f"Best epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title(f"{model_name} — Validation AUC per Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)


def save_confusion_and_roc(out_dir, model_name, bin_labels, bin_preds, preds, auc, split="Test"):
    cm = confusion_matrix(bin_labels, bin_preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred No-Match", "Pred Match"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No-Match", "Actual Match"])
    ax.set_title(f"{model_name} — Confusion ({split})")
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
    ax.set_title(f"{model_name} — ROC ({split})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=130)
    plt.close(fig)


def save_test_predictions(out_dir, rows, preds, threshold):
    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["resume_id", "jd_id", "fit_class", "similarity_label", "pred_score", "pred_match"])
        for row, p in zip(rows, preds):
            w.writerow([row["resume_id"], row["jd_id"], row["resume_category"],
                        row["similarity_label"], round(float(p), 5), int(p >= threshold)])


def fit_loop(model_label, train_step, eval_step, max_epochs, patience, scheduler=None):
    """Generic epoch loop with val-AUC model selection + early stopping.
    Returns history and the best state dict / val predictions."""
    train_losses, val_losses, val_aucs = [], [], []
    best = {"auc": -1.0, "epoch": 0, "state": None, "val_preds": None, "val_loss": None}
    epochs_without_improve = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = train_step()
        val_loss, val_preds, val_labels, state_fn = eval_step()
        _, _, _, _, auc, _, _ = binary_metrics(val_preds, val_labels)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(auc)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Model selection tracks val AUC rather than val MSE loss: the two can
        # diverge (a model can become more polarised, raising MSE on
        # borderline pairs while ranking pairs better), and ranking quality is
        # what we report.
        if auc > best["auc"] + 1e-5:
            best.update(auc=auc, epoch=epoch, state=state_fn(), val_preds=val_preds,
                        val_labels=val_labels, val_loss=val_loss)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        print(f"[{model_label}] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_auc={auc:.4f}", flush=True)
        if epochs_without_improve >= patience:
            print(f"[{model_label}] early stopping at epoch {epoch}, best epoch {best['epoch']}")
            break
    return train_losses, val_losses, val_aucs, best


def finalize(out_dir, model_key, model_label, rows, history, best, test_preds, test_labels,
             n_params, elapsed, seed, extra=None):
    train_losses, val_losses, val_aucs = history
    threshold = tune_threshold(best["val_preds"], best["val_labels"])
    val_acc, val_prec, val_rec, val_f1, val_auc, _, _ = binary_metrics(
        best["val_preds"], best["val_labels"], threshold)
    tm, bin_labels, bin_preds = test_metrics(rows["test"], test_preds, test_labels, threshold)

    metrics = {
        "model": model_label,
        "seed": seed,
        "trainable_parameters": n_params,
        "epochs_run": len(train_losses),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "decision_threshold": threshold,
        "val_accuracy": val_acc, "val_precision": val_prec, "val_recall": val_rec,
        "val_f1": val_f1, "val_auc": val_auc,
        "val_within_resume_auc": within_resume_auc(rows["val"], best["val_preds"])[0],
        **tm,
        "train_seconds": elapsed,
        "device": str(DEVICE),
        **(extra or {}),
    }

    if seed == MAIN_SEED:
        save_training_curves(out_dir, model_label, train_losses, val_losses, val_aucs, best["epoch"])
        save_confusion_and_roc(out_dir, model_label, bin_labels, bin_preds, test_preds, tm["test_auc"])
        save_test_predictions(out_dir, rows["test"], test_preds, threshold)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    seeds_dir = out_dir / "seeds"
    seeds_dir.mkdir(exist_ok=True)
    (seeds_dir / f"seed_{seed}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def train_token_model(model_key, model_label, build_model, lr, data, seed, head, out_name,
                      warm_start=True):
    cfg = DATA_CONFIG[data]
    preprocessing.MAX_LEN = cfg["max_len"]
    set_seed(seed)

    rows = load_pairs(data)
    results_dir = RESULTS_ROOT / data
    vocab_path = results_dir / "vocab.json"
    vocab = Vocab.build([r["resume_text"] for r in rows["train"]] + [r["jd_text"] for r in rows["train"]],
                        vocab_size=cfg["vocab_size"])
    results_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)

    encoded = {split: encode_split(rows[split], vocab) for split in ("train", "val", "test")}
    model = build_model(len(vocab)).to(DEVICE)

    # Warm-start the embedding layer from the trained MLP's embeddings
    # instead of random init. The CNN and BiLSTM encoders are prone to
    # "Siamese collapse" (predicting a near-constant similarity regardless
    # of input) when trained from scratch via plain MSE regression; MLP's
    # mean-pooling encoder reliably avoids this and learns a solid token
    # embedding space, so reusing it gives the other encoders a better start.
    pretrained_embed_path = results_dir / "mlp" / "embedding.pt"
    warm_started = False
    if warm_start and model_key != "mlp" and pretrained_embed_path.exists():
        pretrained = torch.load(pretrained_embed_path)
        if pretrained.shape == model.embedding.weight.shape:
            model.embedding.weight.data.copy_(pretrained)
            model.embedding.weight.requires_grad = False
            warm_started = True
            print(f"[{model_label}] warm-started + froze embeddings from trained MLP")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    out_dir = results_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_step():
        loss, _, _ = run_epoch_token_model(model, encoded["train"], optimizer, train=True)
        return loss

    def eval_step():
        loss, preds, labels = run_epoch_token_model(model, encoded["val"], optimizer, train=False)
        return loss, preds, labels, lambda: {k: v.clone() for k, v in model.state_dict().items()}

    t0 = time.time()
    tl, vl, va, best = fit_loop(model_label, train_step, eval_step,
                                cfg["max_epochs"], cfg["patience"], scheduler)
    elapsed = time.time() - t0

    model.load_state_dict(best["state"])
    _, test_preds, test_labels = run_epoch_token_model(model, encoded["test"], optimizer, train=False)

    metrics = finalize(out_dir, model_key, model_label, rows, (tl, vl, va), best,
                       test_preds, test_labels, n_params, elapsed, seed,
                       extra={"head": head, "max_len": cfg["max_len"], "vocab_size": len(vocab),
                              "embedding_warm_started_from_mlp": warm_started})
    if seed == MAIN_SEED:
        torch.save(model.state_dict(), out_dir / "model.pt")
        if model_key == "mlp":
            torch.save(model.embedding.weight.data.clone().cpu(), out_dir / "embedding.pt")
    return metrics


def train_sbert_model(data, seed, head, out_name, backbone="minilm", epochs=None, lr=2e-5):
    from models import SiameseSBERT, SBERT_BACKBONES

    cfg = DATA_CONFIG[data]
    set_seed(seed)
    rows = load_pairs(data)
    max_len, batch_size = cfg["sbert_max_len"], 16
    if backbone == "mpnet":
        # all-mpnet-base-v2 was trained with max_seq_length 384; at that length
        # and batch 8 the 12-layer 768-d model fits an 8 GB GPU (transformers'
        # MPNet has no gradient-checkpointing support)
        max_len, batch_size = min(max_len, 384), 8
    model = SiameseSBERT(SBERT_BACKBONES[backbone], max_len=max_len, head=head).to(DEVICE)
    # the interaction head starts from random init, so it gets a larger lr
    head_params = list(model.head.parameters()) if model.head is not None else []
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr, "weight_decay": 0.01},
        {"params": head_params, "lr": 1e-3, "weight_decay": 0.01},
    ])
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    out_dir = RESULTS_ROOT / data / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label = f"SBERT ({'MPNet' if backbone == 'mpnet' else 'MiniLM'}, {head})"

    def train_step():
        loss, _, _ = run_epoch_sbert(model, rows["train"], optimizer, scaler, train=True,
                                     batch_size=batch_size)
        return loss

    def eval_step():
        loss, preds, labels = run_epoch_sbert(model, rows["val"], optimizer, scaler,
                                              train=False, batch_size=32)
        return loss, preds, labels, lambda: {k: v.detach().clone() for k, v in model.state_dict().items()}

    t0 = time.time()
    tl, vl, va, best = fit_loop(label, train_step, eval_step, epochs or 6, 2)
    elapsed = time.time() - t0

    model.load_state_dict(best["state"])
    _, test_preds, test_labels = run_epoch_sbert(model, rows["test"], optimizer, scaler,
                                                 train=False, batch_size=32)
    metrics = finalize(out_dir, "sbert", label, rows, (tl, vl, va), best,
                       test_preds, test_labels, n_params, elapsed, seed,
                       extra={"head": head, "backbone": SBERT_BACKBONES[backbone],
                              "max_len": max_len, "lr": lr, "batch_size": batch_size})
    if seed == MAIN_SEED:
        model.backbone.save_pretrained(out_dir / "backbone")
        model.tokenizer.save_pretrained(out_dir / "backbone")
        if model.head is not None:
            torch.save(model.head.state_dict(), out_dir / "head.pt")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp", "cnn", "bilstm", "sbert"])
    parser.add_argument("--data", default="real", choices=["real", "synthetic"])
    parser.add_argument("--seed", type=int, default=MAIN_SEED)
    parser.add_argument("--head", default="interaction", choices=["interaction", "cosine"])
    parser.add_argument("--backbone", default="minilm", choices=["minilm", "mpnet"], help="SBERT only")
    parser.add_argument("--out", default=None,
                        help="results subdir under results/<data>/ (default: the model name)")
    parser.add_argument("--epochs", type=int, default=None, help="SBERT max epochs")
    args = parser.parse_args()

    from models import SiameseMLP, SiameseCNN, SiameseBiLSTM

    out = args.out or args.model
    h = args.head
    print(f"device: {DEVICE}  head: {h}  out: results/{args.data}/{out}")
    if args.model == "mlp":
        train_token_model("mlp", "MLP", lambda vs: SiameseMLP(vs, head=h), 1e-3, args.data, args.seed, h, out)
    elif args.model == "cnn":
        train_token_model("cnn", "1D-CNN", lambda vs: SiameseCNN(vs, head=h), 1e-3, args.data, args.seed, h, out)
    elif args.model == "bilstm":
        train_token_model("bilstm", "BiLSTM", lambda vs: SiameseBiLSTM(vs, head=h), 1e-3, args.data, args.seed, h, out)
    elif args.model == "sbert":
        train_sbert_model(args.data, args.seed, h, out, args.backbone, args.epochs)


if __name__ == "__main__":
    main()
