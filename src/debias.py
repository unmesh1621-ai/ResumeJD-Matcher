"""
Adversarial debiasing of the best encoder (SBERT) with a gradient-reversal
layer (GRL), following Zhang, Lemoine & Mitchell (2018) / Ganin & Lempitsky
(2015).

Setup:
  - start from the fine-tuned SBERT checkpoint (results/real/sbert/backbone)
  - during training, every resume gets a random first name from group A or
    group B prepended (only the FIRST half of each name list, so the audit's
    held-out names stay unseen)
  - main loss: MSE(cosine(resume, jd), fit label)          (unchanged task)
  - adversary: small MLP predicting the name group from the resume
    embedding, attached through a GRL. The adversary minimises its
    cross-entropy; the reversed gradient pushes the encoder to REMOVE the
    group signal from the embedding.
  - lambda (GRL strength) is ramped from 0 to LAMBDA_MAX during training.

To separate the effect of the adversary from the effect of simply training
more on name-injected resumes, a CONTROL run uses exactly the same
procedure with lambda = 0.

Outputs:
  results/real/sbert_debiased/   backbone/ + metrics.json
  results/real/sbert_control/    backbone/ + metrics.json
  results/real/fairness_report.json   before / control / after audit
  results/real/fairness_comparison.png

Usage: python src/debias.py [--epochs 2] [--lambda-max 1.0]
"""
import argparse
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train import (load_pairs, set_seed, binary_metrics, tune_threshold, test_metrics,
                   run_epoch_sbert, save_test_predictions, DEVICE, RESULTS_ROOT, MAIN_SEED)
from models import SiameseSBERT
from model_io import load_encoder
from bias_probe import (audit_encoder, build_audit_set, inject_name, load_test_rows,
                        GROUP_A, GROUP_B, HALF)

DATA = "real"
TRAIN_NAMES = [(n, 0) for n in GROUP_A[:HALF]] + [(n, 1) for n in GROUP_B[:HALF]]


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class Adversary(nn.Module):
    def __init__(self, dim=384, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 2))

    def forward(self, emb, lambd):
        return self.net(GradReverse.apply(emb, lambd))


def train_variant(name, lambda_max, epochs, lr=2e-5):
    set_seed(MAIN_SEED)
    rows = load_pairs(DATA)
    base_dir = RESULTS_ROOT / DATA / "sbert"
    base_metrics = json.loads((base_dir / "metrics.json").read_text(encoding="utf-8"))
    head = base_metrics.get("head", "cosine")
    model = SiameseSBERT(model_name=str(base_dir / "backbone"), max_len=base_metrics["max_len"],
                         head=head).to(DEVICE)
    if model.head is not None:
        model.head.load_state_dict(torch.load(base_dir / "head.pt", map_location=DEVICE))
    batch_size = base_metrics.get("batch_size", 16)
    adversary = Adversary(dim=model.backbone.config.hidden_size).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": lr, "weight_decay": 0.01},
        {"params": adversary.parameters(), "lr": 1e-3},
    ])
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    use_amp = DEVICE.type == "cuda"
    rng = random.Random(MAIN_SEED)

    train_rows = rows["train"]
    steps_per_epoch = (len(train_rows) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    history = []
    best = {"auc": -1.0}
    step = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train(); adversary.train()
        idx = list(range(len(train_rows)))
        rng.shuffle(idx)
        task_losses, adv_losses, adv_accs = [], [], []
        for s in range(0, len(idx), batch_size):
            batch = [train_rows[i] for i in idx[s:s + batch_size]]
            names = [rng.choice(TRAIN_NAMES) for _ in batch]
            resumes = [inject_name(n, r["resume_text"]) for (n, _), r in zip(names, batch)]
            jds = [r["jd_text"] for r in batch]
            labels = torch.tensor([r["similarity_label"] for r in batch], dtype=torch.float32, device=DEVICE)
            groups = torch.tensor([g for _, g in names], dtype=torch.long, device=DEVICE)
            # ramp lambda with the standard DANN schedule 2/(1+e^{-10p}) - 1
            p = step / max(total_steps - 1, 1)
            lambd = lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                r_emb = model.encode_texts(resumes, DEVICE)
                j_emb = model.encode_texts(jds, DEVICE)
            r_emb, j_emb = r_emb.float(), j_emb.float()
            task_loss = F.mse_loss(model.score_vectors(r_emb, j_emb).float(), labels)
            logits = adversary(r_emb, lambd)
            adv_loss = F.cross_entropy(logits, groups)
            # For the control run (lambda_max = 0) the adversary still trains
            # (so we can report how well it does) but no gradient reaches the
            # encoder, since GRL scales it by 0.
            loss = task_loss + adv_loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            task_losses.append(task_loss.item()); adv_losses.append(adv_loss.item())
            adv_accs.append((logits.argmax(1) == groups).float().mean().item())
            step += 1

        val_loss, val_preds, val_labels = run_epoch_sbert(model, rows["val"], None, None, train=False, batch_size=32)
        val_auc = binary_metrics(val_preds, val_labels)[4]
        history.append({"epoch": epoch, "task_loss": float(np.mean(task_losses)),
                        "adv_loss": float(np.mean(adv_losses)), "adv_train_acc": float(np.mean(adv_accs)),
                        "val_loss": val_loss, "val_auc": val_auc, "lambda_end": float(lambd)})
        print(f"[{name}] epoch {epoch}: {history[-1]}", flush=True)
        if val_auc > best["auc"]:
            best = {"auc": val_auc, "epoch": epoch, "val_preds": val_preds, "val_labels": val_labels,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    elapsed = time.time() - t0

    model.load_state_dict(best["state"])
    _, test_preds, test_labels = run_epoch_sbert(model, rows["test"], None, None, train=False, batch_size=32)
    threshold = tune_threshold(best["val_preds"], best["val_labels"])
    tm, _, _ = test_metrics(rows["test"], test_preds, test_labels, threshold)

    out_dir = RESULTS_ROOT / DATA / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out_dir / "backbone")
    model.tokenizer.save_pretrained(out_dir / "backbone")
    if model.head is not None:
        torch.save(model.head.state_dict(), out_dir / "head.pt")
    save_test_predictions(out_dir, rows["test"], test_preds, threshold)
    metrics = {"model": name, "lambda_max": lambda_max, "epochs": epochs, "best_epoch": best["epoch"],
               "decision_threshold": threshold, "val_auc": best["auc"], **tm,
               "head": head, "backbone": base_metrics.get("backbone"), "batch_size": batch_size,
               "max_len": base_metrics["max_len"], "train_seconds": elapsed, "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def plot(report, path):
    variants = list(report["variants"])
    panels = [("probe_test_accuracy", "Probe accuracy (chance 0.5)"),
              ("mean_abs_score_diff", "Mean |score(A) - score(B)|"),
              ("tpr_gap_abs", "|TPR gap|"),
              ("test_auc", "Matching test AUC")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (key, title) in zip(axes, panels):
        vals = [report["variants"][v][key] for v in variants]
        ax.bar(variants, vals, color=["tab:gray", "tab:blue", "tab:green", "tab:olive", "tab:cyan"][:len(variants)])
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("SBERT fairness: before vs. control (no adversary) vs. adversarially debiased")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 3.0],
                        help="GRL strengths to try; the final one is chosen on VALIDATION")
    parser.add_argument("--max-val-auc-drop", type=float, default=0.01)
    args = parser.parse_args()

    test_rows = load_test_rows(DATA)
    audit_set = build_audit_set(test_rows)
    # selection uses a separate audit built from VALIDATION resumes/pairs
    val_rows = load_pairs(DATA)["val"]
    val_audit_set = build_audit_set(val_rows)
    report = {"method": "gradient-reversal adversary on resume embedding (Zhang et al. 2018)",
              "epochs": args.epochs, "lambdas_tried": args.lambdas,
              "selection_rule": (f"lowest VALIDATION probe accuracy among debiased variants whose "
                                 f"validation AUC is within {args.max_val_auc_drop} of the undebiased model"),
              "variants": {}}

    def record(label, model_dir, metrics):
        enc = load_encoder("sbert", DATA, model_dir=model_dir)
        a = audit_encoder(enc, test_rows, audit_set)
        a["tpr_gap_abs"] = abs(a["tpr_gap_A_minus_B"])
        a["test_auc"] = metrics["test_auc"]
        a["test_accuracy"] = metrics["test_accuracy"]
        a["test_f1"] = metrics["test_f1"]
        a["val_auc"] = metrics["val_auc"]
        v = audit_encoder(enc, val_rows, val_audit_set)
        a["val_probe_accuracy"] = v["probe_test_accuracy"]
        a["dir"] = model_dir.relative_to(RESULTS_ROOT / DATA).as_posix()
        report["variants"][label] = a
        print(label, json.dumps(a, indent=2), flush=True)
        del enc
        torch.cuda.empty_cache()
        return a

    base_dir = RESULTS_ROOT / DATA / "sbert"
    before = record("before", base_dir, json.loads((base_dir / "metrics.json").read_text(encoding="utf-8")))
    m = train_variant("sbert_control", 0.0, args.epochs)
    record("control (λ=0)", RESULTS_ROOT / DATA / "sbert_control", m)
    candidates = []
    for lam in args.lambdas:
        name = f"sbert_debiased_l{lam:g}"
        m = train_variant(name, lam, args.epochs)
        candidates.append((f"debiased (λ={lam:g})", record(f"debiased (λ={lam:g})", RESULTS_ROOT / DATA / name, m)))

    ok = [c for c in candidates if c[1]["val_auc"] >= before["val_auc"] - args.max_val_auc_drop]
    chosen = (min(ok, key=lambda c: c[1]["val_probe_accuracy"]) if ok
              else max(candidates, key=lambda c: c[1]["val_auc"]))
    report["selected"] = chosen[0]
    print(f"selected on validation: {chosen[0]}")
    import shutil
    final_dir = RESULTS_ROOT / DATA / "sbert_debiased"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(RESULTS_ROOT / DATA / chosen[1]["dir"], final_dir)

    out = RESULTS_ROOT / DATA / "fairness_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot(report, RESULTS_ROOT / DATA / "fairness_comparison.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
