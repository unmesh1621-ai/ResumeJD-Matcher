"""
Non-trained / trivial reference points, evaluated with exactly the same
protocol as the four encoders (val-tuned threshold, same test metrics):

  resume_prior   score = the resume's mean fit label in TRAIN (JD ignored).
                 474 of 477 test resumes also occur in train (a property of
                 the source dataset; JDs never overlap), so this measures
                 how far one gets by memorising resumes instead of matching.
                 Its within-resume AUC is 0.5 by construction.
  tfidf_cosine   classical IR: TF-IDF (fit on train) + cosine similarity.
  zero_shot_*    pretrained sentence encoders, cosine, no fine-tuning.

Writes results/<data>/baselines/<name>/{metrics.json,test_predictions.csv}.

Usage: python src/baselines.py [--data real] [--skip-zero-shot]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from train import (load_pairs, tune_threshold, binary_metrics, test_metrics, within_resume_auc,
                   save_test_predictions, RESULTS_ROOT, DEVICE, DATA_CONFIG)
from models import SiameseSBERT, SBERT_BACKBONES


def evaluate(name, rows, val_scores, test_scores, out_root, extra=None):
    val_labels = np.array([r["similarity_label"] for r in rows["val"]])
    test_labels = np.array([r["similarity_label"] for r in rows["test"]])
    threshold = tune_threshold(val_scores, val_labels)
    val_acc, _, _, val_f1, val_auc, _, _ = binary_metrics(val_scores, val_labels, threshold)
    tm, _, _ = test_metrics(rows["test"], test_scores, test_labels, threshold)
    metrics = {"model": name, "decision_threshold": threshold, "val_accuracy": val_acc,
               "val_f1": val_f1, "val_auc": val_auc,
               "val_within_resume_auc": within_resume_auc(rows["val"], val_scores)[0],
               **tm, **(extra or {})}
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_test_predictions(out_dir, rows["test"], test_scores, threshold)
    print(f"{name:<22} val_auc={val_auc:.4f} test_auc={tm['test_auc']:.4f} "
          f"within_resume_auc={tm['test_within_resume_auc']:.4f} P@3={tm['test_p_at_3']:.4f}")
    return metrics


def resume_prior(rows):
    sums, counts = {}, {}
    for r in rows["train"]:
        sums[r["resume_id"]] = sums.get(r["resume_id"], 0.0) + r["similarity_label"]
        counts[r["resume_id"]] = counts.get(r["resume_id"], 0) + 1
    global_mean = np.mean([r["similarity_label"] for r in rows["train"]])
    score = lambda split: np.array([sums[r["resume_id"]] / counts[r["resume_id"]]
                                    if r["resume_id"] in sums else global_mean for r in rows[split]])
    return score("val"), score("test")


def tfidf_cosine(rows):
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000, stop_words="english")
    vec.fit([r["resume_text"] for r in rows["train"]] + [r["jd_text"] for r in rows["train"]])

    def score(split):
        R = normalize(vec.transform([r["resume_text"] for r in rows[split]]))
        J = normalize(vec.transform([r["jd_text"] for r in rows[split]]))
        return np.asarray(R.multiply(J).sum(axis=1)).ravel()
    return score("val"), score("test")


@torch.no_grad()
def zero_shot(rows, backbone, max_len):
    model = SiameseSBERT(SBERT_BACKBONES[backbone], max_len=max_len).to(DEVICE).eval()

    def score(split, bs=32):
        out = []
        for i in range(0, len(rows[split]), bs):
            chunk = rows[split][i:i + bs]
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                r = model.encode_texts([c["resume_text"] for c in chunk], DEVICE)
                j = model.encode_texts([c["jd_text"] for c in chunk], DEVICE)
            out.append(F.cosine_similarity(r.float(), j.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)
    return score("val"), score("test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="real", choices=["real", "synthetic"])
    parser.add_argument("--skip-zero-shot", action="store_true")
    args = parser.parse_args()

    rows = load_pairs(args.data)
    out_root = RESULTS_ROOT / args.data / "baselines"
    evaluate("resume_prior", rows, *resume_prior(rows), out_root,
             extra={"note": "uses NO job-description information"})
    evaluate("tfidf_cosine", rows, *tfidf_cosine(rows), out_root)
    if not args.skip_zero_shot:
        max_len = DATA_CONFIG[args.data]["sbert_max_len"]
        for backbone in ("minilm", "mpnet"):
            evaluate(f"zero_shot_{backbone}", rows, *zero_shot(rows, backbone, max_len), out_root,
                     extra={"backbone": SBERT_BACKBONES[backbone]})


if __name__ == "__main__":
    main()
