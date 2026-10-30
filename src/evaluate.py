"""
Cross-model comparison + error analysis on the held-out test split.

Reads results/<data>/<model>/metrics.json, seeds/*.json and
test_predictions.csv written by src/train.py, and produces in
results/<data>/:
  comparison.csv / comparison.md   single comparison table (guidelines Sec. 5)
  comparison_roc.png               all four ROC curves on one plot
  comparison_bars.png              test accuracy / F1 / AUC / P@3 per model
  score_distributions.png          predicted score per true fit class
  error_analysis/summary.json      error rates by fit class, length bucket,
                                   FP vs FN, and cross-model error overlap
  error_analysis/<model>_worst.csv the most confidently wrong test pairs

Usage: python src/evaluate.py [--data real]
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

MODELS = [("mlp", "MLP"), ("cnn", "1D-CNN"), ("bilstm", "BiLSTM"), ("sbert", "SBERT")]
PAIRS_FILE = {"real": "pairs_real.csv", "synthetic": "pairs.csv"}
FIT_ORDER = ["No Fit", "Potential Fit", "Good Fit"]

TABLE_METRICS = [
    ("test_accuracy", "Accuracy"), ("test_precision", "Precision"), ("test_recall", "Recall"),
    ("test_f1", "F1"), ("test_auc", "ROC-AUC"), ("test_p_at_1", "P@1"), ("test_p_at_3", "P@3"),
    ("test_p_at_5", "P@5"), ("test_p_at_3_random", "P@3 random"), ("test_spearman", "Spearman"),
]


def load_seed_metrics(model_dir):
    seeds = sorted((model_dir / "seeds").glob("seed_*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in seeds]


def fmt(mean, std, n):
    return f"{mean:.4f} ± {std:.4f}" if n > 1 else f"{mean:.4f}"


def build_comparison(results_dir):
    rows, md_rows = [], []
    for key, label in MODELS:
        model_dir = results_dir / key
        if not (model_dir / "metrics.json").exists():
            print(f"skipping {key}: no metrics.json")
            continue
        main = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        seeds = load_seed_metrics(model_dir) or [main]
        row = {"model": label, "n_seeds": len(seeds),
               "trainable_parameters": main["trainable_parameters"],
               "train_seconds": np.mean([s["train_seconds"] for s in seeds]),
               "best_epoch": main["best_epoch"]}
        md = [label, str(len(seeds)), f"{main['trainable_parameters']:,}"]
        for m, _ in TABLE_METRICS:
            vals = np.array([s[m] for s in seeds], dtype=float)
            row[m] = vals.mean()
            row[m + "_std"] = vals.std()
            md.append(fmt(vals.mean(), vals.std(), len(vals)))
        md.append(f"{row['train_seconds']:.0f}")
        rows.append(row)
        md_rows.append(md)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "comparison.csv", index=False)
    header = ["Model", "Seeds", "Params"] + [h for _, h in TABLE_METRICS] + ["Train s"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in md_rows]
    note = ("\nAll metrics on the held-out TEST split (JDs unseen in training). "
            "Binary metrics use each model's validation-tuned threshold. "
            "Mean ± std over seeds where more than one seed was run.\n")
    (results_dir / "comparison.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")
    print("\n".join(lines))
    return df


def load_predictions(results_dir, pairs):
    preds = {}
    text_len = pairs.set_index(["resume_id", "jd_id"])
    for key, label in MODELS:
        path = results_dir / key / "test_predictions.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["true_match"] = (df["similarity_label"] >= 0.5).astype(int)
            df["correct"] = (df["pred_match"] == df["true_match"]).astype(int)
            df["resume_words"] = [
                len(text_len.loc[(r, j), "resume_text"].split()) if (r, j) in text_len.index else np.nan
                for r, j in zip(df["resume_id"], df["jd_id"])
            ]
            preds[label] = df
    return preds


def plot_roc(results_dir, preds):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, df in preds.items():
        fpr, tpr, _ = roc_curve(df["true_match"], df["pred_score"])
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(df['true_match'], df['pred_score']):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC on held-out test split")
    ax.legend()
    fig.tight_layout(); fig.savefig(results_dir / "comparison_roc.png", dpi=140); plt.close(fig)


def plot_bars(results_dir, comp):
    metrics = [("test_accuracy", "Accuracy"), ("test_f1", "F1"), ("test_auc", "ROC-AUC"), ("test_p_at_3", "P@3")]
    x = np.arange(len(metrics))
    w = 0.8 / max(len(comp), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (_, row) in enumerate(comp.iterrows()):
        ax.bar(x + i * w, [row[m] for m, _ in metrics], w,
               yerr=[row[m + "_std"] for m, _ in metrics], capsize=3, label=row["model"])
    ax.set_xticks(x + w * (len(comp) - 1) / 2)
    ax.set_xticklabels([h for _, h in metrics])
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Test-set comparison across encoders")
    ax.legend(ncol=4, loc="lower center")
    fig.tight_layout(); fig.savefig(results_dir / "comparison_bars.png", dpi=140); plt.close(fig)


def plot_distributions(results_dir, preds):
    fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (label, df) in zip(axes, preds.items()):
        data = [df.loc[df["fit_class"] == c, "pred_score"].values for c in FIT_ORDER if c in set(df["fit_class"])]
        ax.boxplot(data, tick_labels=[c for c in FIT_ORDER if c in set(df["fit_class"])])
        ax.set_title(label); ax.set_ylabel("Predicted cosine score")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Predicted score by true fit class (test)")
    fig.tight_layout(); fig.savefig(results_dir / "score_distributions.png", dpi=140); plt.close(fig)


def error_analysis(results_dir, preds, pairs):
    out_dir = results_dir / "error_analysis"
    out_dir.mkdir(exist_ok=True)
    summary = {}
    # length buckets computed once from the test resumes
    lengths = next(iter(preds.values()))["resume_words"]
    bins = np.nanquantile(lengths, [0, 1 / 3, 2 / 3, 1])
    bucket_names = [f"short (<{bins[1]:.0f} words)", f"medium ({bins[1]:.0f}-{bins[2]:.0f})", f"long (>{bins[2]:.0f})"]
    text_by_key = pairs.set_index(["resume_id", "jd_id"])

    for label, df in preds.items():
        err = 1 - df["correct"]
        by_class = {c: float(err[df["fit_class"] == c].mean()) for c in FIT_ORDER if (df["fit_class"] == c).any()}
        bucket = pd.cut(df["resume_words"], bins=bins, labels=bucket_names, include_lowest=True)
        by_len = {str(b): float(err[bucket == b].mean()) for b in bucket_names}
        fp = int(((df["pred_match"] == 1) & (df["true_match"] == 0)).sum())
        fn = int(((df["pred_match"] == 0) & (df["true_match"] == 1)).sum())
        summary[label] = {
            "error_rate": float(err.mean()),
            "error_rate_by_fit_class": by_class,
            "error_rate_by_resume_length": by_len,
            "false_positives": fp, "false_negatives": fn,
        }
        # most confidently wrong: largest |score - label| among errors
        wrong = df[df["correct"] == 0].copy()
        wrong["abs_err"] = (wrong["pred_score"] - wrong["similarity_label"]).abs()
        wrong = wrong.sort_values("abs_err", ascending=False).head(15)
        wrong["resume_snippet"] = [text_by_key.loc[(r, j), "resume_text"][:300] for r, j in zip(wrong["resume_id"], wrong["jd_id"])]
        wrong["jd_snippet"] = [text_by_key.loc[(r, j), "jd_text"][:300] for r, j in zip(wrong["resume_id"], wrong["jd_id"])]
        wrong.to_csv(out_dir / f"{label.lower().replace('-', '')}_worst.csv", index=False)

    # cross-model error overlap: how many test pairs each pair of models
    # both get wrong, and how many pairs every model gets wrong.
    labels = list(preds)
    wrong_sets = {l: set(np.where(preds[l]["correct"].values == 0)[0]) for l in labels}
    overlap = {}
    for a, b in combinations(labels, 2):
        inter = len(wrong_sets[a] & wrong_sets[b])
        union = len(wrong_sets[a] | wrong_sets[b])
        overlap[f"{a} & {b}"] = {"both_wrong": inter, "jaccard": round(inter / union, 3) if union else 0.0}
    all_wrong = set.intersection(*wrong_sets.values()) if wrong_sets else set()
    first = preds[labels[0]]
    summary["cross_model"] = {
        "pairwise_error_overlap": overlap,
        "wrong_for_all_models": len(all_wrong),
        "wrong_for_all_by_fit_class": first.iloc[sorted(all_wrong)]["fit_class"].value_counts().to_dict(),
        "n_test_pairs": len(first),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="real", choices=["real", "synthetic"])
    args = parser.parse_args()
    results_dir = ROOT / "results" / args.data

    comp = build_comparison(results_dir)
    pairs = pd.read_csv(DATA_DIR / PAIRS_FILE[args.data])
    pairs = pairs[pairs["split"] == "test"].drop_duplicates(["resume_id", "jd_id"])
    preds = load_predictions(results_dir, pairs)
    if not preds:
        raise SystemExit("no test_predictions.csv found; run src/train.py first")
    plot_roc(results_dir, preds)
    plot_bars(results_dir, comp)
    plot_distributions(results_dir, preds)
    error_analysis(results_dir, preds, pairs)


if __name__ == "__main__":
    main()
