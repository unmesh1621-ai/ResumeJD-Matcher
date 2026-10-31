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
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

MODELS = [("mlp", "MLP"), ("cnn", "1D-CNN"), ("bilstm", "BiLSTM"), ("sbert", "SBERT")]
PAIRS_FILE = {"real": "pairs_real.csv", "synthetic": "pairs.csv"}
FIT_ORDER = ["No Fit", "Potential Fit", "Good Fit"]

TABLE_METRICS = [
    ("val_auc", "Val AUC"),
    ("test_accuracy", "Accuracy"), ("test_precision", "Precision"), ("test_recall", "Recall"),
    ("test_f1", "F1"), ("test_auc", "ROC-AUC"), ("test_within_resume_auc", "Within-resume AUC"),
    ("test_p_at_1", "P@1"), ("test_p_at_3", "P@3"), ("test_p_at_5", "P@5"),
    ("test_p_at_3_random", "P@3 random"), ("test_spearman", "Spearman"),
]


def load_seed_metrics(model_dir):
    seeds = sorted((model_dir / "seeds").glob("seed_*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in seeds]


def fmt(mean, std, n):
    if np.isnan(mean):
        return "—"
    return f"{mean:.4f} ± {std:.4f}" if n > 1 else f"{mean:.4f}"


def summarise(model_dir, label):
    """One table row: mean/std over seeds of every metric in TABLE_METRICS."""
    main = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    seeds = load_seed_metrics(model_dir) or [main]
    row = {"model": label, "dir": model_dir.relative_to(ROOT).as_posix(), "n_seeds": len(seeds),
           "head": main.get("head", "cosine"), "backbone": main.get("backbone", ""),
           "trainable_parameters": main.get("trainable_parameters", 0),
           "train_seconds": float(np.mean([s.get("train_seconds", 0.0) for s in seeds]))}
    md = [label, str(len(seeds)), f"{row['trainable_parameters']:,}" if row["trainable_parameters"] else "—"]
    if all("test_within_resume_auc" not in s for s in seeds) and (model_dir / "test_predictions.csv").exists():
        # runs made before this metric existed: derive it from the saved predictions
        p = pd.read_csv(model_dir / "test_predictions.csv")
        y = (p["similarity_label"] >= 0.5).astype(int)
        aucs = [roc_auc_score(g_y, g_s) for (_, g_y), (_, g_s) in
                zip(y.groupby(p["resume_id"]), p["pred_score"].groupby(p["resume_id"])) if 0 < g_y.sum() < len(g_y)]
        for s in seeds:
            s["test_within_resume_auc"] = float(np.mean(aucs))
    for m, _ in TABLE_METRICS:
        vals = np.array([s.get(m, np.nan) for s in seeds], dtype=float)
        row[m] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan
        row[m + "_std"] = float(np.nanstd(vals)) if not np.all(np.isnan(vals)) else np.nan
        md.append(fmt(row[m], row[m + "_std"], int((~np.isnan(vals)).sum())))
    md.append(f"{row['train_seconds']:.0f}" if row["train_seconds"] else "—")
    return row, md


def md_table(md_rows):
    header = ["Model", "Seeds", "Params"] + [h for _, h in TABLE_METRICS] + ["Train s"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(r) + " |" for r in md_rows]


def collect(results_dir, entries):
    rows, md_rows = [], []
    for model_dir, label in entries:
        if (model_dir / "metrics.json").exists():
            row, md = summarise(model_dir, label)
            rows.append(row)
            md_rows.append(md)
    return rows, md_rows


def variant_label(path, results_dir):
    m = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return f"{m.get('model', path.name)} [{path.relative_to(results_dir).as_posix()}]"


def build_comparison(results_dir):
    main_rows, main_md = collect(results_dir, [(results_dir / k, l) for k, l in MODELS])
    pd.DataFrame(main_rows).to_csv(results_dir / "comparison.csv", index=False)

    # every other trained variant = ablation (head / backbone choices)
    main_dirs = {results_dir / k for k, _ in MODELS}
    ablation_dirs = [p.parent for p in results_dir.glob("**/metrics.json")
                     if p.parent not in main_dirs
                     and "baselines" not in p.parts
                     and not p.parent.name.startswith(("sbert_control", "sbert_debiased"))]
    # the selected SBERT is a copy of one candidate, so it is not listed twice;
    # the selected scratch models are listed with their ablation counterparts
    ablation_dirs += [d for d in main_dirs if d.name != "sbert" and (d / "metrics.json").exists()]
    abl_rows, abl_md = collect(results_dir, [(d, variant_label(d, results_dir)) for d in sorted(set(ablation_dirs))])
    pd.DataFrame(abl_rows).to_csv(results_dir / "comparison_ablations.csv", index=False)

    base_dirs = sorted(p.parent for p in (results_dir / "baselines").glob("*/metrics.json"))
    base_rows, base_md = collect(results_dir, [(d, d.name) for d in base_dirs])
    pd.DataFrame(base_rows).to_csv(results_dir / "comparison_baselines.csv", index=False)

    text = ["# Model comparison (held-out TEST split)", "",
            "All metrics on the held-out test split (its JDs never appear in train/val). Binary metrics use "
            "each model's validation-tuned threshold. Mean ± std over seeds where more than one seed was run. "
            "Scorer (cosine vs. interaction head) and SBERT backbone were chosen per model on VALIDATION AUC. "
            "Within-resume AUC = mean AUC over the JDs paired with each single resume; a model that ignores "
            "the JD scores exactly 0.5 there.", "",
            "## Main comparison (4 architectures)", ""] + md_table(main_md) + \
           ["", "## Ablations (all trained variants, incl. the selected ones)", ""] + md_table(abl_md) + \
           ["", "## Baselines (no training of a matching model)", ""] + md_table(base_md)
    (results_dir / "comparison.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))
    return pd.DataFrame(main_rows)


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
