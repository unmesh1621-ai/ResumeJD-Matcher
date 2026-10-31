"""
Aggregate the grouped K-fold cross-validation runs (results/real/cv/) and
choose, for each architecture, the configuration with the highest MEAN
cross-validated AUC. The held-out test split plays no part in the choice.

Writes results/real/model_selection.json and model_selection.md.

Usage: python src/cv_select.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "real"
CV_DIR = RESULTS / "cv"

# architecture -> candidate configs (cv dir name -> where its main-split run lives)
CANDIDATES = {
    "mlp": {"mlp_cosine": "mlp/cosine", "mlp_interaction": "mlp/interaction"},
    "cnn": {"cnn_cosine": "cnn/cosine", "cnn_interaction": "cnn/interaction"},
    "bilstm": {"bilstm_cosine": "bilstm/cosine", "bilstm_interaction": "bilstm/interaction"},
    "sbert": {"sbert_minilm_cosine": "minilm/cosine", "sbert_minilm_interaction": "minilm/interaction",
              "sbert_mpnet_interaction": "mpnet/interaction"},
}


def load_folds(name):
    folds = sorted((CV_DIR / name).glob("fold_*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in folds]


def main():
    selection, lines = {}, ["# Model selection by grouped 3-fold cross-validation (train+val only)", "",
                            "| Architecture | Config | Folds | CV AUC (mean ± std) | CV within-resume AUC | Selected |",
                            "|---|---|---|---|---|---|"]
    for arch, configs in CANDIDATES.items():
        scored = []
        for name in configs:
            folds = load_folds(name)
            if not folds:
                continue
            aucs = np.array([f["val_auc"] for f in folds])
            wr = np.array([f["val_within_resume_auc"] for f in folds])
            scored.append({"config": name, "n_folds": len(folds), "cv_auc_mean": float(aucs.mean()),
                           "cv_auc_std": float(aucs.std()), "cv_auc_per_fold": aucs.tolist(),
                           "cv_within_resume_auc_mean": float(wr.mean())})
        complete = [s for s in scored if s["n_folds"] == max(x["n_folds"] for x in scored)] if scored else []
        best = max(complete, key=lambda s: s["cv_auc_mean"]) if complete else None
        selection[arch] = {"selected": best["config"] if best else None, "candidates": scored}
        for s in scored:
            lines.append(f"| {arch} | {s['config']} | {s['n_folds']} | {s['cv_auc_mean']:.4f} ± {s['cv_auc_std']:.4f} "
                         f"| {s['cv_within_resume_auc_mean']:.4f} | {'**yes**' if best and s is best else ''} |")
    lines += ["", "Each fold holds out ~93 job descriptions (vs. 42 in the original validation split); "
                  "the held-out test split is never used for any choice."]
    (RESULTS / "model_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (RESULTS / "model_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
