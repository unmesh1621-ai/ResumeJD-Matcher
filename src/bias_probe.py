"""
Bias audit (synopsis Objective 4) via counterfactual name injection,
following the name-based methodology of De-Arteaga et al. (2019).

The real resumes are anonymised, so we construct a controlled audit set:
every test resume is copied twice, once with a group-A first name and once
with a group-B first name prepended (name lists from
data/generate_dataset.py). The two copies are otherwise IDENTICAL, so any
difference the model shows between them can only come from the name.

Two measurements per encoder:
 1. Probe leakage: a logistic-regression probe is trained to predict the
    name group from the resume embedding. Probe-train and probe-test use
    disjoint resumes AND disjoint first names, so the probe must pick up a
    general group signal, not memorise particular names. Chance = 50%.
 2. Counterfactual fairness of the fit score on test pairs: mean |score(A)
    - score(B)|, decision flip rate, and the true-positive-rate gap
    TPR_A - TPR_B (De-Arteaga et al.'s metric) on truly-matching pairs.

Usage: python src/bias_probe.py [--data real] [--models mlp cnn bilstm sbert]
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from generate_dataset import MALE_PROXY_NAMES, FEMALE_PROXY_NAMES
from model_io import load_encoder, available_models, best_model_key, LABELS, RESULTS_ROOT

ROOT = Path(__file__).parent.parent
PAIRS_FILE = {"real": "pairs_real.csv", "synthetic": "pairs.csv"}
SEED = 42

# Names split in half: first half used to fit the probe, second half to test it.
GROUP_A, GROUP_B = MALE_PROXY_NAMES, FEMALE_PROXY_NAMES
HALF = len(GROUP_A) // 2


def inject_name(name, text):
    return f"{name}\n{text}"


def load_test_rows(data):
    with open(ROOT / "data" / PAIRS_FILE[data], encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def build_audit_set(rows):
    """Deterministically assign each unique test resume one A name and one
    B name, and a probe split (train/test) by resume."""
    rng = random.Random(SEED)
    resumes = sorted({r["resume_id"]: r["resume_text"] for r in rows}.items())
    rng.shuffle(resumes)
    audit = {}
    for i, (rid, text) in enumerate(resumes):
        probe_split = "train" if i < len(resumes) // 2 else "test"
        names_a = GROUP_A[:HALF] if probe_split == "train" else GROUP_A[HALF:]
        names_b = GROUP_B[:HALF] if probe_split == "train" else GROUP_B[HALF:]
        audit[rid] = {"text": text, "probe_split": probe_split,
                      "name_a": rng.choice(names_a), "name_b": rng.choice(names_b)}
    return audit


def probe_leakage(enc, audit):
    rids = sorted(audit)
    texts, groups, splits = [], [], []
    for rid in rids:
        a = audit[rid]
        texts += [inject_name(a["name_a"], a["text"]), inject_name(a["name_b"], a["text"])]
        groups += [0, 1]
        splits += [a["probe_split"]] * 2
    X = enc.embed(texts).numpy()
    y = np.array(groups)
    tr = np.array([s == "train" for s in splits])
    scaler = StandardScaler().fit(X[tr])
    probe = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X[tr]), y[tr])
    acc = accuracy_score(y[~tr], probe.predict(scaler.transform(X[~tr])))
    # direct embedding shift caused by the name swap alone
    Xa, Xb = X[0::2], X[1::2]
    cos = (Xa * Xb).sum(1) / (np.linalg.norm(Xa, axis=1) * np.linalg.norm(Xb, axis=1) + 1e-9)
    return {"probe_test_accuracy": float(acc), "chance_accuracy": 0.5,
            "leakage_gap": float(acc - 0.5),
            "mean_cosine_between_name_swapped_embeddings": float(cos.mean())}


def counterfactual_fairness(enc, audit, rows):
    resume_a = [inject_name(audit[r["resume_id"]]["name_a"], r["resume_text"]) for r in rows]
    resume_b = [inject_name(audit[r["resume_id"]]["name_b"], r["resume_text"]) for r in rows]
    jds = [r["jd_text"] for r in rows]
    jd_vecs = enc.embed(jds)
    import torch.nn.functional as F
    sa = F.cosine_similarity(enc.embed(resume_a), jd_vecs, dim=-1).numpy()
    sb = F.cosine_similarity(enc.embed(resume_b), jd_vecs, dim=-1).numpy()
    t = enc.threshold
    true_match = np.array([float(r["similarity_label"]) >= 0.5 for r in rows])
    tpr_a = float((sa[true_match] >= t).mean())
    tpr_b = float((sb[true_match] >= t).mean())
    from sklearn.metrics import roc_auc_score
    return {
        "mean_abs_score_diff": float(np.abs(sa - sb).mean()),
        "max_abs_score_diff": float(np.abs(sa - sb).max()),
        "mean_signed_score_diff_A_minus_B": float((sa - sb).mean()),
        "decision_flip_rate": float(((sa >= t) != (sb >= t)).mean()),
        "tpr_group_A": tpr_a, "tpr_group_B": tpr_b, "tpr_gap_A_minus_B": tpr_a - tpr_b,
        "test_auc_with_names_A": float(roc_auc_score(true_match, sa)),
        "test_auc_with_names_B": float(roc_auc_score(true_match, sb)),
    }


def audit_encoder(enc, rows, audit=None):
    audit = audit or build_audit_set(rows)
    return {"model": enc.label, **probe_leakage(enc, audit), **counterfactual_fairness(enc, audit, rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="real", choices=["real", "synthetic"])
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    rows = load_test_rows(args.data)
    audit = build_audit_set(rows)
    keys = args.models or available_models(args.data)
    best, best_auc = best_model_key(args.data)
    report = {"best_encoder_by_val_auc": LABELS.get(best), "best_val_auc": best_auc,
              "n_test_pairs": len(rows), "n_audit_resumes": len(audit), "encoders": {}}
    for key in keys:
        print(f"auditing {key} ...", flush=True)
        enc = load_encoder(key, args.data)
        report["encoders"][LABELS[key]] = audit_encoder(enc, rows, audit)
        print(json.dumps(report["encoders"][LABELS[key]], indent=2))

    out = RESULTS_ROOT / args.data / "bias_probe_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
