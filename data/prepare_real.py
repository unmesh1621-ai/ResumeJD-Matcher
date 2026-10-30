"""
Preprocess the real resume-JD dataset into data/pairs_real.csv, using the
same column schema as the synthetic data/pairs.csv so every downstream
script (train / evaluate / explain / bias) works on either dataset.

Steps:
  1. clean text: unicode-normalise, mask emails / URLs / phone numbers
     (privacy + they carry no matching signal), collapse whitespace
  2. drop empty texts and exact-duplicate (resume, JD) pairs
  3. map labels: No Fit -> 0.0, Potential Fit -> 0.5, Good Fit -> 1.0
     (binary "match" = label >= 0.5, same threshold as the synthetic data)
  4. split: the dataset's own test.csv is kept as the held-out TEST set.
     Its job descriptions never appear in train.csv, so the test set
     measures generalisation to unseen jobs. We carve VALIDATION out of
     train.csv the same way -- grouped by JD, so no validation JD is seen
     in training. (Resumes are reused across pairs in the source data, so
     resume-level disjointness is not possible; JD-level disjointness is
     the meaningful guarantee here.)

Usage: python data/prepare_real.py
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
REAL_DIR = DATA_DIR / "real"
RESULTS_DIR = DATA_DIR.parent / "results" / "real"

LABEL_MAP = {"No Fit": 0.0, "Potential Fit": 0.5, "Good Fit": 1.0}
VAL_FRACTION = 0.15
SEED = 42

EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{8,}\d")
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = EMAIL_RE.sub(" [EMAIL] ", text)
    text = URL_RE.sub(" [URL] ", text)
    text = PHONE_RE.sub(" [PHONE] ", text)
    return SPACE_RE.sub(" ", text).strip()


def short_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def load_split(name):
    df = pd.read_csv(REAL_DIR / f"{name}.csv")
    df = df.rename(columns={"job_description_text": "jd_text"})
    df["resume_text"] = df["resume_text"].map(clean_text)
    df["jd_text"] = df["jd_text"].map(clean_text)
    df = df[(df["resume_text"].str.len() > 50) & (df["jd_text"].str.len() > 50)]
    df = df.drop_duplicates(subset=["resume_text", "jd_text"])
    df["similarity_label"] = df["label"].map(LABEL_MAP)
    assert df["similarity_label"].notna().all(), "unexpected label value"
    df["resume_id"] = "R" + df["resume_text"].map(short_hash)
    df["jd_id"] = "J" + df["jd_text"].map(short_hash)
    # resume_category / jd_category keep the 3-class label so error
    # analysis can break results down by fit class.
    df["resume_category"] = df["label"]
    df["jd_category"] = df["label"]
    return df


def main():
    train_full = load_split("train")
    test = load_split("test")

    rng = np.random.default_rng(SEED)
    jd_ids = np.array(sorted(train_full["jd_id"].unique()))
    rng.shuffle(jd_ids)
    n_val = int(round(len(jd_ids) * VAL_FRACTION))
    val_jds = set(jd_ids[:n_val])

    train = train_full[~train_full["jd_id"].isin(val_jds)].copy()
    val = train_full[train_full["jd_id"].isin(val_jds)].copy()
    train["split"], val["split"], test["split"] = "train", "val", "test"

    # leakage guards: no JD shared between any two splits
    assert not set(train["jd_id"]) & set(val["jd_id"])
    assert not set(train["jd_id"]) & set(test["jd_id"])
    assert not set(val["jd_id"]) & set(test["jd_id"])

    cols = ["resume_id", "jd_id", "resume_text", "jd_text",
            "resume_category", "jd_category", "similarity_label", "split"]
    pairs = pd.concat([train, val, test])[cols]
    pairs.to_csv(DATA_DIR / "pairs_real.csv", index=False)

    stats = {"source": "huggingface.co/datasets/cnamuangtoun/resume-job-description-fit"}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        stats[name] = {
            "pairs": len(df),
            "unique_resumes": int(df["resume_id"].nunique()),
            "unique_jds": int(df["jd_id"].nunique()),
            "label_counts": df["resume_category"].value_counts().to_dict(),
            "binary_match_rate": float((df["similarity_label"] >= 0.5).mean()),
            "median_resume_words": float(df["resume_text"].str.split().str.len().median()),
            "median_jd_words": float(df["jd_text"].str.split().str.len().median()),
        }
    stats["resumes_shared_train_test"] = len(set(train["resume_id"]) & set(test["resume_id"]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
