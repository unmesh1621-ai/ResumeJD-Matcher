"""
Explainability layer: for a resume-JD pair, reports matched skills, missing
skills (skill gaps), and a keyword-overlap ratio. This gives a
human-readable reason behind a fit score, on top of whichever encoder
produced it.

Skill vocabularies:
  synthetic -> data/skill_vocab.csv (the generator's own skill list)
  real      -> data/skill_vocab_real.csv (~200 common real-world skills,
               each with optional '|'-separated aliases mapped back to the
               canonical skill name, e.g. "k8s" -> "kubernetes")

Usage: python src/explainability.py [--data real]
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_ROOT = ROOT / "results"

VOCAB_FILE = {"synthetic": "skill_vocab.csv", "real": "skill_vocab_real.csv"}
PAIRS_FILE = {"synthetic": "pairs.csv", "real": "pairs_real.csv"}


def load_skill_vocab(data="real"):
    """Returns a list of (canonical_skill, compiled_regex) pairs, longest
    surface forms first so e.g. "machine learning" wins over "learning"."""
    surface_to_skill = {}
    with open(DATA_DIR / VOCAB_FILE[data], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            skill = row["skill"].strip().lower()
            surface_to_skill[skill] = skill
            for alias in (row.get("aliases") or "").split("|"):
                if alias.strip():
                    surface_to_skill[alias.strip().lower()] = skill
    vocab = []
    for surface in sorted(surface_to_skill, key=len, reverse=True):
        # word-boundary match that still allows symbols inside the skill
        # (c++, c#, .net, node.js, ci/cd)
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(surface) + r"(?![a-z0-9+#])")
        vocab.append((surface_to_skill[surface], pattern))
    return vocab


def find_skills(text, skill_vocab):
    text = " " + re.sub(r"\s+", " ", text.lower()) + " "
    return {skill for skill, pattern in skill_vocab if pattern.search(text)}


def explain_pair(resume_text, jd_text, skill_vocab):
    resume_skills = find_skills(resume_text, skill_vocab)
    jd_skills = find_skills(jd_text, skill_vocab)
    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    overlap_ratio = len(matched) / len(jd_skills) if jd_skills else 0.0
    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": matched,
        "missing_skills": missing,
        "keyword_overlap_ratio": round(overlap_ratio, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="real", choices=["real", "synthetic"])
    args = parser.parse_args()

    skill_vocab = load_skill_vocab(args.data)
    with open(DATA_DIR / PAIRS_FILE[args.data], encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

    # How well does raw keyword overlap alone track the fit label? This is a
    # non-neural reference point for the encoders' Spearman correlation.
    from scipy.stats import spearmanr
    overlaps, labels = [], []
    for row in rows:
        overlaps.append(explain_pair(row["resume_text"], row["jd_text"], skill_vocab)["keyword_overlap_ratio"])
        labels.append(float(row["similarity_label"]))
    rho = spearmanr(overlaps, labels).correlation

    # Worked examples: two per fit class, for the report / viva.
    examples, per_class = [], {}
    for row in rows:
        cls = row["resume_category"]
        if per_class.get(cls, 0) >= 2:
            continue
        explanation = explain_pair(row["resume_text"], row["jd_text"], skill_vocab)
        if not explanation["jd_skills"]:
            continue
        per_class[cls] = per_class.get(cls, 0) + 1
        examples.append({
            "resume_id": row["resume_id"],
            "jd_id": row["jd_id"],
            "fit_class": cls,
            "similarity_label": float(row["similarity_label"]),
            **explanation,
        })

    out = {
        "data": args.data,
        "n_test_pairs": len(rows),
        "keyword_overlap_vs_label_spearman": float(rho),
        "mean_overlap_by_class": {
            c: float(sum(o for o, r in zip(overlaps, rows) if r["resume_category"] == c)
                     / max(1, sum(1 for r in rows if r["resume_category"] == c)))
            for c in sorted({r["resume_category"] for r in rows})
        },
        "examples": examples,
    }
    out_dir = RESULTS_ROOT / args.data
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "explainability_examples.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"keyword-overlap vs label Spearman = {rho:.4f}; mean overlap by class = {out['mean_overlap_by_class']}")
    for ex in examples:
        print(f"{ex['fit_class']:>13}  matched={ex['matched_skills'][:8]}  missing={ex['missing_skills'][:8]}")


if __name__ == "__main__":
    main()
