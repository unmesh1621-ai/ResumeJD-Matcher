"""
Explainability layer: for a resume-JD pair, reports matched skills, missing
skills (skill gaps), and a raw keyword-overlap ratio, using the same fixed
skill vocabulary the dataset generator used. This gives a human-readable
reason behind a fit score, on top of whichever encoder produced it.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_skill_vocab():
    skills = set()
    with open(DATA_DIR / "skill_vocab.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills.add(row["skill"])
    return sorted(skills, key=len, reverse=True)  # longer phrases matched first


def find_skills(text, skill_vocab):
    # Normalize to space-joined word tokens so punctuation (e.g. "java,")
    # doesn't break a literal "surrounded by spaces" substring match.
    normalized = " " + " ".join(TOKEN_RE.findall(text.lower())) + " "
    found = set()
    for skill in skill_vocab:
        if f" {skill} " in normalized:
            found.add(skill)
    return found


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
    skill_vocab = load_skill_vocab()
    rows = []
    with open(DATA_DIR / "pairs.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test":
                rows.append(row)

    # A handful of worked examples spanning same-category (likely matches)
    # and cross-category (likely non-matches) pairs, for the report.
    examples = []
    for row in rows[:40]:
        if len(examples) >= 6:
            break
        explanation = explain_pair(row["resume_text"], row["jd_text"], skill_vocab)
        examples.append({
            "resume_id": row["resume_id"],
            "jd_id": row["jd_id"],
            "resume_category": row["resume_category"],
            "jd_category": row["jd_category"],
            "similarity_label": float(row["similarity_label"]),
            **explanation,
        })

    out_path = RESULTS_DIR / "explainability_examples.json"
    out_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(examples)} examples)")
    for ex in examples:
        print(f"{ex['resume_id']} ({ex['resume_category']}) vs {ex['jd_id']} ({ex['jd_category']}) "
              f"label={ex['similarity_label']:.2f} matched={ex['matched_skills']} missing={ex['missing_skills']}")


if __name__ == "__main__":
    main()
