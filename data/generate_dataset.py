"""
Synthetic resume-JD pair dataset generator.

Stand-in for the three Kaggle datasets named in the project synopsis
(Resume Dataset, LinkedIn Job Postings, Resume-vs-JD Matching Dataset).
Those require a Kaggle API token that isn't available in this environment,
so this script builds a smaller, fully self-contained dataset with the same
shape: categorized resumes, categorized job descriptions, and resume-JD
pairs labelled with a graded fit score.

It also builds a name -> demographic-proxy-group map used ONLY by the bias
probe later (src/bias_probe.py). The proxy group is assigned independently
of the skills/category sampled for a resume, so any correlation the probe
later finds between embeddings and proxy group reflects genuine leakage
from the name/text, not a confound baked into the data.
"""
import csv
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent

CATEGORIES = {
    "Information Technology": [
        "python", "java", "sql", "aws", "docker", "kubernetes", "react",
        "linux", "git", "machine learning", "rest api", "agile development",
        "ci cd pipelines", "cloud computing", "javascript",
    ],
    "Human Resources": [
        "recruitment", "onboarding", "payroll processing", "employee relations",
        "hr policy design", "performance management", "talent acquisition",
        "conflict resolution", "benefits administration", "labor law",
        "training and development", "compliance auditing", "hris systems",
        "workforce planning",
    ],
    "Sales": [
        "lead generation", "crm software", "salesforce", "negotiation",
        "account management", "cold calling", "pipeline management",
        "b2b sales", "quota attainment", "customer retention",
        "sales forecasting", "prospecting", "closing deals",
        "territory management",
    ],
    "Healthcare": [
        "patient care", "clinical documentation", "ehr systems",
        "hipaa compliance", "medical terminology", "nursing care",
        "diagnosis support", "treatment planning", "triage", "vital signs monitoring",
        "infection control", "patient safety", "care coordination",
        "medical records management",
    ],
    "Engineering": [
        "autocad", "solidworks", "structural analysis", "project management",
        "matlab", "six sigma", "quality control", "manufacturing processes",
        "cad design", "mechanical design", "thermodynamics",
        "materials science", "blueprint reading", "process improvement",
    ],
    "Finance": [
        "financial modeling", "excel", "accounting", "budgeting",
        "forecasting", "gaap standards", "financial analysis",
        "risk management", "auditing", "bloomberg terminal",
        "variance analysis", "investment analysis", "tax preparation",
        "reconciliation",
    ],
}

EXPERIENCE_LEVELS = [
    "1 year", "2 years", "3 years", "4 years", "5 years", "6 years",
    "7 years", "8 years", "10 years",
]

# Generic cross-cutting skills shared across every category. These dilute
# pure category-keyword separability so a bag-of-words model can't solve
# the task by trivially detecting which disjoint vocabulary a document
# uses -- it has to weigh *which* skills overlap, not just *that* some do.
GENERIC_SKILLS = [
    "communication", "teamwork", "problem solving", "project management",
    "time management", "leadership", "microsoft excel", "attention to detail",
    "critical thinking", "adaptability",
]

# Short paraphrase variants per skill so literal keyword matching is noisy
# (e.g. "python" vs "python programming" vs "coding in python").
SKILL_PARAPHRASES = {
    "python": ["python", "python programming", "coding in python"],
    "communication": ["communication", "verbal and written communication", "strong communication skills"],
    "project management": ["project management", "managing projects", "project coordination"],
    "leadership": ["leadership", "leading teams", "team leadership"],
}


def paraphrase(skill):
    options = SKILL_PARAPHRASES.get(skill, [skill])
    return random.choice(options)


FILLER_SENTENCES = [
    "Known for strong attendance and a positive attitude in the workplace.",
    "Enjoys mentoring junior colleagues and participating in company events.",
    "Based in a major metropolitan area and open to relocation.",
    "Holds a bachelor's degree and completed relevant continuing education.",
    "Comfortable working both independently and as part of a larger team.",
]

MALE_PROXY_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard",
    "Joseph", "Thomas", "Charles", "Daniel", "Matthew", "Andrew", "Mark",
]
FEMALE_PROXY_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Sandra",
]

N_RESUMES_PER_CATEGORY = 60
N_JDS_PER_CATEGORY = 60
SKILLS_PER_RESUME = (5, 8)
SKILLS_PER_JD = (4, 7)


def make_resume_text(name, category, skills):
    # Deliberately does NOT mention `category` anywhere in the text: an
    # earlier version embedded the literal category name in every resume
    # and JD, which let every model trivially learn "same category name
    # appears in both documents" instead of reasoning about actual skill
    # overlap. Matching must be inferable from skills alone.
    exp = random.choice(EXPERIENCE_LEVELS)
    phrased = [paraphrase(s) for s in skills]
    skills_sentence = ", ".join(phrased[:-1]) + f", and {phrased[-1]}" if len(phrased) > 1 else phrased[0]
    text = (
        f"{name} is a professional with {exp} of experience. "
        f"Core skills include {skills_sentence}. "
        f"Proven track record of delivering results, "
        f"with hands-on experience applying {phrased[0]} and {phrased[-1]} "
        f"in fast-paced team environments."
    )
    if random.random() < 0.5:
        text += " " + random.choice(FILLER_SENTENCES)
    return text


def make_jd_text(category, skills):
    phrased = [paraphrase(s) for s in skills]
    skills_sentence = ", ".join(phrased[:-1]) + f", and {phrased[-1]}" if len(phrased) > 1 else phrased[0]
    text = (
        f"We are seeking a professional with strong skills in "
        f"{skills_sentence}. The ideal candidate has hands-on experience with "
        f"{phrased[0]} and {phrased[-1]}, and will be responsible for applying "
        f"these skills in a collaborative team setting."
    )
    if random.random() < 0.3:
        text += " " + random.choice(FILLER_SENTENCES)
    return text


def generate_documents():
    resumes = []  # (resume_id, name, category, skills(list), text, proxy_group)
    jds = []  # (jd_id, category, skills(list), text)

    resume_id = 0
    jd_id = 0
    for category, skill_pool in CATEGORIES.items():
        for _ in range(N_RESUMES_PER_CATEGORY):
            k = random.randint(*SKILLS_PER_RESUME)
            n_generic = random.randint(1, 3)
            skills = random.sample(skill_pool, max(k - n_generic, 2)) + random.sample(GENERIC_SKILLS, n_generic)
            random.shuffle(skills)
            proxy_group = random.choice(["A", "B"])
            name = random.choice(MALE_PROXY_NAMES if proxy_group == "A" else FEMALE_PROXY_NAMES)
            text = make_resume_text(name, category, skills)
            resumes.append((f"R{resume_id}", name, category, skills, text, proxy_group))
            resume_id += 1

        for _ in range(N_JDS_PER_CATEGORY):
            k = random.randint(*SKILLS_PER_JD)
            n_generic = random.randint(1, 2)
            skills = random.sample(skill_pool, max(k - n_generic, 2)) + random.sample(GENERIC_SKILLS, n_generic)
            random.shuffle(skills)
            text = make_jd_text(category, skills)
            jds.append((f"J{jd_id}", category, skills, text))
            jd_id += 1

    return resumes, jds


def skill_overlap_score(resume_skills, jd_skills):
    rs, js = set(resume_skills), set(jd_skills)
    if not js:
        return 0.0
    return len(rs & js) / len(js)


def build_pairs(resumes, jds):
    rows = []
    for r_id, name, r_cat, r_skills, r_text, proxy_group in resumes:
        # same-category JDs: positive pairs, graded by actual skill overlap
        same_cat_jds = [j for j in jds if j[1] == r_cat]
        diff_cat_jds = [j for j in jds if j[1] != r_cat]

        for j_id, j_cat, j_skills, j_text in random.sample(same_cat_jds, min(6, len(same_cat_jds))):
            overlap = skill_overlap_score(r_skills, j_skills)
            label = 0.45 + 0.5 * overlap + random.uniform(-0.08, 0.08)
            rows.append([r_id, j_id, r_text, j_text, r_cat, j_cat, round(min(max(label, 0.0), 1.0), 3)])

        for j_id, j_cat, j_skills, j_text in random.sample(diff_cat_jds, min(6, len(diff_cat_jds))):
            overlap = skill_overlap_score(r_skills, j_skills)
            label = 0.15 + 0.45 * overlap + random.uniform(-0.08, 0.08)
            rows.append([r_id, j_id, r_text, j_text, r_cat, j_cat, round(min(max(label, 0.0), 1.0), 3)])

    random.shuffle(rows)
    return rows


def split_rows(rows, train_frac=0.70, val_frac=0.15):
    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    for i, row in enumerate(rows):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        row.append(split)
    return rows


def main():
    resumes, jds = generate_documents()
    rows = build_pairs(resumes, jds)
    rows = split_rows(rows)

    pairs_path = DATA_DIR / "pairs.csv"
    with open(pairs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resume_id", "jd_id", "resume_text", "jd_text",
                          "resume_category", "jd_category", "similarity_label", "split"])
        writer.writerows(rows)

    proxy_path = DATA_DIR / "gender_proxy.csv"
    with open(proxy_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resume_id", "name", "proxy_group"])
        for r_id, name, _cat, _skills, _text, proxy_group in resumes:
            writer.writerow([r_id, name, proxy_group])

    skills_path = DATA_DIR / "skill_vocab.csv"
    with open(skills_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "skill"])
        for category, skill_pool in CATEGORIES.items():
            for skill in skill_pool:
                writer.writerow([category, skill])

    print(f"resumes={len(resumes)} jds={len(jds)} pairs={len(rows)}")
    print(f"wrote {pairs_path}")
    print(f"wrote {proxy_path}")
    print(f"wrote {skills_path}")


if __name__ == "__main__":
    main()
