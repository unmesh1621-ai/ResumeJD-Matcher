"""
Score one resume against one job description with every trained encoder,
plus the explainability layer's matched / missing skills.

Usage:
  python src/predict.py --resume path/to/resume.pdf --jd path/to/jd.txt
  python src/predict.py --resume resume.txt --jd-text "We are hiring a data analyst ..."
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from model_io import load_encoder, available_models
from explainability import load_skill_vocab, explain_pair
from prepare_real import clean_text  # same cleaning as the training data

_ENCODER_CACHE = {}


def read_document(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if path.suffix.lower() == ".docx":
        import zipfile, re
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
        return re.sub(r"<[^>]+>", " ", xml.replace("</w:p>", "\n"))
    return path.read_text(encoding="utf-8", errors="ignore")


def get_encoder(key, data="real"):
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = load_encoder(key, data)
    return _ENCODER_CACHE[key]


def score_pair(resume_text, jd_text, data="real", models=None):
    resume_text, jd_text = clean_text(resume_text), clean_text(jd_text)
    results = []
    for key in models or available_models(data):
        enc = get_encoder(key, data)
        score = float(enc.score([resume_text], [jd_text])[0])
        results.append({"model": enc.label, "score": score, "threshold": enc.threshold,
                        "match": score >= enc.threshold,
                        "test_auc": enc.metrics.get("test_auc")})
    explanation = explain_pair(resume_text, jd_text, load_skill_vocab(data))
    return results, explanation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jd")
    group.add_argument("--jd-text")
    args = parser.parse_args()

    resume_text = read_document(args.resume)
    jd_text = args.jd_text if args.jd_text else read_document(args.jd)
    results, explanation = score_pair(resume_text, jd_text)

    print(f"\n{'Model':<10} {'Score':>7} {'Thresh':>7}  Decision   (test AUC)")
    for r in results:
        print(f"{r['model']:<10} {r['score']:>7.3f} {r['threshold']:>7.3f}  "
              f"{'MATCH   ' if r['match'] else 'no match'}   ({r['test_auc']:.3f})")
    print(f"\nMatched skills : {', '.join(explanation['matched_skills']) or '-'}")
    print(f"Missing skills : {', '.join(explanation['missing_skills']) or '-'}")
    print(f"Keyword overlap: {explanation['keyword_overlap_ratio']:.0%} of JD skills found in resume")


if __name__ == "__main__":
    main()
