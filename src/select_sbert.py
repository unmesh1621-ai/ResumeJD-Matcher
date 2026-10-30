"""
Choose the final SBERT configuration (backbone x scorer) by VALIDATION AUC
and copy it to results/<data>/sbert/, the directory every downstream script
(evaluate, bias probe, debiasing, demo) reads. Test metrics are printed for
information only and are never used for the choice.

Usage: python src/select_sbert.py [--data real]
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

CANDIDATES = [
    "ablation_cosine/sbert",               # MiniLM, cosine (first run)
    "ablation_interaction/sbert_minilm",   # MiniLM, interaction head
    "sbert_mpnet_cosine",                  # MPNet, cosine
    "sbert_mpnet_interaction",             # MPNet, interaction head
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="real")
    args = parser.parse_args()
    results = ROOT / "results" / args.data

    found = []
    for rel in CANDIDATES:
        path = results / rel / "metrics.json"
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))
            found.append((m["val_auc"], rel, m))
            print(f"{rel:<38} val_auc={m['val_auc']:.4f}  (test_auc={m['test_auc']:.4f}, info only)")
    if not found:
        raise SystemExit("no SBERT candidates found")
    val_auc, rel, m = max(found, key=lambda t: t[0])
    print(f"\nselected by val AUC: {rel} ({val_auc:.4f})")

    dest = results / "sbert"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(results / rel, dest)
    m["selected_from"] = rel
    m["selection_rule"] = "max validation AUC over SBERT candidates " + ", ".join(CANDIDATES)
    (dest / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    for seed_file in (dest / "seeds").glob("*.json"):
        s = json.loads(seed_file.read_text(encoding="utf-8"))
        s["selected_from"] = rel
        seed_file.write_text(json.dumps(s, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
