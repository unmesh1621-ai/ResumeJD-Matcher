"""
Install the final SBERT configuration (backbone x scorer) as
results/<data>/sbert/, the directory every downstream script (evaluate,
bias probe, debiasing, demo) reads.

The choice comes from grouped 3-fold cross-validation over train+val
(results/real/model_selection.json, written by src/cv_select.py). If no CV
result exists yet, it falls back to the single validation split. Test
metrics are printed for information only and never used for the choice.

Usage: python src/select_sbert.py [--data real]
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

# CV config name -> directory of its run(s) on the main train/val/test split
MAIN_SPLIT_DIR = {
    "sbert_minilm_cosine": "ablation_cosine/sbert",
    "sbert_minilm_interaction": "ablation_interaction/sbert_minilm",
    "sbert_mpnet_interaction": "sbert_mpnet_interaction",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="real")
    args = parser.parse_args()
    results = ROOT / "results" / args.data

    for name, rel in MAIN_SPLIT_DIR.items():
        path = results / rel / "metrics.json"
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))
            print(f"{name:<26} [{rel}] val_auc={m['val_auc']:.4f}  (test_auc={m['test_auc']:.4f}, info only)")

    selection_file = results / "model_selection.json"
    selection = json.loads(selection_file.read_text(encoding="utf-8")) if selection_file.exists() else {}
    chosen = selection.get("sbert", {}).get("selected")
    if chosen:
        rule = "max mean AUC over grouped 3-fold CV on train+val (results/real/model_selection.json)"
    else:
        found = [(json.loads((results / rel / "metrics.json").read_text(encoding="utf-8"))["val_auc"], name)
                 for name, rel in MAIN_SPLIT_DIR.items() if (results / rel / "metrics.json").exists()]
        chosen = max(found)[1]
        rule = "max validation AUC (no CV results available)"
    rel = MAIN_SPLIT_DIR[chosen]
    print(f"\nselected: {chosen} -> {rel}  ({rule})")

    dest = results / "sbert"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(results / rel, dest)
    m = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
    m.update(selected_from=rel, selected_config=chosen, selection_rule=rule)
    (dest / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    for seed_file in (dest / "seeds").glob("*.json"):
        s = json.loads(seed_file.read_text(encoding="utf-8"))
        s.update(selected_from=rel, selected_config=chosen)
        seed_file.write_text(json.dumps(s, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
