"""
Run the full pipeline end to end on the real dataset:

  1. download + preprocess data
  2. baselines (resume-prior shortcut, TF-IDF, zero-shot sentence encoders)
  3. train all model variants (scripts/train_all.ps1 on Windows, the same
     sequence inline elsewhere): scratch models x 3 seeds with the cosine
     scorer, interaction-head ablations, SBERT backbone/scorer candidates
  4. pick the final SBERT on validation AUC
  5. comparison + error analysis, explainability
  6. bias audit of all encoders, adversarial debiasing of SBERT

Usage: python run_all.py [--skip-data] [--skip-train]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(*args):
    cmd = [sys.executable, *args]
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_all():
    for model in ("mlp", "cnn", "bilstm"):
        for seed in (42, 7, 123):
            run("src/train.py", "--model", model, "--seed", str(seed), "--head", "cosine")
    for model in ("mlp", "cnn", "bilstm"):
        run("src/train.py", "--model", model, "--head", "interaction", "--out", f"ablation_interaction/{model}")
    run("src/train.py", "--model", "sbert", "--backbone", "minilm", "--head", "cosine",
        "--out", "ablation_cosine/sbert")
    run("src/train.py", "--model", "sbert", "--backbone", "minilm", "--head", "interaction",
        "--out", "ablation_interaction/sbert_minilm")
    run("src/train.py", "--model", "sbert", "--backbone", "mpnet", "--head", "cosine",
        "--out", "sbert_mpnet_cosine", "--epochs", "5")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if not args.skip_data:
        run("data/download_real.py")
        run("data/prepare_real.py")
    run("src/baselines.py")
    if not args.skip_train:
        train_all()
    run("src/select_sbert.py")
    run("src/evaluate.py")
    run("src/explainability.py")
    run("src/bias_probe.py")
    run("src/debias.py")


if __name__ == "__main__":
    main()
