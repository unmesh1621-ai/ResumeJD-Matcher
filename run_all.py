"""
Run the full pipeline end to end on the real dataset:

  1. download + preprocess data
  2. baselines (resume-prior shortcut, TF-IDF, zero-shot sentence encoders)
  3. grouped 3-fold CV over train+val for every candidate config (model
     selection; test never touched), then train the variants on the main
     split: scratch models x 3 seeds, ablations, SBERT candidates
  4. cv_select.py picks each architecture's config on mean CV AUC;
     select_sbert.py installs the chosen SBERT
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
    """Main-split runs. Scorer per model as chosen by grouped 3-fold CV:
    MLP, 1D-CNN and BiLSTM cosine; SBERT = MPNet + interaction head."""
    seeds = ("42", "7", "123")
    for s in seeds:  # MLP first: its seed-42 embeddings warm-start CNN / BiLSTM
        run("src/train.py", "--model", "mlp", "--seed", s, "--head", "cosine")
    run("src/train.py", "--model", "mlp", "--head", "interaction", "--out", "ablation_interaction/mlp")
    for s in seeds:
        run("src/train.py", "--model", "cnn", "--seed", s, "--head", "cosine")
        run("src/train.py", "--model", "cnn", "--seed", s, "--head", "interaction", "--out", "ablation_interaction/cnn")
    for s in seeds:
        run("src/train.py", "--model", "bilstm", "--seed", s, "--head", "cosine")
    run("src/train.py", "--model", "bilstm", "--head", "interaction", "--out", "ablation_interaction/bilstm")
    run("src/train.py", "--model", "sbert", "--backbone", "minilm", "--head", "cosine",
        "--out", "ablation_cosine/sbert")
    run("src/train.py", "--model", "sbert", "--backbone", "minilm", "--head", "interaction",
        "--out", "ablation_interaction/sbert_minilm")
    for s in seeds:
        run("src/train.py", "--model", "sbert", "--backbone", "mpnet", "--head", "interaction",
            "--out", "sbert_mpnet_interaction", "--epochs", "5", "--seed", s)


def cross_validate():
    """Grouped 3-fold CV over train+val for every candidate configuration
    (same list as scripts/cv_select.ps1); test is never touched."""
    configs = [("mlp_cosine", ["--model", "mlp", "--head", "cosine"]),
               ("mlp_interaction", ["--model", "mlp", "--head", "interaction"]),
               ("cnn_cosine", ["--model", "cnn", "--head", "cosine"]),
               ("cnn_interaction", ["--model", "cnn", "--head", "interaction"]),
               ("bilstm_cosine", ["--model", "bilstm", "--head", "cosine"]),
               ("bilstm_interaction", ["--model", "bilstm", "--head", "interaction"]),
               ("sbert_minilm_cosine", ["--model", "sbert", "--backbone", "minilm", "--head", "cosine"]),
               ("sbert_minilm_interaction", ["--model", "sbert", "--backbone", "minilm", "--head", "interaction"]),
               ("sbert_mpnet_interaction", ["--model", "sbert", "--backbone", "mpnet", "--head", "interaction",
                                            "--epochs", "5"])]
    for _, args in configs:
        for fold in range(3):
            run("src/train.py", *args, "--cv-folds", "3", "--fold", str(fold))


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
        cross_validate()
        train_all()
    run("src/cv_select.py")
    run("src/select_sbert.py")
    run("src/evaluate.py")
    run("src/explainability.py")
    run("src/bias_probe.py")
    run("src/debias.py")


if __name__ == "__main__":
    main()
