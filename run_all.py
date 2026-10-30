"""
Run the full pipeline end to end on the real dataset:

  1. download + preprocess data
  2. train the 4 encoders (MLP first: its embeddings warm-start CNN/BiLSTM)
     - scratch models: 3 seeds each (42, 7, 123), SBERT: seed 42
  3. cross-model comparison + error analysis
  4. explainability examples
  5. bias audit of all encoders, then adversarial debiasing of SBERT

Usage: python run_all.py [--skip-data] [--seeds 42 7 123]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 123])
    args = parser.parse_args()

    if not args.skip_data:
        run("data/download_real.py")
        run("data/prepare_real.py")

    # seed 42 first for every model: it is the "main" run whose checkpoint,
    # plots and predictions are kept (and MLP-42 provides the warm-start).
    for model in ("mlp", "cnn", "bilstm"):
        for seed in args.seeds:
            run("src/train.py", "--model", model, "--seed", str(seed))
    run("src/train.py", "--model", "sbert")

    run("src/evaluate.py")
    run("src/explainability.py")
    run("src/bias_probe.py")
    run("src/debias.py")


if __name__ == "__main__":
    main()
