"""
Download the real resume-JD fit dataset used for the final evaluation.

Source: HuggingFace dataset `cnamuangtoun/resume-job-description-fit`
(public, ungated). Each row is a real resume text, a real job description
text, and a human-assigned label in {No Fit, Potential Fit, Good Fit}.
This replaces the Kaggle sources named in the synopsis, which could not be
fetched without Kaggle API credentials.

Usage: python data/download_real.py
"""
import urllib.request
from pathlib import Path

REPO = "cnamuangtoun/resume-job-description-fit"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
OUT_DIR = Path(__file__).parent / "real"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "test.csv"):
        dest = OUT_DIR / name
        if dest.exists():
            print(f"{dest} already exists, skipping")
            continue
        print(f"downloading {BASE_URL}/{name} -> {dest}")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)
        print(f"  {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
