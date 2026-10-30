# Resume–JD Matcher (ICT 4442 Deep Learning Mini Project)

**Comparative Analysis of Deep Learning Architectures for Bias-Aware Resume–Job Description Matching**

Team: Unmesh Vardhan (230911030) · ShanmugSrinivas (230911394) · Tanmay Sharma (230911294)

Four document encoders from four architecture families are trained as Siamese (shared-weight) twin
networks under one common protocol. Each one maps a resume and a job description to vectors, and
the **cosine similarity** between them is the fit score:

| # | Model | Family | Owner |
|---|---|---|---|
| 1 | Siamese **MLP** over mean-pooled word embeddings | feed-forward baseline | Unmesh Vardhan |
| 2 | Siamese **1D-CNN** (multi-kernel, Kim 2014) | convolutional | ShanmugSrinivas |
| 3 | Siamese **BiLSTM** (masked mean-pooling) | recurrent | Tanmay Sharma |
| 4 | Fine-tuned **SBERT** bi-encoder (all-MiniLM-L6-v2) | Transformer | Unmesh Vardhan |

On top of the best encoder we add:
- an **explainability layer** (matched skills, missing skills and keyword overlap);
- a **bias audit** (counterfactual name-swap probe plus TPR gap);
- **adversarial debiasing** with a gradient-reversal layer.

## Results (held-out test split)

All numbers come from `results/real/` and were produced by the scripts in this repo. The scratch models
are mean ± std over 3 seeds; SBERT is one seed. Full table: `results/real/comparison.md`.

| Model | Params | Accuracy | F1 | ROC-AUC | P@1 | P@3 | Spearman ρ |
|---|---|---|---|---|---|---|---|
| MLP | 2.57 M | 0.582 ± 0.004 | 0.630 ± 0.053 | 0.616 ± 0.018 | 0.484 ± 0.047 | 0.462 ± 0.030 | 0.188 ± 0.032 |
| 1D-CNN | 0.14 M* | 0.569 ± 0.029 | 0.615 ± 0.055 | 0.580 ± 0.046 | 0.394 ± 0.035 | 0.412 ± 0.005 | 0.112 ± 0.076 |
| BiLSTM | 0.11 M* | 0.592 ± 0.012 | 0.624 ± 0.016 | 0.630 ± 0.015 | 0.455 ± 0.018 | 0.430 ± 0.021 | 0.226 ± 0.026 |
| **SBERT (fine-tuned)** | 22.7 M | **0.683** | **0.728** | **0.740** | **0.676** | **0.594** | **0.408** |

\* trainable parameters only. CNN and BiLSTM reuse the MLP's trained word embeddings as a frozen warm start.

Reference points:
- A random ranking scores P@3 = 0.430.
- The same SBERT checkpoint **before fine-tuning** reaches test AUC 0.622.
- Raw keyword overlap from the explainability layer correlates with the label at only ρ = 0.06.

**Discussion (short).** Fine-tuned SBERT wins on every metric. It is the only model whose ranking of
candidates per job is clearly better than random (P@1 0.68 vs. 0.43). This points to the value of
pretrained language knowledge with only ~640 distinct training resumes. The three scratch models
overfit quickly: train MSE keeps falling while validation peaks within 3–12 epochs, because the same
resumes recur across many pairs. They end up close to each other, with the BiLSTM marginally best.
The 1D-CNN is the least stable across seeds (AUC 0.52–0.62). Local n-gram features add little when
fit depends on overall profile and seniority rather than on specific phrases.

**Error analysis** (`results/real/error_analysis/`):
- Every model's dominant failure is **false positives on "No Fit" pairs**.
- 235 test pairs are misclassified by all four models, and 225 of them are *No Fit*.
- Errors fall as resumes get longer. For SBERT the error rate is 38% on short resumes vs 27% on long ones.
- The scratch models make very similar mistakes (MLP and BiLSTM error Jaccard 0.67). SBERT's errors overlap
  much less with the others (Jaccard ≈ 0.35).

### Bias audit and adversarial debiasing (`results/real/fairness_report.json`)

Each test resume is duplicated with a group-A or group-B first name prepended; otherwise the two copies are
identical. A linear probe trained on disjoint resumes *and disjoint names* tries to recover the group.

| Encoder | Probe acc. (chance 0.50) | Mean \|Δscore\| | TPR gap (A−B) | Test AUC |
|---|---|---|---|---|
| MLP / 1D-CNN / BiLSTM | 0.504 / 0.513 / 0.506 | ≤ 0.015 | ≤ 0.008 | — |
| SBERT (before) | **0.579** | 0.0045 | +0.0022 | 0.740 |
| SBERT control (λ = 0, same extra training) | 0.554 | 0.0028 | −0.0022 | 0.755 |
| **SBERT adversarially debiased (GRL, λ = 1)** | **0.517** | 0.0026 | **0.0000** | **0.762** |

- The scratch models carry no name signal, because names are mostly out of their vocabulary.
- SBERT's embedding leaks the name group: its pretrained knowledge of first names gives 0.58 on unseen names.
- Gradient-reversal training pushes probe accuracy almost to chance **without hurting matching AUC**.
  The control run shows that only part of the drop comes from the extra training itself.
- Caveat: the score-level effects were already small (under 1% TPR gap), and the probe test set is 954
  embeddings, roughly ±3% at 95% confidence.

## Data

The final results use **real** resume–JD pairs from the public HuggingFace dataset
[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit).
It has 8,000 pairs of real resumes and real job postings, each labelled *No Fit*, *Potential Fit* or
*Good Fit*. The Kaggle datasets named in the synopsis needed API credentials that weren't available,
and this dataset provides the direct pair-level supervision the synopsis asked for.

Preprocessing is in `data/prepare_real.py`:
- Clean the text: unicode-normalise; mask e-mails, URLs and phone numbers; drop duplicates.
- Map labels to graded targets: No Fit → 0.0, Potential → 0.5, Good → 1.0. A pair counts as a match if its label is ≥ 0.5.
- Split: the dataset's own test file is the **held-out test set**, and validation (15%) is carved from train.
  **No job description appears in more than one split**, so the test set measures generalisation to unseen jobs.

| Split | Pairs | Unique resumes | Unique JDs |
|---|---|---|---|
| train | 5,338 | 636 | 238 |
| val | 896 | 336 | 42 |
| test | 1,759 | 477 | 71 |

The earlier synthetic dataset from the interim report is still reproducible: `python data/generate_dataset.py`,
then `--data synthetic` on the scripts. Its results are in `results/` (top level, interim).

## Common evaluation protocol

All four models use:
- the same split;
- the same objective: MSE between cosine score and graded label;
- checkpoint selection on validation ROC-AUC;
- a decision threshold tuned on **validation** and applied unchanged to test.

Reported test metrics are Accuracy, Precision, Recall, F1, ROC-AUC, **Precision@k** (resumes ranked
per job, k = 1/3/5) and Spearman ρ against the graded label. The three scratch-trained models are run
with 3 seeds (mean ± std). The scratch models share one tokenizer and vocabulary (20k tokens,
512-token sequences) built on the train split only. SBERT uses its own pretrained WordPiece tokenizer.

## Layout

```
data/
  download_real.py      fetch the real dataset
  prepare_real.py       cleaning, label mapping, JD-disjoint split -> pairs_real.csv
  skill_vocab_real.csv  ~200 skills + aliases for the explainability layer
  generate_dataset.py   (interim) synthetic dataset generator
src/
  models.py             the 4 Siamese encoders
  preprocessing.py      shared tokenizer / vocab for the scratch models
  train.py              common training + test evaluation loop
  evaluate.py           comparison table, ROC/bar plots, error analysis
  explainability.py     matched / missing skills
  bias_probe.py         counterfactual name-swap bias audit (all encoders)
  debias.py             gradient-reversal adversarial debiasing of SBERT
  model_io.py           load any trained encoder
  predict.py            CLI: score a resume (pdf/docx/txt) against a JD
app.py                  Streamlit demo
run_all.py              whole pipeline in one command
results/real/           all final metrics, plots, predictions, reports
```

## Reproducing

```bash
pip install -r requirements.txt          # GPU: install the CUDA torch build first
python run_all.py                        # data -> 4 models -> evaluation -> fairness
# or step by step:
python data/download_real.py && python data/prepare_real.py
python src/train.py --model mlp          # MLP first (its embeddings warm-start CNN/BiLSTM)
python src/train.py --model cnn
python src/train.py --model bilstm
python src/train.py --model sbert
python src/evaluate.py
python src/explainability.py
python src/bias_probe.py
python src/debias.py
# demo
python src/predict.py --resume my_resume.pdf --jd-text "We are hiring a data analyst ..."
streamlit run app.py
```

## Use of external code / LLM assistance

The guidelines (§9) require disclosure. Pretrained weights come from `sentence-transformers/all-MiniLM-L6-v2`
(HuggingFace). The dataset is cited above. An LLM assistant (Claude) helped write parts of the code.
<!-- Team: list exactly which files/functions here before submission. -->
