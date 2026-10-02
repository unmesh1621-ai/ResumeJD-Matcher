const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, BorderStyle, ImageRun, AlignmentType, PageBreak,
  ShadingType, VerticalAlign,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const RESULTS = path.join(ROOT, "results");

function readJSON(p) {
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}

const metrics = {
  mlp: readJSON(path.join(RESULTS, "mlp", "metrics.json")),
  cnn: readJSON(path.join(RESULTS, "cnn", "metrics.json")),
  bilstm: readJSON(path.join(RESULTS, "bilstm", "metrics.json")),
  sbert: readJSON(path.join(RESULTS, "sbert", "metrics.json")),
};
const biasReport = readJSON(path.join(RESULTS, "bias_probe_report.json"));
const explainExamples = readJSON(path.join(RESULTS, "explainability_examples.json"));

const pct = (x) => (x * 100).toFixed(2) + "%";
const fmtAuc = (x) => x.toFixed(4);

// ---------- helpers ----------

function para(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts.run })],
    spacing: { after: 160 },
    ...opts.paragraph,
  });
}

function heading(text, level) {
  return new Paragraph({ text, heading: level, spacing: { before: 280, after: 160 } });
}

function bold(text) {
  return new TextRun({ text, bold: true });
}

function cell(text, opts = {}) {
  const children = Array.isArray(text) ? text : [new Paragraph({ children: [new TextRun({ text, bold: !!opts.bold, size: opts.size })] })];
  return new TableCell({
    children,
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.shade ? { type: ShadingType.CLEAR, fill: opts.shade } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
  });
}

function imageParagraph(filePath, width, height) {
  const data = fs.readFileSync(filePath);
  return new Paragraph({
    children: [new ImageRun({ data, transformation: { width, height }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
  });
}

function sideBySideImages(leftPath, rightPath, w, h) {
  const leftData = fs.readFileSync(leftPath);
  const rightData = fs.readFileSync(rightPath);
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      insideHorizontal: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: 4500, type: WidthType.DXA },
            children: [new Paragraph({
              children: [new ImageRun({ data: leftData, transformation: { width: w, height: h }, type: "png" })],
              alignment: AlignmentType.CENTER,
            })],
          }),
          new TableCell({
            width: { size: 4500, type: WidthType.DXA },
            children: [new Paragraph({
              children: [new ImageRun({ data: rightData, transformation: { width: w, height: h }, type: "png" })],
              alignment: AlignmentType.CENTER,
            })],
          }),
        ],
      }),
    ],
  });
}

function figureCaption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20 })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
  });
}

// ---------- Title page ----------

const titlePage = [
  new Paragraph({ text: "", spacing: { after: 800 } }),
  new Paragraph({
    children: [new TextRun({ text: "COMPARATIVE ANALYSIS OF DEEP LEARNING ARCHITECTURES FOR", bold: true, size: 32 })],
    alignment: AlignmentType.CENTER, spacing: { after: 120 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "BIAS-AWARE RESUME–JOB DESCRIPTION MATCHING", bold: true, size: 32 })],
    alignment: AlignmentType.CENTER, spacing: { after: 600 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Interim report on", size: 22 })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Deep Learning Project [ICT-4442]", size: 22 })],
    alignment: AlignmentType.CENTER, spacing: { after: 600 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Submitted By", bold: true, size: 24 })],
    alignment: AlignmentType.CENTER, spacing: { after: 240 },
  }),
  new Paragraph({ children: [new TextRun({ text: "Unmesh Vardhan (230911030)", size: 22 })], alignment: AlignmentType.CENTER, spacing: { after: 80 } }),
  new Paragraph({ children: [new TextRun({ text: "ShanmugSrinivas (230911394)", size: 22 })], alignment: AlignmentType.CENTER, spacing: { after: 80 } }),
  new Paragraph({ children: [new TextRun({ text: "Tanmay Sharma (230911294)", size: 22 })], alignment: AlignmentType.CENTER, spacing: { after: 800 } }),
  new Paragraph({
    children: [new TextRun({ text: "SCHOOL OF COMPUTER ENGINEERING", bold: true, size: 22 })],
    alignment: AlignmentType.CENTER, spacing: { after: 40 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "MANIPAL INSTITUTE OF TECHNOLOGY,", bold: true, size: 22 })],
    alignment: AlignmentType.CENTER, spacing: { after: 40 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "MANIPAL ACADEMY OF HIGHER EDUCATION", bold: true, size: 22 })],
    alignment: AlignmentType.CENTER, spacing: { after: 600 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "October 2026", size: 22 })],
    alignment: AlignmentType.CENTER,
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- Part B header ----------

const partBHeader = [
  new Paragraph({
    children: [new TextRun({ text: "PART B: INTERIM REPORT", bold: true, size: 28 })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Phase 2", size: 20 })],
    alignment: AlignmentType.CENTER, spacing: { after: 300 },
  }),
  para("Team No. and Names (with Registration Numbers):", { run: { bold: true } }),
  para("1. Unmesh Vardhan (230911030)"),
  para("2. ShanmugSrinivas (230911394)"),
  para("3. Tanmay Sharma (230911294)"),
  new Paragraph({ text: "", spacing: { after: 160 } }),
  para("Title of the Project:", { run: { bold: true } }),
  para("Comparative Analysis of Deep Learning Architectures for Bias-Aware Resume–Job Description Matching"),
  new Paragraph({ text: "", spacing: { after: 160 } }),
  para("GitHub Repository Link:", { run: { bold: true } }),
  new Paragraph({
    children: [new TextRun({ text: "[GITHUB-REPO-URL]", color: "0563C1", underline: {} })],
    spacing: { after: 300 },
  }),
];

// ---------- 1. Literature review ----------

const litRows = [
  ["Paper (Author, Year)", "Method", "Dataset", "Key Result", "Relevance to Project"],
  ["Zhu et al. (2018)", "PJFNN (CNN-based bipartite Person-Job Fit)", "Historical job applications", "Learns joint candidate/job representations from past applications", "Established neural person-job matching as a representation-learning problem"],
  ["Qin et al. (2018)", "Ability-aware attention network", "Resume/JD pairs", "Attention over ability requirements improves person-job fit prediction", "Motivates attention-style weighting over raw skill overlap"],
  ["Kim (2014)", "Multi-kernel 1D-CNN for sentence classification", "SST-2, MR, TREC", "Multi-kernel CNN matches deeper models on several benchmarks", "Basis for the multi-kernel (3, 4, 5) 1D-CNN encoder we implement"],
  ["Reimers & Gurevych (2019)", "Sentence-BERT (Siamese bi-encoder)", "NLI, STS benchmarks", "Siamese bi-encoders give efficient, high-quality sentence embeddings without pairwise cross-attention", "Direct basis for our fine-tuned SBERT bi-encoder (all-MiniLM-L6-v2)"],
  ["Sanh et al. (2019)", "DistilBERT (knowledge distillation)", "GLUE", "Keeps ~97% of BERT accuracy at 40% fewer parameters", "Motivates using a small, fast pretrained Transformer backbone under our time/compute budget"],
  ["De-Arteaga et al. (2019)", "Bias in Bios (TPR gender gap)", "Biography/occupation text", "Occupation classifiers leak gender bias even without explicit gender features", "Methodology adapted for our bias-probing audit on resume embeddings"],
  ["Zhang, Lemoine & Mitchell (2018)", "Adversarial debiasing (gradient reversal)", "UCI Adult, word embeddings", "An adversary trained to predict a protected attribute, opposed via gradient reversal, reduces encoded bias", "Target method for the adversarial debiasing extension planned as remaining work"],
];

function litTable() {
  const colWidths = [1800, 1800, 1500, 2400, 2500];
  const header = new TableRow({
    tableHeader: true,
    children: litRows[0].map((t, i) => cell(t, { bold: true, shade: "D9E2F3", width: colWidths[i] })),
  });
  const body = litRows.slice(1).map((r) => new TableRow({
    children: r.map((t, i) => cell(t, { width: colWidths[i] })),
  }));
  return new Table({ width: { size: 10000, type: WidthType.DXA }, rows: [header, ...body] });
}

const literatureSection = [
  heading("1. Literature Review", HeadingLevel.HEADING_1),
  para(
    "Neural approaches to resume–job matching were introduced by Zhu et al. [1], whose Person-Job Fit " +
    "Neural Network (PJFNN) uses a CNN-based bipartite architecture to learn joint representations of " +
    "candidates and jobs from historical applications, and extended by Qin et al. [2] with an " +
    "ability-aware attention mechanism. Matching pipelines have since moved toward pretrained sentence " +
    "encoders: Sentence-BERT [4] established the Siamese bi-encoder approach used in this project, " +
    "enabling efficient similarity computation between independently encoded documents rather than " +
    "expensive pairwise cross-attention; Sanh et al. [5] showed that a distilled Transformer keeps most " +
    "of a full model's accuracy at a fraction of the compute, which is why our SBERT encoder fine-tunes a " +
    "small pretrained MiniLM checkpoint rather than a full-size model. Kim [3] showed that multiple 1D " +
    "convolution kernels of different widths over word embeddings can match deeper networks on sentence " +
    "classification, which directly informed the three-kernel (3, 4, 5) design of our 1D-CNN encoder."
  ),
  para(
    "On the fairness side, De-Arteaga et al. [6] demonstrated in their “Bias in Bios” study that " +
    "occupation classifiers trained on biography text exhibit significant gender bias even when explicit " +
    "gender indicators are removed — a methodology this project adapts for auditing resume embeddings " +
    "via a probing classifier (Section 5). Zhang, Lemoine, and Mitchell [7] proposed adversarial " +
    "debiasing, training a secondary adversary to predict a protected attribute from a model's " +
    "representation while the main model is trained to fool it; this is the method earmarked for the " +
    "debiasing extension listed under Remaining Work."
  ),
  para("Table 1 groups these seven papers by method, dataset, and their relevance to this project.", { run: { italics: true } }),
  litTable(),
  new Paragraph({ text: "", spacing: { after: 300 } }),
];

// ---------- 2. Dataset & preprocessing ----------

const datasetSection = [
  heading("2. Dataset Acquisition & Preprocessing", HeadingLevel.HEADING_1),
  para(
    "Dataset. The project synopsis specifies three Kaggle datasets (a 2,400+ resume corpus, 124,000+ " +
    "LinkedIn job postings, and a resume-vs-JD matching dataset) as the source of resume–JD pairs. " +
    "Kaggle API credentials were not available in the development environment used for this interim " +
    "pass, so a smaller, self-contained synthetic dataset was generated locally with the same shape and " +
    "task: 150 synthetic resumes and 150 synthetic job descriptions across 6 job categories " +
    "(Information Technology, Human Resources, Sales, Healthcare, Engineering, Finance), each built from " +
    "a hand-written pool of 12–15 category skills plus 10 cross-cutting generic skills (communication, " +
    "project management, leadership, etc.) shared across all categories. The generic skills were added " +
    "deliberately so the matching task could not be solved by trivially detecting which disjoint " +
    "category-specific vocabulary a document uses — an early version of the dataset without them let a " +
    "bag-of-words baseline reach 100% validation accuracy, which would not have exercised any real " +
    "difference between the four architectures."
  ),
  para(
    "1,200 resume–JD pairs were built: same-category pairs received a graded similarity label of " +
    "0.45–0.95 scaled by actual skill overlap plus small random jitter, and cross-category pairs " +
    "received a label of 0.07–0.60 scaled the same way, so some cross-category pairs with coincidental " +
    "skill overlap (e.g. shared generic skills) score higher than some weak same-category pairs. Pairs " +
    "were split 70/15/15 into train/validation/test (840/180/180). This mirrors the real datasets' " +
    "category structure and the graded-fit-score supervision described in the synopsis, at a scale that " +
    "fits the compute and time available for this interim pass; it is documented here as an explicit " +
    "stand-in, not a substitute for running the full pipeline on the real Kaggle sources before final " +
    "submission (see Section 7)."
  ),
  para(
    "Bias-probing labels. Each synthetic resume is independently assigned a first name drawn from one of " +
    "two name lists associated with different demographic-proxy groups, chosen at random and independent " +
    "of the resume's skills/category. This name → proxy-group mapping is stored separately and is never " +
    "given to any matching model — it is used only afterward by the bias probe (Section 5), following the " +
    "counterfactual, name-based auditing methodology of De-Arteaga et al. [6]."
  ),
  para(
    "Preprocessing pipeline. The three scratch-trained models (MLP, 1D-CNN, BiLSTM) share one tokenizer: " +
    "a regex word tokenizer, a vocabulary of up to 5,000 tokens plus an <OOV> token built from the " +
    "training split only, and integer-encoded sequences padded/truncated to 128 tokens. The fitted " +
    "vocabulary is saved to data/vocab.json and reused unchanged by all three training runs, so every " +
    "scratch-trained model sees identical integer-encoded input. The SBERT encoder uses its own " +
    "pretrained WordPiece tokenizer (from sentence-transformers/all-MiniLM-L6-v2) instead, consistent " +
    "with how a pretrained checkpoint must be tokenized the way it was pretrained."
  ),
  new Paragraph({ text: "", spacing: { after: 200 } }),
];

// ---------- 3. Models implemented ----------

function modelSection(letter, title, key, description, figPrefix) {
  const m = metrics[key];
  const base = RESULTS;
  const trainingCurves = path.join(base, key, "training_curves.png");
  const confMatrix = path.join(base, key, "confusion_matrix.png");
  const rocCurve = path.join(base, key, "roc_curve.png");

  return [
    new Paragraph({
      children: [new TextRun({ text: `${letter}. ${title}`, bold: true, italics: true, size: 24 })],
      spacing: { before: 240, after: 120 },
    }),
    para(description),
    para(
      `Total trainable parameters: ${m.trainable_parameters.toLocaleString()}. Training ran ` +
      `${m.epochs_run} epochs (early stopping patience applied), with the best checkpoint at epoch ` +
      `${m.best_epoch} (val loss = ${m.best_val_loss.toFixed(4)}), taking ${m.train_seconds.toFixed(1)}s ` +
      `on ${m.device.toUpperCase()}. On the held-out validation pairs (binarizing the graded similarity ` +
      `label and the predicted cosine similarity at 0.5 to define match/no-match), it reached ` +
      `${pct(m.val_accuracy)} accuracy, ${pct(m.val_precision)} precision, ${pct(m.val_recall)} recall, ` +
      `${pct(m.val_f1)} F1, and AUC = ${fmtAuc(m.val_auc)}.`
    ),
    imageParagraph(trainingCurves, 540, 197),
    figureCaption(`Figure ${figPrefix}a: ${title} — training/validation loss and accuracy per epoch (best epoch marked).`),
    sideBySideImages(confMatrix, rocCurve, 260, 231),
    figureCaption(`Figure ${figPrefix}b/c: ${title} — validation confusion matrix and ROC curve (AUC = ${fmtAuc(m.val_auc)}).`),
  ];
}

const statusRows = [
  ["Model", "Owner (Member)", "Status", "Val Accuracy", "Val AUC", "Notes"],
  ["MLP", "Unmesh Vardhan", "Complete", pct(metrics.mlp.val_accuracy), fmtAuc(metrics.mlp.val_auc), `${metrics.mlp.epochs_run} epochs; bag-of-averaged-embeddings baseline, no sequence modelling`],
  ["1D-CNN", "ShanmugSrinivas", "Complete", pct(metrics.cnn.val_accuracy), fmtAuc(metrics.cnn.val_auc), `${metrics.cnn.epochs_run} epochs; multi-kernel (3,4,5) architecture, batch-normalized`],
  ["BiLSTM", "Tanmay Sharma", "Complete", pct(metrics.bilstm.val_accuracy), fmtAuc(metrics.bilstm.val_auc), `${metrics.bilstm.epochs_run} epochs; bidirectional recurrent model, mean-pooled over timesteps`],
  ["SBERT", "Unmesh Vardhan", "Complete", pct(metrics.sbert.val_accuracy), fmtAuc(metrics.sbert.val_auc), `${metrics.sbert.epochs_run} epochs; fine-tuned all-MiniLM-L6-v2 pretrained Transformer`],
];

function statusTable() {
  const widths = [1300, 2000, 1300, 1500, 1200, 3200];
  const header = new TableRow({
    tableHeader: true,
    children: statusRows[0].map((t, i) => cell(t, { bold: true, shade: "D9E2F3", width: widths[i] })),
  });
  const body = statusRows.slice(1).map((r) => new TableRow({ children: r.map((t, i) => cell(t, { width: widths[i] })) }));
  return new Table({ width: { size: 10500, type: WidthType.DXA }, rows: [header, ...body] });
}

const modelsSection = [
  heading("3. Models Implemented So Far", HeadingLevel.HEADING_1),
  para(
    "All four proposed encoder architectures have been implemented as shared-weight (Siamese) PyTorch " +
    "modules, trained, and evaluated on the validation split under a common cosine-similarity regression " +
    "objective (MSE loss against the graded similarity label). Each produces a fixed-length vector per " +
    "document; the fit score is the cosine similarity between the resume vector and the JD vector."
  ),
  ...modelSection("A", "Multi-Layer Perceptron (MLP)", "mlp",
    "The MLP baseline embeds each token (128-dim, trainable from scratch), masks and mean-pools the " +
    "embeddings over the real (non-padding) tokens — word order is discarded entirely — then passes the " +
    "averaged vector through Dense(64, ReLU) and a second Dense(64) layer with dropout (0.2 on the " +
    "embeddings, 0.4 before the final layer). This establishes the baseline achievable without any " +
    "sequence modelling.", "2"),
  ...modelSection("B", "1D Convolutional Neural Network (1D-CNN)", "cnn",
    "Following Kim (2014), three parallel Conv1d branches with kernel sizes 3, 4, and 5 (64 filters " +
    "each, 'same' padding) scan the embedded token sequence as bigram/trigram/four-gram detectors. " +
    "Padding positions are masked to −∞ before max-pooling so they cannot be selected as the strongest " +
    "activation, and each branch is batch-normalized before its ReLU — both were necessary fixes after an " +
    "initial version collapsed to predicting a near-constant similarity regardless of input, a known " +
    "failure mode for small-data Siamese CNNs without normalization. The three 64-dim pooled vectors are " +
    "concatenated, passed through Dense(64, ReLU) and LayerNorm, with dropout at 0.3/0.4/0.3.", "3"),
  ...modelSection("C", "Bidirectional LSTM (BiLSTM)", "bilstm",
    "A single Bidirectional LSTM(64) layer reads the embedded sequence in both directions " +
    "(mask_zero via packed sequences so padding is ignored). Rather than using only the final hidden " +
    "state — an initial version that bottlenecked badly and also collapsed on this small dataset — the " +
    "encoder mean-pools the LSTM's per-timestep outputs over the real tokens, giving a more robust " +
    "128-dim (64 forward + 64 backward) document vector that is then passed through Dense(64, ReLU) and " +
    "LayerNorm.", "4"),
  ...modelSection("D", "Fine-tuned SBERT (all-MiniLM-L6-v2)", "sbert",
    "The SBERT encoder wraps a pretrained sentence-transformers/all-MiniLM-L6-v2 checkpoint (6-layer " +
    "MiniLM, mean-pooled token embeddings) and fine-tunes all parameters end-to-end with AdamW " +
    "(lr = 2×10⁻⁵, weight decay 0.01) under the same cosine-similarity objective. This is a smaller, " +
    "faster pretrained backbone than the full DistilBERT named in the synopsis, chosen to keep " +
    "fine-tuning practical on CPU within this interim pass's time budget, while remaining a genuinely " +
    "fine-tuned pretrained Transformer in the same spirit as the original design.", "5"),
  heading("Summary", HeadingLevel.HEADING_2),
  para("Table 2 summarizes status and validation results for all four models as of this interim submission."),
  statusTable(),
  new Paragraph({ text: "", spacing: { after: 200 } }),
  para(
    "SBERT achieves the best validation AUC (0.9769) and accuracy (93.3%), consistent with the advantage " +
    "pretrained language representations typically bring over models trained from scratch on a few " +
    "hundred documents. The MLP and BiLSTM baselines perform close behind (AUC 0.9526 and 0.9575), " +
    "suggesting that on this short, keyword-dense synthetic task, a well-regularized averaging or " +
    "mean-pooling encoder captures most of the useful signal. The 1D-CNN trails the other three " +
    "(AUC 0.8367) even after fixing its initial training collapse; our working hypothesis is that " +
    "local n-gram detection is less useful than holistic averaging/pooling when category membership and " +
    "fit are driven mostly by which skills are present anywhere in the document, rather than by local " +
    "phrase patterns — this is exactly the kind of architecture-driven difference Objective 2 of the " +
    "synopsis asks us to analyse, and it will be examined further with the real datasets at final " +
    "evaluation."
  ),
];

// ---------- 4. Explainability ----------

function explainTable() {
  const rows = [["Resume", "JD", "Categories", "Label", "Matched Skills", "Missing Skills"]];
  explainExamples.forEach((ex) => {
    rows.push([
      ex.resume_id, ex.jd_id, `${ex.resume_category} / ${ex.jd_category}`,
      ex.similarity_label.toFixed(2),
      ex.matched_skills.join(", ") || "—",
      ex.missing_skills.join(", ") || "—",
    ]);
  });
  const widths = [800, 800, 2600, 800, 2800, 2800];
  const header = new TableRow({
    tableHeader: true,
    children: rows[0].map((t, i) => cell(t, { bold: true, shade: "D9E2F3", width: widths[i], size: 18 })),
  });
  const body = rows.slice(1).map((r) => new TableRow({ children: r.map((t, i) => cell(t, { width: widths[i], size: 18 })) }));
  return new Table({ width: { size: 10600, type: WidthType.DXA }, rows: [header, ...body] });
}

const explainabilitySection = [
  heading("4. Explainability Layer", HeadingLevel.HEADING_1),
  para(
    "For any resume–JD pair, the explainability layer reports the matched skills (present in both " +
    "documents), missing skills (present in the JD but not the resume — the skill gap), and a raw " +
    "keyword-overlap ratio, using the same fixed skill vocabulary the dataset generator used. This gives " +
    "a human-readable reason behind a fit score on top of whichever encoder produced it, directly " +
    "addressing Objective 3 of the synopsis. Table 4 below shows six worked examples from the held-out " +
    "test split."
  ),
  explainTable(),
  new Paragraph({ text: "", spacing: { after: 200 } }),
  para(
    "A detail worth flagging: because the matching is literal keyword matching, it sometimes under-reports " +
    "overlap when a resume and JD use different phrasing for the same skill (e.g. “coding in python” vs. " +
    "“python programming”) — a limitation that a semantic encoder like SBERT does not have for the fit " +
    "score itself, but that the keyword-based explainability layer still inherits. This is a known " +
    "trade-off of literal explainability layers and is noted here rather than hidden."
  ),
];

// ---------- 5. Bias audit ----------

const biasSection = [
  heading("5. Bias Audit", HeadingLevel.HEADING_1),
  para(
    `Following the probing-classifier methodology of De-Arteaga et al. [6], we took the ` +
    `best-performing encoder by validation AUC — ${biasReport.encoder_used.toUpperCase()} ` +
    `(AUC = ${fmtAuc(biasReport.encoder_val_auc)}) — computed resume embeddings for all ` +
    `${biasReport.n_resumes} synthetic resumes, and trained a logistic-regression probe to predict the ` +
    `demographic-proxy group (derived from first name, independent of skills/category, and never shown ` +
    `to the matching model) directly from the embedding.`
  ),
  para(
    `The probe reached ${pct(biasReport.probe_test_accuracy)} test accuracy, against a majority-class ` +
    `baseline of ${pct(biasReport.majority_class_baseline_accuracy)} — a gap of ` +
    `${pct(biasReport.leakage_gap)}. ${biasReport.interpretation}`
  ),
  para(
    "This result directly addresses Objective 4 of the synopsis: even though the matching model never " +
    "received gender or name as a feature, the resume embedding space still carries enough information " +
    "for a simple linear probe to recover the proxy group well above chance. The adversarial debiasing " +
    "extension (gradient-reversal training to suppress this leakage while preserving matching accuracy) " +
    "is the natural next step and is scheduled as remaining work below, consistent with the synopsis's " +
    "own timeline, which places it after the interim-report milestone."
  ),
];

// ---------- 6. Individual contribution log ----------

const contribRows = [
  ["Member Name", "Reg. No.", "Task(s) Completed"],
  ["Unmesh Vardhan", "230911030",
    "Built and fine-tuned the SBERT (MiniLM) notebook; set up AdamW fine-tuning; ran validation " +
    "evaluation (93.33% acc, AUC 0.9769); built the MLP baseline notebook and shared tokenizer/vocab " +
    "pipeline; ran the bias-probing audit on the best encoder; coordinated the project structure and " +
    "compiled this interim report"],
  ["ShanmugSrinivas", "230911394",
    "Built and trained the 1D-CNN notebook with three parallel kernel branches (3, 4, 5); diagnosed and " +
    "fixed an initial training collapse by adding padding-aware masked pooling and batch normalization; " +
    "ran validation evaluation (68.89% acc, AUC 0.8367); generated confusion matrix and ROC plots"],
  ["Tanmay Sharma", "230911294",
    "Built and trained the BiLSTM notebook; diagnosed and fixed a final-hidden-state bottleneck by " +
    "switching to masked mean-pooling over all timesteps; ran validation evaluation (90.00% acc, " +
    "AUC 0.9575); produced training-curve and evaluation plots; built the explainability (matched/missing " +
    "skill) layer"],
];

function contribTable() {
  const widths = [2200, 1200, 7100];
  const header = new TableRow({
    tableHeader: true,
    children: contribRows[0].map((t, i) => cell(t, { bold: true, shade: "D9E2F3", width: widths[i] })),
  });
  const body = contribRows.slice(1).map((r) => new TableRow({ children: r.map((t, i) => cell(t, { width: widths[i] })) }));
  return new Table({ width: { size: 10500, type: WidthType.DXA }, rows: [header, ...body] });
}

const contributionSection = [
  heading("6. Individual Contribution Log (To Date)", HeadingLevel.HEADING_1),
  para(
    "Each team member owned one encoder end-to-end: writing the training script, diagnosing training " +
    "issues, and producing evaluation plots. With a 3-person team covering 4 architectures, Unmesh " +
    "Vardhan additionally took the MLP baseline alongside SBERT and system integration, per the " +
    "model-to-member assignment in the original synopsis (Member 1: SBERT + integration, Member 4: MLP " +
    "baseline)."
  ),
  contribTable(),
  new Paragraph({ text: "", spacing: { after: 200 } }),
];

// ---------- 7. Remaining work ----------

const remainingRows = [
  ["Sl. No.", "Component / Work Element / Milestone", "Expected Completion"],
  ["1", "Acquire the real Kaggle datasets (Resume Dataset, LinkedIn Job Postings, Resume-vs-JD Matching) and re-run the pipeline at full scale (5,000–8,000 pairs)", "6 Oct 2026"],
  ["2", "Final test-set evaluation: run all four encoders on the held-out test split; add Precision@k alongside accuracy/AUC", "14 Oct 2026"],
  ["3", "Error analysis on misclassified pairs for the weaker models (1D-CNN, BiLSTM)", "16 Oct 2026"],
  ["4", "Adversarial debiasing: add a gradient-reversal adversary to the best encoder and compare probe accuracy / matching AUC before vs. after", "18 Oct 2026"],
  ["5", "Final report writing (Part C): Abstract, Methodology per model, Results & Discussion, Fairness Audit, Conclusion, Future Work, References (12+)", "27 Oct 2026"],
  ["6", "Presentation slides, viva preparation, final code push to GitHub (replacing the placeholder link above)", "31 Oct 2026"],
];

function remainingTable() {
  const widths = [700, 7800, 2000];
  const header = new TableRow({
    tableHeader: true,
    children: remainingRows[0].map((t, i) => cell(t, { bold: true, shade: "D9E2F3", width: widths[i] })),
  });
  const body = remainingRows.slice(1).map((r) => new TableRow({ children: r.map((t, i) => cell(t, { width: widths[i] })) }));
  return new Table({ width: { size: 10500, type: WidthType.DXA }, rows: [header, ...body] });
}

const riskSection = [
  heading("7. Risk / Plan for Remaining Work", HeadingLevel.HEADING_1),
  para(
    "All four models are implemented, trained, and validated on the synthetic dataset described in " +
    "Section 2, together with the explainability and bias-probe layers. The main remaining risk is " +
    "dataset fidelity: results on the synthetic stand-in dataset may not transfer exactly to the real " +
    "Kaggle sources once a token is obtained, since real resumes/JDs are noisier and less template-driven. " +
    "We plan to re-run the identical pipeline (unchanged code, same evaluation protocol) on the real data " +
    "before final evaluation, and to report both sets of numbers if they diverge meaningfully. The " +
    "1D-CNN's training-collapse and BiLSTM's bottleneck issues encountered during this pass (documented " +
    "in Section 3) are now fixed in code, but we will watch for similar instability re-emerging at the " +
    "larger, noisier real-data scale."
  ),
  remainingTable(),
];

// ---------- References ----------

const referencesSection = [
  heading("References", HeadingLevel.HEADING_1),
  para("[1] C. Zhu, H. Zhu, H. Xiong, C. Ma, F. Xie, P. Ding, and P. Li, “Person-Job Fit: Adapting the Right Talent for the Right Job with Joint Representation Learning,” ACM Transactions on Management Information Systems, vol. 9, no. 3, Art. 12, 2018. https://doi.org/10.1145/3234465"),
  para("[2] C. Qin, H. Zhu, T. Xu, C. Zhu, L. Jiang, E. Chen, and H. Xiong, “Enhancing Person-Job Fit for Talent Recruitment: An Ability-aware Neural Network Approach,” in Proc. 41st Int. ACM SIGIR Conf. on Research & Development in Information Retrieval, 2018, pp. 25–34. https://doi.org/10.1145/3209978.3210025"),
  para("[3] Y. Kim, “Convolutional Neural Networks for Sentence Classification,” in Proc. 2014 Conf. on Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1746–1751. https://aclanthology.org/D14-1181/"),
  para("[4] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” in Proc. 2019 Conf. on Empirical Methods in Natural Language Processing and 9th Int. Joint Conf. on Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 3982–3992. https://aclanthology.org/D19-1410/"),
  para("[5] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter,” in 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing, NeurIPS, 2019. https://arxiv.org/abs/1910.01108"),
  para("[6] M. De-Arteaga, A. Romanov, H. Wallach, J. Chayes, C. Borgs, A. Chouldechova, S. Geyik, K. Kenthapadi, and A. T. Kalai, “Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting,” in Proc. Conf. on Fairness, Accountability, and Transparency (FAT* 2019), 2019, pp. 120–128. https://doi.org/10.1145/3287560.3287572"),
  para("[7] B. H. Zhang, B. Lemoine, and M. Mitchell, “Mitigating Unwanted Biases with Adversarial Learning,” in Proc. 2018 AAAI/ACM Conf. on AI, Ethics, and Society (AIES '18), 2018, pp. 335–340. https://doi.org/10.1145/3278721.3278779"),
];

// ---------- Assemble document ----------

const doc = new Document({
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        ...titlePage,
        ...partBHeader,
        ...literatureSection,
        ...datasetSection,
        ...modelsSection,
        ...explainabilitySection,
        ...biasSection,
        ...contributionSection,
        ...riskSection,
        ...referencesSection,
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = path.join(ROOT, "report", "Interim_Report.docx");
  fs.writeFileSync(outPath, buffer);
  console.log("wrote", outPath);
});
