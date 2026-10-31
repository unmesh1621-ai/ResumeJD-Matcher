# Train every model/variant on the real data. Logs go to results/real/logs/.
#
# Protocol: the scorer (cosine vs interaction head) and the SBERT backbone
# (MiniLM vs MPNet) are hyper-parameters chosen on VALIDATION AUC only
# (mean over seeds where several were run). The resulting choices were:
#   MLP: cosine   1D-CNN: interaction   BiLSTM: cosine   SBERT: MiniLM + interaction
# Losing variants stay under results/real/ablation_*/ for the report.
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force results/real/logs/ablation_interaction, results/real/logs/ablation_cosine | Out-Null

function Train($name, $trainArgs) {
    python src/train.py @trainArgs *> "results/real/logs/$name.log"
    Write-Output "done $name (exit $LASTEXITCODE)"
}

# 1. MLP first: its seed-42 embeddings warm-start CNN / BiLSTM
foreach ($s in 42, 7, 123) {
    Train "mlp_seed$s" @("--model", "mlp", "--seed", $s, "--head", "cosine")
}
Train "ablation_interaction/mlp_seed42" @("--model", "mlp", "--head", "interaction", "--out", "ablation_interaction/mlp")
foreach ($s in 42, 7, 123) {
    Train "cnn_seed$s" @("--model", "cnn", "--seed", $s, "--head", "interaction")
    Train "ablation_cosine/cnn_seed$s" @("--model", "cnn", "--seed", $s, "--head", "cosine", "--out", "ablation_cosine/cnn")
}
foreach ($s in 42, 7, 123) {
    Train "bilstm_seed$s" @("--model", "bilstm", "--seed", $s, "--head", "cosine")
}
Train "ablation_interaction/bilstm_seed42" @("--model", "bilstm", "--head", "interaction", "--out", "ablation_interaction/bilstm")

# 2. SBERT candidates (seed 42); select_sbert.py copies the val-best to results/real/sbert
Train "ablation_cosine/sbert_minilm_seed42" @("--model", "sbert", "--backbone", "minilm", "--head", "cosine", "--out", "ablation_cosine/sbert")
Train "ablation_interaction/sbert_minilm_seed42" @("--model", "sbert", "--backbone", "minilm", "--head", "interaction", "--out", "ablation_interaction/sbert_minilm")
Train "sbert_mpnet_interaction_seed42" @("--model", "sbert", "--backbone", "mpnet", "--head", "interaction", "--out", "sbert_mpnet_interaction", "--epochs", "5")
python src/select_sbert.py
