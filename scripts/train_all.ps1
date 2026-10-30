# Train every model/variant on the real data. Logs go to results/real/logs/.
#
# Protocol: the scorer (cosine vs interaction head) and the SBERT backbone
# (MiniLM vs MPNet) are hyper-parameters chosen on VALIDATION AUC only; the
# losing variants are kept under results/real/ablation_*/ for the report.
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force results/real/logs/ablation_interaction | Out-Null

function Train($name, $trainArgs) {
    python src/train.py @trainArgs *> "results/real/logs/$name.log"
    Write-Output "done $name (exit $LASTEXITCODE)"
}

# 1. scratch models, cosine scorer (selected on val), 3 seeds; MLP first
#    because its seed-42 embeddings warm-start CNN / BiLSTM
foreach ($m in "mlp", "cnn", "bilstm") {
    foreach ($s in 42, 7, 123) {
        Train "${m}_seed$s" @("--model", $m, "--seed", $s, "--head", "cosine")
    }
}

# 2. interaction-head ablation for the scratch models (seed 42)
foreach ($m in "mlp", "cnn", "bilstm") {
    Train "ablation_interaction/${m}_seed42" @("--model", $m, "--head", "interaction", "--out", "ablation_interaction/$m")
}

# 3. SBERT candidates (seed 42): MiniLM interaction, MPNet cosine
Train "ablation_interaction/sbert_minilm_seed42" @("--model", "sbert", "--backbone", "minilm", "--head", "interaction", "--out", "ablation_interaction/sbert_minilm")
Train "sbert_mpnet_cosine_seed42" @("--model", "sbert", "--backbone", "mpnet", "--head", "cosine", "--out", "sbert_mpnet_cosine", "--epochs", "5")
