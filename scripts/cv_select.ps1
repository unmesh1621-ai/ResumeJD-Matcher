# Model selection by 3-fold cross-validation grouped by job description over
# train+val (the held-out test split is never touched). Each config writes
# results/real/cv/<config>/fold_<i>.json; src/cv_select.py aggregates them.
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force results/real/logs/cv | Out-Null
$K = 3

function CV($name, $cvArgs) {
    for ($f = 0; $f -lt $K; $f++) {
        if (Test-Path "results/real/cv/$name/fold_$f.json") { Write-Output "skip $name fold $f (done)"; continue }
        python src/train.py @cvArgs --cv-folds $K --fold $f *> "results/real/logs/cv/${name}_fold$f.log"
        Write-Output "done $name fold $f (exit $LASTEXITCODE)"
    }
}

# MLP-cosine first: its per-fold embeddings warm-start the fold's CNN / BiLSTM
CV "mlp_cosine"         @("--model", "mlp", "--head", "cosine")
CV "mlp_interaction"    @("--model", "mlp", "--head", "interaction")
CV "cnn_cosine"         @("--model", "cnn", "--head", "cosine")
CV "cnn_interaction"    @("--model", "cnn", "--head", "interaction")
CV "bilstm_cosine"      @("--model", "bilstm", "--head", "cosine")
CV "bilstm_interaction" @("--model", "bilstm", "--head", "interaction")
CV "sbert_minilm_cosine"      @("--model", "sbert", "--backbone", "minilm", "--head", "cosine")
CV "sbert_minilm_interaction" @("--model", "sbert", "--backbone", "minilm", "--head", "interaction")
CV "sbert_mpnet_interaction"  @("--model", "sbert", "--backbone", "mpnet", "--head", "interaction", "--epochs", "5")
