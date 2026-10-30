# Train all four encoders on the real data (3 seeds for scratch models, 1 for SBERT).
# Logs go to results/real/logs/.
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force results/real/logs | Out-Null
foreach ($m in "mlp", "cnn", "bilstm") {
    foreach ($s in 42, 7, 123) {
        python src/train.py --model $m --seed $s *> "results/real/logs/${m}_seed$s.log"
        Write-Output "done $m seed $s (exit $LASTEXITCODE)"
    }
}
python src/train.py --model sbert *> "results/real/logs/sbert_seed42.log"
Write-Output "done sbert (exit $LASTEXITCODE)"
