# After CV selection: extra seeds for the selected SBERT, install it, then
# evaluation + fairness. Logs go to results/real/logs/.
Set-Location (Join-Path $PSScriptRoot "..")
foreach ($s in 7, 123) {
    if (Test-Path "results/real/sbert_mpnet_interaction/seeds/seed_$s.json") { Write-Output "skip mpnet seed $s"; continue }
    python src/train.py --model sbert --backbone mpnet --head interaction --out sbert_mpnet_interaction --epochs 5 --seed $s *> "results/real/logs/sbert_mpnet_interaction_seed$s.log"
    Write-Output "done mpnet seed $s (exit $LASTEXITCODE)"
}
python src/cv_select.py *> results/real/logs/cv_select.log;   Write-Output "done cv_select (exit $LASTEXITCODE)"
python src/select_sbert.py *> results/real/logs/select_sbert.log; Write-Output "done select_sbert (exit $LASTEXITCODE)"
python src/evaluate.py *> results/real/logs/evaluate.log;       Write-Output "done evaluate (exit $LASTEXITCODE)"
python src/bias_probe.py *> results/real/logs/bias_probe.log;   Write-Output "done bias_probe (exit $LASTEXITCODE)"
python src/debias.py *> results/real/logs/debias.log;           Write-Output "done debias (exit $LASTEXITCODE)"
