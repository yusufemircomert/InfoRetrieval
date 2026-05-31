# Long-running BERT CV with logging (all folds)
Set-Location $PSScriptRoot\..

$log = "results\bert_overnight.log"
Write-Host "Logging to $log"
Write-Host "Estimated time: 1.5-2.5 hours for all 5 folds on GPU"

$env:PYTHONUNBUFFERED = "1"
& .\.venv\Scripts\python.exe -u scripts\run_bert_cv.py --folds all --skip-done 2>&1 |
    Tee-Object -FilePath $log

Write-Host "`nDone. Check results\bert\all_folds_metrics.csv"
Read-Host "Press Enter to close"
