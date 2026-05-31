# Long-running LLM CV with logging (zero-shot + few-shot, all folds)
Set-Location $PSScriptRoot\..

$log = "results\llm_overnight.log"
Write-Host "Logging to $log"
Write-Host "Estimated time: 10-20 hours for all folds on GPU"

$env:PYTHONUNBUFFERED = "1"
& .\.venv\Scripts\python.exe -u scripts\run_llm_cv.py --folds all --skip-done 2>&1 |
    Tee-Object -FilePath $log

Write-Host "`nDone. Check results\llm\zero_shot\all_folds_metrics.csv"
Read-Host "Press Enter to close"
