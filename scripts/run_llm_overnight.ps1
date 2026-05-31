# Overnight LLM run (PowerShell — may need: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
Set-Location $PSScriptRoot\..

$log = "results\llm_overnight.log"
Write-Host "Logging to $log"
Write-Host "Estimated time: 8-16 hours for folds 1-4 (zero-shot + few-shot)"

$env:PYTHONUNBUFFERED = "1"
& .\.venv\Scripts\python.exe -u scripts\run_llm_cv.py --folds 1,2,3,4 --skip-done *> $log

Write-Host "`nDone. Check results\llm\zero_shot\all_folds_metrics.csv"
Read-Host "Press Enter to close"
