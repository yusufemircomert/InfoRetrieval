@echo off
REM Overnight LLM run: folds 1-4, zero-shot + few-shot (fold 0 already done)
REM Double-click this file or run from cmd. No PowerShell execution policy needed.

cd /d "%~dp0.."
echo Logging to results\llm_overnight.log
echo Estimated time: 8-16 hours for folds 1-4 (both modes)
echo.

set PYTHONUNBUFFERED=1
".venv\Scripts\python.exe" -u scripts\run_llm_cv.py --folds 1,2,3,4 --skip-done > results\llm_overnight.log 2>&1

echo.
echo Done. Check results\llm_overnight.log and results\llm\zero_shot\all_folds_metrics.csv
pause
