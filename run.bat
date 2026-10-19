@echo off
setlocal
chcp 65001 >nul
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Project environment is missing. Run setup.ps1 first.
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
exit /b %errorlevel%
