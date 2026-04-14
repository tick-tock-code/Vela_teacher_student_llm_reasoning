@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "PYTHON_EXE=C:\Users\joelb\.conda\envs\vela_TRL\python.exe"

if /I "%~1"=="--check" goto check_only

if not exist "%PYTHON_EXE%" (
    echo ERROR: Expected interpreter not found:
    echo   %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

if not exist "%REPO_ROOT%\src\gui\run_launcher.py" (
    echo ERROR: Could not find GUI launcher from repo root:
    echo   %REPO_ROOT%
    echo.
    pause
    exit /b 1
)

start "Feature Repository Pipeline Launcher" /D "%REPO_ROOT%" "%PYTHON_EXE%" -m src.gui.run_launcher
exit /b 0

:check_only
echo REPO_ROOT=%REPO_ROOT%
echo PYTHON_EXE=%PYTHON_EXE%
if not exist "%PYTHON_EXE%" exit /b 1
if not exist "%REPO_ROOT%\src\gui\run_launcher.py" exit /b 1
exit /b 0
