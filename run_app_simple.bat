@echo off
REM ============================================================
REM run_app_simple.bat
REM Auto-launches the Solar Performance Analyzer (Simple View)
REM Place this file in the SAME folder as app_simple.py
REM ============================================================

cd /d "%~dp0"

echo ============================================
echo   Solar Performance Analyzer - Simple View
echo ============================================
echo.

REM --- Check Python is available ---
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM --- Check Streamlit is installed; install requirements if not ---
python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing required packages...
    if exist requirements.txt (
        pip install -r requirements.txt
    ) else (
        pip install streamlit pandas numpy plotly
    )
)

echo.
echo Launching app_simple.py ...
echo (A browser tab will open automatically. Close this window to stop the app.)
echo.

streamlit run app_simple.py

pause
