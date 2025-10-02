@echo off
echo ========================================
echo   AI Agent for Name Extraction
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or newer
    pause
    exit /b 1
)

echo ✅ Python is available
echo.

echo Starting the program...
python run.py

echo.
echo Program finished
pause
