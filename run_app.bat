@echo off
setlocal enabledelayedexpansion
title Tourism Experience Analytics Launcher
cd /d "%~dp0"

echo =====================================================================
echo           Tourism Experience Analytics Web Application
echo =====================================================================
echo.

:: Detect Python executable
set "PY_CMD="
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PY_CMD=py"
    ) else (
        where python3 >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            set "PY_CMD=python3"
        )
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python was not found on this system!
    echo Please install Python 3.10+ from https://www.python.org/
    echo and make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/3] Using Python: %PY_CMD%
%PY_CMD% --version

:: Check if required packages are installed
echo [2/3] Checking required libraries...
%PY_CMD% -c "import streamlit, pandas, numpy, sklearn, joblib, matplotlib" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [INFO] Missing required libraries. Installing from requirements.txt...
    echo [INFO] This happens only on first setup. Please wait a moment...
    echo.
    %PY_CMD% -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Failed to install dependencies.
        echo Please run: pip install -r requirements.txt manually.
        pause
        exit /b %ERRORLEVEL%
    )
    echo [INFO] Dependencies installed successfully!
) else (
    echo [INFO] All required libraries are ready.
)

:: Launch the Streamlit application
echo.
echo [3/3] Starting Streamlit dashboard...
echo =====================================================================
echo  App URL: http://localhost:8501
echo  Press Ctrl+C in this window anytime to stop the server.
echo =====================================================================
echo.

start "" http://localhost:8501
%PY_CMD% -m streamlit run app/app.py --server.port 8501

pause
