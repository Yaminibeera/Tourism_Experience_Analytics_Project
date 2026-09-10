@echo off
cd /d "%~dp0"
echo ========================================================
echo  Running Tourism Experience Analytics Pipeline
echo ========================================================
echo.
echo [1/2] Cleaning data and engineering features...
python src/data_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during data pipeline execution!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Training Regression, Classification, and Recommender models...
python src/train_models.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during model training!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Pipeline finished successfully!
pause
