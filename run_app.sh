#!/usr/bin/env bash
set -e

# Change to script directory
cd "$(dirname "$0")"

echo "====================================================================="
echo "          🧳 Tourism Experience Analytics Web Application"
echo "====================================================================="
echo ""

# Find python command
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 was not found on your system."
    echo "Please install Python 3.10+ from https://www.python.org/"
    exit 1
fi

echo "[1/3] Using Python: $($PYTHON_CMD --version)"

# Check dependencies
echo "[2/3] Checking required dependencies..."
if ! $PYTHON_CMD -c "import streamlit, pandas, numpy, sklearn, joblib, matplotlib" &>/dev/null; then
    echo "[INFO] Installing dependencies from requirements.txt..."
    $PYTHON_CMD -m pip install -r requirements.txt
else
    echo "[INFO] All required libraries are ready."
fi

# Launch Streamlit app
echo ""
echo "[3/3] Starting Streamlit dashboard..."
echo "====================================================================="
echo " App URL: http://localhost:8501"
echo " Press Ctrl+C in this terminal to stop the server."
echo "====================================================================="
echo ""

$PYTHON_CMD -m streamlit run app/app.py --server.port 8501
