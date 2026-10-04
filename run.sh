#!/bin/bash
# ── Steel Billet Inspection System — Linux/macOS Launcher ──
echo "============================================"
echo " Steel Billet Inspection & Traceability"
echo " CodeUtsava X.0 - Team INIT"
echo "============================================"

# Check if venv already exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    # Check Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo "ERROR: Python 3 not found. Install Python 3.8+."
        exit 1
    fi
    PY_BIN=$(command -v python3 || command -v python)
    echo "Creating virtual environment ..."
    $PY_BIN -m venv venv
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        source venv/Scripts/activate
    fi
fi

# Install
echo "Installing dependencies ..."
pip install -r requirements.txt --quiet

# .env
if [ ! -f ".env" ]; then
    cp .env.example .env 2>/dev/null
fi

# Launch
echo ""
echo "Starting application at http://localhost:8501"
echo ""
python -m streamlit run app.py --server.headless true
