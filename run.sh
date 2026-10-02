#!/usr/bin/env bash
set -e

if [ ! -d "venv" ]; then
    echo "Creating isolated Python environment..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi

echo "============================================"
echo " Starting Airgap AI at http://127.0.0.1:8080"
echo "============================================"
./venv/bin/python app.py
