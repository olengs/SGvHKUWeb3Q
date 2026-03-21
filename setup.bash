#!/bin/bash

if command -v python3 &>/dev/null; then
    #python 3 found
    python3 -m venv .venv &&
    source activate .venv/bin/activate &&
    pip3 install -r requirements.txt &&
    python3 app.py
else if command -v python &>/dev/null; then
    python -m venv .venv &&
    source activate .venv/bin/activate &&
    pip install -r requirements.txt &&
    python app.py
else
    echo "error: unable to find python"
fi