#!/bin/bash
set -e

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Upgrading pip inside venv..."
.venv/bin/pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    .venv/bin/pip install -r requirements.txt
else
    echo "No requirements.txt found."
fi

echo "Done! You can activate the venv with:"
echo "source .venv/bin/activate"