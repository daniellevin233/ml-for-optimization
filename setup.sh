#!/usr/bin/env bash

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
PYTHON_VERSION="python3.13"

echo "Setting up Python environment..."

# Check if Python 3.13 is available
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "Error: $PYTHON_VERSION not found. Please install Python 3.13 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with $PYTHON_VERSION..."
    $PYTHON_VERSION -m venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    echo "Requirements installed successfully."
else
    echo "Warning: requirements.txt not found."
fi

echo ""
echo "Setup complete!"
echo "To activate the virtual environment manually, run:"
echo "  source venv/bin/activate"
