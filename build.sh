#!/bin/bash
# Quick build script for vinyl scratch removal

set -e  # Exit on error

echo "===================================="
echo "Vinyl Scratch Removal - Build Script"
echo "===================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found!"
    echo "Please install Python 3.7 or later"
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

# Build core library
echo "[1/3] Building core library (Cython)..."
cd core
if ! python3 setup.py build_ext --inplace; then
    echo "Error: Core library build failed!"
    echo "Check that you have Cython, NumPy, and SciPy installed:"
    echo "  pip install cython numpy scipy soundfile"
    exit 1
fi
cd ..
echo "✓ Core library built successfully"
echo ""

# Build LV2 plugin
echo "[2/3] Building LV2 plugin..."
cd lv2
if command -v make &> /dev/null; then
    if make; then
        echo "✓ LV2 plugin built successfully"
    else
        echo "⚠ LV2 plugin build failed (optional component)"
        echo "  Install lv2-dev package if you need the LV2 plugin"
    fi
else
    echo "⚠ make not found, skipping LV2 plugin (optional)"
fi
cd ..
echo ""

# Setup CLI tool
echo "[3/3] Setting up CLI tool..."
chmod +x cli/vinyl
if ./cli/vinyl --version &> /dev/null; then
    echo "✓ CLI tool ready"
else
    echo "⚠ CLI tool setup incomplete"
fi
echo ""

echo "===================================="
echo "Build Complete!"
echo "===================================="
echo ""
echo "Usage:"
echo "  CLI tool:    ./cli/vinyl input.wav output.wav"
echo "  Python:      python3 -c 'from vinyl_core import VinylProcessor'"
echo "  LV2 plugin:  cd lv2 && make install"
echo ""
echo "See BUILD.md for detailed instructions"
