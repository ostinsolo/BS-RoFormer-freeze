#!/bin/bash
#
# Build frozen executable for Music Source Separation
# Uses ONLY inference dependencies (no training deps)
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"
BUILD_DIR="$SCRIPT_DIR/dist"

echo "============================================================"
echo "Building Music Source Separation Executable"
echo "============================================================"

# Create fresh virtual environment
BUILD_VENV="$SCRIPT_DIR/build_venv"
rm -rf "$BUILD_VENV"
echo "Creating clean environment..."
python3 -m venv "$BUILD_VENV"
source "$BUILD_VENV/bin/activate"
pip install --upgrade pip

# Install ONLY inference dependencies from our minimal requirements
echo "Installing minimal inference dependencies..."
pip install -r "$SCRIPT_DIR/requirements_freeze.txt"

# Clean previous build
rm -rf "$BUILD_DIR"

# Build with cx_Freeze
cd "$SCRIPT_DIR"
echo "Building executable..."
cxfreeze main.py \
    --target-dir="$BUILD_DIR" \
    --target-name=mss-separate \
    --packages=torch,numpy,scipy,soundfile,librosa,tqdm,yaml,omegaconf,ml_collections,einops,rotary_embedding_torch,beartype,loralib

# Copy required project files
echo "Copying project files..."
cp -r "$PROJECT_DIR/configs" "$BUILD_DIR/"
cp -r "$PROJECT_DIR/models" "$BUILD_DIR/"
cp -r "$PROJECT_DIR/utils" "$BUILD_DIR/"
mkdir -p "$BUILD_DIR/weights"

# Copy soundfile data
SOUNDFILE_DATA=$(python -c "import soundfile; import os; print(os.path.dirname(soundfile.__file__))" 2>/dev/null)/_soundfile_data
if [ -d "$SOUNDFILE_DATA" ]; then
    mkdir -p "$BUILD_DIR/lib"
    cp -r "$SOUNDFILE_DATA" "$BUILD_DIR/lib/"
fi

# Copy download scripts
cp "$PROJECT_DIR/download_models.js" "$BUILD_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/download_models.py" "$BUILD_DIR/" 2>/dev/null || true

echo ""
echo "============================================================"
echo "BUILD COMPLETE!"
echo "============================================================"
echo "Output: $BUILD_DIR"
echo "Size: $(du -sh "$BUILD_DIR" | cut -f1)"
echo ""
echo "Test: $BUILD_DIR/mss-separate --list-models"
echo ""

deactivate
