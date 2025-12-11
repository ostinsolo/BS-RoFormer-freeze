#!/bin/bash
#
# Build frozen executable for Music Source Separation
# Uses ONLY inference dependencies (no training deps)
# Uses uv for fast, reliable package installation
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"
BUILD_DIR="$SCRIPT_DIR/dist"

echo "============================================================"
echo "Building Music Source Separation Executable"
echo "============================================================"

# Check if uv is available, install if not
if ! command -v uv &> /dev/null; then
    echo "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create fresh virtual environment with uv
BUILD_VENV="$SCRIPT_DIR/build_venv"
rm -rf "$BUILD_VENV"
echo "Creating clean environment with uv..."
uv venv "$BUILD_VENV" --python 3.10
source "$BUILD_VENV/bin/activate"

# Install ONLY inference dependencies using uv
# Force pre-built wheels only to avoid llvmlite/numba compilation issues
echo "Installing minimal inference dependencies with uv..."
echo "  (forcing pre-built wheels only)"

# cx-Freeze needs setuptools
uv pip install "setuptools<70,>=62.6" cx-Freeze==6.15.16

uv pip install --only-binary :all: -r "$SCRIPT_DIR/requirements_freeze.txt" || {
    echo ""
    echo "Some packages don't have wheels. Trying with pinned versions..."
    # These versions have known working wheels
    uv pip install torch==2.2.2 torchaudio==2.2.2 "numpy<2" scipy soundfile \
        librosa==0.10.1 llvmlite==0.41.1 numba==0.58.1 \
        tqdm pyyaml omegaconf ml_collections \
        einops rotary-embedding-torch beartype loralib matplotlib
}

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
cp "$SCRIPT_DIR/models.json" "$BUILD_DIR/" 2>/dev/null || true
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
