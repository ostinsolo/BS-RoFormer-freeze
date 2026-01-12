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

# Clean previous build and any cached .pyc files that could have embedded paths
rm -rf "$BUILD_DIR"
rm -rf "$SCRIPT_DIR/__pycache__"
rm -rf "$PROJECT_DIR/__pycache__"
find "$PROJECT_DIR/models" -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_DIR/utils" -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_DIR/models" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_DIR/utils" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "Cleaned previous build and __pycache__ directories"

# Build with cx_Freeze
cd "$SCRIPT_DIR"
echo "Building executable..."

# Add project root to PYTHONPATH so models/ and utils/ can be found
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Build using cxfreeze with explicit packages
# Include all torch subpackages to prevent partial imports causing "bad magic number" errors
cxfreeze main.py \
    --target-dir="$BUILD_DIR" \
    --target-name=mss-separate \
    --packages=torch,torch.nn,torch.nn.modules,torch.nn.functional,torch.utils,torch.utils.data,torch.fft,torch.linalg,torch.autograd,torch.backends,torch.backends.mkl,torch.backends.mkldnn,torch.backends.cudnn,torch.cuda,torch.package,torch.package.analyze,torch._C,torch._jit_internal,torch.jit,torch.onnx,torch.optim,torch.distributions,torch.sparse,torch.special,torch.serialization,numpy,scipy,scipy.signal,scipy.fft,soundfile,librosa,tqdm,yaml,omegaconf,ml_collections,einops,rotary_embedding_torch,beartype,loralib,numba

# The executable is in BUILD_DIR now
if [ ! -f "$BUILD_DIR/mss-separate" ]; then
    echo "Error: Build failed - executable not found"
    exit 1
fi
echo "Executable built successfully"

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

# ============================================================
# CRITICAL: Clean up .pth files and path references that could break portability
# .pth files can contain absolute paths that cause "bad magic number" errors
# ============================================================
echo "Cleaning up .pth files and path references..."
find "$BUILD_DIR/lib" -name "*.pth" -delete 2>/dev/null || true
find "$BUILD_DIR/lib" -name "easy-install.pth" -delete 2>/dev/null || true
# Remove any __pycache__ directories that might have wrong paths
find "$BUILD_DIR/lib" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  Cleanup complete"

# ============================================================
# CRITICAL: Remove duplicate OpenMP library to prevent conflicts
# Keep only Intel's libiomp5.dylib, remove LLVM's libomp.dylib
# This prevents thread scheduling conflicts and improves performance by ~34%
# ============================================================
echo "Checking for OpenMP library conflicts..."
# Remove all libomp.dylib (LLVM) - keep only libiomp5.dylib (Intel)
# This prevents thread scheduling conflicts and improves performance
REMOVED_COUNT=0
for omp_file in $(find "$BUILD_DIR" -name "libomp.dylib" 2>/dev/null); do
    echo "  Removing: $omp_file"
    rm -f "$omp_file"
    REMOVED_COUNT=$((REMOVED_COUNT + 1))
done
if [ $REMOVED_COUNT -gt 0 ]; then
    echo "  Removed $REMOVED_COUNT conflicting libomp.dylib files"
else
    echo "  No conflicting libomp.dylib found"
fi

# Verify OpenMP setup
echo "OpenMP libraries after cleanup:"
ls -la "$BUILD_DIR/lib/"*omp* 2>/dev/null || echo "  None found"

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
