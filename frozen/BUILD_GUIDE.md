# Frozen Executable Build Guide

This guide explains how to build a standalone executable for Music Source Separation.

**Current Version: v1.8.3 (December 2025)**

---

## ✅ Achieved

### Core Functionality
- [x] Frozen executable working on all platforms (macOS Intel/ARM, Windows CPU/CUDA)
- [x] Models loaded from external directory (not bundled)
- [x] Flexible model registry via `models.json`
- [x] Support for custom user-trained models
- [x] Lowercase output filenames for consistency

### v1.8.0+ Features
- [x] `--fast` - **2x speedup** via vectorized chunking (works on ANY file length)
- [x] `--two-stems X` - Demucs-style output (stem + no_stem)
- [x] `--stems X,Y` - Select specific stems to output
- [x] `--extract-instrumental` - Generate instrumental from vocal models
- [x] `--flac` / `--pcm-type` - Output format options
- [x] `--overlap` / `--batch-size` / `--use-tta` - Quality tuning
- [x] `--threads` - CPU thread control with auto-detection
- [x] `--precision` - Matmul precision control (high/medium/low)
- [x] `--ensemble` - Combine multiple models for higher quality
- [x] Intel threading optimizations (KMP_AFFINITY, KMP_BLOCKTIME)
- [x] Fixed OpenMP library conflict (removed libomp.dylib)
- [x] Short audio padding - `--fast` automatically pads short files

### Model Support (29 models)
- [x] **Logic RoFormer** - 6-stem with BEST bass separation (40x less bleed)
- [x] **SCNet XL IHF** - 4-stem with highest SDR (10.08)
- [x] **Apollo** - Vocal restoration/enhancement
- [x] BS-RoFormer (all variants including Resurrection, Revive)
- [x] MelBand RoFormer (Gabox, Karaoke, Aspiration, etc.)
- [x] MDX23C
- [x] Custom BS-RoFormer with `use_shared_bias` support

### Executable Sizes
| Platform | Size | Notes |
|----------|------|-------|
| macOS Intel | ~426 MB | Built locally |
| macOS ARM | ~226 MB | Built on GitHub Actions |
| Windows CPU | ~331 MB | |
| Windows CUDA | ~1.3 GB | Includes CUDA libs |

---

## ⚠️ Intel Mac Build Issue & Solution

### The Problem
On Intel Macs, `pip install librosa` pulls in `numba` → `llvmlite`, which tries to **compile from source** and fails due to LLVM/Xcode version conflicts:

```
error: 'get<int, int, llvm::MCRegister>' is unavailable: introduced in macOS 10.13
```

### The Solution
Use `--only-binary :all:` to force pre-built wheels:

```bash
pip install --only-binary :all: torch==2.2.2 torchaudio==2.2.2 "numpy<2" scipy soundfile librosa tqdm pyyaml omegaconf ml_collections einops rotary-embedding-torch beartype loralib matplotlib cx-Freeze==6.15.16
```

Or use `uv` (faster, smarter about wheels):

```bash
uv pip install --only-binary :all: -r requirements_freeze.txt
```

### Why GitHub Actions ARM Works
- macOS ARM runners (macos-14) have pre-built wheels available
- Intel Mac may not have wheels for specific Python/macOS/Xcode combinations

---

## Prerequisites

- Python 3.10
- ~5GB disk space for build
- cx_Freeze 6.15.16 (IMPORTANT: specific version)
- setuptools < 70 (required by cx_Freeze)

---

## Build Steps (macOS/Linux)

### Option 1: Using build.sh (Recommended)

The `build.sh` script now uses `uv` for faster, more reliable builds:

```bash
cd frozen
chmod +x build.sh
./build.sh
```

### Option 2: Manual Build

```bash
cd frozen

# Create virtual environment
python3.10 -m venv build_venv
source build_venv/bin/activate

# Install dependencies (IMPORTANT: use --only-binary for Intel Mac)
pip install --upgrade pip
pip install "setuptools<70,>=62.6" cx-Freeze==6.15.16
pip install --only-binary :all: torch==2.2.2 torchaudio==2.2.2 "numpy<2" scipy soundfile librosa tqdm pyyaml omegaconf ml_collections einops rotary-embedding-torch beartype loralib matplotlib

# Build
cxfreeze main.py \
  --target-dir=dist \
  --target-name=mss-separate \
  --packages=torch,numpy,scipy,soundfile,librosa,tqdm,yaml,omegaconf,ml_collections,einops,rotary_embedding_torch,beartype,loralib

# Copy required files
cp -r ../configs dist/
cp -r ../models dist/
cp -r ../utils dist/
cp models.json dist/
mkdir -p dist/weights
```

### Option 3: Using uv (Fastest)

```bash
# Install uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

cd frozen
uv venv build_venv --python 3.10
source build_venv/bin/activate

# Install with pre-built wheels only
uv pip install "setuptools<70,>=62.6" cx-Freeze==6.15.16
uv pip install --only-binary :all: -r requirements_freeze.txt

# Build
cxfreeze main.py --target-dir=dist --target-name=mss-separate --packages=torch,numpy,scipy,soundfile,librosa,tqdm,yaml,omegaconf,ml_collections,einops,rotary_embedding_torch,beartype,loralib

# Copy files
cp -r ../configs ../models ../utils dist/
cp models.json dist/
mkdir -p dist/weights
```

### Test the build

```bash
# List models
./dist/mss-separate --list-models

# Test separation
./dist/mss-separate \
    --model vocals_melband \
    --models-dir /path/to/models \
    --input song.wav \
    --output output_dir
```

---

## Build Steps (Windows with CUDA)

### 1. Create virtual environment

```cmd
cd frozen
python -m venv build_venv
build_venv\Scripts\activate
pip install --upgrade pip
```

### 2. Install CUDA-enabled PyTorch

```cmd
pip install torch==2.2.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_freeze.txt
```

### 3. Build

```cmd
cxfreeze main.py --target-dir=dist --target-name=mss-separate --packages=torch,numpy,scipy,soundfile,librosa,tqdm,yaml,omegaconf,ml_collections,einops,rotary_embedding_torch,beartype,loralib
```

### 4. Copy project files

```cmd
xcopy /E /I ..\configs dist\configs
xcopy /E /I ..\models dist\models
xcopy /E /I ..\utils dist\utils
copy models.json dist\
mkdir dist\weights
```

---

## Available Models (v1.8.2)

### 🏆 Best Models by Category

| Category | Model | Quality | Speed |
|----------|-------|---------|-------|
| **Bass Separation** | `logic_roformer` | 🏆 BEST | ~60s |
| **4-Stem Quality** | `scnet_xl_ihf` | SDR 10.08 | ~90s |
| **Vocal Extraction** | `vocals_melband` | SDR 10.98 | ~54s |
| **6-Stem** | `bsrofo_sw` | High | ~48s |
| **Instrumental** | `gabox_inst_fv7z` | Fullness 29.96 | Medium |
| **Denoising** | `denoise` | SDR 27.99 | Medium |

### All 28 Models

**6-Stem (bass/drums/vocals/other/guitar/piano)**
- `logic_roformer` - **BEST BASS** (0.6% bleed vs 26% others)
- `bsrofo_sw` - By jarredou

**4-Stem (bass/drums/vocals/other)**
- `scnet_xl_ihf` - **HIGHEST SDR** (10.08)
- `bsroformer_4stem` - SDR 9.65
- `aname_4stem_large` - BEST Drums SDR 9.72

**Vocals**
- `vocals_melband` - **BEST** SDR 10.98
- `vocals_bsroformer_viperx` - SDR 10.87
- `vocals_mdx23c` - Fastest
- `resurrection_vocals` - BS-RoFormer Resurrection
- `revive2_vocals` - HIGHEST Bleedless 40.07
- `revive3e_vocals` - Maximum Fullness

**Instrumental**
- `resurrection_inst` - BS-RoFormer Resurrection
- `gabox_inst_fv7z/fv8/fv4` - Gabox variants
- `inst_v1e_plus` - Unwa V1e+

**Karaoke**
- `karaoke_becruily` - BEST vocals/instrumental
- `karaoke_aufr33` - SDR 10.19

**Processing**
- `denoise` - Remove noise (SDR 27.99)
- `dereverb` - Remove reverb (SDR 19.17)
- `bleed_suppressor` - Remove vocal bleed
- `denoise_debleed` - Gabox debleed

**Special**
- `aspiration` - De-breathe vocals
- `chorus_male_female` - Separate duets
- `crowd` - Extract crowd noise
- `guitar_becruily` - Guitar isolation

**Drum Separation**
- `drumsep_mdx23c_aufr33` - 6-stem
- `drumsep_mdx23c_jarredou` - 5-stem BEST

---

## Minimal Dependencies

```
torch==2.2.2
torchaudio==2.2.2
numpy<2
scipy
soundfile
librosa
tqdm
pyyaml
omegaconf
ml_collections
einops
rotary-embedding-torch
beartype
loralib
matplotlib

# Build tools (installed separately)
setuptools<70,>=62.6
cx-Freeze==6.15.16
```

**CRITICAL:** 
- Use cx-Freeze 6.15.16 (newer versions have bugs!)
- Use NumPy < 2 (ABI compatibility)
- Use `--only-binary :all:` on Intel Macs

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| IndexError during build | Use cx-Freeze==6.15.16 with setuptools<70 |
| llvmlite compilation fails | Use `--only-binary :all:` or `uv` |
| No module 'encodings' | Rebuild with setuptools<70 |
| numpy 2.x error | pip install 'numpy<2' |
| Model not found | Use --models-dir to specify location |
| use_shared_bias error | Ensure custom bs_roformer.py is used |
| Bad CPU type in executable | Use correct architecture build (Intel vs ARM) |

---

## Performance (CPU)

### Standard vs Fast Mode

| Model | Audio | Standard | --fast | Speedup |
|-------|-------|----------|--------|---------|
| `vocals_melband` | 5.86s | 35.9s | 18.8s | **1.9x** |
| `bsroformer_4stem` | 5.86s | 68.9s | 35.2s | **2.0x** |
| `logic_roformer` | 5.86s | 86.1s | fallback* | - |

*Fast mode now **pads short audio** automatically - works on ANY file length!

### When to use --fast
- **Any audio length** - short files are automatically padded
- Best results with `--overlap 2` or higher
- May produce slightly different (sometimes better!) output than standard mode

### Baseline Times (Standard Mode)

| Model | Task | Time (Intel Mac) |
|-------|------|------------------|
| Logic RoFormer | 6 stems | ~60s |
| BS-ROFO-SW | 6 stems | ~48s |
| SCNet XL IHF | 4 stems | ~90s |
| MelBand RoFormer | Vocals | ~54s |

---

## GitHub Actions

- **ARM builds** (macos-14): Automatic, works reliably
- **Intel builds**: Must be done locally and uploaded manually

See `.github/workflows/build-mac.yml` for the ARM build workflow.

---

## Credits

- https://github.com/ZFTurbo/Music-Source-Separation-Training
- https://github.com/lucidrains/BS-RoFormer
- https://github.com/stemrollerapp/demucs-cxfreeze
- Logic RoFormer community training

---

*Last updated: December 13, 2025*
