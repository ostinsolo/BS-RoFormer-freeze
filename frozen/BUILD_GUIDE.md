# Frozen Executable Build Guide

This guide explains how to build a standalone executable for Music Source Separation.

---

## ✅ Achieved

- [x] Frozen executable working on macOS (Intel)
- [x] BS-ROFO-SW 6-stem model (~57s)
- [x] MelBand RoFormer vocals model (~41s)
- [x] Models loaded from external directory (not bundled)
- [x] Executable size: ~1.5GB (without models)
- [x] Minimal dependencies (no training packages)

## 🚧 Roadmap / TODO

- [ ] Windows build with CUDA support
- [ ] Apple Silicon (M1/M2) native build
- [ ] Linux build
- [ ] FFmpeg integration for MP3/M4A input support
- [ ] Reduce executable size
- [ ] GPU acceleration on macOS (MPS)

---

## Prerequisites

- Python 3.10
- ~5GB disk space for build

---

## Build Steps (macOS/Linux)

### 1. Prepare utils for inference-only

**utils/settings.py** - Comment out wandb:
```python
# import wandb  # Optional for training
```

**utils/model_utils.py** - Comment out muon:
```python
# from .muon import Muon, AdaGO  # Optional for training
```

### 2. Run the build script

```bash
cd frozen
chmod +x build.sh
./build.sh
```

### 3. Test the build

```bash
./dist/mss-separate --list-models

./dist/mss-separate \
    --model bsrofo_sw \
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
mkdir dist\weights
```

### 5. Run with CUDA

```cmd
mss-separate --model bsrofo_sw --input song.wav --output output --device cuda
```

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
cx-Freeze==6.15.16
```

**IMPORTANT:** Use cx-Freeze 6.15.16 (newer versions have bugs!)

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| IndexError during build | Use cx-Freeze==6.15.16 |
| No module 'wandb' | Comment out in utils/settings.py |
| numpy 2.x error | pip install 'numpy<2' |
| Unicode filename error | Use ASCII filenames |
| Model not found | Use --models-dir to specify location |

---

## Performance (Intel Mac CPU)

| Model | Task | Time |
|-------|------|------|
| BS-ROFO-SW | 6 stems | ~57s |
| MelBand RoFormer KJ | Vocals | ~41s |

---

## FFmpeg Note

NOT required because we use librosa+soundfile (WAV only).
For MP3/M4A support, FFmpeg would be needed.

---

## Credits

- https://github.com/ZFTurbo/Music-Source-Separation-Training
- https://github.com/stemrollerapp/demucs-cxfreeze
