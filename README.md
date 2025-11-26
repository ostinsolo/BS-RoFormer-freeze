# BS-RoFormer Freeze

Pre-built executable for Music Source Separation using BS-RoFormer and MelBand RoFormer models.

**No Python installation required!**

## Features

- 🎵 **6-stem separation**: vocals, drums, bass, guitar, piano, other
- 🎤 **Vocal extraction**: High-quality vocal isolation
- 📁 **Supports**: WAV, MP3, FLAC, M4A, OGG
- 💻 **CPU-only**: Works on any Mac (Intel/Apple Silicon)

## Quick Start

1. Download the latest release for your platform
2. Extract the archive
3. Download models (see below)
4. Run separation

```bash
# List available models
./mss-separate --list-models

# Separate audio (6 stems)
./mss-separate -m bsrofo_sw --models-dir /path/to/models -i song.mp3 -o output/

# Separate vocals only
./mss-separate -m vocals_melband --models-dir /path/to/models -i song.mp3 -o output/
```

## Available Models

| Model | Stems | Quality | Speed (per minute) |
|-------|-------|---------|-------------------|
| `bsrofo_sw` | 6 (vocals, drums, bass, guitar, piano, other) | ⭐⭐⭐⭐⭐ | ~8s |
| `bsroformer_4stem` | 4 (vocals, drums, bass, other) | ⭐⭐⭐⭐ | ~11s |
| `vocals_melband` | 2 (vocals, other) | ⭐⭐⭐⭐⭐ | ~9s |

## Downloading Models

Models are NOT included in the release (too large). Download separately:

### Option 1: Use download script
```bash
node download_models.js --download bsrofo_sw
```

### Option 2: Manual download
- **bsrofo_sw** (6-stem): [HuggingFace](https://huggingface.co/jarredou/BS-ROFO-SW-Fixed)
- **vocals_melband** (vocals): [HuggingFace](https://huggingface.co/KimberleyJensen/Kim_Mel_Band_Roformer)

Place `.ckpt` files in the `weights/` folder.

## Directory Structure

```
BS-RoFormer-freeze/
├── mss-separate          # Main executable
├── lib/                  # Python libraries (bundled)
├── configs/              # Model configurations
├── models/               # Model architectures
├── utils/                # Utility functions
├── weights/              # Place model weights here
└── download_models.js    # Model download script
```

## Performance

Tested on Intel Mac (CPU):

| Audio Length | 6-stem Time | Vocals Time |
|--------------|-------------|-------------|
| 6 seconds    | ~48s        | ~41s        |
| 30 seconds   | ~6 min      | ~3.5 min    |
| 3 minutes    | ~30 min     | ~20 min     |

## Credits

- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
- Model weights by jarredou, KimberleyJensen, and community

## License

MIT License
