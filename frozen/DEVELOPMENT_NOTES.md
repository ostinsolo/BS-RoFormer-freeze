# BS-RoFormer Frozen Executable - Development Notes

## Project Goal
Create a standalone frozen executable for music source separation that:
1. Works on macOS Intel, macOS Apple Silicon, Windows CPU, and Windows CUDA
2. Loads models from an external directory (not bundled)
3. Supports custom user-trained models via `models.json` registry
4. Integrates with Max/MSP via Node.js scripts

---

## Current Version: v1.8.3 (December 2025)

### Major Achievements
- ✅ **`--fast` mode** - Vectorized chunking for **2x speedup** on longer audio
- ✅ **Intel threading optimizations** - KMP_AFFINITY, KMP_BLOCKTIME
- ✅ **OpenMP conflict fix** - Removed libomp.dylib conflict
- ✅ **Ensemble mode** - Combine multiple models with `--ensemble`
- ✅ **Apollo enhancer** - Vocal restoration model
- ✅ **Logic RoFormer** - 6-stem model with **40x better bass separation**
- ✅ **SCNet XL IHF** - Highest SDR 4-stem (10.08)
- ✅ **Custom BS-RoFormer** - Added `use_shared_bias` support for community models
- ✅ **HTDemucs separated** - Now handled by dedicated Demucs executable
- ✅ **Aname 4-stem Large** - Best drums (9.72) and bass (9.40)
- ✅ **Revive 2** - Highest vocal bleedless (40.07)
- ✅ **Becruily Karaoke** - Best lead/backing vocal separation
- ✅ **Gabox Instrumental Models** - Multiple fullness options
- ✅ **5 NEW Special Models** - Aspiration, Chorus, Bleed Suppressor, Crowd, Denoise-Debleed
- ✅ **27 models total** in registry

### NEW in v1.7.0: Special Purpose Models
- **Aspiration** - De-breathe vocals (SDR 18.98) by Sucial
- **Chorus Male-Female** - Separate duets (SDR 24.12) by Sucial
- **Bleed Suppressor** - Remove vocal bleed from instrumentals by unwa-97chris
- **Crowd** - Extract crowd/audience noise (SDR 8.71) by aufr33-viperx
- **Denoise-Debleed** - Clean fullness model artifacts by Gabox

---

## All Available Models (v1.6.0)

### 🏆 Best Models by Category

| Category | Model | Quality | Notes |
|----------|-------|---------|-------|
| **Bass Separation** | `logic_roformer` | 0.6% bleed | 40x better than others |
| **4-Stem Quality** | `scnet_xl_ihf` | SDR 10.08 | Highest overall |
| **4-Stem Drums** | `aname_4stem_large` | SDR 9.72 | Best drums |
| **Vocal Bleedless** | `revive2_vocals` | 40.07 | Cleanest separation |
| **Vocal Fullness** | `revive3e_vocals` | Maximum | Preserves most audio |
| **Vocal SDR** | `vocals_melband` | SDR 10.98 | Best overall vocals |
| **Karaoke** | `karaoke_becruily` | - | Best lead/backing |
| **Instrumental Full** | `inst_v1e_plus` | 37.89 | Less noise |

### Complete Model List

#### 6-Stem (bass/drums/vocals/other/guitar/piano)
| Model | Description | Quality |
|-------|-------------|---------|
| `logic_roformer` | 🏆 BEST BASS | 0.6% bleed |
| `bsrofo_sw` | Standard 6-stem | By jarredou |

#### 4-Stem (bass/drums/vocals/other)
| Model | Description | Quality |
|-------|-------------|---------|
| `scnet_xl_ihf` | 🏆 HIGHEST SDR | 10.08 |
| `aname_4stem_large` | 🏆 BEST DRUMS | 9.72 |
| `bsroformer_4stem` | Standard | 9.65 |

#### Instrumental (2-stem)
| Model | Description | Fullness | Bleedless |
|-------|-------------|----------|-----------|
| `inst_v1e_plus` | 🏆 Less noise | 37.89 | 36.53 |
| `gabox_inst_fv7z` | High fullness | 29.96 | 44.61 |
| `gabox_inst_fv8` | Updated | - | - |
| `gabox_inst_fv4` | Not muddy | - | - |
| `resurrection_inst` | Karaoke | - | - |

#### Vocals
| Model | Description | Bleedless | Fullness | SDR |
|-------|-------------|-----------|----------|-----|
| `revive2_vocals` | 🏆 CLEANEST | 40.07 | 15.13 | 10.97 |
| `revive3e_vocals` | Maximum Full | - | Highest | - |
| `vocals_melband` | 🏆 BEST SDR | - | - | 10.98 |
| `resurrection_vocals` | Alternative | - | - | - |
| `vocals_bsroformer_viperx` | High quality | - | - | 10.87 |
| `vocals_mdx23c` | Fast | - | - | 10.17 |

#### Karaoke (Lead/Backing Separation)
| Model | Description | Notes |
|-------|-------------|-------|
| `karaoke_becruily` | 🏆 BEST | Best harmony, LV/BV differentiation |
| `karaoke_aufr33` | SDR 10.19 | Good alternative |

#### Processing
| Model | Description | SDR |
|-------|-------------|-----|
| `denoise` | Remove noise | 27.99 |
| `dereverb` | Remove reverb | 19.17 |
| `bleed_suppressor` | Remove vocal bleed | - |
| `denoise_debleed` | Clean fullness artifacts | - |

#### Special Purpose (NEW v1.7.0)
| Model | Description | SDR | Output |
|-------|-------------|-----|--------|
| `aspiration` | De-breathe vocals | 18.98 | no_breath, breath |
| `chorus_male_female` | Separate duets | 24.12 | male, female |
| `crowd` | Extract crowd noise | 8.71 | crowd, other |

#### Drum Separation
| Model | Description |
|-------|-------------|
| `drumsep_mdx23c_aufr33` | 6-stem |
| `drumsep_mdx23c_jarredou` | 5-stem BEST |

---

## Model Sources

| Model | Source | URL |
|-------|--------|-----|
| Logic RoFormer | Community | https://drive.google.com/drive/folders/1ee9HBdwygactWLi_7hdZiFgFNv45Y22m |
| SCNet XL IHF | ZFTurbo | https://github.com/ZFTurbo/Music-Source-Separation-Training/releases |
| Aname 4-stem | Aname-Tommy | https://huggingface.co/Aname-Tommy/melbandroformer4stems |
| Revive 2/3e | pcunwa | https://huggingface.co/pcunwa/BS-Roformer-Revive |
| V1e+ | pcunwa | https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst |
| Becruily Karaoke | becruily | https://huggingface.co/becruily/mel-band-roformer-karaoke |
| Gabox Models | GaboxR67 | https://huggingface.co/GaboxR67/MelBandRoformers |
| MelBand Roformer KJ | KimberleyJensen | https://huggingface.co/KimberleyJSN/melbandroformer |
| Resurrection | pcunwa | https://huggingface.co/pcunwa/BS-Roformer-Resurrection |
| Special Models | Sucial/aufr33 | python-audio-separator releases |

### Alternative: python-audio-separator
For access to 78+ Roformer models: https://github.com/nomadkaraoke/python-audio-separator

---

## MVSEP-Exclusive Models

The following models are **only available via mvsep.com** and not downloadable:
- Strings (BS-RoFormer) - SDR 5.41
- Saxophone (BS-RoFormer) - SDR 9.77
- Flute (BS-RoFormer) - SDR 9.45
- Wind (BS-RoFormer)
- Acoustic Guitar (BS-RoFormer)
- Trumpet (BS-RoFormer)
- Organ (BS-RoFormer) - SDR 5.08
- Cello, Viola, Double Bass, Harp, Mandolin, Trombone

---

## Directory Structure

### GitHub Repo (for releases)
```
/Users/ostinsolo/Documents/Code/BS-RoFormer-freeze/
├── frozen/
│   ├── main.py
│   └── models.json
├── configs/           # All config YAMLs
├── models/bs_roformer/
│   ├── bs_roformer.py  # Custom with use_shared_bias
│   └── attend.py
└── .github/workflows/  # CI/CD
```

### Max/MSP Integration
```
/Users/ostinsolo/Documents/Max 9/
├── SplitWizard/ThirdPartyApps/
│   ├── bsroformer/dist/
│   └── Models/bsroformer/
│       ├── weights/
│       ├── configs/
│       └── models.json
└── Max for Live Devices/test_for_node Project/code/
    ├── bsroformer_models.js
    ├── setupSW+.js
    └── models.js
```

---

## Version History

### v1.8.3 (December 2025)
- [x] Added `--fast` mode for 2x speedup (vectorized chunking)
- [x] Added Intel threading optimizations (KMP_AFFINITY, KMP_BLOCKTIME)
- [x] Fixed OpenMP library conflict (removed libomp.dylib)
- [x] Added `--ensemble` mode to combine multiple models
- [x] Added Apollo vocal enhancer model
- [x] **Short audio padding** - `--fast` now works on ANY file length
- [x] Added `--precision` flag for matmul precision control
- [x] Total: **28 models**

### v1.7.0 (December 2025)
- [x] Added 5 Special Purpose Models (from python-audio-separator)
  - Aspiration (de-breathe) - SDR 18.98
  - Chorus Male-Female (duets) - SDR 24.12
  - Bleed Suppressor
  - Crowd extraction - SDR 8.71
  - Denoise-Debleed
- [x] Total: **27 models**

### v1.6.0 (December 2025)
- [x] Added Aname 4-stem Large (best drums/bass)
- [x] Added Revive 2 (highest bleedless 40.07)
- [x] Added Revive 3e (maximum fullness)
- [x] Added V1e+ instrumental (less noise)
- [x] Added Becruily Karaoke (best lead/backing)
- [x] Added Aufr33 Karaoke
- [x] Total: 22 models

### v1.5.2 (December 2025)
- [x] Added Gabox instrumental models (Fv7z, Fv8, Fv4)

### v1.5.1 (December 2025)
- [x] Added BS-RoFormer Resurrection (Vocals + Inst)
- [x] Fixed use_shared_bias conditional bug

### v1.5.0 (December 2025)
- [x] Added Logic RoFormer (best bass, 40x less bleed)
- [x] Added custom BS-RoFormer with use_shared_bias

### v1.4.0 (December 2025)
- [x] Added SCNet XL IHF (SDR 10.08)

### v1.3.0 (November 2025)
- [x] Removed HTDemucs (separate executable)
- [x] Split workflows by platform

---

## Performance: `--fast` Mode

### What it does
Uses `torch.unfold` for vectorized chunk extraction instead of Python loops.
Pre-computes per-chunk windows for correct boundary handling.

### Speed Improvement

| Model | Audio | Standard | Fast | Speedup |
|-------|-------|----------|------|---------|
| `vocals_melband` | 5.86s | 35.9s | 18.8s | **1.9x** |
| `bsroformer_4stem` | 5.86s | 68.9s | 35.2s | **2.0x** |
| `logic_roformer` | 5.86s | 86.1s | fallback | - |

### When Fast Mode Works
- **Any audio length** - short audio is automatically padded to chunk_size
- Best with `--overlap 2` or higher
- Vectorized chunk extraction for parallel processing

### When Fast Mode Falls Back to Standard
- HTDemucs models (different chunking logic)

### Short Audio Handling
- Audio shorter than chunk_size is **padded with reflection** to fit at least one chunk
- Padding is **automatically removed** after processing
- Output length matches original input exactly

### Model Chunk Sizes
| Model | Chunk Size | Duration |
|-------|------------|----------|
| `vocals_melband` | 352800 | ~8s |
| `bsroformer_4stem` | 352800 | ~8s |
| `logic_roformer` | 485100 | ~11s |

### Quality Notes
- Fast mode produces slightly different output than standard
- User testing indicates fast mode may sound **better** in some cases
- SNR difference ~18-20 dB (differences mainly at chunk boundaries)
- Recommended to keep as **optional flag** (not default)

### Usage
```bash
mss-separate -m vocals_melband -i input.wav -o output/ --overlap 2 --fast
```

---

## Known Issues & Solutions

1. **Logic RoFormer requires custom BS-RoFormer** - `use_shared_bias` support
2. **Aname 4-stem Large is 3.5GB** - Large download
3. **Becruily Karaoke is 1.6GB** - Dual model
4. **MVSEP instrument models not downloadable** - Use mvsep.com

---

## Files to Keep in Sync

1. `/Users/ostinsolo/Documents/Code/BS-RoFormer-freeze/frozen/models.json`
2. `/Users/ostinsolo/Documents/Max 9/SplitWizard/ThirdPartyApps/Models/bsroformer/models.json`
3. `/Users/ostinsolo/Documents/Max 9/Max for Live Devices/test_for_node Project/code/bsroformer_models.js`

---

*Last updated: December 13, 2025*
