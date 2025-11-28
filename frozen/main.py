#!/usr/bin/env python3
"""
Music Source Separation - Frozen Executable
Models loaded from external directory (not bundled).
Supports custom user-trained models via models.json registry.
"""

import argparse
import os
import sys
import time
import glob
import json

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

import torch
import numpy as np
import soundfile as sf
import librosa
from tqdm.auto import tqdm

# ============================================================================
# DEFAULT MODELS (built-in, used if models.json doesn't exist)
# ============================================================================

DEFAULT_MODELS = {
    # === 4-STEM ===
    "htdemucs_4stem": {
        "type": "htdemucs",
        "config": "configs/config_musdb18_htdemucs.yaml",
        "checkpoint": "weights/htdemucs_4stem.th",
        "stems": ["bass", "drums", "vocals", "other"],
        "description": "HTDemucs 4-stem FASTEST"
    },
    "bsroformer_4stem": {
        "type": "bs_roformer",
        "config": "configs/config_musdb18_bs_roformer.yaml",
        "checkpoint": "weights/bsroformer_4stem.ckpt",
        "stems": ["bass", "drums", "vocals", "other"],
        "description": "BS-RoFormer 4-stem HIGH QUALITY"
    },
    
    # === 6-STEM ===
    "htdemucs_6stem": {
        "type": "htdemucs",
        "config": "configs/config_htdemucs_6stems.yaml",
        "checkpoint": "weights/htdemucs_6stem.th",
        "stems": ["bass", "drums", "vocals", "other", "piano", "guitar"],
        "description": "HTDemucs 6-stem FAST"
    },
    "bsrofo_sw": {
        "type": "bs_roformer",
        "config": "configs/config_bsrofo_sw_fixed.yaml",
        "checkpoint": "weights/bsrofo_sw_fixed.ckpt",
        "stems": ["bass", "drums", "vocals", "other", "guitar", "piano"],
        "description": "BS-ROFO-SW 6-stem BEST"
    },
    
    # === VOCALS ===
    "vocals_melband": {
        "type": "mel_band_roformer",
        "config": "configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        "checkpoint": "weights/MelBandRoformer_kj.ckpt",
        "stems": ["vocals", "other"],
        "description": "MelBand RoFormer vocals BEST"
    },
    "vocals_bsroformer_viperx": {
        "type": "bs_roformer",
        "config": "configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "checkpoint": "weights/vocals_bsroformer_viperx.ckpt",
        "stems": ["vocals", "other"],
        "description": "BS-RoFormer vocals HIGH QUALITY"
    },
    "vocals_mdx23c": {
        "type": "mdx23c",
        "config": "configs/config_vocals_mdx23c.yaml",
        "checkpoint": "weights/vocals_mdx23c.ckpt",
        "stems": ["vocals", "other"],
        "description": "MDX23C vocals FAST"
    },
    
    # === AUDIO PROCESSING ===
    "denoise": {
        "type": "mel_band_roformer",
        "config": "configs/config_denoise_mel_band_roformer.yaml",
        "checkpoint": "weights/denoise.ckpt",
        "stems": ["dry", "other"],
        "description": "Remove noise (dry=clean, other=noise)"
    },
    "dereverb": {
        "type": "mel_band_roformer",
        "config": "configs/config_dereverb_mel_band_roformer.yaml",
        "checkpoint": "weights/dereverb.ckpt",
        "stems": ["noreverb", "reverb"],
        "description": "Remove reverb"
    },
    
    # === SPECIALIZED ===
    "drums_htdemucs": {
        "type": "htdemucs",
        "config": "configs/config_musdb18_htdemucs.yaml",
        "checkpoint": "weights/drums_htdemucs.th",
        "stems": ["drums"],
        "description": "Drum extraction"
    },
    "bass_htdemucs": {
        "type": "htdemucs",
        "config": "configs/config_musdb18_htdemucs.yaml",
        "checkpoint": "weights/bass_htdemucs.th",
        "stems": ["bass"],
        "description": "Bass extraction"
    },
    
    # === DRUM KIT SEPARATION ===
    "drumsep_htdemucs": {
        "type": "htdemucs",
        "config": "configs/config_drumsep.yaml",
        "checkpoint": "weights/drumsep_htdemucs.th",
        "stems": ["kick", "snare", "cymbals", "toms"],
        "description": "Drum kit 4-stem"
    },
    "drumsep_mdx23c_aufr33": {
        "type": "mdx23c",
        "config": "configs/config_drumsep_mdx23c_aufr33.yaml",
        "checkpoint": "weights/drumsep_mdx23c_aufr33.ckpt",
        "stems": ["kick", "snare", "toms", "hh", "ride", "crash"],
        "description": "Drum kit 6-stem"
    },
    "drumsep_mdx23c_jarredou": {
        "type": "mdx23c",
        "config": "configs/config_drumsep_mdx23c_jarredou.yaml",
        "checkpoint": "weights/drumsep_mdx23c_jarredou.ckpt",
        "stems": ["kick", "snare", "toms", "hh", "cymbals"],
        "description": "Drum kit 5-stem BEST"
    },
}

# ============================================================================
# MODEL REGISTRY (loads from models.json or uses defaults)
# ============================================================================

def get_models_json_path(models_dir=None):
    """Get path to models.json registry file"""
    base = models_dir if models_dir else BASE_DIR
    return os.path.join(base, "models.json")

def load_models_registry(models_dir=None):
    """
    Load models from models.json if it exists, otherwise use defaults.
    This allows users to add custom models without rebuilding the executable.
    """
    json_path = get_models_json_path(models_dir)
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                custom_models = json.load(f)
            print(f"Loaded {len(custom_models)} models from models.json")
            return custom_models
        except Exception as e:
            print(f"Warning: Could not load models.json: {e}")
            print("Using default models")
    
    return DEFAULT_MODELS

def save_models_registry(models, models_dir=None):
    """Save models registry to models.json"""
    json_path = get_models_json_path(models_dir)
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(models, f, indent=2, ensure_ascii=False)
        print(f"Saved models registry to: {json_path}")
    except Exception as e:
        print(f"Warning: Could not save models.json: {e}")

def init_models_json(models_dir=None):
    """Initialize models.json with default models if it doesn't exist"""
    json_path = get_models_json_path(models_dir)
    
    if not os.path.exists(json_path):
        save_models_registry(DEFAULT_MODELS, models_dir)
        print(f"Created default models.json at: {json_path}")
    else:
        print(f"models.json already exists at: {json_path}")

def add_custom_model(name, model_type, config_path, checkpoint_path, stems, description="Custom model", models_dir=None):
    """
    Add a custom model to the registry.
    
    Usage:
        add_custom_model(
            name="my_vocal_model",
            model_type="mel_band_roformer",
            config_path="configs/my_config.yaml",
            checkpoint_path="weights/my_model.ckpt",
            stems=["vocals", "other"],
            description="My custom vocal model"
        )
    """
    models = load_models_registry(models_dir)
    
    models[name] = {
        "type": model_type,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "stems": stems,
        "description": description
    }
    
    save_models_registry(models, models_dir)
    print(f"Added custom model: {name}")

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def list_models(models_dir=None):
    """List all available models"""
    models = load_models_registry(models_dir)
    base = models_dir if models_dir else BASE_DIR
    
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)
    
    # Group by type
    categories = {}
    for name, info in models.items():
        cat = info.get("type", "unknown")
        if cat not in categories:
            categories[cat] = []
        
        # Check if checkpoint exists
        checkpoint_path = os.path.join(base, info["checkpoint"])
        installed = "[OK]" if os.path.exists(checkpoint_path) else "[--]"
        
        categories[cat].append((name, info, installed))
    
    for cat, items in categories.items():
        print(f"\n{cat.upper()}:")
        for name, info, installed in items:
            desc = info.get("description", "")
            stems = ", ".join(info.get("stems", []))
            print(f"  {installed} {name}: {desc}")
            print(f"      Stems: {stems}")
    
    print("\n" + "=" * 60)
    print("Legend: [OK] = installed, [--] = not installed (download weights)")
    print("=" * 60 + "\n")

def load_model(model_name, models_dir=None):
    """Load a model by name from the registry"""
    models = load_models_registry(models_dir)
    
    if model_name not in models:
        print(f"Error: Unknown model '{model_name}'")
        print("\nAvailable models:")
        for name in sorted(models.keys()):
            print(f"  - {name}")
        print("\nTo add a custom model, edit models.json or use --init-registry")
        sys.exit(1)
    
    model_info = models[model_name]
    weights_base = models_dir if models_dir else BASE_DIR
    
    # Weights come from models_dir (external)
    checkpoint_path = os.path.join(weights_base, model_info["checkpoint"])
    
    # Configs: try bundled first (BASE_DIR), then models_dir
    config_rel = model_info["config"]
    config_path = os.path.join(BASE_DIR, config_rel)
    if not os.path.exists(config_path) and models_dir:
        config_path = os.path.join(models_dir, config_rel)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print(f"\nDownload the model weights and place them at:")
        print(f"  {checkpoint_path}")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"Error: Config not found: {config_path}")
        print(f"\nMake sure the config file exists at:")
        print(f"  {os.path.join(BASE_DIR, config_rel)}")
        print(f"  or: {os.path.join(models_dir, config_rel) if models_dir else 'N/A'}")
        sys.exit(1)
    
    from utils.settings import get_model_from_config
    
    print(f"Loading: {model_name}")
    print(f"  Type: {model_info['type']}")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    model, config = get_model_from_config(model_info["type"], config_path)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'state' in state_dict:
        state_dict = state_dict['state']
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, config, model_info

def separate(model, config, audio_path, output_dir, model_info, device="cpu"):
    """Separate audio into stems"""
    from utils.model_utils import demix
    
    print(f"\nProcessing: {audio_path}")
    sample_rate = getattr(config.audio, 'sample_rate', 44100)
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=0)
    
    print("Separating...")
    start = time.time()
    waveforms = demix(config, model, audio, device, model_type=model_info["type"], pbar=True)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    out_subdir = os.path.join(output_dir, basename)
    os.makedirs(out_subdir, exist_ok=True)
    
    for stem, wav in waveforms.items():
        path = os.path.join(out_subdir, f"{stem}.wav")
        sf.write(path, wav.T, sample_rate)
        print(f"  Saved: {path}")
    
    return elapsed

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Music Source Separation - Supports custom models via models.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  %(prog)s --list-models
  
  # Separate with default model (bsrofo_sw)
  %(prog)s -i song.wav -o output/
  
  # Separate with specific model
  %(prog)s -m vocals_melband -i song.wav -o output/
  
  # Use external models directory
  %(prog)s -m my_model --models-dir /path/to/models -i song.wav -o output/
  
  # Initialize models.json registry
  %(prog)s --init-registry --models-dir /path/to/models

Custom Models:
  To add your own trained model:
  1. Place your checkpoint in: models_dir/weights/my_model.ckpt
  2. Place your config in: models_dir/configs/my_config.yaml
  3. Edit models_dir/models.json to add your model entry
  
  Example models.json entry:
  {
    "my_custom_model": {
      "type": "mel_band_roformer",
      "config": "configs/my_config.yaml",
      "checkpoint": "weights/my_model.ckpt",
      "stems": ["vocals", "other"],
      "description": "My custom vocal model"
    }
  }
        """
    )
    
    parser.add_argument("--model", "-m", default="bsrofo_sw",
                        help="Model name (default: bsrofo_sw)")
    parser.add_argument("--input", "-i",
                        help="Input audio file or folder")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models")
    parser.add_argument("--device", "-d", default="cpu",
                        help="Device: cpu, cuda, mps (default: cpu)")
    parser.add_argument("--models-dir",
                        help="External models directory (contains weights/, configs/, models.json)")
    parser.add_argument("--init-registry", action="store_true",
                        help="Initialize models.json with default models")
    
    args = parser.parse_args()
    
    # Initialize registry if requested
    if args.init_registry:
        init_models_json(args.models_dir)
        return
    
    # List models
    if args.list_models:
        list_models(args.models_dir)
        return
    
    # Require input for separation
    if not args.input:
        parser.print_help()
        print("\nError: --input is required for separation")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)
    
    # Load model
    model, config, model_info = load_model(args.model, args.models_dir)
    model = model.to(args.device)
    
    # Get input files
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = glob.glob(os.path.join(args.input, "*.*"))
        files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))]
    
    if not files:
        print("Error: No audio files found")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\nProcessing {len(files)} file(s)...")
    total_time = 0
    
    for f in files:
        elapsed = separate(model, config, f, args.output, model_info, args.device)
        total_time += elapsed
    
    print(f"\nDONE! Total time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
