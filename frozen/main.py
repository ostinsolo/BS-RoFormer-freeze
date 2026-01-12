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
import subprocess

def get_physical_cores():
    """Get number of physical CPU cores (not logical/hyperthreaded)"""
    import platform
    system = platform.system()
    
    try:
        if system == 'Darwin':  # macOS
            result = subprocess.run(['sysctl', '-n', 'hw.physicalcpu'], 
                                    capture_output=True, text=True)
            return int(result.stdout.strip())
        elif system == 'Linux':
            # Count physical cores from /proc/cpuinfo
            with open('/proc/cpuinfo') as f:
                content = f.read()
            # Count unique physical id + core id combinations
            cores = set()
            physical_id = core_id = None
            for line in content.split('\n'):
                if 'physical id' in line:
                    physical_id = line.split(':')[1].strip()
                elif 'core id' in line:
                    core_id = line.split(':')[1].strip()
                    if physical_id is not None:
                        cores.add((physical_id, core_id))
            return len(cores) if cores else os.cpu_count() // 2
        elif system == 'Windows':
            result = subprocess.run(['wmic', 'cpu', 'get', 'NumberOfCores'], 
                                    capture_output=True, text=True)
            lines = [l.strip() for l in result.stdout.split('\n') if l.strip().isdigit()]
            return sum(int(l) for l in lines)
    except:
        pass
    
    # Fallback: assume half of logical CPUs (hyperthreading)
    return max(1, os.cpu_count() // 2)

# ============================================================================
# DEFAULT MODELS (built-in, used if models.json doesn't exist)
# ============================================================================

DEFAULT_MODELS = {
    # === 4-STEM ===
    # NOTE: htdemucs models removed - use Demucs executable for those
    "bsroformer_4stem": {
        "type": "bs_roformer",
        "config": "configs/config_musdb18_bs_roformer.yaml",
        "checkpoint": "weights/bsroformer_4stem.ckpt",
        "stems": ["bass", "drums", "vocals", "other"],
        "description": "BS-RoFormer 4-stem HIGH QUALITY"
    },
    
    # === 6-STEM ===
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
    
    # === DRUM KIT SEPARATION ===
    # NOTE: drumsep_htdemucs removed - use Demucs executable for that
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

def separate(model, config, audio_path, output_dir, model_info, device="cpu", 
              overlap=None, batch_size=None, use_tta=False,
              output_format='wav', pcm_type='FLOAT', extract_instrumental=False,
              selected_stems=None, two_stems=None, use_fast=False):
    """Separate audio into stems
    
    Args:
        model: Loaded model
        config: Model configuration
        audio_path: Path to input audio
        output_dir: Output directory
        model_info: Model info dict
        device: cpu, cuda, or mps
        overlap: Number of overlapping chunks (higher = better quality, slower)
                 2 = 50% overlap (default), 4 = 75% overlap, 8 = 87.5% overlap
        batch_size: Chunks to process at once (affects VRAM)
        use_tta: Enable test-time augmentation (3x slower but better quality)
        output_format: 'wav' or 'flac'
        pcm_type: 'PCM_16', 'PCM_24', or 'FLOAT'
        extract_instrumental: Generate instrumental by subtracting vocals
        selected_stems: List of stems to output (None = all stems)
        two_stems: Demucs-style mode - output specified stem + 'no_{stem}'
    """
    from utils.model_utils import demix, demix_fast, apply_tta
    
    print(f"\nProcessing: {audio_path}")
    sample_rate = getattr(config.audio, 'sample_rate', 44100)
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=0)
    
    # Keep original mix for instrumental extraction
    mix_orig = audio.copy()
    
    # Override config inference parameters if specified
    if overlap is not None:
        config.inference.num_overlap = overlap
        print(f"  Overlap: {overlap} (step = chunk_size / {overlap})")
    if batch_size is not None:
        config.inference.batch_size = batch_size
        print(f"  Batch size: {batch_size}")
    
    print("Separating...")
    start = time.time()
    
    # Use fast demix if enabled (vectorized chunking)
    if use_fast:
        waveforms = demix_fast(config, model, audio, device, model_type=model_info["type"], pbar=True)
    else:
        waveforms = demix(config, model, audio, device, model_type=model_info["type"], pbar=True)
    
    # Apply TTA if requested
    if use_tta:
        print("Applying TTA (test-time augmentation)...")
        waveforms = apply_tta(config, model, audio, waveforms, device, model_info["type"])
    
    # Extract instrumental by subtracting vocals (case-insensitive)
    if extract_instrumental:
        # Find vocals stem (case-insensitive)
        vocals_key = None
        for key in waveforms.keys():
            if key.lower() == 'vocals':
                vocals_key = key
                break
        if vocals_key:
            print("Extracting instrumental...")
            waveforms['instrumental'] = mix_orig - waveforms[vocals_key]
        else:
            print("  Warning: No 'vocals' stem found, cannot extract instrumental")
            print(f"  Available stems: {', '.join(waveforms.keys())}")
    
    # Two-stems mode: output target stem + "no_{stem}" (everything else combined)
    if two_stems:
        # Find the target stem (case-insensitive)
        target_key = None
        for key in waveforms.keys():
            if key.lower() == two_stems.lower():
                target_key = key
                break
        
        if target_key:
            print(f"Two-stems mode: {target_key} + no_{target_key.lower()}")
            # Combine all other stems into "no_{stem}"
            other_stems = [waveforms[k] for k in waveforms.keys() if k != target_key]
            if other_stems:
                no_stem = sum(other_stems)
            else:
                # If only one stem, compute from original mix
                no_stem = mix_orig - waveforms[target_key]
            
            # Replace waveforms with just the two outputs
            waveforms = {
                target_key: waveforms[target_key],
                f'no_{target_key.lower()}': no_stem
            }
        else:
            print(f"  Warning: Stem '{two_stems}' not found, outputting all stems")
            print(f"  Available stems: {', '.join(waveforms.keys())}")
    
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    out_subdir = os.path.join(output_dir, basename)
    os.makedirs(out_subdir, exist_ok=True)
    
    # Validate PCM type for codec
    codec = output_format
    subtype = pcm_type
    available = sf.available_subtypes(codec)
    if subtype not in available:
        default_subtype = sf.default_subtype(codec)
        print(f"  Warning: {codec} doesn't support {subtype}, using {default_subtype}")
        subtype = default_subtype
    
    # Filter stems if specified (case-insensitive matching)
    stems_to_save = list(waveforms.keys())
    if selected_stems is not None:
        # Create case-insensitive mapping
        stem_map = {s.lower(): s for s in waveforms.keys()}
        stems_to_save = []
        skipped = []
        for requested in selected_stems:
            if requested.lower() in stem_map:
                stems_to_save.append(stem_map[requested.lower()])
            else:
                skipped.append(requested)
        if skipped:
            print(f"  Warning: Stems not found: {', '.join(skipped)}")
            print(f"  Available stems: {', '.join(waveforms.keys())}")
    
    for stem in stems_to_save:
        wav = waveforms[stem]
        # Always use lowercase filenames for consistency
        stem_filename = stem.lower()
        path = os.path.join(out_subdir, f"{stem_filename}.{output_format}")
        sf.write(path, wav.T, sample_rate, subtype=subtype)
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
  
  # High quality mode (more overlap, TTA)
  %(prog)s -m vocals_melband -i song.wav -o output/ --overlap 4 --use-tta
  
  # Fast mode (less overlap)
  %(prog)s -m vocals_melband -i song.wav -o output/ --overlap 2 --batch-size 4
  
  # Use external models directory
  %(prog)s -m my_model --models-dir /path/to/models -i song.wav -o output/
  
  # Initialize models.json registry
  %(prog)s --init-registry --models-dir /path/to/models

Quality/Performance Tuning:
  --overlap N    Overlap factor. Higher = better quality, slower processing.
                 1 = no overlap (fastest, default)
                 2 = 50%% overlap (balanced)
                 4 = 75%% overlap (high quality)
  
  --batch-size N Chunks processed at once. Higher = faster but more VRAM.
                 1 = low VRAM (default for most models)
                 4 = faster if you have enough VRAM
  
  --use-tta      Test-time augmentation. ~3x slower but better quality.
                 Applies channel inversion and polarity inversion.

  --threads N    CPU threads for processing (0 = auto-detect).
                 Auto-detect uses physical cores (recommended).
                 On laptops, fewer threads may prevent thermal throttling.

Output Format:
  --flac         Output FLAC instead of WAV (smaller files).
  --pcm-type X   Bit depth: PCM_16 (smallest), PCM_24, FLOAT (default).
  --extract-instrumental  Generate instrumental by subtracting vocals.
  --stems X,Y    Only output specific stems (e.g., 'vocals,drums').
                 Default: output all stems from the model.
  --two-stems X  Demucs-style mode: output stem X + 'no_X' (everything else).
                 Example: --two-stems bass → bass.wav + no_bass.wav

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
    
    # Quality/Performance tuning
    parser.add_argument("--overlap", type=int, default=1,
                        help="Overlap factor (1=none/fastest, 2=50%%, 4=75%%). Default: 1 (no overlap, fastest).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for inference (affects VRAM). Default from config.")
    parser.add_argument("--use-tta", action="store_true",
                        help="Enable test-time augmentation (3x slower but better quality)")
    parser.add_argument("--fast", action="store_true",
                        help="Use optimized vectorized chunking (faster with --overlap 2+, experimental)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Number of CPU threads (0=auto-detect physical cores). "
                             "Recommended: leave at 0 or set to number of physical cores.")
    parser.add_argument("--precision", type=str, choices=['high', 'medium'], default='high',
                        help="CPU matmul precision: 'high' (default, best quality) or 'medium' (faster, ~10%% speedup)")
    
    # Output format options
    parser.add_argument("--flac", action="store_true",
                        help="Output FLAC instead of WAV (smaller file size)")
    parser.add_argument("--pcm-type", type=str, choices=['PCM_16', 'PCM_24', 'FLOAT'], default='FLOAT',
                        help="Audio bit depth: PCM_16 (smallest), PCM_24 (balanced), FLOAT (highest quality, default)")
    parser.add_argument("--extract-instrumental", action="store_true",
                        help="Generate instrumental stem by subtracting vocals (for vocal models)")
    parser.add_argument("--stems", type=str, default=None,
                        help="Comma-separated list of stems to output (e.g., 'vocals,drums'). "
                             "Default: output all stems. Use --list-models to see available stems.")
    parser.add_argument("--two-stems", type=str, default=None,
                        help="Demucs-style two-stems mode: output specified stem + 'no_{stem}' "
                             "(e.g., '--two-stems bass' outputs bass.wav + no_bass.wav). "
                             "Works with any multi-stem model.")
    
    # Ensemble Options
    parser.add_argument("--ensemble", 
                        help="List of models to ensemble, comma-separated (e.g. 'vocals_melband,vocals_bsroformer_viperx')")
    parser.add_argument("--ensemble-type", default="avg_wave",
                        choices=["avg_wave", "median_wave", "min_wave", "max_wave", "avg_fft", "median_fft", "min_fft", "max_fft"],
                        help="Ensemble algorithm (default: avg_wave)")
    parser.add_argument("--ensemble-weights", 
                        help="Weights for ensemble, comma-separated (e.g. '1.0,0.8'). Default: equal weights.")

    args = parser.parse_args()
    
    # Set thread count (do this early, before any heavy computation)
    if args.threads == 0:
        # Auto-detect: use physical cores
        num_threads = get_physical_cores()
        print(f"Auto-detected {num_threads} physical cores")
    else:
        num_threads = args.threads
        print(f"Using {num_threads} threads (user-specified)")
    
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(2, num_threads // 2))  # Half threads for inter-op
    
    # Intel CPU/OpenMP/BLAS optimizations for maximum parallelization
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'  # Thread pinning
    os.environ['KMP_BLOCKTIME'] = '0'  # Immediate thread release
    
    # CPU matmul precision (optional: trade precision for speed)
    if hasattr(torch, 'set_float32_matmul_precision') and args.precision == 'medium':
        torch.set_float32_matmul_precision('medium')
        print(f"Using medium precision matmul (faster, ~10% speedup)")
    
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
        
    # Get input files (common for both modes)
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = glob.glob(os.path.join(args.input, "*.*"))
        files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))]
    
    if not files:
        print("Error: No audio files found")
        sys.exit(1)

    output_format = 'flac' if args.flac else 'wav'
    
    # Parse stems selection
    selected_stems = None
    if args.stems:
        selected_stems = [s.strip().lower() for s in args.stems.split(',')]
        print(f"Selected stems: {', '.join(selected_stems)}")

    # ==========================================
    # ENSEMBLE MODE
    # ==========================================
    if args.ensemble:
        from utils.ensemble import average_waveforms
        import shutil
        import tempfile
        
        models = [m.strip() for m in args.ensemble.split(',')]
        weights = [float(w) for w in args.ensemble_weights.split(',')] if args.ensemble_weights else [1.0] * len(models)
        
        if len(weights) != len(models):
            print("Error: Number of weights must match number of models")
            sys.exit(1)
            
        print(f"\n=== Running Ensemble Mode ===")
        print(f"Models: {', '.join(models)}")
        print(f"Algorithm: {args.ensemble_type}")
        print(f"Weights: {weights}")
        print(f"Input: {len(files)} file(s)")
        
        base_temp_dir = tempfile.mkdtemp(prefix="mss_ensemble_")
        print(f"Temp dir: {base_temp_dir}")
        
        try:
            total_time = 0
            
            # 1. Run separation for each model
            for i, model_name in enumerate(models):
                print(f"\n--- [Model {i+1}/{len(models)}] {model_name} ---")
                model_out_dir = os.path.join(base_temp_dir, model_name)
                
                try:
                    model, config, model_info = load_model(model_name, args.models_dir)
                    model = model.to(args.device)
                    
                    for f in files:
                        # Use WAV/FLOAT for intermediate results to preserve quality
                        elapsed = separate(
                            model, config, f, model_out_dir, model_info, args.device,
                            overlap=args.overlap,
                            batch_size=args.batch_size,
                            use_tta=args.use_tta,
                            output_format='wav',
                            pcm_type='FLOAT',
                            extract_instrumental=args.extract_instrumental,
                            selected_stems=selected_stems,
                            two_stems=args.two_stems,
                            use_fast=args.fast
                        )
                        total_time += elapsed
                    
                    # Free memory
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing model {model_name}: {e}")
                    raise e

            # 2. Average results
            print("\n--- Merging Ensemble Results ---")
            
            for f in files:
                basename = os.path.splitext(os.path.basename(f))[0]
                final_out_subdir = os.path.join(args.output, basename)
                os.makedirs(final_out_subdir, exist_ok=True)
                
                # Determine stems from first successful model output
                first_model_dir = os.path.join(base_temp_dir, models[0], basename)
                if not os.path.exists(first_model_dir):
                    print(f"Warning: No output found for {basename}")
                    continue
                    
                stems = [f.replace('.wav', '') for f in os.listdir(first_model_dir) if f.endswith('.wav')]
                
                for stem in stems:
                    print(f"  Ensembling stem: {stem}...")
                    waveforms = []
                    valid_weights = []
                    
                    for i, model_name in enumerate(models):
                        stem_path = os.path.join(base_temp_dir, model_name, basename, f"{stem}.wav")
                        if os.path.exists(stem_path):
                            wav, sr = librosa.load(stem_path, sr=None, mono=False)
                            if len(wav.shape) == 1: 
                                wav = np.stack([wav, wav], axis=0)
                            waveforms.append(wav)
                            valid_weights.append(weights[i])
                        else:
                            print(f"    Warning: Stem {stem} missing from {model_name}")
                            
                    if waveforms:
                        # Ensure shapes match (min length)
                        min_len = min(w.shape[1] for w in waveforms)
                        waveforms = [w[:, :min_len] for w in waveforms]
                        
                        # Average
                        merged_wav = average_waveforms(np.array(waveforms), valid_weights, args.ensemble_type)
                        
                        # Save final result
                        out_path = os.path.join(final_out_subdir, f"{stem}.{output_format}")
                        sf.write(out_path, merged_wav.T, sr, subtype=args.pcm_type if not args.flac else None)
                        print(f"    Saved: {out_path}")
            
            print(f"\nEnsemble complete! Total processing time: {total_time:.2f}s")
            
        finally:
            print(f"Cleaning up temp dir: {base_temp_dir}")
            shutil.rmtree(base_temp_dir, ignore_errors=True)
            
        return

    # ==========================================
    # SINGLE MODEL MODE
    # ==========================================
    
    # Load model
    model, config, model_info = load_model(args.model, args.models_dir)
    model = model.to(args.device)
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\nProcessing {len(files)} file(s)...")
    if args.overlap:
        print(f"Overlap: {args.overlap}")
    if args.batch_size:
        print(f"Batch size: {args.batch_size}")
    if args.use_tta:
        print("TTA: enabled (will take ~3x longer)")
    
    print(f"Output format: {output_format.upper()} ({args.pcm_type})")
    if args.extract_instrumental:
        print("Extract instrumental: enabled")
    if args.two_stems:
        print(f"Two-stems mode: {args.two_stems} + no_{args.two_stems.lower()}")
    
    total_time = 0
    
    for f in files:
        elapsed = separate(
            model, config, f, args.output, model_info, args.device,
            overlap=args.overlap,
            batch_size=args.batch_size,
            use_tta=args.use_tta,
            output_format=output_format,
            pcm_type=args.pcm_type,
            extract_instrumental=args.extract_instrumental,
            selected_stems=selected_stems,
            two_stems=args.two_stems,
            use_fast=args.fast
        )
        total_time += elapsed
    
    print(f"\nDONE! Total time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
