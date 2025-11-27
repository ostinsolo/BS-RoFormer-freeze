#!/usr/bin/env python3
"""
Music Source Separation - Frozen Executable
Models loaded from external directory (not bundled).
"""

import argparse
import os
import sys
import time
import glob

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

MODELS = {
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
        "config": "configs/config_vocals_mel_band_roformer.yaml",
        "checkpoint": "weights/denoise.ckpt",
        "stems": ["clean", "noise"],
        "description": "Remove noise"
    },
    "dereverb": {
        "type": "mel_band_roformer",
        "config": "configs/config_vocals_mel_band_roformer.yaml",
        "checkpoint": "weights/dereverb.ckpt",
        "stems": ["dry", "reverb"],
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

def list_models():
    print("\nAVAILABLE MODELS\n")
    for name, info in MODELS.items():
        print(f"  {name}: {info['description']}")
        print(f"    Stems: {', '.join(info['stems'])}\n")

def load_model(model_name, models_dir=None):
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print("Available models:")
        for name in MODELS.keys():
            print(f"  - {name}")
        sys.exit(1)
    
    model_info = MODELS[model_name]
    base = models_dir if models_dir else BASE_DIR
    config_path = os.path.join(base, model_info["config"])
    checkpoint_path = os.path.join(base, model_info["checkpoint"])
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Download with: node download_models.js --download " + model_name)
        sys.exit(1)
    
    from utils.settings import get_model_from_config
    
    print(f"Loading: {model_name}")
    model, config = get_model_from_config(model_info["type"], config_path)
    
    # Load checkpoint directly
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'state' in state_dict:
        state_dict = state_dict['state']
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, model_info

def separate(model, config, audio_path, output_dir, model_info, device="cpu"):
    from utils.model_utils import demix
    
    print(f"\nProcessing: {audio_path}")
    sample_rate = getattr(config.audio, 'sample_rate', 44100)
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=0)
    
    print("Separating...")
    start = time.time()
    waveforms = demix(config, model, audio, device, model_type=model_info["type"], pbar=True)
    print(f"Time: {time.time() - start:.2f}s")
    
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    out_subdir = os.path.join(output_dir, basename)
    os.makedirs(out_subdir, exist_ok=True)
    
    for stem, wav in waveforms.items():
        path = os.path.join(out_subdir, f"{stem}.wav")
        sf.write(path, wav.T, sample_rate)
        print(f"  Saved: {path}")

def main():
    parser = argparse.ArgumentParser(description="Music Source Separation")
    parser.add_argument("--model", "-m", default="bsrofo_sw")
    parser.add_argument("--input", "-i")
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--device", "-d", default="cpu")
    parser.add_argument("--models-dir")
    args = parser.parse_args()
    
    if args.list_models:
        list_models()
        return
    
    if not args.input or not os.path.exists(args.input):
        print("Error: --input required")
        sys.exit(1)
    
    model, config, model_info = load_model(args.model, args.models_dir)
    model = model.to(args.device)
    
    files = [args.input] if os.path.isfile(args.input) else glob.glob(os.path.join(args.input, "*.*"))
    os.makedirs(args.output, exist_ok=True)
    
    for f in files:
        separate(model, config, f, args.output, model_info, args.device)
    
    print("\nDONE!")

if __name__ == "__main__":
    main()
