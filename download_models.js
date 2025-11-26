#!/usr/bin/env node
/**
 * Music Source Separation - Model Downloader
 * ==========================================
 * 
 * A Node.js script for downloading pretrained models.
 * Designed for integration with Max MSP via Node for Max.
 * 
 * Usage:
 *   node download_models.js --list                    # List all available models
 *   node download_models.js --model htdemucs_4stem   # Download specific model
 *   node download_models.js --model vocals_melband_kj --model htdemucs_4stem  # Multiple models
 *   node download_models.js --category vocals        # Download all vocal models
 *   node download_models.js --info bsrofo_sw_fixed   # Get model info without downloading
 * 
 * For Max MSP (Node for Max):
 *   const maxApi = require('max-api');
 *   const downloader = require('./download_models.js');
 *   maxApi.addHandler('download', (modelName) => downloader.downloadModel(modelName));
 *   maxApi.addHandler('list', () => downloader.listModels());
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

// ============================================================================
// MODEL REGISTRY
// ============================================================================

const MODELS = {
    // =========================================================================
    // 4-STEM SEPARATION (bass / drums / vocals / other)
    // =========================================================================
    
    "htdemucs_4stem": {
        description: "HTDemucs 4-stem - Fast and good quality",
        architecture: "HTDemucs (Hybrid Transformer Demucs)",
        quality: "SDR avg: 9.16",
        speed: "fast (~21s on CPU)",
        category: "4stem",
        stems: ["bass", "drums", "vocals", "other"],
        files: {
            checkpoint: {
                url: "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
                path: "weights/htdemucs_4stem.th"
            },
            config: {
                url: "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
                path: "configs/config_htdemucs_4stem.yaml"
            }
        }
    },
    
    "bsroformer_4stem": {
        description: "BS-RoFormer 4-stem - High quality",
        architecture: "BS-RoFormer (Band Split RoFormer)",
        quality: "SDR avg: 9.65 (bass: 8.48, drums: 11.61, vocals: 11.08, other: 7.44)",
        speed: "medium (~67s on CPU)",
        category: "4stem",
        stems: ["bass", "drums", "vocals", "other"],
        files: {
            checkpoint: {
                url: "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
                path: "weights/bsroformer_4stem.ckpt"
            },
            config: {
                url: "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml",
                path: "configs/config_bs_roformer_4stem.yaml"
            }
        }
    },
    
    "scnet_xl_4stem": {
        description: "SCNet XL - Highest quality 4-stem",
        architecture: "SCNet (Source Separation Conformer Network)",
        quality: "SDR avg: 10.08 (bass: 9.23, drums: 11.81, vocals: 11.42, other: 7.88) 🏆",
        speed: "slow (~94s on CPU)",
        category: "4stem",
        stems: ["bass", "drums", "vocals", "other"],
        files: {
            checkpoint: {
                url: "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt",
                path: "weights/scnet_xl_4stem.ckpt"
            },
            config: {
                url: "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml",
                path: "configs/config_scnet_xl_4stem.yaml"
            }
        }
    },
    
    // =========================================================================
    // 6-STEM SEPARATION (+ piano / guitar)
    // =========================================================================
    
    "htdemucs_6stem": {
        description: "HTDemucs 6-stem - Official Facebook model",
        architecture: "HTDemucs (Hybrid Transformer Demucs)",
        quality: "Good",
        speed: "fast (~25s on CPU)",
        category: "6stem",
        stems: ["bass", "drums", "vocals", "other", "guitar", "piano"],
        files: {
            checkpoint: {
                url: "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/htdemucs_6s-cec6f0ff-18c7c1c7.th",
                path: "weights/htdemucs_6stem.th"
            },
            config: {
                url: "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_htdemucs_6stems.yaml",
                path: "configs/config_htdemucs_6stem.yaml"
            }
        }
    },
    
    "bsrofo_sw_fixed": {
        description: "BS-ROFO-SW-Fixed 6-stem - High quality by jarredou",
        architecture: "BS-RoFormer (Band Split RoFormer)",
        quality: "High - excellent for guitar/piano separation",
        speed: "medium (~48s on CPU)",
        category: "6stem",
        stems: ["bass", "drums", "vocals", "other", "guitar", "piano"],
        files: {
            checkpoint: {
                url: "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/bs-rofo_sw_fixed.ckpt",
                path: "weights/bsrofo_sw_fixed.ckpt"
            },
            config: {
                url: "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/config_bs_roformer_6_stems.yaml",
                path: "configs/config_bsrofo_sw_fixed.yaml"
            }
        }
    },
    
    // =========================================================================
    // VOCAL EXTRACTION
    // =========================================================================
    
    "vocals_melband_kj": {
        description: "MelBand RoFormer KJ - BEST vocal quality",
        architecture: "MelBand RoFormer",
        quality: "SDR: 10.98 🏆 - Highest vocal quality",
        speed: "medium (~54s on CPU)",
        category: "vocals",
        stems: ["vocals", "other"],
        files: {
            checkpoint: {
                url: "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
                path: "weights/vocals_melband_kj.ckpt"
            },
            config: {
                url: "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/config_vocals_mel_band_roformer_kj.yaml",
                path: "configs/config_vocals_melband_kj.yaml"
            }
        }
    },
    
    "vocals_bs_roformer_viperx": {
        description: "BS-RoFormer ViperX - High quality vocals",
        architecture: "BS-RoFormer",
        quality: "SDR: 10.87",
        speed: "slow (~125s on CPU)",
        category: "vocals",
        stems: ["vocals", "other"],
        files: {
            checkpoint: {
                url: "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                path: "weights/vocals_bs_roformer_viperx.ckpt"
            },
            config: {
                url: "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
                path: "configs/config_vocals_bs_roformer_viperx.yaml"
            }
        }
    },
    
    "vocals_mdx23c": {
        description: "MDX23C Vocals - Good speed/quality balance",
        architecture: "MDX23C (TFC-TDF)",
        quality: "SDR: 10.17",
        speed: "fast (~30s on CPU)",
        category: "vocals",
        stems: ["vocals", "other"],
        files: {
            checkpoint: {
                url: "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
                path: "weights/vocals_mdx23c.ckpt"
            },
            config: {
                url: "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_mdx23c.yaml",
                path: "configs/config_vocals_mdx23c.yaml"
            }
        }
    },
    
    // =========================================================================
    // SPECIAL PURPOSE
    // =========================================================================
    
    "dereverb": {
        description: "De-Reverb model - Remove reverb from audio",
        architecture: "BS-RoFormer",
        quality: "High",
        speed: "medium",
        category: "effects",
        stems: ["dry", "reverb"],
        files: {
            checkpoint: {
                url: "https://huggingface.co/jarredou/MVSEP_models/resolve/main/dereverb_bs_roformer_8_256dim_8depth.ckpt",
                path: "weights/dereverb.ckpt"
            },
            config: {
                url: "https://huggingface.co/jarredou/MVSEP_models/resolve/main/dereverb_bs_roformer_8_256dim_8depth.yaml",
                path: "configs/config_dereverb.yaml"
            }
        }
    },
    
    "denoise": {
        description: "De-Noise model - Remove noise from audio",
        architecture: "BS-RoFormer",
        quality: "High",
        speed: "medium",
        category: "effects",
        stems: ["clean", "noise"],
        files: {
            checkpoint: {
                url: "https://huggingface.co/jarredou/MVSEP_models/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
                path: "weights/denoise.ckpt"
            },
            config: {
                url: "https://huggingface.co/jarredou/MVSEP_models/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.yaml",
                path: "configs/config_denoise.yaml"
            }
        }
    }
};

// ============================================================================
// DOWNLOAD FUNCTIONS
// ============================================================================

/**
 * Download a file with progress tracking and redirect handling
 */
function downloadFile(url, destPath, progressCallback) {
    return new Promise((resolve, reject) => {
        const parsedUrl = new URL(url);
        const protocol = parsedUrl.protocol === 'https:' ? https : http;
        
        // Ensure directory exists
        const dir = path.dirname(destPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        
        const request = protocol.get(url, { 
            headers: { 'User-Agent': 'Mozilla/5.0' }
        }, (response) => {
            // Handle redirects
            if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
                downloadFile(response.headers.location, destPath, progressCallback)
                    .then(resolve)
                    .catch(reject);
                return;
            }
            
            if (response.statusCode !== 200) {
                reject(new Error(`Failed to download: HTTP ${response.statusCode}`));
                return;
            }
            
            const totalSize = parseInt(response.headers['content-length'], 10) || 0;
            let downloadedSize = 0;
            
            const file = fs.createWriteStream(destPath);
            
            response.on('data', (chunk) => {
                downloadedSize += chunk.length;
                if (progressCallback && totalSize > 0) {
                    progressCallback(downloadedSize, totalSize);
                }
            });
            
            response.pipe(file);
            
            file.on('finish', () => {
                file.close();
                resolve({ path: destPath, size: downloadedSize });
            });
            
            file.on('error', (err) => {
                fs.unlink(destPath, () => {});
                reject(err);
            });
        });
        
        request.on('error', reject);
        request.setTimeout(30000, () => {
            request.destroy();
            reject(new Error('Download timeout'));
        });
    });
}

/**
 * Format bytes to human readable string
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * Download a specific model by name
 */
async function downloadModel(modelName, basePath = '.') {
    const model = MODELS[modelName];
    if (!model) {
        throw new Error(`Unknown model: ${modelName}. Use --list to see available models.`);
    }
    
    console.log('\n' + '='.repeat(60));
    console.log(`Model: ${modelName}`);
    console.log(`Description: ${model.description}`);
    console.log(`Architecture: ${model.architecture}`);
    console.log(`Quality: ${model.quality}`);
    console.log(`Speed: ${model.speed}`);
    console.log(`Stems: ${model.stems.join(', ')}`);
    console.log('='.repeat(60));
    
    const results = [];
    
    for (const [fileType, fileInfo] of Object.entries(model.files)) {
        const destPath = path.join(basePath, fileInfo.path);
        
        // Skip if file already exists
        if (fs.existsSync(destPath)) {
            console.log(`  ✓ Already exists: ${fileInfo.path}`);
            results.push({ type: fileType, path: destPath, status: 'exists' });
            continue;
        }
        
        console.log(`  ↓ Downloading ${fileType}: ${fileInfo.url.substring(0, 60)}...`);
        console.log(`    → ${fileInfo.path}`);
        
        try {
            const result = await downloadFile(fileInfo.url, destPath, (downloaded, total) => {
                const percent = Math.round((downloaded / total) * 100);
                process.stdout.write(`\r    Progress: ${percent}% (${formatBytes(downloaded)} / ${formatBytes(total)})`);
            });
            console.log(`\n  ✓ Downloaded: ${formatBytes(result.size)}`);
            results.push({ type: fileType, path: destPath, status: 'downloaded', size: result.size });
        } catch (err) {
            console.log(`\n  ✗ Failed: ${err.message}`);
            results.push({ type: fileType, path: destPath, status: 'failed', error: err.message });
        }
    }
    
    return { model: modelName, files: results };
}

/**
 * List all available models
 */
function listModels(category = null) {
    console.log('\n' + '='.repeat(70));
    console.log('AVAILABLE MODELS');
    console.log('='.repeat(70));
    
    const categories = {};
    
    for (const [name, model] of Object.entries(MODELS)) {
        if (category && model.category !== category) continue;
        
        if (!categories[model.category]) {
            categories[model.category] = [];
        }
        categories[model.category].push({ name, ...model });
    }
    
    for (const [cat, models] of Object.entries(categories)) {
        console.log(`\n### ${cat.toUpperCase()} ###\n`);
        
        for (const model of models) {
            console.log(`  ${model.name}`);
            console.log(`    ${model.description}`);
            console.log(`    Quality: ${model.quality}`);
            console.log(`    Speed: ${model.speed}`);
            console.log(`    Stems: ${model.stems.join(', ')}`);
            console.log('');
        }
    }
    
    return Object.keys(MODELS);
}

/**
 * Get model info without downloading
 */
function getModelInfo(modelName) {
    const model = MODELS[modelName];
    if (!model) {
        return null;
    }
    return { name: modelName, ...model };
}

/**
 * Get all model names
 */
function getModelNames() {
    return Object.keys(MODELS);
}

/**
 * Get models by category
 */
function getModelsByCategory(category) {
    return Object.entries(MODELS)
        .filter(([_, model]) => model.category === category)
        .map(([name, model]) => ({ name, ...model }));
}

// ============================================================================
// CLI INTERFACE
// ============================================================================

async function main() {
    const args = process.argv.slice(2);
    
    if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
        console.log(`
Music Source Separation - Model Downloader
==========================================

Usage:
  node download_models.js --list                     List all available models
  node download_models.js --list --category vocals   List models in category
  node download_models.js --model <name>             Download a specific model
  node download_models.js --model <n1> --model <n2>  Download multiple models
  node download_models.js --info <name>              Get model info (no download)
  node download_models.js --all                      Download all models

Categories: 4stem, 6stem, vocals, effects

Examples:
  node download_models.js --model htdemucs_4stem
  node download_models.js --model vocals_melband_kj --model bsrofo_sw_fixed
  node download_models.js --list --category 4stem
`);
        return;
    }
    
    // Parse arguments
    const modelsToDownload = [];
    let listOnly = false;
    let category = null;
    let infoOnly = null;
    let downloadAll = false;
    
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--list') {
            listOnly = true;
        } else if (args[i] === '--category' && args[i + 1]) {
            category = args[++i];
        } else if (args[i] === '--model' && args[i + 1]) {
            modelsToDownload.push(args[++i]);
        } else if (args[i] === '--info' && args[i + 1]) {
            infoOnly = args[++i];
        } else if (args[i] === '--all') {
            downloadAll = true;
        }
    }
    
    // Execute commands
    if (listOnly) {
        listModels(category);
        return;
    }
    
    if (infoOnly) {
        const info = getModelInfo(infoOnly);
        if (info) {
            console.log(JSON.stringify(info, null, 2));
        } else {
            console.error(`Unknown model: ${infoOnly}`);
            process.exit(1);
        }
        return;
    }
    
    if (downloadAll) {
        for (const modelName of Object.keys(MODELS)) {
            await downloadModel(modelName);
        }
        return;
    }
    
    if (modelsToDownload.length > 0) {
        for (const modelName of modelsToDownload) {
            try {
                await downloadModel(modelName);
            } catch (err) {
                console.error(`Error downloading ${modelName}: ${err.message}`);
            }
        }
        return;
    }
    
    console.log('No action specified. Use --help for usage information.');
}

// ============================================================================
// EXPORTS FOR MAX MSP (Node for Max)
// ============================================================================

module.exports = {
    MODELS,
    downloadModel,
    downloadFile,
    listModels,
    getModelInfo,
    getModelNames,
    getModelsByCategory,
    formatBytes
};

// Run CLI if called directly
if (require.main === module) {
    main().catch(console.error);
}

