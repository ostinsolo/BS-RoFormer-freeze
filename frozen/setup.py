"""
cx_Freeze setup for Music Source Separation
Fixes hardcoded path issues by properly configuring package includes.
"""

import sys
import os
from cx_Freeze import setup, Executable

# Add parent directory to path so we can import project modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# All torch submodules that need to be explicitly included
# This prevents partial imports that cause "bad magic number" errors
TORCH_PACKAGES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.modules",
    "torch.utils",
    "torch.utils.data",
    "torch._C",
    "torch._jit_internal",
    "torch.package",
    "torch.package._mangling",
    "torch.package.analyze",
    "torch.package.package_exporter",
    "torch.functional",
    "torch.autograd",
    "torch.cuda",
    "torch.backends",
    "torch.backends.mkl",
    "torch.backends.mkldnn",
    "torch.backends.openmp",
    "torch.backends.cudnn",
    "torch.fft",
    "torch.linalg",
    "torch.special",
    "torch.sparse",
    "torch.distributions",
    "torch.optim",
    "torch.serialization",
    "torch.onnx",
    "torch.profiler",
    "torch.ao",
]

# Other packages to include
OTHER_PACKAGES = [
    "numpy",
    "scipy",
    "scipy.signal",
    "scipy.fft",
    "soundfile",
    "librosa",
    "tqdm",
    "yaml",
    "omegaconf",
    "ml_collections",
    "einops",
    "rotary_embedding_torch",
    "beartype",
    "loralib",
    "numba",
    "llvmlite",
    # Project packages
    "models",
    "models.bs_roformer",
    "models.bandit",
    "models.bandit.core",
    "models.scnet",
    "models.scnet_unofficial",
    "utils",
]

ALL_PACKAGES = TORCH_PACKAGES + OTHER_PACKAGES

# Modules to exclude (reduce size, not needed at runtime)
EXCLUDES = [
    "tkinter",
    "unittest",
    "test",
    "tests",
    "distutils",
    "setuptools",
    "pip",
    "wheel",
    "pkg_resources",
    "pydoc_data",
    "curses",
    "IPython",
    "jupyter",
    "notebook",
    "matplotlib.backends.backend_qt5agg",
    "PyQt5",
    "PySide2",
]

build_options = {
    "packages": ALL_PACKAGES,
    "excludes": EXCLUDES,
    "include_files": [
        # These are copied separately in build.sh, but ensure they're available
    ],
    # Critical: Don't compress .pyc files - torch doesn't like it
    "zip_include_packages": [],
    "zip_exclude_packages": "*",  # Don't zip anything - prevents path issues
    "optimize": 0,  # Don't optimize bytecode - can cause issues with torch
    # Build in lib/ subdirectory
    "build_exe": "dist",
    # Replace paths to make bundle portable
    "replace_paths": [("*", "")],  # Remove all absolute paths
}

base = None
if sys.platform == "win32":
    base = "Console"

executables = [
    Executable(
        "main.py",
        base=base,
        target_name="mss-separate",
    )
]

setup(
    name="mss-separate",
    version="1.8.3",
    description="Music Source Separation - Frozen Executable",
    options={"build_exe": build_options},
    executables=executables,
)
