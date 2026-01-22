@echo off
setlocal enabledelayedexpansion

REM Build frozen executable for Music Source Separation - Windows
REM Mirrors the GitHub Actions build-windows.yml workflow
REM
REM Usage: build.bat [cpu|cuda]
REM   - cpu: CPU-only PyTorch (smaller, works everywhere)
REM   - cuda: CUDA-enabled PyTorch (faster on NVIDIA GPUs, larger)

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=cpu

echo ============================================================
echo Building Music Source Separation Executable
echo Platform: Windows %BUILD_TYPE%
echo ============================================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10 and add to PATH
    exit /b 1
)

REM Create fresh virtual environment
if exist build_venv rd /s /q build_venv
echo Creating clean Python environment...
python -m venv build_venv
call build_venv\Scripts\activate.bat

REM Skip pip upgrade to avoid WinError 5 permission issue
REM pip install --upgrade pip
echo Using pip version:
python -m pip --version

REM Install dependencies based on build type
if "%BUILD_TYPE%"=="cuda" (
    echo Installing dependencies for CUDA...
    REM PyTorch 2.10.0 with CUDA 12.6 support (latest stable)
    pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
) else (
    echo Installing dependencies for CPU...
    REM PyTorch 2.10.0 CPU-only (latest stable)
    pip install torch==2.10.0 torchaudio==2.10.0
)

REM Install other dependencies
pip install "numpy<2" scipy soundfile librosa tqdm pyyaml omegaconf ml_collections einops rotary-embedding-torch beartype loralib matplotlib
pip install numba llvmlite
pip install cx-Freeze==6.15.16

REM Show installed versions
echo.
echo Installed versions:
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'  Torchaudio: {torchaudio.__version__}')"
echo.

REM Clean previous build
if exist dist rd /s /q dist
echo Cleaned previous build

REM Build with cx_Freeze
echo Building executable...
cxfreeze main.py --target-dir=dist --target-name=mss-separate --packages=torch,torch.nn,torch.nn.modules,torch.nn.functional,torch.utils,torch.utils.data,torch.fft,torch.linalg,torch.autograd,torch.backends,torch.backends.mkl,torch.backends.mkldnn,torch.backends.cudnn,torch.cuda,torch.package,torch.package.analyze,torch._C,torch._jit_internal,torch.jit,torch.onnx,torch.optim,torch.distributions,torch.sparse,torch.special,torch.serialization,numpy,scipy,scipy.signal,scipy.fft,soundfile,librosa,tqdm,yaml,omegaconf,ml_collections,einops,rotary_embedding_torch,beartype,loralib,numba,llvmlite

if not exist dist\mss-separate.exe (
    echo ERROR: Build failed - executable not found
    exit /b 1
)
echo Executable built successfully

REM Copy resources
echo Copying project files...
xcopy /E /I /Y ..\configs dist\configs
xcopy /E /I /Y ..\models dist\models
xcopy /E /I /Y ..\utils dist\utils
copy models.json dist\ 2>nul
if not exist dist\weights mkdir dist\weights

REM Copy soundfile data
echo Copying soundfile data...
for /f "delims=" %%i in ('python -c "import soundfile; import os; print(os.path.dirname(soundfile.__file__))"') do set SOUNDFILE_DIR=%%i
if exist "%SOUNDFILE_DIR%\_soundfile_data" (
    if not exist dist\lib mkdir dist\lib
    xcopy /E /I /Y "%SOUNDFILE_DIR%\_soundfile_data" dist\lib\_soundfile_data
    echo Copied soundfile data
)

REM Copy llvmlite DLLs (critical for librosa)
echo Copying llvmlite DLLs...
for /f "delims=" %%i in ('python -c "import llvmlite; import os; print(os.path.dirname(llvmlite.__file__))"') do set LLVMLITE_DIR=%%i
if exist "%LLVMLITE_DIR%\binding" (
    if not exist dist\lib\llvmlite\binding mkdir dist\lib\llvmlite\binding
    xcopy /E /I /Y "%LLVMLITE_DIR%\binding\*" dist\lib\llvmlite\binding\
    echo Copied llvmlite binding files
)

REM Copy llvmlite.libs folder (contains msvcp140-*.dll required on Windows)
echo Copying llvmlite.libs (MSVC runtime)...
for /f "delims=" %%i in ('python -c "import site; print(site.getsitepackages()[0])"') do set SITE_PACKAGES=%%i
if exist "%SITE_PACKAGES%\llvmlite.libs" (
    if not exist dist\lib\llvmlite.libs mkdir dist\lib\llvmlite.libs
    xcopy /E /I /Y "%SITE_PACKAGES%\llvmlite.libs\*" dist\lib\llvmlite.libs\
    echo Copied llvmlite.libs folder
) else (
    echo WARNING: llvmlite.libs not found at %SITE_PACKAGES%\llvmlite.libs
    REM Try venv location
    if exist "build_venv\Lib\site-packages\llvmlite.libs" (
        if not exist dist\lib\llvmlite.libs mkdir dist\lib\llvmlite.libs
        xcopy /E /I /Y "build_venv\Lib\site-packages\llvmlite.libs\*" dist\lib\llvmlite.libs\
        echo Copied llvmlite.libs from venv
    )
)

REM Verify build
echo.
echo Verifying build...
dist\mss-separate.exe --list-models 2>&1 | findstr /n "." | findstr "^[1-9]:" | findstr /n "." | findstr "^[1-2][0-9]:" >nul
dist\mss-separate.exe --list-models

echo.
echo ============================================================
echo BUILD COMPLETE!
echo ============================================================
echo Output: dist\
echo.
echo Test: dist\mss-separate.exe --list-models
echo.
echo To package for distribution:
if "%BUILD_TYPE%"=="cuda" (
    echo   Use 7-Zip: 7z a mss-separate-win-cuda.7z dist
) else (
    echo   powershell Compress-Archive -Path dist -DestinationPath mss-separate-win-cpu.zip
)
echo.

call build_venv\Scripts\deactivate.bat
