@echo off
REM TerrainAI Deployment Script for Avanish's Machine (C:\Users\avani\terrainai)
REM Run this as Administrator for best results

setlocal enabledelayedexpansion

echo ================================
echo TerrainAI Deployment Starting
echo ================================
echo.

REM Step 1: Verify Python version
echo [1/8] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    exit /b 1
)
python --version

REM Step 2: Create virtual environment if not exists
echo [2/8] Setting up virtual environment...
if not exist ".venv" (
    echo Creating new venv...
    python -m venv .venv
) else (
    echo venv already exists
)

REM Step 3: Activate venv
echo [3/8] Activating venv...
call .venv\Scripts\activate.bat

REM Step 4: Upgrade pip
echo [4/8] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Step 5: Install requirements
echo [5/8] Installing dependencies ^(this may take a few minutes^)...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed
    exit /b 1
)

REM Step 6: Verify dataset structure
echo [6/8] Checking dataset structure...
if not exist "data\train\images" (
    echo.
    echo ^^! WARNING: Dataset not found in data\train\
    echo    Please ensure the data is downloaded to: C:\Users\avani\terrainai\data\
    echo    Structure should be:
    echo    data\
    echo    ├── train\images\
    echo    ├── train\masks\
    echo    ├── val\images\
    echo    ├── val\masks\
    echo    └── testImages\
    echo.
)

REM Step 7: Create necessary directories
echo [7/8] Creating output directories...
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
if not exist "predictions" mkdir predictions

REM Step 8: Verify key dependencies
echo [8/8] Verifying key packages...
python -c "import torch; print('✓ torch:', torch.__version__)"
python -c "import segmentation_models_pytorch; print('✓ segmentation_models_pytorch ok')"
python -c "import transformers; print('✓ transformers ok')"
python -c "import albumentations; print('✓ albumentations ok')"
python -c "import pandas; print('✓ pandas ok')"

echo.
echo ================================
echo ✓ Deployment Complete!
echo ================================
echo.
echo Ready to start training!
echo.
echo For DEBUG mode ^(quick test on CPU^):
echo   python src/train.py --debug --root C:\Users\avani\terrainai
echo.
echo For FULL training ^(20 epochs on GPU^):
echo   python src/train.py --root C:\Users\avani\terrainai
echo.
pause
