#!/bin/bash
# TerrainAI Deployment Script for Avanish's Machine (C:\Users\avani\terrainai)

echo "================================"
echo "TerrainAI Deployment Starting"
echo "================================"

# Step 1: Verify Python version
echo "[1/8] Checking Python version..."
python --version || { echo "Python not found!"; exit 1; }

# Step 2: Create virtual environment if not exists
echo "[2/8] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python -m venv .venv
    echo "Created new venv"
else
    echo "venv already exists"
fi

# Step 3: Activate venv
echo "[3/8] Activating venv..."
source .venv/Scripts/activate 2>/dev/null || .venv\\Scripts\\activate.bat

# Step 4: Upgrade pip
echo "[4/8] Upgrading pip..."
python -m pip install --upgrade pip

# Step 5: Install requirements
echo "[5/8] Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
[ $? -eq 0 ] || { echo "Dependency installation failed"; exit 1; }

# Step 6: Verify dataset structure
echo "[6/8] Checking dataset structure..."
if [ ! -d "data/train/images" ] || [ ! -d "data/train/masks" ]; then
    echo "⚠️  WARNING: Dataset not found in data/train/"
    echo "   Please ensure the data is downloaded to: C:\\Users\\avani\\terrainai\\data\\"
    echo "   Structure should be:"
    echo "   data/"
    echo "   ├── train/images/"
    echo "   ├── train/masks/"
    echo "   ├── val/images/"
    echo "   ├── val/masks/"
    echo "   └── testImages/"
fi

# Step 7: Create necessary directories
echo "[7/8] Creating output directories..."
mkdir -p checkpoints logs predictions

# Step 8: Verify key dependencies
echo "[8/8] Verifying key packages..."
python -c "import torch; print('✓ torch:', torch.__version__)"
python -c "import segmentation_models_pytorch; print('✓ segmentation_models_pytorch ok')"
python -c "import transformers; print('✓ transformers ok')"
python -c "import albumentations; print('✓ albumentations ok')"
python -c "import pandas; print('✓ pandas ok')"

echo ""
echo "================================"
echo "✅ Deployment Complete!"
echo "================================"
echo ""
echo "Ready to start training!"
echo ""
echo "For DEBUG mode (quick test on CPU):"
echo "  python src/train.py --debug --root C:\\Users\\avani\\terrainai"
echo ""
echo "For FULL training (20 epochs on GPU):"
echo "  python src/train.py --root C:\\Users\\avani\\terrainai"
echo ""
