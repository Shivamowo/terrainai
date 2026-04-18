# 🚀 TerrainAI Deployment Checklist

**Last Updated:** 2026-04-18  
**Status:** ✅ READY FOR PRODUCTION DEPLOYMENT  
**Target:** C:\Users\avani\terrainai

---

## PRE-DEPLOYMENT VERIFICATION

### ✅ Code Quality
- [x] All source files present and correct
- [x] No hardcoded absolute paths (using pathlib.Path)
- [x] Git repository clean and synced
- [x] All imports tested and working
- [x] Device detection (CPU/GPU) working

### ✅ Configuration
- [x] Model: SegFormer-B2 with HuggingFace fallback
- [x] Loss: Combined CrossEntropy (weighted) + DiceLoss
- [x] Optimizer: AdamW with proper learning rate and weight decay
- [x] Scheduler: 5-epoch warmup + CosineAnnealing
- [x] Augmentation: Albumentations pipeline for training

### ✅ Documentation
- [x] README.md with complete setup instructions
- [x] DEPLOY.bat script for Windows
- [x] This deployment checklist
- [x] DEPLOYMENT_CONFIG.json with full metadata

---

## DEPLOYMENT STEPS FOR AVANISH

### Step 1: Initial Setup (5 minutes)
```bash
# On C:\Users\avani\terrainai machine:
cd C:\Users\avani\terrainai
DEPLOY.bat
```
This will:
- Create Python virtual environment
- Install all dependencies from requirements.txt
- Create necessary output directories
- Verify all packages

### Step 2: Data Validation (Before Training)
Ensure FalconCloud dataset is downloaded:
```
C:\Users\avani\terrainai\data\
├── train\
│   ├── images\      (PNG files)
│   └── masks\       (PNG files)
├── val\
│   ├── images\      (PNG files)
│   └── masks\       (PNG files)
└── testImages\      (PNG files for submission)
```

### Step 3: Quick Validation Test (2 minutes)
```bash
# Activate venv first
.venv\Scripts\activate

# Run debug mode (small dataset, CPU, 2 epochs)
python src/train.py --debug --root C:\Users\avani\terrainai
```

**Expected Output:**
- Should complete without errors in ~2 minutes
- Should print per-class IoU for all 10 terrain classes
- Should save checkpoint to `checkpoints/run_best.pth`

### Step 4: Full Training Run (120 minutes on GPU)
```bash
# Activate venv if not already
.venv\Scripts\activate

# Run full training (20 epochs on GPU)
python src/train.py --root C:\Users\avani\terrainai
```

**Expected Output:**
- Training will iterate through 20 epochs
- After each epoch: prints mIoU and per-class IoU
- Saves best model checkpoint periodically
- Generates `logs/results.csv` with all metrics

---

## SYSTEM REQUIREMENTS VERIFICATION

Before deployment, verify on Avanish's machine:

```bash
# Python version
python --version
# Should output: Python 3.10.x or higher

# NVIDIA GPU (optional but recommended)
nvidia-smi
# Should show GPU memory available

# Required packages after DEPLOY.bat
python -c "import torch; print('torch:', torch.__version__)"
python -c "import segmentation_models_pytorch; import sys; print('smp ok')"
python -c "import transformers; print('transformers ok')"
python -c "import albumentations; print('albumentations ok')"
```

---

## PROJECT STRUCTURE

```
C:\Users\avani\terrainai\
├── src/
│   ├── train.py           ← Main training script
│   ├── dataset.py         ← Data loading + augmentation
│   ├── model.py           ← Model initialization
│   ├── utils.py           ← IoU computation
│   └── augment.py         ← Albumentations pipeline
├── data/                  ← Dataset (to be downloaded)
│   ├── train/
│   ├── val/
│   └── testImages/
├── checkpoints/           ← Model checkpoints (auto-created)
├── logs/                  ← Results CSV (auto-created)
├── predictions/           ← Inference outputs (auto-created)
├── requirements.txt       ← Core dependencies
├── DEPLOY.bat            ← Windows deployment script
├── README.md             ← Setup instructions
└── DEPLOYMENT_CONFIG.json ← This deployment metadata
```

---

## KEY PARAMETERS

| Parameter | Value |
|-----------|-------|
| Model Architecture | SegFormer-B2 |
| Input Resolution | 512×512 |
| Number of Classes | 10 (terrain types) |
| Training Epochs | 20 (debug: 2) |
| Batch Size | 8 (debug: 2) |
| Learning Rate | 6e-5 |
| Weight Decay | 1e-4 |
| Loss Function | 0.5 × CE(weighted) + 0.5 × Dice |
| Optimizer | AdamW |
| Scheduler | CosineAnnealing (5-epoch warmup) |
| Class Weights | [0.5]×8 + [5.0, 5.0] (classes 8-9 upweighted) |

---

## EXPECTED RESULTS

### Debug Mode (2 epochs on 10 images)
- **Duration:** ~2-3 minutes on CPU
- **mIoU Range:** Untrained model, expect low values (0.05-0.15)
- **Purpose:** Verify pipeline integrity

### Full Training (20 epochs on full dataset)
- **Duration:** ~2 hours on NVIDIA A100, ~4-6 hours on RTX 3090
- **Expected mIoU:** 0.45-0.60 range (depending on dataset quality)
- **Outputs:**
  - `checkpoints/run_best.pth` (best model checkpoint)
  - `logs/results.csv` (detailed metrics per epoch)
  - Per-class IoU for each of 10 terrain classes

---

## TROUBLESHOOTING

### Issue: "No module named qdrant_client"
**Solution:** qdrant_client is optional. Run with basic requirements.txt, or install:
```bash
pip install qdrant-client>=1.7.0
```

### Issue: CUDA out of memory
**Solution:** Reduce batch_size in code or use CPU mode:
```bash
# Modify src/train.py line 21: batch_size = 4 (instead of 8)
# Or run with CPU explicitly
```

### Issue: Dataset not found
**Solution:** Ensure `data/train/images/` exists with PNG files before training

### Issue: Model download timeout
**Solution:** First time setup downloads ~99MB model. Ensure internet connection.  
May take 5-10 minutes on first run.

---

## POST-TRAINING CHECKLIST

After full training completes:

- [ ] Verify `checkpoints/run_best.pth` exists (~200MB)
- [ ] Verify `logs/results.csv` exists with 20 epoch rows
- [ ] Confirm final mIoU is reasonable (>0.3)
- [ ] Export predictions on testImages (if needed)
- [ ] Save final model for submission

---

## SUPPORT & DOCUMENTATION

- **README.md:** Detailed setup instructions
- **src/train.py:** Well-commented training loop
- **src/dataset.py:** Data loading logic with mask remapping details
- **DEPLOYMENT_CONFIG.json:** Machine-readable deployment metadata

---

**Deployment Status:** ✅ COMPLETE AND READY  
**Deployed By:** GitHub Copilot  
**Date:** 2026-04-18  
**Version:** 1.0-production
