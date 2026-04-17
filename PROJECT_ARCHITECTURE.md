# TerrainAI Project Architecture Overview
**Last Updated:** April 17, 2026 | **Status:** Training Complete + Backend Deployed

---

## 🏗️ PROJECT STRUCTURE (COMPLETE BREAKDOWN)

```
C:\Users\avani\terrainai\
│
├── 📁 src/                           [TRAINING CORE]
│   ├── train.py                      (158 lines) — Main training loop
│   ├── model.py                      (42 lines)  — SegFormer-B2 model loading
│   ├── dataset.py                    (82 lines)  — TerrainDataset class
│   ├── utils.py                      (44 lines)  — IoU computation helpers
│   ├── augment.py                    (16 lines)  — Albumentations pipeline
│   ├── schemas.py                    (172 lines) — Qdrant + epoch log schemas
│   ├── qdrant_miner.py              (229 lines) — Hard example mining
│   └── app.py                        (202 lines) — Streamlit dashboard
│
├── 📁 backend/                       [🆕 NEW: API + VECTOR STORE]
│   ├── 📁 api/
│   │   ├── server.py                (~100 lines) — FastAPI /health, /epochs, /failures, /search
│   │   └── __init__.py               (empty)
│   │
│   ├── 📁 dashboard/
│   │   └── app.py                   (custom styled Streamlit dashboard)
│   │
│   ├── 📁 vector/
│   │   ├── qdrant_store.py          (vector store wrapper, in-process Qdrant)
│   │   └── test_qdrant.py           (validation script)
│   │
│   ├── ingest.py                     (batch ingest script)
│   ├── qdrant_data/                  (in-process Qdrant database)
│   └── outputs/                      (training logs storage)
│
├── 📁 data/
│   ├── train/
│   │   ├── images/                  (2,857 PNG files)
│   │   └── masks/                   (2,857 uint16 PNG files)
│   ├── val/
│   │   ├── images/                  (317 PNG files)
│   │   └── masks/                   (317 uint16 PNG files)
│   └── testImages/                  (0 files — for submission)
│
├── 📁 checkpoints/
│   └── run_best.pth                 (109.6 MB model weights)
│
├── 📁 logs/
│   ├── epoch_metrics.jsonl          (22 epochs, Qdrant mining logs)
│   └── results.csv                  (20 epochs, mIoU + per-class IoU)
│
├── 📄 requirements.txt               (10 core ML dependencies)
├── 📄 DIAGNOSTIC_REPORT.md          (full system analysis)
├── 📄 README.md                      (project setup)
├── 📄 results.md                     (run results table)
└── .gitignore                        (newly added)
```

---

## 🎯 TRAINING PIPELINE (src/)

### **1. Training (`train.py` — 158 lines)**
```python
# Main training loop: 20 epochs
# ✅ Loads SegFormer-B2 model
# ✅ Combined loss: 0.5*CE + 0.5*DiceLoss
# ✅ Class weights: [0.5, 0.5, ..., 5.0, 5.0] (rare classes upweighted)
# ✅ Qdrant hard-example mining after each epoch
# ✅ Logs to: logs/epoch_metrics.jsonl + logs/results.csv
```

**Key outputs after training:**
- `logs/epoch_metrics.jsonl` — Per-epoch performance data (JSON Lines format)
- `logs/results.csv` — Structured metrics table (mIoU + per-class IoU)
- `checkpoints/run_best.pth` — Best model weights (saved at epoch 19)

---

### **2. Dataset (`dataset.py` — 82 lines)**
```python
class TerrainDataset(Dataset):
    # Loads RGB images + uint16 masks
    # Remaps mask values: 100→0, 200→1, ..., 10000→9 (10 classes)
    # Resizes to 512x512
    # Applies Albumentations augmentation (train split only)
    # Normalizes with ImageNet stats
```

**Class Mapping:**
```
100 → Class 0: Trees
200 → Class 1: Lush Bushes
300 → Class 2: Dry Grass
500 → Class 3: Dry Bushes
550 → Class 4: Ground Clutter
600 → Class 5: Flowers
700 → Class 6: Logs
800 → Class 7: Rocks
7100 → Class 8: Landscape ⭐ RARE
10000 → Class 9: Sky ⭐ RARE
```

---

### **3. Model (`model.py` — 42 lines)**
```python
def get_model(num_classes=10, device='cuda'):
    # Try: SMP SegFormer-B2
    # Fallback: HuggingFace nvidia/mit-b2
    # Wrapped in ModelWrapper to handle both backends
```

---

### **4. Utilities (`utils.py` — 44 lines)**
```python
def compute_iou_per_class():  # Global IoU per class
def compute_miou():           # Mean IoU from dict
def get_class_names():        # Class name lookup
```

---

### **5. Augmentation (`augment.py` — 16 lines)**
```python
# Albumentations pipeline:
- RandomBrightnessContrast (p=0.5)
- GaussNoise (p=0.4)
- MotionBlur (p=0.3)
- HueSaturationValue (p=0.4)
- HorizontalFlip (p=0.5)
- CoarseDropout (p=0.3)
```

---

### **6. Qdrant Mining (`qdrant_miner.py` — 229 lines)**
```python
# Hard example identification:
# - Identifies validation images where rare classes (8,9) have IoU < 0.2
# - Stores encoder embeddings (512-D vectors) in Qdrant
# - Provides WeightedRandomSampler for oversampling hard examples
# - From epoch 2 onwards, training uses sampler instead of uniform shuffle
```

---

### **7. Schemas (`schemas.py` — 172 lines)**
```python
# TypedDict definitions for standardization:

class QdrantPayload:
    image_path: str
    class_id: int
    iou: float
    epoch: int
    split: str
    run_id: str
    is_rare_class: bool
    ...

class EpochLog:
    epoch: int
    run_id: str
    metrics: MetricsBlock
    training: TrainingMetrics
    failure_analysis: FailureAnalysis
    timestamp: str
```

---

## 🚀 BACKEND SYSTEM (backend/)

### **📡 API Server (`backend/api/server.py` — ~100 lines)**

**FastAPI endpoints:**

```python
GET  /health                    → Status + Qdrant point count
GET  /epochs                    → All training epochs
GET  /epoch/{epoch_id}          → Single epoch details
GET  /failures/{class_id}       → Hard examples for class
GET  /search                    → Vector similarity search
```

**Example responses:**
```json
# GET /health
{
  "status": "ok",
  "qdrant_points": 450
}

# GET /epoch/10
{
  "epoch": 10,
  "miou": 0.5902,
  "class_ious": [...],
  "training": {"loss": 0.2857, "val_loss": 0.2401}
}

# GET /failures/8
[
  {
    "image_path": "data/val/images/img_001.png",
    "iou": 0.15,
    "epoch": 8,
    "class_id": 8
  }, ...
]
```

---

### **🎨 Dashboard (`backend/dashboard/app.py`)**

**Features:**
- Custom CSS injection (dark theme with accent colors)
- Real-time API polling
- Failure browser
- Training metrics visualization
- Responsive design for presentation

**Styling:**
- Accent: `#00ff94` (neon green)
- Secondary: `#00c8ff` (cyan)
- Background: `#0a0c10` (dark)
- Font: JetBrains Mono + Syne Sans

---

### **🧠 Vector Store (`backend/vector/qdrant_store.py`)**

**Design: In-process Qdrant (no Docker required!)**

```python
class TerrainVectorStore:
    def __init__(self):
        # Uses local file storage: ./qdrant_data/
        self.client = QdrantClient(path="./qdrant_data")
    
    def _create_collection(self):
        # Collection: "terrain_patches"
        # Embedding dim: 512 (SegFormer-B2 encoder output)
        # Distance metric: COSINE
    
    def index_batch(batch_points):
        # Called after each epoch by train.py
        # Stores: embeddings + metadata (iou, class_id, epoch, etc.)
    
    def get_hard_examples(class_id, limit=16):
        # Returns lowest-IoU examples for a class
```

**Indexes:**
- `class_id` (INTEGER)
- `is_rare_class` (BOOLEAN)
- `iou_score` (FLOAT)

---

### **📥 Data Ingestion (`backend/ingest.py`)**

```python
def ingest():
    store = TerrainVectorStore()
    data = load_data()  # from outputs/hard_examples.json
    store.index_batch(data)
```

---

### **✅ Testing (`backend/vector/test_qdrant.py`)**

```python
def check_logs():
    # Verifies training_logs.json exists
    # Shows total epochs, last epoch mIoU, loss
    
def check_qdrant():
    # Counts total vectors stored
    # Shows sample retrievals
```

---

## 📊 TRAINING RESULTS SUMMARY

### **Final Performance (Epoch 19/20)**
```
mIoU:              61.09% ✅
Training Loss:     0.2592 ✅ (decreasing, no overfitting)
Validation Loss:   0.2239 ✅ (decreasing)
Run ID:            run_main
```

### **Per-Class IoU (Epoch 19)**
```
Class 0 (Trees):            82.32% 🟢 EXCELLENT
Class 1 (Lush Bushes):      66.88% 🟢 GOOD
Class 2 (Dry Grass):        60.42% 🟡 FAIR
Class 3 (Dry Bushes):       48.05% 🟡 FAIR
Class 4 (Ground Clutter):   25.38% 🔴 WEAK
Class 5 (Flowers):          61.97% 🟢 GOOD
Class 6 (Logs):             55.82% 🟡 FAIR
Class 7 (Rocks):            47.02% 🟡 FAIR
Class 8 (Landscape):        65.04% 🟢 GOOD
Class 9 (Sky):              98.01% 🟢 EXCEPTIONAL
```

---

## 🔄 DATA FLOW ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (train.py)                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Load TerrainDataset (2,857 train / 317 val images)          │
│  2. Initialize SegFormer-B2 model                               │
│  3. For each epoch (20 epochs):                                 │
│     a) Forward pass with combined loss (CE + Dice)              │
│     b) Backprop + optimizer step                                │
│     c) Validation: compute global IoU per class                 │
│     d) QDRANT MINING: extract embeddings + identify hard cases  │
│     e) Save metrics → logs/epoch_metrics.jsonl + results.csv   │
│     f) Checkpoint best model → checkpoints/run_best.pth         │
│     g) From epoch 2+: Resample next epoch using hard sampler    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              QDRANT VECTOR STORE (backend/vector/)               │
├─────────────────────────────────────────────────────────────────┤
│  • Collection: "terrain_patches"                                 │
│  • Stores: 512-D embeddings + metadata                           │
│  • Queryable by: class_id, iou_score, epoch, run_id             │
│  • Used for: Hard example analysis                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────┬──────────────────┬──────────────────┐
│  FastAPI Server  │  Streamlit Dash  │  Streamlit Dash  │
│  (backend/api)   │  (backend/dash)  │  (src/app.py)    │
├──────────────────┼──────────────────┼──────────────────┤
│ • /health        │ • Training logs  │ • Live predict   │
│ • /epochs        │ • Metrics graph  │ • Metrics table  │
│ • /failures      │ • Failure browser│ • Results summary│
│ • /search        │                  │                  │
└──────────────────┴──────────────────┴──────────────────┘
```

---

## 🔧 HOW TO RUN EVERYTHING

### **1. Training** (creates logs, embeddings)
```bash
cd C:\Users\avani\terrainai
conda activate terrainai
python src/train.py --root . --run_id run_main
```

### **2. API Server** (serves metrics, embeddings)
```bash
cd backend/
pip install fastapi uvicorn
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### **3. Dashboard (backend styled)**
```bash
cd backend/
pip install streamlit
streamlit run dashboard/app.py
```

### **4. Dashboard (original, in src)**
```bash
cd src/
streamlit run app.py
```

### **5. Ingest hard examples**
```bash
cd backend/
python ingest.py
```

### **6. Test Qdrant**
```bash
cd backend/vector/
python test_qdrant.py
```

---

## 📦 DEPENDENCIES

**Core (requirements.txt):**
```
torch>=1.9.0
torchvision>=0.10.0
segmentation-models-pytorch>=0.3.0
Pillow>=8.0.0
numpy>=1.21.0
albumentations>=1.3.0
timm>=0.9.0
pandas>=1.3.0
transformers>=4.20.0
qdrant-client>=1.7.0
```

**Additional (for backend):**
```
pip install fastapi uvicorn requests streamlit
```

---

## 🎯 WHAT'S MISSING FOR HACKATHON SUBMISSION

| Item | Status | Notes |
|------|--------|-------|
| ✅ Trained model | DONE | 61.09% mIoU, epoch 19 |
| ✅ Training logs | DONE | epoch_metrics.jsonl + results.csv |
| ✅ API server | DONE | FastAPI with 5 endpoints |
| ✅ Dashboard | DONE | 2 Streamlit apps ready |
| ✅ Vector store | DONE | In-process Qdrant, no Docker needed |
| ❌ Test inference | TODO | Generate predictions on testImages/ |
| ❌ 8-page report | TODO | Methodology + failure analysis |
| ❌ README instructions | TODO | Reproduction steps |
| ❌ Visualization graphs | TODO | Loss curves, confusion matrix |

---

## 🚀 NEXT STEPS (PRIORITY ORDER)

1. **Write 8-page hackathon report** (methodology, results, failures)
2. **Generate visualization graphs** (loss curves, per-class IoU charts)
3. **Create test.py** for inference on testImages/
4. **Update README.md** with reproduction instructions
5. **Commit backend changes** to git
6. **Package final submission** folder

---

## 📝 GIT HISTORY

```
2b66dbe (HEAD → main)  Merge branch 'main' of https://github.com/Shivamowo/terrainai
860f93d                feat: add Streamlit dashboard for live predictions and training metrics visualization
7b2b322 (origin/main)  did a lil backend, API and dashboard stuff ✨
64129ce                feat: standardize Qdrant/epoch-metrics schemas via shared schemas.py
b1cbcc1                feat: add Qdrant hard example mining for rare classes (Logs, Flowers)
```

**Status:** 3 commits ahead of origin/main

---

## 💾 KEY FILES BY ROLE

**For Judges:**
- `README.md` — Setup & reproduction
- `DIAGNOSTIC_REPORT.md` — System analysis
- `results.md` — Results table
- `logs/results.csv` — Per-epoch metrics
- `checkpoints/run_best.pth` — Model weights

**For Developers:**
- `src/train.py` — Training script
- `src/dataset.py` — Data loading
- `src/model.py` — Model architecture
- `backend/api/server.py` — API endpoints
- `backend/vector/qdrant_store.py` — Vector storage

**For Deployment:**
- `backend/dashboard/app.py` — Main dashboard
- `backend/ingest.py` — Data ingestion
- `.gitignore` — VCS exclusions
- `requirements.txt` — Dependencies

---

**Generated:** April 17, 2026  
**Project:** TerrainAI Semantic Segmentation (CodeWizards 2.0 Hackathon)  
**Team:** Avanish, Shivam, Rohan, Shaurya
