# 🚀 TerrainAI Complete Startup Guide

**Quick Reference:** Run everything in ~5 minutes  
**Status:** All components ready  
**Environment:** Python 3.10, PyTorch 2.7.1 + CUDA

---

## 🎯 FULL SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                   YOUR MACHINE (Windows)                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📊 Training Loop (train.py)                               │
│  ↓ (generates logs + embeddings)                            │
│  ├─→ logs/epoch_metrics.jsonl                              │
│  ├─→ logs/results.csv                                      │
│  └─→ backend/qdrant_data/ (embeddings)                     │
│                                                               │
│  📡 FastAPI Server (backend/api/server.py)                 │
│  ├─→ http://localhost:8000                                 │
│  ├─→ /health, /epochs, /failures, /search                 │
│  └─→ Reads from: qdrant_data/ + logs/                     │
│                                                               │
│  🎨 Streamlit Dashboard #1 (backend/dashboard/app.py)      │
│  ├─→ http://localhost:8501                                 │
│  ├─→ Styled interface with metrics                         │
│  └─→ Calls API server on http://localhost:8000            │
│                                                               │
│  🎨 Streamlit Dashboard #2 (src/app.py)                    │
│  ├─→ http://localhost:8502 (or next available)            │
│  ├─→ Live predictions + training metrics                   │
│  └─→ Calls API server on http://localhost:8000            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 PREREQUISITE: Verify Everything is Installed

```bash
# Activate venv
.venv\Scripts\activate

# Check Python
python --version
# Expected: Python 3.10.11

# Check key packages
python -c "import torch; print('torch:', torch.__version__)"
python -c "import fastapi; print('fastapi ok')"
python -c "import streamlit; print('streamlit ok')"
python -c "import qdrant_client; print('qdrant_client ok')"
```

If any are missing:
```bash
pip install fastapi uvicorn requests streamlit
```

---

## 🏃 OPTION A: QUICKSTART (Everything in 1 Terminal)

**Use this if you just want to see it work fast:**

```bash
cd C:\Users\avani\terrainai

# 1. Activate venv
.venv\Scripts\activate

# 2. Run API in background + both dashboards
start python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
timeout 2
start streamlit run backend/dashboard/app.py --logger.level=warning
timeout 2
start streamlit run src/app.py --logger.level=warning

# 3. Your browser will open automatically. If not:
# API Dashboard:      http://localhost:8501
# Prediction Dashboard: http://localhost:8502
# API Health:         http://localhost:8000/health
```

---

## 🎯 OPTION B: PROFESSIONAL SETUP (Separate Terminals - RECOMMENDED)

**Use this for a proper dev environment:**

### **Terminal 1: API Server** 
```bash
cd C:\Users\avani\terrainai
.venv\Scripts\activate

# FastAPI server
python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --reload

# Output should show:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

**✅ What it does:**
- Serves `/health`, `/epochs`, `/failures/{class_id}`, `/search`
- Reads from `logs/epoch_metrics.jsonl`
- Reads from `backend/qdrant_data/` (embeddings)
- Accessible at: `http://localhost:8000`

**Test it:**
```bash
# In another terminal:
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","qdrant_points":450}
```

---

### **Terminal 2: Styled Dashboard** (🎨 Better UX)
```bash
cd C:\Users\avani\terrainai
.venv\Scripts\activate

# Backend dashboard (custom styled)
streamlit run backend/dashboard/app.py

# Browser opens automatically at:
# http://localhost:8501
```

**✅ Features:**
- 📊 Training logs visualization
- 📈 Metrics graphs
- 🔍 Failure browser (search hard examples)
- Dark theme with neon accents

---

### **Terminal 3: Prediction Dashboard** (Optional)
```bash
cd C:\Users\avani\terrainai
.venv\Scripts\activate

# Original prediction dashboard
streamlit run src/app.py

# Browser opens at:
# http://localhost:8502 (or next available port)
```

**✅ Features:**
- 🔍 Live image upload & prediction
- 📈 Training metrics table
- 📋 Results summary

---

### **Terminal 4: Training** (If you want to retrain)
```bash
cd C:\Users\avani\terrainai
.venv\Scripts\activate

# Run training (20 epochs, ~1 hour)
python src/train.py --root . --run_id run_main

# Or debug mode (2 epochs, ~2 min)
python src/train.py --root . --run_id debug_test --debug

# Output files created:
# - logs/epoch_metrics.jsonl (appended each epoch)
# - logs/results.csv (final)
# - checkpoints/run_best.pth (best model)
```

---

## 📍 ACCESS POINTS

Once everything is running:

| Component | URL | Port | Terminal |
|-----------|-----|------|----------|
| **API Server** | http://localhost:8000 | 8000 | Terminal 1 |
| **Dashboard (Styled)** | http://localhost:8501 | 8501 | Terminal 2 |
| **Dashboard (Predictions)** | http://localhost:8502 | 8502 | Terminal 3 |
| **API Docs** | http://localhost:8000/docs | 8000 | Terminal 1 |

---

## 🔄 HOW THEY COMMUNICATE

```
Dashboard #1 (8501)  ─────────┐
                               │ HTTP GET requests
Dashboard #2 (8502)  ─────────→ API Server (8000)
                               │
                               ↓
                    Reads from disk:
                    - logs/epoch_metrics.jsonl
                    - logs/results.csv
                    - backend/qdrant_data/ (embeddings)
```

---

## 📊 API ENDPOINTS REFERENCE

### **1. Health Check**
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "qdrant_points": 450
}
```

### **2. Get All Epochs**
```bash
curl http://localhost:8000/epochs
```
**Response:**
```json
[
  {
    "epoch": 1,
    "miou": 0.2661,
    "class_ious": [0.371, 0.334, ...],
    "loss": 0.6676,
    "val_loss": 0.3984
  },
  { ... epoch 2-20 ... }
]
```

### **3. Get Single Epoch**
```bash
curl http://localhost:8000/epoch/10
```
**Response:**
```json
{
  "epoch": 10,
  "miou": 0.5902,
  "training": {"loss": 0.2857, "val_loss": 0.2401}
}
```

### **4. Get Hard Examples (Failures)**
```bash
curl "http://localhost:8000/failures/8?limit=10"
```
**Response:**
```json
[
  {
    "image_path": "data/val/images/img_001.png",
    "iou": 0.15,
    "epoch": 8,
    "class_id": 8
  },
  ...
]
```

### **5. Vector Search**
```bash
curl http://localhost:8000/search
```
**Response:**
```json
[
  {
    "class_id": 8,
    "iou": 0.12,
    "image_path": "data/val/images/img_042.png"
  },
  ...
]
```

---

## 🧪 TESTING EVERYTHING

### **Test 1: Verify Logs Exist**
```bash
# Check training metrics
type logs\epoch_metrics.jsonl
type logs\results.csv

# Expected: JSON lines + CSV with 20 epochs
```

### **Test 2: Verify API Server Starts**
```bash
python -m uvicorn backend.api.server:app --port 8000

# Expected output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

### **Test 3: Query API from Terminal**
```powershell
# PowerShell:
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Expected: {"status":"ok","qdrant_points":450}
```

### **Test 4: Start Dashboard**
```bash
streamlit run backend/dashboard/app.py

# Expected:
# You can now view your Streamlit app in your browser
# Local URL: http://localhost:8501
```

---

## 🎬 COMPLETE WORKFLOW (Step-by-Step)

### **Step 1: Open 4 Terminals**
```
Terminal 1: [API Server]
Terminal 2: [Styled Dashboard]
Terminal 3: [Prediction Dashboard]
Terminal 4: [Optional: Training]
```

### **Step 2: Terminal 1 - Start API**
```bash
cd C:\Users\avani\terrainai
.venv\Scripts\activate
python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --reload
```
✅ Wait for: `Application startup complete`

### **Step 3: Terminal 2 - Start Main Dashboard**
```bash
cd C:\Users\avani\terrainai
.venv\Scripts\activate
streamlit run backend/dashboard/app.py
```
✅ Browser opens at `http://localhost:8501`

### **Step 4: Terminal 3 - Start Prediction Dashboard** (Optional)
```bash
cd C:\Users\avani\terrainai
.venv\Scripts\activate
streamlit run src/app.py
```
✅ Browser opens at `http://localhost:8502`

### **Step 5: Terminal 4 - Retrain** (Optional, takes ~1 hour)
```bash
cd C:\Users\avani\terrainai
.venv\Scripts\activate
python src/train.py --root . --run_id run_main
```
✅ Logs appear in `logs/epoch_metrics.jsonl` in real-time

### **Step 6: Explore**
- Dashboard 1 (8501): See training progress
- Dashboard 2 (8502): Upload test images for predictions
- API (8000/docs): Interactive API documentation

---

## 🔥 TROUBLESHOOTING

### **Issue: Port 8000 already in use**
```bash
# Kill the process using port 8000
netstat -ano | findstr :8000
# Note the PID, then:
taskkill /PID <PID> /F

# Or use different port:
python -m uvicorn backend.api.server:app --port 8001
```

### **Issue: Streamlit port conflicts**
```bash
# Streamlit uses 8501 by default
# If taken, specify a different port:
streamlit run backend/dashboard/app.py --server.port 8503
```

### **Issue: "ModuleNotFoundError: backend"**
```bash
# Make sure you're in the project root:
cd C:\Users\avani\terrainai
# And activate venv first
.venv\Scripts\activate
```

### **Issue: "Qdrant connection refused"**
```bash
# Qdrant uses LOCAL file storage now (backend/qdrant_data/)
# No Docker needed! It should work automatically
# If not, run this to initialize:
python -c "from backend.vector.qdrant_store import TerrainVectorStore; TerrainVectorStore()"
```

### **Issue: "requests module not found"**
```bash
pip install requests
```

---

## 📋 QUICK REFERENCE CARD

```bash
# ACTIVATE ENVIRONMENT (always first)
.venv\Scripts\activate

# START API
python -m uvicorn backend.api.server:app --port 8000

# START DASHBOARDS
streamlit run backend/dashboard/app.py              # Port 8501
streamlit run src/app.py                             # Port 8502

# TEST API
curl http://localhost:8000/health
curl http://localhost:8000/epochs

# RETRAIN MODEL
python src/train.py --root . --run_id run_main

# DEBUG MODE (fast)
python src/train.py --debug

# CHECK LOGS
type logs\epoch_metrics.jsonl
type logs\results.csv

# INGEST HARD EXAMPLES
cd backend && python ingest.py

# TEST QDRANT
cd backend/vector && python test_qdrant.py
```

---

## 🎯 EXPECTED OUTPUT

### **After API starts (Terminal 1):**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### **After Dashboard starts (Terminal 2):**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### **After Dashboard starts (Terminal 3):**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8502
Network URL: http://192.168.x.x:8502
```

### **During Training (Terminal 4):**
```
Using device: cuda
[Epoch 1/20] Mining hard examples from validation set...
[QdrantMiner] Upserted 30 hard example points (threshold IoU < 0.2, mean_hard_iou=0.0)
Epoch 1/20: mIoU = 0.2661  train_loss=0.6676  val_loss=0.3984
  Trees (Class 0): IoU = 0.3712
  Lush Bushes (Class 1): IoU = 0.3345
  ...
  Saved best model with mIoU 0.2661
```

---

## 🚀 SUMMARY

### **To run EVERYTHING:**

```bash
# Terminal 1
.venv\Scripts\activate
python -m uvicorn backend.api.server:app --port 8000

# Terminal 2
.venv\Scripts\activate
streamlit run backend/dashboard/app.py

# Terminal 3 (Optional)
.venv\Scripts\activate
streamlit run src/app.py

# Terminal 4 (Optional - for retraining)
.venv\Scripts\activate
python src/train.py --root . --run_id run_main
```

### **Then open in browser:**
- API: http://localhost:8000
- Dashboard 1: http://localhost:8501 ← **Main**
- Dashboard 2: http://localhost:8502 ← **Predictions**

---

## 📁 FILE STRUCTURE REMINDER

```
terrainai/
├── backend/
│   ├── api/server.py              ← API endpoints
│   ├── dashboard/app.py            ← Main dashboard
│   ├── vector/qdrant_store.py     ← Vector DB
│   └── qdrant_data/               ← Embeddings storage
├── src/
│   ├── train.py                   ← Training script
│   ├── app.py                     ← Prediction dashboard
│   └── model.py, dataset.py, ...
├── logs/
│   ├── epoch_metrics.jsonl        ← Training logs
│   └── results.csv                ← Metrics table
└── checkpoints/run_best.pth       ← Model weights
```

---

**Questions? Check PROJECT_ARCHITECTURE.md for detailed explanations!**
