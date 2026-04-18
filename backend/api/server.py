from fastapi import FastAPI, UploadFile, File
from backend.vector.qdrant_store import TerrainVectorStore
from pathlib import Path
import json
import base64
import io
import cv2
import numpy as np
import torch
from typing import Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Import inference functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.inference import (
    load_model, preprocess_frame, predict_frame,
    render_overlay, get_terrain_stats, get_zone_map, CLASSES, ABSENT_CLASS_IDS
)

app = FastAPI(title="TerrainAI API")

store = TerrainVectorStore()
LOG_PATH = Path("logs/epoch_metrics.jsonl")
RESULTS_PATH = Path("logs/results.csv")

# Load model globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_PATH = "checkpoints/ft_v2_best.pth"

def load_model_on_startup():
    """Load model on first request"""
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model(MODEL_PATH, DEVICE)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            MODEL = None


# ---------------- UTIL ----------------
def load_logs():
    """Load epoch metrics from JSONL file (one JSON object per line)"""
    if not LOG_PATH.exists():
        return []
    logs = []
    with open(LOG_PATH, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs


# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "qdrant_points": store.client.count(
            collection_name="terrain_patches",
            exact=True
        ).count
    }


# ---------------- ALL EPOCHS ----------------
@app.get("/epochs")
def get_epochs():
    return load_logs()


# ---------------- SINGLE EPOCH ----------------
@app.get("/epoch/{epoch_id}")
def get_epoch(epoch_id: int):
    logs = load_logs()
    for e in logs:
        if e["epoch"] == epoch_id:
            return e
    return {"error": "epoch not found"}


# ---------------- FAILURE EXAMPLES ----------------
@app.get("/failures/{class_id}")
def get_failures(class_id: int, limit: int = 10):

    results = store.client.search(
        collection_name="terrain_patches",
        query_vector=[1.0] * 512,
        limit=limit,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="class_id",
                    match=MatchValue(value=class_id)
                )
            ]
        ),
        with_payload=True
    )

    return [
        {
            "image_path": r.payload["image_path"],
            "iou": r.payload["iou"],
            "epoch": r.payload["epoch"],
            "class_id": r.payload["class_id"]
        }
        for r in results
    ]

# ---------------- VECTOR SEARCH DEMO ----------------
@app.get("/search")
def search():
    results = store.client.search(
        collection_name="terrain_patches",
        query_vector=[1.0] * 512,
        limit=5,
        with_payload=True
    )

    return [
        {
            "class_id": r.payload["class_id"],
            "iou": r.payload["iou"],
            "image_path": r.payload["image_path"]
        }
        for r in results
    ]


# ──────────────────── INFERENCE HELPERS ────────────────────

def bgr_to_b64(frame: np.ndarray) -> str:
    """Convert BGR numpy array to base64 PNG string (raw, no data URI prefix)"""
    success, buffer = cv2.imencode('.png', frame)
    if not success:
        raise ValueError("Failed to encode image")
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64


def bytes_to_b64(data: bytes) -> str:
    """Convert bytes to base64 data URI"""
    return f"data:application/octet-stream;base64,{base64.b64encode(data).decode('utf-8')}"


def generate_tactical_analysis(stats: dict, zones: list) -> dict:
    """Generate tactical analysis from terrain stats and zone map"""
    per_class = stats.get('per_class', {})
    total_traversable = stats.get('total_traversable_pct', 0)
    alerts = stats.get('active_alerts', [])
    
    # Compute traversability score (0-100)
    traversability_score = min(100, int(total_traversable))
    
    # Rating
    if traversability_score >= 70:
        rating = "EXCELLENT"
    elif traversability_score >= 50:
        rating = "GOOD"
    elif traversability_score >= 30:
        rating = "CAUTION"
    else:
        rating = "DANGER"
    
    # Threat level based on alerts and non-traversable terrain
    total_non_traversable = stats.get('total_non_traversable_pct', 0)
    if len(alerts) > 1 or total_non_traversable > 50:
        threat_level = "HIGH"
    elif len(alerts) > 0 or total_non_traversable > 30:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"
    
    # Generate alerts
    threat_alerts = []
    for alert in alerts:
        threat_alerts.append({
            'name': alert.get('name', 'Unknown'),
            'percentage': alert.get('percentage', 0),
            'message': f"Detected {alert.get('name', 'hazard')} at {alert.get('percentage', 0):.1f}% coverage"
        })
    
    # Zone safety assessment (simple: zones with >80% traversability are safe)
    safe_zones = []
    avoid_zones = []
    for row_zones in zones:
        for zone in row_zones:
            score = zone.get('traversability_score', 0)
            row = zone.get('zone_row', 0)
            col = zone.get('zone_col', 0)
            if score >= 80:
                safe_zones.append({'row': row, 'col': col, 'score': score})
            elif score < 30:
                avoid_zones.append({'row': row, 'col': col, 'score': score})
    
    # Primary action recommendation
    if traversability_score >= 70 and threat_level == "LOW":
        primary_action = "ADVANCE"
        reasoning = f"Terrain is highly traversable ({traversability_score}%) with low threat. Safe to proceed."
    elif traversability_score >= 50:
        primary_action = "PROCEED_CAUTION"
        reasoning = f"Moderate traversability ({traversability_score}%). Avoid zones marked AVOID and monitor alerts."
    else:
        primary_action = "HALT"
        reasoning = f"Low traversability ({traversability_score}%). Significant obstacles present. Recommend route deviation."
    
    return {
        'traversability': {
            'score': traversability_score,
            'rating': rating,
            'percentage': round(total_traversable, 2),
        },
        'threat': {
            'threat_level': threat_level,
            'alerts': threat_alerts,
            'hazard_coverage': round(total_non_traversable, 2),
        },
        'recommendation': {
            'primary_action': primary_action,
            'reasoning': reasoning,
            'safe_zones': safe_zones,
            'avoid_zones': avoid_zones,
        }
    }


# ──────────────────── IMAGE ANALYSIS ────────────────────

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze a single terrain image and return full tactical data"""
    load_model_on_startup()
    
    if MODEL is None:
        return {"error": "Model not loaded"}
    
    try:
        # Read image from upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid image file"}
        
        original_h, original_w = frame.shape[:2]
        
        # Run inference
        tensor = preprocess_frame(frame)
        mask = predict_frame(MODEL, tensor, DEVICE, original_h, original_w)
        overlay = render_overlay(frame, mask)
        stats = get_terrain_stats(mask)
        zones = get_zone_map(mask)
        
        # Generate tactical analysis
        analysis = generate_tactical_analysis(stats, zones)
        
        # Encode outputs as base64
        original_b64 = bgr_to_b64(frame)
        overlay_b64 = bgr_to_b64(overlay)
        
        # Generate session ID
        import uuid
        session_id = str(uuid.uuid4())
        
        return {
            "status": "success",
            "session_id": session_id,
            "original_b64": original_b64,
            "overlay_b64": overlay_b64,
            "terrain_stats": stats,
            "analysis": analysis,
            "model": MODEL_PATH.split('/')[-1],
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze a video frame-by-frame and return first frame analysis"""
    load_model_on_startup()
    
    if MODEL is None:
        return {"error": "Model not loaded"}
    
    try:
        # Save temp video file
        contents = await file.read()
        temp_path = Path(f"/tmp/{file.filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(contents)
        
        # Open video
        cap = cv2.VideoCapture(str(temp_path))
        ret, frame = cap.read()
        cap.release()
        temp_path.unlink()
        
        if not ret:
            return {"error": "Failed to read video"}
        
        original_h, original_w = frame.shape[:2]
        
        # Run inference on first frame
        tensor = preprocess_frame(frame)
        mask = predict_frame(MODEL, tensor, DEVICE, original_h, original_w)
        overlay = render_overlay(frame, mask)
        stats = get_terrain_stats(mask)
        zones = get_zone_map(mask)
        
        # Generate tactical analysis
        analysis = generate_tactical_analysis(stats, zones)
        
        # Encode outputs
        original_b64 = bgr_to_b64(frame)
        overlay_b64 = bgr_to_b64(overlay)
        
        import uuid
        session_id = str(uuid.uuid4())
        
        return {
            "status": "success",
            "session_id": session_id,
            "frame_number": 0,
            "original_b64": original_b64,
            "overlay_b64": overlay_b64,
            "terrain_stats": stats,
            "analysis": analysis,
            "model": MODEL_PATH.split('/')[-1],
        }
    
    except Exception as e:
        return {"error": f"Video analysis failed: {str(e)}"}


@app.get("/report")
def report():
    """Return training results summary"""
    try:
        results_path = Path("results.md")
        if results_path.exists():
            content = results_path.read_text()
            return {"report": content}
        return {"report": "No report available"}
    except Exception as e:
        return {"error": f"Failed to load report: {str(e)}"}