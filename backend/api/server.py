from fastapi import FastAPI
from backend.vector.qdrant_store import TerrainVectorStore
from pathlib import Path
import json
from qdrant_client.models import Filter, FieldCondition, MatchValue

app = FastAPI(title="Training Debug API")

store = TerrainVectorStore()
LOG_PATH = Path("logs/epoch_metrics.jsonl")
RESULTS_PATH = Path("logs/results.csv")


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