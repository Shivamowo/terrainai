"""
TerrainAI Tactical — FastAPI server.
Endpoints: /health, /predict, /analyze/image, /analyze/video, /report/{session_id}
"""

import sys
import uuid
import base64
import json
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import load_model, process_image, process_video, CLASSES
from src.tactical import analyze_frame, analyze_video_summary

# ─── Model global ────────────────────────────────────────────────────────────

MODEL = None
DEVICE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = Path(__file__).parent.parent / 'checkpoints' / 'run_best.pth'
    if ckpt.exists():
        MODEL = load_model(str(ckpt), DEVICE)
        print(f'Model loaded on {DEVICE}')
    else:
        print(f'WARNING: checkpoint not found at {ckpt}')
    yield


# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(title='TerrainAI Tactical API', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount('/outputs', StaticFiles(directory=str(OUTPUTS_DIR)), name='outputs')


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE),
        'checkpoint': 'run_best.pth',
        'miou': 0.6109,
        'class_8_iou': 0.6504,
        'class_9_iou': 0.9801,
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """Legacy endpoint — returns mask array compatible with original app.py."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    suffix = Path(file.filename).suffix or '.png'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        orig, overlay, mask, stats, zones = process_image(tmp_path, MODEL, DEVICE)
        return JSONResponse({
            'mask': mask.tolist(),
            'shape': list(mask.shape),
            'stats': stats,
        })
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post('/analyze/image')
async def analyze_image(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    suffix = Path(file.filename).suffix or '.png'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        orig, overlay, mask, stats, zones = process_image(tmp_path, MODEL, DEVICE)
        analysis = analyze_frame(stats, zones)

        _, buf = cv2.imencode('.png', overlay)
        overlay_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        _, buf2 = cv2.imencode('.png', orig)
        original_b64 = base64.b64encode(buf2.tobytes()).decode('utf-8')

        session_id = str(uuid.uuid4())
        session_data = {
            'session_id': session_id,
            'type': 'image',
            'filename': file.filename,
            'stats': stats,
            'analysis': analysis,
            'overlay_b64': overlay_b64,
            'original_b64': original_b64,
        }
        with open(OUTPUTS_DIR / f'{session_id}_data.json', 'w') as f:
            json.dump(session_data, f, default=str)

        legend = [
            {
                'class_id': cid,
                'name': info['name'],
                'color_hex': '#{:02x}{:02x}{:02x}'.format(*info['color'][::-1]),
                'traversable': info['traversable'],
                'present_in_dataset': info['present_in_dataset'],
            }
            for cid, info in CLASSES.items()
        ]

        return JSONResponse({
            'session_id': session_id,
            'overlay_b64': overlay_b64,
            'original_b64': original_b64,
            'terrain_stats': stats,
            'analysis': analysis,
            'class_legend': legend,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post('/analyze/video')
async def analyze_video_endpoint(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    session_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix or '.mp4'

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    output_video_path = OUTPUTS_DIR / f'{session_id}_output.mp4'

    try:
        summary = process_video(
            tmp_path,
            str(output_video_path),
            MODEL,
            DEVICE,
            frame_skip=3,
        )
        tactical_summary = analyze_video_summary(
            [summary['avg_terrain_stats']] if summary['avg_terrain_stats'] else []
        )

        session_data = {
            'session_id': session_id,
            'type': 'video',
            'filename': file.filename,
            'summary': summary,
            'tactical_summary': tactical_summary,
        }
        with open(OUTPUTS_DIR / f'{session_id}_data.json', 'w') as f:
            json.dump(session_data, f, default=str)

        return JSONResponse({
            'session_id': session_id,
            'video_url': f'/outputs/{session_id}_output.mp4',
            'summary': summary,
            'tactical_summary': tactical_summary,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get('/report/{session_id}')
def get_report(session_id: str):
    data_path = OUTPUTS_DIR / f'{session_id}_data.json'
    if not data_path.exists():
        raise HTTPException(status_code=404, detail='Session not found')

    with open(data_path) as f:
        session_data = json.load(f)

    from src.report import generate_pdf_report
    pdf_path = OUTPUTS_DIR / f'{session_id}_report.pdf'
    generate_pdf_report(session_data, str(pdf_path))

    return FileResponse(
        str(pdf_path),
        media_type='application/pdf',
        filename=f'terrainai_report_{session_id}.pdf',
    )
