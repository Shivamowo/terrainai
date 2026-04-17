"""
TerrainAI — Streamlit Dashboard
Tabs:
  1. Live Prediction   — upload an image, call FastAPI /predict, show original + colored mask
  2. Training Metrics   — plot mIoU and rare-class IoU from logs/results.csv
  3. Results Summary    — hardcoded run table + results.md contents
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from io import BytesIO
import json

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TerrainAI Dashboard",
    page_icon="🏜️",
    layout="wide",
)

# ─── Constants ──────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "Ground",
    1: "Sand",
    2: "Rock",
    3: "Vegetation",
    4: "Shrub",
    5: "Road",
    6: "Building",
    7: "Sky",
    8: "Logs",
    9: "Flowers",
}

# 10 distinct colors (RGB) — one per class
CLASS_COLORS = np.array([
    [60,  60,  60 ],   # 0  Ground     — dark gray
    [210, 180, 120],   # 1  Sand       — sandy tan
    [140, 100,  70],   # 2  Rock       — brown
    [30,  160,  30],   # 3  Vegetation — green
    [100, 200, 100],   # 4  Shrub      — light green
    [80,  80,  80 ],   # 5  Road       — asphalt gray
    [200,  50,  50],   # 6  Building   — red
    [135, 206, 235],   # 7  Sky        — sky blue
    [160, 100,  20],   # 8  Logs       — dark brown/orange  ★ rare
    [255, 100, 200],   # 9  Flowers    — pink/magenta       ★ rare
], dtype=np.uint8)

API_URL = "http://localhost:8000/predict"
ROOT = Path(__file__).parent.parent
LOGS_CSV = ROOT / "logs" / "results.csv"
RESULTS_MD = ROOT / "results.md"


# ─── Helpers ────────────────────────────────────────────────────────────────

def colorize_mask(mask_array: np.ndarray) -> np.ndarray:
    """Convert a HxW class-index mask to an HxWx3 RGB image using CLASS_COLORS."""
    h, w = mask_array.shape[:2]
    color_mask = CLASS_COLORS[mask_array.flatten()].reshape(h, w, 3)
    return color_mask


def build_legend() -> None:
    """Render a small color legend in the sidebar."""
    st.sidebar.markdown("### 🎨 Class Legend")
    for idx, name in CLASS_NAMES.items():
        r, g, b = CLASS_COLORS[idx]
        swatch = f'<span style="display:inline-block;width:14px;height:14px;background:rgb({r},{g},{b});border:1px solid #555;margin-right:6px;vertical-align:middle;border-radius:2px;"></span>'
        rare = " ⭐" if idx in (8, 9) else ""
        st.sidebar.markdown(f"{swatch} **{idx}** — {name}{rare}", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🏜️ TerrainAI")
st.sidebar.caption("Synthetic-to-Real Terrain Segmentation")
build_legend()

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Live Prediction", "📈 Training Metrics", "📋 Results Summary"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Live Prediction")
    st.markdown("Upload a terrain image to get a segmentation prediction from the deployed model.")

    uploaded = st.file_uploader("Upload a terrain image (.png / .jpg)", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        # Show the uploaded image
        original = Image.open(uploaded).convert("RGB")

        col_orig, col_pred = st.columns(2)
        with col_orig:
            st.subheader("Original Image")
            st.image(original, use_container_width=True)

        # Call the FastAPI endpoint
        with st.spinner("Calling prediction API..."):
            try:
                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/png")}
                resp = requests.post(API_URL, files=files, timeout=30)
                resp.raise_for_status()

                pred_data = resp.json()

                # Expect the API to return {"mask": [[int, ...], ...], "shape": [H, W]}
                mask_array = np.array(pred_data["mask"], dtype=np.uint8)
                color_mask = colorize_mask(mask_array)

                with col_pred:
                    st.subheader("Predicted Mask")
                    st.image(color_mask, use_container_width=True)

                # Class distribution bar
                st.markdown("---")
                st.subheader("Class Distribution")
                unique, counts = np.unique(mask_array, return_counts=True)
                dist_df = pd.DataFrame({
                    "Class": [f"{c} — {CLASS_NAMES.get(c, '?')}" for c in unique],
                    "Pixels": counts,
                })
                st.bar_chart(dist_df.set_index("Class"))

            except requests.exceptions.ConnectionError:
                with col_pred:
                    st.error(
                        "⚠️ Cannot connect to prediction API at `localhost:8000`.\n\n"
                        "Make sure the FastAPI server is running:\n"
                        "```bash\nuvicorn src.api:app --host 0.0.0.0 --port 8000\n```"
                    )
            except requests.exceptions.HTTPError as e:
                with col_pred:
                    st.error(f"API returned error: {e}")
            except Exception as e:
                with col_pred:
                    st.error(f"Unexpected error: {e}")
    else:
        st.info("👆 Upload an image above to get started.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING METRICS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Training Metrics")

    if LOGS_CSV.exists():
        df = pd.read_csv(LOGS_CSV)

        # ── mIoU over epochs ──
        st.subheader("mIoU Over Epochs")
        if "epoch" in df.columns and "miou" in df.columns:
            chart_miou = df[["epoch", "miou"]].set_index("epoch")
            st.line_chart(chart_miou, use_container_width=True)
        else:
            st.warning("CSV missing 'epoch' or 'miou' columns.")

        st.markdown("---")

        # ── Rare-class IoU (Class 8 = Logs, Class 9 = Flowers) ──
        st.subheader("⭐ Rare Class IoU — Logs (8) & Flowers (9)")
        rare_cols = []
        if "iou_class_8" in df.columns:
            rare_cols.append("iou_class_8")
        if "iou_class_9" in df.columns:
            rare_cols.append("iou_class_9")

        if rare_cols and "epoch" in df.columns:
            chart_rare = df[["epoch"] + rare_cols].set_index("epoch")
            st.line_chart(chart_rare, use_container_width=True)
        else:
            st.warning("CSV missing 'iou_class_8' / 'iou_class_9' columns.")

        st.markdown("---")

        # ── Full per-class IoU table ──
        st.subheader("Per-Class IoU (Latest Epoch)")
        latest = df.iloc[-1]
        iou_data = {}
        for c in range(10):
            col_name = f"iou_class_{c}"
            if col_name in df.columns:
                val = latest[col_name]
                iou_data[f"{c} — {CLASS_NAMES[c]}"] = round(val, 4) if pd.notna(val) else "N/A"
        if iou_data:
            st.table(pd.DataFrame(iou_data.items(), columns=["Class", "IoU"]))

    else:
        st.warning(
            "📂 No training log found at `logs/results.csv`.\n\n"
            "Training metrics will appear here once a training run writes CSV output.\n\n"
            "Expected CSV columns: `epoch, miou, iou_class_0, ..., iou_class_9`"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Results Summary")

    # Hardcoded run table
    runs = pd.DataFrame({
        "Run":          [1, 2, 3, 4, 5],
        "Change":       [
            "Baseline CE only",
            "+ Class weights",
            "+ Augmentation",
            "+ DiceLoss + warmup",
            "+ Qdrant mining",
        ],
        "mIoU":         ["TBD", "TBD", "TBD", "TBD", "TBD"],
        "Class 8 IoU":  ["TBD", "TBD", "TBD", "TBD", "TBD"],
        "Class 9 IoU":  ["TBD", "TBD", "TBD", "TBD", "TBD"],
        "Delta":        ["—",   "TBD", "TBD", "TBD", "TBD"],
    })

    st.dataframe(runs, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Show results.md if it exists and has real data
    if RESULTS_MD.exists():
        content = RESULTS_MD.read_text(encoding="utf-8")
        if "TBD" not in content:
            st.subheader("📄 results.md")
            st.markdown(content)
        else:
            st.info("⏳ Training in progress... results.md will be updated as runs complete.")
    else:
        st.info("⏳ Training in progress... results.md not yet created.")

    st.markdown("---")
    st.caption("TerrainAI — Hackathon 2026 | SegFormer-B2 + Qdrant Hard Example Mining")
