"""
TerrainAI — Streamlit Dashboard
Tabs:
  1. Live Prediction   — upload image/video, full tactical analysis
  2. Training Metrics   — mIoU and rare-class IoU from logs/results.csv
  3. Results Summary    — run table + results.md
"""

import base64
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from pathlib import Path
from PIL import Image

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TerrainAI Dashboard",
    page_icon="🏜️",
    layout="wide",
)

# ─── API endpoints ───────────────────────────────────────────────────────────
API_BASE          = 'http://localhost:8000'
ANALYZE_IMAGE_URL = f'{API_BASE}/analyze/image'
ANALYZE_VIDEO_URL = f'{API_BASE}/analyze/video'
REPORT_URL        = f'{API_BASE}/report'
HEALTH_URL        = f'{API_BASE}/health'

# ─── Constants ───────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "Sand", 1: "Gravel", 2: "Rocks", 3: "Dirt", 4: "Grass",
    5: "Trees", 6: "Water", 7: "Sky", 8: "Logs", 9: "Flowers",
}

ABSENT_CLASS_IDS = frozenset({5, 6})

CLASS_COLORS = np.array([
    [210, 180, 120],   # 0  Sand
    [160, 155, 140],   # 1  Gravel
    [140, 100,  70],   # 2  Rocks
    [101,  67,  33],   # 3  Dirt
    [ 30, 160,  30],   # 4  Grass
    [ 34, 110,  34],   # 5  Trees  (absent)
    [  0, 100, 200],   # 6  Water  (absent)
    [135, 206, 235],   # 7  Sky
    [160, 100,  20],   # 8  Logs   ★ rare
    [255, 100, 200],   # 9  Flowers ★ rare
], dtype=np.uint8)

# Zone traversability color thresholds
ZONE_HIGH  = 60
ZONE_LOW   = 30

ROOT      = Path(__file__).parent.parent
LOGS_CSV  = ROOT / "logs" / "results.csv"
RESULTS_MD = ROOT / "results.md"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def colorize_mask(mask_array: np.ndarray) -> np.ndarray:
    h, w = mask_array.shape[:2]
    return CLASS_COLORS[mask_array.flatten()].reshape(h, w, 3)


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


def zone_color(score: float) -> str:
    if score >= ZONE_HIGH:
        return "#10b981"   # green
    if score >= ZONE_LOW:
        return "#f59e0b"   # amber
    return "#ef4444"       # red


def action_color(action: str) -> str:
    a = action.upper()
    if "ADVANCE" in a:
        return "#10b981"
    if "CAUTION" in a:
        return "#f59e0b"
    return "#ef4444"


def threat_color(level: str) -> str:
    if level == "LOW":
        return "#10b981"
    if level == "MEDIUM":
        return "#f59e0b"
    return "#ef4444"


def build_legend() -> None:
    st.sidebar.markdown("### 🎨 Class Legend")
    for idx, name in CLASS_NAMES.items():
        r, g, b = CLASS_COLORS[idx]
        swatch = (
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:rgb({r},{g},{b});border:1px solid #555;'
            f'margin-right:6px;vertical-align:middle;border-radius:2px;"></span>'
        )
        if idx in ABSENT_CLASS_IDS:
            st.sidebar.markdown(
                f'<span style="opacity:0.45;">{swatch}'
                f'<s>{idx} — {name}</s>'
                f' <em style="font-size:0.75em;">(not in training data)</em></span>',
                unsafe_allow_html=True,
            )
        else:
            rare = " ⭐" if idx in (8, 9) else ""
            st.sidebar.markdown(f"{swatch} **{idx}** — {name}{rare}", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🏜️ TerrainAI")
st.sidebar.caption("Synthetic-to-Real Terrain Segmentation")

# Health check
try:
    health = requests.get(HEALTH_URL, timeout=2).json()
    if health.get('model_loaded'):
        st.sidebar.success(f"API ONLINE — {health.get('device', '?').upper()}")
    else:
        st.sidebar.warning("API: model not loaded")
except Exception:
    st.sidebar.error("API OFFLINE — start server first")

build_legend()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Live Prediction", "📈 Training Metrics", "📋 Results Summary"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Live Terrain Intelligence")
    st.markdown("Upload a terrain image or video for full tactical analysis.")

    uploaded = st.file_uploader(
        "Upload terrain image or video",
        type=["png", "jpg", "jpeg", "mp4"],
    )

    if uploaded is not None:
        is_video = uploaded.name.lower().endswith(".mp4")

        if not is_video:
            # ── IMAGE FLOW ────────────────────────────────────────────────
            col_orig, col_overlay = st.columns(2)

            with col_orig:
                st.subheader("Original")
                original_pil = Image.open(uploaded).convert("RGB")
                st.image(original_pil, use_container_width=True)

            with st.spinner("Analysing terrain..."):
                try:
                    uploaded.seek(0)
                    resp = requests.post(
                        ANALYZE_IMAGE_URL,
                        files={"file": (uploaded.name, uploaded.getvalue(), "image/png")},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    session_id   = data['session_id']
                    terrain_stats = data['terrain_stats']
                    analysis     = data['analysis']
                    traversability = analysis['traversability']
                    threat         = analysis['threat']
                    recommendation = analysis['recommendation']

                    with col_overlay:
                        st.subheader("Terrain Overlay")
                        overlay_pil = b64_to_pil(data['overlay_b64'])
                        st.image(overlay_pil, use_container_width=True)

                    # ── Tactical summary ──────────────────────────────────
                    st.markdown("---")
                    m1, m2, m3, m4 = st.columns(4)
                    score  = traversability['score']
                    rating = traversability['rating']
                    s_color = "#10b981" if score >= 70 else ("#f59e0b" if score >= 40 else "#ef4444")
                    m1.markdown(
                        f'<div style="text-align:center;">'
                        f'<span style="font-size:2.2rem;font-weight:bold;color:{s_color};">{score:.1f}</span>'
                        f'<br><small>Traversability / 100</small></div>',
                        unsafe_allow_html=True,
                    )
                    m2.markdown(
                        f'<div style="text-align:center;">'
                        f'<span style="font-size:1.4rem;font-weight:bold;color:{s_color};">{rating}</span>'
                        f'<br><small>Rating</small></div>',
                        unsafe_allow_html=True,
                    )
                    t_col = threat_color(threat['threat_level'])
                    m3.markdown(
                        f'<div style="text-align:center;">'
                        f'<span style="font-size:1.4rem;font-weight:bold;color:{t_col};">{threat["threat_level"]}</span>'
                        f'<br><small>Threat Level</small></div>',
                        unsafe_allow_html=True,
                    )
                    action = recommendation['primary_action']
                    a_col = action_color(action)
                    m4.markdown(
                        f'<div style="text-align:center;">'
                        f'<span style="font-size:1.1rem;font-weight:bold;color:{a_col};">{action}</span>'
                        f'<br><small>Primary Action</small></div>',
                        unsafe_allow_html=True,
                    )

                    # Reasoning
                    st.info(recommendation.get('reasoning', ''))

                    # ── Terrain Composition ───────────────────────────────
                    st.markdown("---")
                    st.subheader("Terrain Composition")

                    per_class = terrain_stats.get('per_class', {})
                    comp_rows = []
                    for cls_id_str, cls_data in sorted(per_class.items(), key=lambda x: int(x[0])):
                        cls_id = int(cls_id_str)
                        if cls_id in ABSENT_CLASS_IDS:
                            continue
                        if not cls_data.get('present_in_dataset', True):
                            continue
                        pct = cls_data.get('percentage', 0)
                        if pct <= 0:
                            continue
                        comp_rows.append({
                            "Class": f"{cls_id} — {cls_data['name']}",
                            "Coverage %": round(pct, 2),
                        })

                    if comp_rows:
                        comp_df = pd.DataFrame(comp_rows).set_index("Class")
                        st.bar_chart(comp_df["Coverage %"])

                    st.caption(
                        "ℹ️ **Trees** (class 5) and **Water** (class 6) are excluded — "
                        "zero pixels in the FalconCloud training dataset."
                    )

                    # ── Zone map 3×3 ──────────────────────────────────────
                    st.markdown("---")
                    st.subheader("Zone Map (3×3 Grid)")
                    zones = data.get('analysis', {}).get('recommendation', {})
                    # zone_map comes from the server's terrain analysis stored in session
                    # Re-fetch per-zone data from the traversability breakdown if available
                    # The /analyze/image response doesn't include raw zone_map; use safe/avoid zones
                    safe_coords  = {(z['row'], z['col']) for z in recommendation.get('safe_zones', [])}
                    avoid_coords = {(z['row'], z['col']) for z in recommendation.get('avoid_zones', [])}

                    grid_cols = st.columns(3)
                    for row in range(3):
                        for col in range(3):
                            coord = (row, col)
                            if coord in safe_coords:
                                bg, label = "#d1fae5", "SAFE"
                                txt = "#065f46"
                            elif coord in avoid_coords:
                                bg, label = "#fee2e2", "AVOID"
                                txt = "#991b1b"
                            else:
                                bg, label = "#fef9c3", "MED"
                                txt = "#78350f"
                            grid_cols[col].markdown(
                                f'<div style="background:{bg};border-radius:6px;padding:10px 4px;'
                                f'text-align:center;margin:2px;">'
                                f'<span style="font-weight:bold;color:{txt};font-size:0.8rem;">'
                                f'R{row}C{col}<br>{label}</span></div>',
                                unsafe_allow_html=True,
                            )

                    # ── Active alerts ────────────────────────────────────
                    active_alerts = threat.get('alerts', [])
                    if active_alerts:
                        st.markdown("---")
                        st.subheader("Active Alerts")
                        for alert in active_alerts:
                            st.error(
                                f"**{alert['name']}** — {alert['percentage']:.2f}% coverage  \n"
                                f"{alert.get('message', '')}"
                            )

                    # ── Export PDF ────────────────────────────────────────
                    st.markdown("---")
                    if st.button("Export Tactical Report (PDF)"):
                        with st.spinner("Generating PDF..."):
                            try:
                                pdf_resp = requests.get(
                                    f'{REPORT_URL}/{session_id}', timeout=30
                                )
                                pdf_resp.raise_for_status()
                                st.download_button(
                                    label="Download PDF",
                                    data=pdf_resp.content,
                                    file_name=f"terrainai_report_{session_id}.pdf",
                                    mime="application/pdf",
                                )
                            except Exception as e:
                                st.error(f"Report generation failed: {e}")

                except requests.exceptions.ConnectionError:
                    with col_overlay:
                        st.error(
                            "Cannot connect to API at `localhost:8000`.\n\n"
                            "```bash\nuvicorn src.server:app --host 0.0.0.0 --port 8000\n```"
                        )
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        else:
            # ── VIDEO FLOW ────────────────────────────────────────────────
            st.subheader("Video Analysis")
            with st.spinner("Processing video (this may take a minute)..."):
                try:
                    uploaded.seek(0)
                    resp = requests.post(
                        ANALYZE_VIDEO_URL,
                        files={"file": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    video_url = f'{API_BASE}{data["video_url"]}'
                    summary   = data.get('summary', {})
                    tactical  = data.get('tactical_summary', {})

                    st.video(video_url)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Frames Processed", summary.get('processed_frames', '—'))
                    c2.metric("Duration (s)",      round(summary.get('duration_seconds', 0), 1))
                    c3.metric("Alert Frames %",    f"{tactical.get('alert_frequency_pct', 0):.1f}%")

                    st.info(tactical.get('overall_recommendation', ''))

                    if summary.get('alert_timeline'):
                        st.subheader("Alert Timeline")
                        timeline_df = pd.DataFrame(summary['alert_timeline'])
                        st.dataframe(timeline_df, use_container_width=True)

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API at `localhost:8000`.")
                except Exception as e:
                    st.error(f"Video analysis error: {e}")
    else:
        st.info("👆 Upload an image or video to begin tactical analysis.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING METRICS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Training Metrics")

    if LOGS_CSV.exists():
        df = pd.read_csv(LOGS_CSV)

        st.subheader("mIoU Over Epochs")
        if "epoch" in df.columns and "miou" in df.columns:
            st.line_chart(df[["epoch", "miou"]].set_index("epoch"), use_container_width=True)
        else:
            st.warning("CSV missing 'epoch' or 'miou' columns.")

        st.markdown("---")

        st.subheader("⭐ Rare Class IoU — Logs (8) & Flowers (9)")
        rare_cols = [c for c in ["iou_class_8", "iou_class_9"] if c in df.columns]
        if rare_cols and "epoch" in df.columns:
            st.line_chart(df[["epoch"] + rare_cols].set_index("epoch"), use_container_width=True)
        else:
            st.warning("CSV missing 'iou_class_8' / 'iou_class_9' columns.")

        st.markdown("---")

        st.subheader("Per-Class IoU (Latest Epoch)")
        latest = df.iloc[-1]
        iou_rows = []
        for c in range(10):
            name = CLASS_NAMES[c]
            if c in ABSENT_CLASS_IDS:
                iou_rows.append({"Class": f"{c} — {name}", "IoU": "N/A (not in training data)"})
            else:
                col_name = f"iou_class_{c}"
                val = latest.get(col_name)
                iou_rows.append({
                    "Class": f"{c} — {name}",
                    "IoU": round(val, 4) if val is not None and pd.notna(val) else "N/A",
                })
        st.table(pd.DataFrame(iou_rows))

    else:
        st.warning(
            "📂 No training log found at `logs/results.csv`.\n\n"
            "Expected columns: `epoch, miou, iou_class_0, ..., iou_class_9`"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Results Summary")

    runs = pd.DataFrame({
        "Run":         [1, 2, 3, 4, 5],
        "Change":      [
            "Baseline CE only",
            "+ Class weights",
            "+ Augmentation",
            "+ DiceLoss + warmup",
            "+ Qdrant mining",
        ],
        "mIoU":        ["TBD", "TBD", "TBD", "TBD", "TBD"],
        "Class 8 IoU": ["TBD", "TBD", "TBD", "TBD", "TBD"],
        "Class 9 IoU": ["TBD", "TBD", "TBD", "TBD", "TBD"],
        "Delta":       ["—",   "TBD", "TBD", "TBD", "TBD"],
    })
    st.dataframe(runs, use_container_width=True, hide_index=True)

    st.markdown("---")

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
