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
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
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

# ─────────────────────────────────────────────
#  GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;700;800&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0a0c10;
    --surface:   #0f1117;
    --border:    #1e2230;
    --accent:    #00ff94;
    --accent2:   #00c8ff;
    --warn:      #ff6b35;
    --text:      #cdd6f4;
    --muted:     #6272a4;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Syne', sans-serif;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Top rule banner ── */
.top-banner {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 28px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
}
.top-banner .hex { font-size: 2.2rem; line-height: 1; color: var(--accent); }
.top-banner h1 {
    font-family: var(--sans);
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    color: #fff;
}
.top-banner .sub {
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 3px;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent);
    margin-left: auto;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
}

/* ── Section labels ── */
.section-label {
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    border-left: 2px solid var(--accent);
    padding-left: 10px;
    margin: 40px 0 20px;
    font-family: var(--mono);
}

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 32px;
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent) 0%, transparent 100%);
}
.kpi-card:hover { border-color: #2e3350; }
.kpi-label {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
    letter-spacing: -1px;
}
.kpi-value.accent   { color: var(--accent); }
.kpi-value.accent2  { color: var(--accent2); }
.kpi-value.warn     { color: var(--warn); }
.kpi-delta {
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 8px;
}

/* ── Chart container ── */
.chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 24px;
    margin-bottom: 14px;
}

/* ── Dataframe overrides ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}
[data-testid="stDataFrame"] thead tr th {
    background: #12141c !important;
    color: var(--muted) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 10px 14px !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: #1a1d28 !important;
}
[data-testid="stDataFrame"] tbody tr td {
    border-bottom: 1px solid var(--border) !important;
    color: var(--text) !important;
    padding: 9px 14px !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}
[data-testid="stSelectbox"] label, [data-testid="stFileUploader"] label, [data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: 0.6rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── Warning / info ── */
[data-testid="stAlert"] {
    background: #12141c !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--muted) !important;
    font-size: 0.75rem !important;
}

/* ── Metric badge (inline) ── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.65rem;
    font-family: var(--mono);
    letter-spacing: 0.08em;
    font-weight: 500;
}
.badge-green  { background: rgba(0,255,148,0.1); color: var(--accent);  border: 1px solid rgba(0,255,148,0.25); }
.badge-blue   { background: rgba(0,200,255,0.1); color: var(--accent2); border: 1px solid rgba(0,200,255,0.25); }
.badge-orange { background: rgba(255,107,53,0.1); color: var(--warn);   border: 1px solid rgba(255,107,53,0.25); }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 32px 0 !important; }

/* ── Streamlit line chart dark override ── */
[data-testid="stVegaLiteChart"] canvas { filter: saturate(1.3); }
</style>
''', unsafe_allow_html=True)



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
    st.sidebar.markdown('<div class="section-label">Class Legend</div>', unsafe_allow_html=True)
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
            rare = " *" if idx in (8, 9) else ""
            st.sidebar.markdown(f"{swatch} **{idx}** — {name}{rare}", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style=\"font-family:var(--sans);color:#fff;\">⬡ TerrainAI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<div class=\"sub\">Synthetic-to-Real Terrain Segmentation</div>", unsafe_allow_html=True)

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

# ─────────────────────────────────────────────
#  HEADER BANNER
# ─────────────────────────────────────────────
st.markdown('''
<div class="top-banner">
    <div class="hex">⬡</div>
    <div>
        <h1>TerrainAI Dashboard</h1>
        <div class="sub">Tactical Analysis · Training Metrics</div>
    </div>
    <div class="status-dot" title="Live"></div>
</div>
''', unsafe_allow_html=True)


# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Live Prediction", "Training Metrics", "Results Summary"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    

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
                st.markdown('<div class="section-label">Original</div>', unsafe_allow_html=True)
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
                        st.markdown('<div class="section-label">Terrain Overlay</div>', unsafe_allow_html=True)
                        overlay_pil = b64_to_pil(data['overlay_b64'])
                        st.image(overlay_pil, use_container_width=True)

                    # ── Tactical summary ──────────────────────────────────
                    score  = traversability['score']
                    
                    # Dynamically update related elements based on new score
                    if score >= 70:
                        rating = "Safe"
                        threat['threat_level'] = "LOW"
                        recommendation['primary_action'] = "ADVANCE"
                    elif score >= 40:
                        rating = "Caution"
                        threat['threat_level'] = "MEDIUM"
                        recommendation['primary_action'] = "PROCEED CAREFULLY"
                    else:
                        rating = "Danger"
                        threat['threat_level'] = "HIGH"
                        recommendation['primary_action'] = "STOP"
                        
                    s_color = "accent" if score >= 70 else ("warn" if score < 40 else "accent2")
                    t_col = "accent" if threat['threat_level'] == "LOW" else ("warn" if threat['threat_level'] == "HIGH" else "accent2")
                    action = recommendation['primary_action']
                    a_col = "accent" if "ADVANCE" in action.upper() else ("warn" if "STOP" in action.upper() else "accent2")
                    
                    st.markdown(f'''
                    <div class="kpi-grid">
                        <div class="kpi-card">
                            <div class="kpi-label">Traversability</div>
                            <div class="kpi-value {s_color}">{score:.1f}</div>
                            <div class="kpi-delta">/ 100 Score</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-label">Rating</div>
                            <div class="kpi-value {s_color}">{rating}</div>
                            <div class="kpi-delta">Status</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-label">Threat Level</div>
                            <div class="kpi-value {t_col}">{threat['threat_level']}</div>
                            <div class="kpi-delta">Current Assessment</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-label">Primary Action</div>
                            <div class="kpi-value {a_col}" style="font-size:1.4rem;">{action}</div>
                            <div class="kpi-delta">Recommendation</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                    # Reasoning
                    st.info(recommendation.get('reasoning', ''))

                    # ── Terrain Composition ───────────────────────────────
                    st.markdown("---")
                    st.markdown('<div class="section-label">Terrain Composition</div>', unsafe_allow_html=True)

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
                        "**Trees** (class 5) and **Water** (class 6) are excluded — "
                        "zero pixels in the FalconCloud training dataset."
                    )

                    # ── Zone map 3×3 ──────────────────────────────────────
                    st.markdown("---")
                    st.markdown('<div class="section-label">Zone Map (3×3 Grid)</div>', unsafe_allow_html=True)
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
                        st.markdown('<div class="section-label">Active Alerts</div>', unsafe_allow_html=True)
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
                    try:
                        detail = e.response.json().get('detail', str(e))
                    except Exception:
                        detail = str(e)
                    st.error(f"API error: {detail}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        else:
            # ── VIDEO FLOW ────────────────────────────────────────────────
            st.markdown('<div class="section-label">Video Analysis</div>', unsafe_allow_html=True)
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
                        st.markdown('<div class="section-label">Alert Timeline</div>', unsafe_allow_html=True)
                        timeline_df = pd.DataFrame(summary['alert_timeline'])
                        st.dataframe(timeline_df, use_container_width=True)

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API at `localhost:8000`.")
                except Exception as e:
                    st.error(f"Video analysis error: {e}")
    else:
        st.info("Upload an image or video to begin tactical analysis.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING METRICS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    

    if LOGS_CSV.exists():
        df = pd.read_csv(LOGS_CSV)

        st.markdown('<div class="section-label">mIoU Over Epochs</div>', unsafe_allow_html=True)
        if "epoch" in df.columns and "miou" in df.columns:
            st.line_chart(df[["epoch", "miou"]].set_index("epoch"), use_container_width=True)
        else:
            st.warning("CSV missing 'epoch' or 'miou' columns.")

        st.markdown("---")

        st.markdown('<div class="section-label">Rare Class IoU — Logs (8) & Flowers (9)</div>', unsafe_allow_html=True)
        rare_cols = [c for c in ["iou_class_8", "iou_class_9"] if c in df.columns]
        if rare_cols and "epoch" in df.columns:
            st.line_chart(df[["epoch"] + rare_cols].set_index("epoch"), use_container_width=True)
        else:
            st.warning("CSV missing 'iou_class_8' / 'iou_class_9' columns.")

        st.markdown("---")

        st.markdown('<div class="section-label">Per-Class IoU (Latest Epoch)</div>', unsafe_allow_html=True)
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
            "No training log found at `logs/results.csv`.\n\n"
            "Expected columns: `epoch, miou, iou_class_0, ..., iou_class_9`"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    

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
            st.markdown('<div class="section-label">results.md</div>', unsafe_allow_html=True)
            st.markdown(content)
        else:
            st.info("Training in progress... results.md will be updated as runs complete.")
    else:
        st.info("Training in progress... results.md not yet created.")

    st.markdown("---")
    st.caption("TerrainAI — Hackathon 2026 | SegFormer-B2 + Qdrant Hard Example Mining")
