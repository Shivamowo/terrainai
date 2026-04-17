from pathlib import Path
import json
import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Training Monitor",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
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
.top-banner .hex { font-size: 2.2rem; line-height: 1; }
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
[data-testid="stSelectbox"] label {
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
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROOT PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Go to project root
LOG_PATH  = BASE_DIR / "logs" / "epoch_metrics.jsonl"
HARD_PATH = BASE_DIR / "logs" / "hard_examples.jsonl"


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
def load_logs():
    if not LOG_PATH.exists():
        return []
    logs = []
    try:
        with open(LOG_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
    except Exception as e:
        print(f"Error loading logs: {e}")
    return logs

def load_hard():
    if not HARD_PATH.exists():
        return []
    hard = []
    try:
        with open(HARD_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    hard.append(json.loads(line))
    except Exception as e:
        print(f"Error loading hard examples: {e}")
    return hard

logs = load_logs()
hard = load_hard()


# ─────────────────────────────────────────────
#  HEADER BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div class="hex">⬡</div>
    <div>
        <h1>Training Monitor</h1>
        <div class="sub">Segmentation · Debug Console · v2.0</div>
    </div>
    <div class="status-dot" title="Live"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  COMPUTE SUMMARY METRICS
# ─────────────────────────────────────────────
if logs:
    df = pd.DataFrame([
        {
            "epoch": l["epoch"],
            "miou":  l["metrics"]["miou"],
            "loss":  l["training"]["loss"],
        }
        for l in logs
    ])
    best_miou   = df["miou"].max()
    last_miou   = df["miou"].iloc[-1]
    last_loss   = df["loss"].iloc[-1]
    total_epochs = len(df)
else:
    best_miou = last_miou = last_loss = total_epochs = None

hard_count  = len(hard)
worst_iou   = min(x["iou"] for x in hard) if hard else None
unique_cls  = len(set(x["class_id"] for x in hard)) if hard else 0


# ─────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────
def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "—"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Best mIoU</div>
        <div class="kpi-value accent">{fmt(best_miou)}</div>
        <div class="kpi-delta">peak across all epochs</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Latest Loss</div>
        <div class="kpi-value accent2">{fmt(last_loss)}</div>
        <div class="kpi-delta">epoch {total_epochs if total_epochs else '—'}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Hard Samples</div>
        <div class="kpi-value warn">{hard_count}</div>
        <div class="kpi-delta">{unique_cls} unique classes affected</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Worst IoU</div>
        <div class="kpi-value" style="color:#ff4560">{fmt(worst_iou)}</div>
        <div class="kpi-delta">minimum sample IoU observed</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TRAINING METRICS CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Training Metrics</div>', unsafe_allow_html=True)

if logs:
    col_chart, col_gap = st.columns([3, 1])
    with col_chart:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.line_chart(
            df.set_index("epoch")[["miou", "loss"]],
            color=["#00ff94", "#00c8ff"],
            height=280,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_gap:
        st.markdown('<br>', unsafe_allow_html=True)
        for _, row in df.tail(5).iloc[::-1].iterrows():
            trend = "▲" if row["miou"] >= df["miou"].mean() else "▼"
            color = "accent" if trend == "▲" else "warn"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        border-bottom:1px solid var(--border);
                        padding:8px 4px;font-size:0.72rem;color:var(--muted)">
                <span>ep {int(row['epoch'])}</span>
                <span class="badge badge-{'green' if trend=='▲' else 'orange'}">{trend} {row['miou']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("No training logs found at the configured path.")


# ─────────────────────────────────────────────
#  FAILURE ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Failure Analysis</div>', unsafe_allow_html=True)

if hard:
    df2 = pd.DataFrame(hard)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("""
        <div style="font-size:0.65rem;color:var(--muted);
                    letter-spacing:0.15em;text-transform:uppercase;
                    margin-bottom:10px">
            Bottom-10 by IoU score
        </div>""", unsafe_allow_html=True)

        worst = df2.sort_values("iou").head(10).copy()
        # Rank column for context
        worst.insert(0, "#", range(1, len(worst) + 1))

        st.dataframe(
            worst[["#", "image_path", "class_id", "iou", "epoch"]],
            use_container_width=True,
            hide_index=True,
        )

    with col_b:
        # Per-class hard-sample count
        class_counts = df2["class_id"].value_counts().reset_index()
        class_counts.columns = ["class_id", "count"]

        st.markdown("""
        <div style="font-size:0.65rem;color:var(--muted);
                    letter-spacing:0.15em;text-transform:uppercase;
                    margin-bottom:10px">
            Hard samples / class
        </div>""", unsafe_allow_html=True)
        st.bar_chart(
            class_counts.set_index("class_id"),
            color="#ff6b35",
            height=220,
        )
else:
    st.warning("No hard examples found at the configured path.")


# ─────────────────────────────────────────────
#  CLASS EXPLORER
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Class Explorer</div>', unsafe_allow_html=True)

if hard:
    class_ids = sorted(set(x["class_id"] for x in hard))

    col_sel, col_info = st.columns([1, 3])

    with col_sel:
        class_id = st.selectbox("Class ID", class_ids, label_visibility="visible")

    filtered = [x for x in hard if x["class_id"] == class_id]
    fdf = pd.DataFrame(filtered)

    with col_info:
        avg_iou = fdf["iou"].mean() if not fdf.empty else 0
        badge_color = "green" if avg_iou > 0.5 else "orange" if avg_iou > 0.25 else "orange"
        st.markdown(f"""
        <div style="display:flex;gap:12px;align-items:center;
                    padding-top:28px;flex-wrap:wrap;">
            <span class="badge badge-blue">class {class_id}</span>
            <span class="badge badge-{badge_color}">avg iou {avg_iou:.4f}</span>
            <span class="badge badge-blue">{len(filtered)} samples</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(fdf, use_container_width=True, hide_index=True)