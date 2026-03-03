import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from ultralytics import YOLO
import pickle
import json
import tempfile
import os
import time
from datetime import datetime, timedelta
from collections import deque
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartTraffic AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

* { font-family: 'Syne', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

/* Dark theme */
.stApp { background: #0a0a0f; color: #e8e8e8; }
section[data-testid="stSidebar"] { background: #0f0f1a; border-right: 1px solid #1e1e3a; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f0f1a 0%, #141428 100%);
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #00ff88; }
.metric-value { font-size: 2rem; font-weight: 800; color: #00ff88; }
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }

/* Congestion badge */
.badge-low    { background: #00ff8822; border: 1px solid #00ff88; color: #00ff88; padding: 4px 16px; border-radius: 20px; font-weight: 700; }
.badge-medium { background: #ffaa0022; border: 1px solid #ffaa00; color: #ffaa00; padding: 4px 16px; border-radius: 20px; font-weight: 700; }
.badge-high   { background: #ff444422; border: 1px solid #ff4444; color: #ff4444; padding: 4px 16px; border-radius: 20px; font-weight: 700; }

/* Section headers */
.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e3a;
}

/* Vehicle pill */
.vehicle-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #141428;
    border: 1px solid #1e1e3a;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 4px;
    font-size: 0.85rem;
}
.vehicle-count { font-weight: 800; color: #00ff88; font-size: 1.1rem; }

/* Alert box */
.alert-box {
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    font-weight: 600;
    font-size: 1rem;
}
.alert-low    { background: #00ff8811; border-left: 4px solid #00ff88; }
.alert-medium { background: #ffaa0011; border-left: 4px solid #ffaa00; }
.alert-high   { background: #ff444411; border-left: 4px solid #ff4444; }

/* Header */
.main-header {
    background: linear-gradient(90deg, #0f0f1a, #141428);
    border-bottom: 1px solid #1e1e3a;
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

/* Hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
PCU_VALUES = {
    'auto rickshaw': 1.2,
    'bicycle':       0.5,
    'bus':           3.0,
    'car':           1.0,
    'motorbike':     0.5,
    'truck':         3.0,
    'van':           1.5,
    'person':        0.0,
}

VEHICLE_EMOJI = {
    'auto rickshaw': '🛺',
    'bicycle':       '🚲',
    'bus':           '🚌',
    'car':           '🚗',
    'motorbike':     '🏍️',
    'truck':         '🚛',
    'van':           '🚐',
    'person':        '🚶',
}
# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
HF_REPO = "yash2024/smarttraffic-ai"
MODELS_DIR = "models"

def download_models_from_hf():
    os.makedirs(MODELS_DIR, exist_ok=True)
    files = ["smarttraffic_best.pt", "best_lstm.keras", "scaler.pkl", "metadata.json"]
    all_present = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in files)
    if all_present:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        os.system("pip install huggingface_hub -q")
        from huggingface_hub import hf_hub_download
    import shutil
    for fname in files:
        dest = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(dest):
            with st.spinner(f"Downloading {fname} from HuggingFace..."):
                path = hf_hub_download(repo_id=HF_REPO, filename=fname, repo_type="model")
                shutil.copy(path, dest)

@st.cache_resource
def load_models():
    download_models_from_hf()
    yolo = YOLO(os.path.join(MODELS_DIR, "smarttraffic_best.pt"))
    lstm = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_lstm.keras"))
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return yolo, lstm, scaler, metadata

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_congestion(pcu):
    if pcu < 20:
        return "Low", "low", "🟢"
    elif pcu < 40:
        return "Moderate", "medium", "🟡"
    else:
        return "High", "high", "🔴"

def predict_future(lstm_model, scaler, metadata, pcu_history):
    FEATURE_COLS = metadata['feature_cols']
    SEQ_LEN      = metadata['sequence_length']

    if len(pcu_history) < SEQ_LEN:
        return None

    # Build feature row from recent PCU values
    rows = []
    now  = datetime.now()
    for i, pcu in enumerate(list(pcu_history)[-SEQ_LEN:]):
        t = now - timedelta(seconds=(SEQ_LEN - i))
        row = {col: 0.0 for col in FEATURE_COLS}
        row['total_pcu']    = pcu
        row['total_vehicles'] = pcu / 1.0  # approximate
        row['hour_sin']     = np.sin(2 * np.pi * t.hour / 24)
        row['hour_cos']     = np.cos(2 * np.pi * t.hour / 24)
        row['day_sin']      = np.sin(2 * np.pi * t.weekday() / 7)
        row['day_cos']      = np.cos(2 * np.pi * t.weekday() / 7)
        rows.append(row)

    df_seq = pd.DataFrame(rows, columns=FEATURE_COLS).fillna(0)

    try:
        scaled = scaler.transform(df_seq.values)
        X      = scaled[np.newaxis, :, :]
        pred   = lstm_model.predict(X, verbose=0)

        # Inverse transform
        dummy  = np.zeros((1, len(FEATURE_COLS)))
        idx    = FEATURE_COLS.index(metadata['target_col'])
        dummy[0, idx] = pred[0, 0]
        result = scaler.inverse_transform(dummy)
        return float(result[0, idx])
    except Exception:
        return None

def draw_boxes(frame, results, model_names):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = model_names[int(box.cls)]
            conf     = float(box.conf)
            color    = (0, 255, 136)  # #00ff88

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚦 SmartTraffic AI")
    st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)

    conf_threshold   = st.slider("Detection Confidence", 0.1, 0.9, 0.35, 0.05)
    sample_every_n   = st.slider("Sample Every N Frames", 1, 30, 5)
    show_boxes       = st.toggle("Show Bounding Boxes", True)

    st.markdown("<div class='section-title'>PCU Reference</div>", unsafe_allow_html=True)
    for v, pcu in PCU_VALUES.items():
        emoji = VEHICLE_EMOJI[v]
        st.markdown(f"`{emoji} {v:<15}` **{pcu}**")

    st.markdown("<div class='section-title'>Model Info</div>", unsafe_allow_html=True)
    st.markdown("""
    - **YOLO:** YOLOv8s · 21.5 MB
    - **LSTM:** BiLSTM · 3.8 MB
    - **Dataset:** IDD · 41,962 imgs
    - **Classes:** 8 vehicle types
    """)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
    <h1 style="font-size: 2rem; font-weight: 800; margin: 0; color: #fff;">
        🚦 SmartTraffic <span style="color: #00ff88;">AI</span>
    </h1>
    <p style="color: #555; margin: 0; font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase;">
        YOLOv8s + BiLSTM · Indian Traffic Intelligence System
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
with st.spinner("Loading AI models..."):
    try:
        yolo_model, lstm_model, scaler, metadata = load_models()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("Make sure `models/` folder has: `smarttraffic_best.pt`, `best_lstm.keras`, `scaler.pkl`, `metadata.json`")
        st.stop()

# ─────────────────────────────────────────────
# VIDEO UPLOAD
# ─────────────────────────────────────────────
st.markdown("<div class='section-title'>Upload Traffic Video</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"], label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style="border: 2px dashed #1e1e3a; border-radius: 12px; padding: 3rem; text-align: center; color: #444;">
        <div style="font-size: 3rem;">🎥</div>
        <div style="font-size: 1rem; margin-top: 1rem;">Upload a traffic video to begin analysis</div>
        <div style="font-size: 0.75rem; color: #333; margin-top: 0.5rem;">MP4 · AVI · MOV supported</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# PROCESS VIDEO
# ─────────────────────────────────────────────
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded.read())
tfile.flush()

cap         = cv2.VideoCapture(tfile.name)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS) or 30

st.markdown(f"<div class='section-title'>Processing · {total_frames} frames · {fps:.0f} FPS</div>", unsafe_allow_html=True)

# Layout
col_video, col_stats = st.columns([3, 2])

with col_video:
    video_placeholder = st.empty()

with col_stats:
    metric_placeholder = st.empty()
    alert_placeholder  = st.empty()
    chart_placeholder  = st.empty()

progress_bar = st.progress(0)
status_text  = st.empty()

# State
frame_count  = 0
pcu_history  = deque(maxlen=metadata['sequence_length'])
pcu_timeline = []
time_labels  = []
all_records  = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    progress = min(frame_count / max(total_frames, 1), 1.0)
    progress_bar.progress(progress)
    status_text.markdown(f"`Frame {frame_count}/{total_frames}`")

    if frame_count % sample_every_n != 0:
        continue

    # YOLO Detection
    results = yolo_model(frame, verbose=False, conf=conf_threshold)

    counts    = {name: 0 for name in yolo_model.names.values()}
    total_pcu = 0.0

    for r in results:
        for box in r.boxes:
            cls_name = yolo_model.names[int(box.cls)]
            counts[cls_name] = counts.get(cls_name, 0) + 1
            total_pcu += PCU_VALUES.get(cls_name, 1.0)

    total_pcu = round(total_pcu, 2)
    pcu_history.append(total_pcu)

    elapsed = frame_count / fps
    ts      = datetime.now() - timedelta(seconds=(total_frames - frame_count) / fps)
    pcu_timeline.append(total_pcu)
    time_labels.append(f"{int(elapsed)}s")

    # Draw boxes
    if show_boxes:
        frame = draw_boxes(frame, results, yolo_model.names)

    # Show frame (resize for display)
    display = cv2.resize(frame, (640, 360))
    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    video_placeholder.image(display, channels="RGB", use_container_width=True)

    # Congestion level
    level, badge_class, emoji = get_congestion(total_pcu)

    # LSTM Prediction
    predicted_pcu = predict_future(lstm_model, scaler, metadata, pcu_history)

    # Stats panel
    with metric_placeholder.container():
        c1, c2, c3 = st.columns(3)
        total_vehicles = sum(v for k, v in counts.items() if k != 'person')
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{total_vehicles}</div>
                <div class='metric-label'>Vehicles</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{total_pcu}</div>
                <div class='metric-label'>PCU Score</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            pred_str = f"{predicted_pcu:.1f}" if predicted_pcu else "—"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{pred_str}</div>
                <div class='metric-label'>Predicted</div>
            </div>""", unsafe_allow_html=True)

        # Vehicle breakdown
        st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
        pills = ""
        for cls, cnt in counts.items():
            if cnt > 0:
                emoji_v = VEHICLE_EMOJI.get(cls, "🚗")
                pills += f"<span class='vehicle-pill'>{emoji_v} {cls} <span class='vehicle-count'>{cnt}</span></span>"
        st.markdown(f"<div style='margin-top: 0.5rem;'>{pills}</div>", unsafe_allow_html=True)

    # Alert
    with alert_placeholder.container():
        alert_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}[badge_class]
        alert_msg   = {
            "low":    "Traffic is flowing smoothly.",
            "medium": "Moderate congestion detected.",
            "high":   "⚠️ High congestion! Alert triggered."
        }[badge_class]
        st.markdown(f"""
        <div class='alert-box alert-{badge_class}'>
            {alert_emoji} <strong>{level} Traffic</strong> — {alert_msg}
        </div>
        """, unsafe_allow_html=True)

    # PCU Chart
    if len(pcu_timeline) > 1:
        with chart_placeholder.container():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_labels[-50:],
                y=pcu_timeline[-50:],
                mode='lines',
                line=dict(color='#00ff88', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,255,136,0.08)',
                name='PCU'
            ))
            if predicted_pcu:
                fig.add_hline(
                    y=predicted_pcu,
                    line_dash="dash",
                    line_color="#ffaa00",
                    annotation_text=f"Predicted: {predicted_pcu:.1f}",
                    annotation_font_color="#ffaa00"
                )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888', size=10),
                margin=dict(l=0, r=0, t=20, b=0),
                height=180,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor='#1e1e3a'),
                showlegend=False,
                title=dict(text="PCU Timeline", font=dict(color='#555', size=11))
            )
            st.plotly_chart(fig, use_container_width=True)

cap.release()
time.sleep(0.5)  # Windows file lock release ke liye
try:
    os.unlink(tfile.name)
except Exception:
    pass  # Windows pe file busy ho toh ignore karo
progress_bar.progress(1.0)
status_text.markdown("`✅ Processing complete!`")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>Session Summary</div>", unsafe_allow_html=True)

if pcu_timeline:
    avg_pcu = np.mean(pcu_timeline)
    max_pcu = np.max(pcu_timeline)
    min_pcu = np.min(pcu_timeline)

    s1, s2, s3, s4 = st.columns(4)
    for col, val, label in [
        (s1, f"{avg_pcu:.1f}", "Avg PCU"),
        (s2, f"{max_pcu:.1f}", "Peak PCU"),
        (s3, f"{min_pcu:.1f}", "Min PCU"),
        (s4, str(len(pcu_timeline)), "Samples"),
    ]:
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    # Full timeline chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=pcu_timeline,
        mode='lines',
        line=dict(color='#00ff88', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0,255,136,0.06)',
        name='PCU'
    ))
    fig2.add_hline(y=20, line_dash="dot", line_color="#00ff88", annotation_text="Low threshold")
    fig2.add_hline(y=40, line_dash="dot", line_color="#ffaa00", annotation_text="High threshold")
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#888'),
        margin=dict(l=0, r=0, t=30, b=0),
        height=250,
        xaxis=dict(showgrid=False, title="Frame Sample"),
        yaxis=dict(showgrid=True, gridcolor='#1e1e3a', title="PCU Score"),
        showlegend=False,
        title=dict(text="Full Session PCU Timeline", font=dict(color='#aaa', size=13))
    )
    st.plotly_chart(fig2, use_container_width=True)
