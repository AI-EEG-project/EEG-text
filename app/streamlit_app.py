"""
streamlit_app.py — EEG-to-Text demo UI
Lupse Ioan Victor — Sapt. 14
Clean English interface, no emojis, dark theme.
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroText | EEG-to-Text",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000"

# ── Design tokens ─────────────────────────────────────────────────────────────
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:           #0f1315;
    --surface:      #1a1d1f;
    --surface-high: #232729;
    --primary:      #00d4e8;
    --primary-dim:  #00bcd0;
    --accent:       #d04fff;
    --on-surface:   #dde3e5;
    --on-muted:     #8fa3a6;
    --outline:      rgba(0,212,232,0.14);
    --outline-v:    #35484a;
    --success:      #2ecc71;
    --warning:      #f39c12;
    --error:        #e74c3c;
    --font-body:    'Inter', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--on-surface) !important;
    font-family: var(--font-body);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(26,29,31,0.97) !important;
    border-right: 1px solid var(--outline) !important;
}
[data-testid="stSidebar"] * { color: var(--on-surface) !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--outline-v) !important;
    color: var(--primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: all 0.18s ease !important;
    padding: 0.5rem 1rem !important;
}
.stButton > button:hover {
    background: rgba(0,212,232,0.07) !important;
    border-color: var(--primary) !important;
    box-shadow: 0 0 12px rgba(0,212,232,0.18) !important;
}
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: #001e22 !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 0 16px rgba(0,212,232,0.28) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 28px rgba(0,212,232,0.45) !important;
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary-dim), var(--primary)) !important;
    border-radius: 2px !important;
}
.stProgress > div > div {
    background: var(--surface-high) !important;
    border-radius: 2px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--outline) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--on-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    color: var(--primary) !important;
    font-size: 1.7rem !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--outline-v) !important;
    color: var(--on-surface) !important;
    border-radius: 6px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,212,232,0.28) !important;
    border-radius: 8px !important;
    background: rgba(0,212,232,0.02) !important;
}

/* Alert boxes */
.stAlert { border-radius: 6px !important; border-width: 1px !important; }
.stInfo    { background: rgba(0,212,232,0.05) !important; border-color: rgba(0,212,232,0.22) !important; }
.stSuccess { background: rgba(46,204,113,0.07) !important; border-color: rgba(46,204,113,0.28) !important; }
.stWarning { background: rgba(243,156,18,0.07) !important; border-color: rgba(243,156,18,0.28) !important; }
.stError   { background: rgba(231,76,60,0.07) !important; border-color: rgba(231,76,60,0.28) !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--primary) !important; }

/* Utility classes */
.mono-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--on-muted);
    margin-bottom: 0.4rem;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 6px;
}
.divider {
    height: 1px;
    background: var(--outline-v);
    margin: 0.75rem 0;
    opacity: 0.5;
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "mode":             "upload",   # "upload" | "test"
    "result":           None,
    "true_word":        None,
    "eeg_array":        None,
    "revealed":         False,
    "_last_example_id": None,
    "eval_result":      None,
    "selected_model":   "eegnet",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── HTTP helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_examples():
    try:
        r = requests.get(f"{BACKEND_URL}/examples", timeout=5)
        r.raise_for_status()
        return r.json()["examples"]
    except Exception:
        return []

@st.cache_data(ttl=30)
def fetch_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def call_predict_upload(file_bytes, filename, model):
    try:
        r = requests.post(
            f"{BACKEND_URL}/predict",
            files={"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")},
            params={"model": model},
            timeout=20,
        )
        if r.status_code == 422:
            st.error(f"Validation error: {r.json().get('detail', r.text)}")
            return None
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Start uvicorn on port 8000.")
        return None
    except Exception as exc:
        st.error(f"Error: {exc}")
        return None

def call_predict_example(example_id, model):
    try:
        r = requests.post(
            f"{BACKEND_URL}/predict/example/{example_id}",
            params={"model": model},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend.")
        return None
    except Exception as exc:
        st.error(f"Error: {exc}")
        return None

def call_evaluate(model):
    try:
        r = requests.get(f"{BACKEND_URL}/evaluate", params={"model": model}, timeout=90)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

# ── EEG waveform plot ─────────────────────────────────────────────────────────
def plot_eeg_signal(eeg, fs=500):
    pz_idx = min(62, eeg.shape[0] - 1)
    signal = eeg[pz_idx]
    n_t = signal.shape[0]
    t_ms = np.linspace(-200, -200 + n_t / fs * 1000, n_t)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor("#0f1315")
    ax.set_facecolor("#1a1d1f")

    ax.plot(t_ms, signal, color="#00d4e8", linewidth=0.85, alpha=0.9)
    ax.fill_between(t_ms, signal, 0, alpha=0.055, color="#00d4e8")

    ax.axvline(x=0,   color="#e74c3c", linestyle="--", linewidth=0.9, alpha=0.65, label="Onset (0 ms)")
    ax.axvline(x=300, color="#f39c12", linestyle=":",  linewidth=0.9, alpha=0.75, label="P300 (~300 ms)")
    ax.axvline(x=400, color="#d04fff", linestyle=":",  linewidth=0.9, alpha=0.75, label="N400 (~400 ms)")
    ax.fill_between(t_ms, signal, 0, where=(t_ms >= 250) & (t_ms <= 350), alpha=0.10, color="#f39c12")
    ax.fill_between(t_ms, signal, 0, where=(t_ms >= 350) & (t_ms <= 500), alpha=0.08, color="#d04fff")

    ax.set_xlabel("Time (ms)", fontsize=8, color="#8fa3a6", fontfamily="monospace")
    ax.set_ylabel("Amplitude (uV)", fontsize=8, color="#8fa3a6", fontfamily="monospace")
    ax.set_title("WAVEFORM — Channel Pz (ch 62)", fontsize=9,
                 color="#00d4e8", fontfamily="monospace", loc="left", pad=8)
    ax.tick_params(colors="#8fa3a6", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#35484a")
    ax.legend(fontsize=7, loc="upper right", facecolor="#1a1d1f",
              edgecolor="#35484a", labelcolor="#8fa3a6", framealpha=0.9)
    ax.grid(True, alpha=0.08, color="#35484a", linestyle="--")
    ax.set_xlim(t_ms[0], t_ms[-1])
    fig.tight_layout(pad=0.4)
    return fig

# ── HTML component helpers ────────────────────────────────────────────────────
def _kpi_color(v, hi=0.6, lo=0.2):
    if v >= hi: return "#2ecc71"
    if v >= lo: return "#f39c12"
    return "#e74c3c"

def html_status_bar(health):
    if health:
        mock    = health.get("using_mock", True)
        c       = "#f39c12" if mock else "#2ecc71"
        tag     = "MOCK" if mock else "REAL"
        ver     = health.get("model_version", "N/A")
        vocab   = health.get("vocab_size", "N/A")
        n_ch    = health.get("n_channels", "N/A")
        dot_bg  = c
    else:
        c = "#e74c3c"; tag = "OFFLINE"; ver = "N/A"; vocab = "N/A"; n_ch = "N/A"
        dot_bg = "#e74c3c"

    return f"""
    <div style="background:rgba(26,29,31,0.9); border:1px solid rgba(0,212,232,0.12);
                border-radius:8px; padding:0.75rem 1.25rem; margin-bottom:1rem;
                display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
      <div style="display:flex; gap:2rem; flex-wrap:wrap;">
        <div>
          <div class="mono-label">Model</div>
          <div style="font-family:var(--font-mono); font-size:13px; color:#00d4e8;">{ver}</div>
        </div>
        <div>
          <div class="mono-label">Vocabulary</div>
          <div style="font-family:var(--font-mono); font-size:13px; color:#00d4e8;">{vocab} words</div>
        </div>
        <div>
          <div class="mono-label">Channels</div>
          <div style="font-family:var(--font-mono); font-size:13px; color:#00d4e8;">{n_ch}</div>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="status-dot" style="background:{dot_bg}; box-shadow:0 0 6px {dot_bg};"></span>
        <span style="font-family:var(--font-mono); font-size:11px; font-weight:600;
                     letter-spacing:0.1em; text-transform:uppercase; color:{c};">
          BACKEND {tag}
        </span>
      </div>
    </div>
    """

def html_word_card(rank, word, score, pos_tag=""):
    pos_colors = {
        "NOUN": "#e74c3c", "VERB": "#3498db", "ADJ": "#2ecc71",
        "PROPN": "#9b59b6", "ADV": "#f39c12", "NUM": "#1abc9c",
        "ADP": "#95a5a6", "PRON": "#e67e22",
    }
    pos_color   = pos_colors.get(pos_tag, "#6b8285")
    bar_w       = int(score * 100)
    is_first    = rank == 1
    border      = "rgba(0,212,232,0.38)" if is_first else "rgba(53,72,74,0.45)"
    bg          = "rgba(0,212,232,0.05)" if is_first else "rgba(26,29,31,0.7)"
    glow        = "box-shadow:0 0 12px rgba(0,212,232,0.16);" if is_first else ""
    rank_label  = f"#{rank}"

    pos_badge = (
        f"<span style='font-family:JetBrains Mono,monospace;font-size:9px;"
        f"letter-spacing:0.1em;color:{pos_color};background:rgba(0,0,0,0.3);"
        f"border:1px solid {pos_color}38;padding:1px 5px;border-radius:3px;'>{pos_tag}</span>"
        if pos_tag else ""
    )

    word_color = "#00d4e8" if is_first else "#dde3e5"

    return f"""
    <div style="background:{bg}; border:1px solid {border}; border-radius:7px;
                padding:0.65rem 0.9rem; margin-bottom:0.45rem; {glow}">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
        <div style="display:flex; align-items:center; gap:7px;">
          <span style="font-family:JetBrains Mono,monospace; font-size:10px; color:#8fa3a6;">{rank_label}</span>
          <span style="font-family:Inter,sans-serif; font-size:14px; font-weight:600; color:{word_color};">{word}</span>
          {pos_badge}
        </div>
        <span style="font-family:JetBrains Mono,monospace; font-size:11px;
                     color:{'#00d4e8' if is_first else '#8fa3a6'}; font-weight:500;">{score:.4f}</span>
      </div>
      <div style="height:2px; background:#1a1d1f; border-radius:1px; overflow:hidden;">
        <div style="height:100%; width:{bar_w}%;
                    background:{'linear-gradient(90deg,#00bcd0,#00d4e8)' if is_first else 'rgba(0,212,232,0.35)'};
                    border-radius:1px;"></div>
      </div>
    </div>
    """

def html_result_card(true_word, top1_word, top5_words):
    correct  = top1_word == true_word
    in_top5  = true_word in top5_words
    rank     = top5_words.index(true_word) + 1 if in_top5 else None

    if correct:
        bg = "rgba(46,204,113,0.07)"; border = "rgba(46,204,113,0.35)"
        label = "Top-1 correct"
        msg   = f"True word: <strong style='color:#2ecc71;'>{true_word}</strong>"
    elif in_top5:
        bg = "rgba(243,156,18,0.07)"; border = "rgba(243,156,18,0.35)"
        label = f"Found at rank #{rank}"
        msg   = (f"True word: <strong style='color:#f39c12;'>{true_word}</strong> "
                 f"— rank #{rank} in top-5. Top-1 predicted: <em>{top1_word}</em>")
    else:
        bg = "rgba(231,76,60,0.07)"; border = "rgba(231,76,60,0.35)"
        label = "Not in top-5"
        msg   = (f"True word: <strong style='color:#e74c3c;'>{true_word}</strong> "
                 f"— not in top-5. Top-1 predicted: <em>{top1_word}</em>")

    return f"""
    <div style="background:{bg}; border:1px solid {border}; border-radius:7px;
                padding:0.85rem 1.1rem; margin-top:0.75rem;">
      <div style="font-family:JetBrains Mono,monospace; font-size:9px; letter-spacing:0.1em;
                  text-transform:uppercase; color:#8fa3a6; margin-bottom:5px;">{label}</div>
      <div style="font-family:Inter,sans-serif; font-size:13px; color:#dde3e5;">{msg}</div>
    </div>
    """

def html_sentence_card(sentence):
    return f"""
    <div style="background:rgba(0,212,232,0.03); border:1px solid rgba(0,212,232,0.17);
                border-radius:8px; padding:1rem 1.25rem; margin-top:0.5rem; position:relative; overflow:hidden;">
      <div style="position:absolute; top:0; left:0; right:0; height:1px;
                  background:linear-gradient(90deg,transparent,#00d4e8,transparent); opacity:0.5;"></div>
      <div class="mono-label" style="margin-bottom:6px;">Reconstructed sequence</div>
      <div style="font-family:Inter,sans-serif; font-size:15px; font-weight:500;
                  color:#dde3e5; line-height:1.6;">{sentence}</div>
    </div>
    """

def html_eval_kpi(top1, top5, cos, sem, n):
    return f"""
    <div style="background:rgba(26,29,31,0.92); border:1px solid rgba(0,212,232,0.12);
                border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.5rem;">
      <div class="mono-label" style="margin-bottom:8px;">n = {n} examples</div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
        {_kpi_tile("Top-1 Acc", f"{top1*100:.0f}%", _kpi_color(top1))}
        {_kpi_tile("Top-5 Acc", f"{top5*100:.0f}%", _kpi_color(top5))}
        {_kpi_tile("Avg Cosine", f"{cos:.3f}", _kpi_color(cos))}
        {_kpi_tile("Sem Sim", f"{sem:.3f}", _kpi_color(sem))}
      </div>
    </div>
    """

def _kpi_tile(label, value, color):
    return f"""
    <div style="background:rgba(0,212,232,0.03); border:1px solid rgba(0,212,232,0.09);
                border-radius:5px; padding:7px; text-align:center;">
      <div style="font-size:1.15rem; font-weight:700; color:{color};
                  font-family:JetBrains Mono,monospace;">{value}</div>
      <div style="font-family:JetBrains Mono,monospace; font-size:8px;
                  color:#8fa3a6; text-transform:uppercase; letter-spacing:0.08em;">{label}</div>
    </div>
    """

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Title
    st.markdown("""
    <div style="padding:0.4rem 0 0.9rem;">
      <div style="font-family:Inter,sans-serif; font-size:1.25rem; font-weight:700;
                  color:#00d4e8; letter-spacing:-0.01em;">NeuroText</div>
      <div style="font-family:JetBrains Mono,monospace; font-size:9px; letter-spacing:0.12em;
                  color:#8fa3a6; text-transform:uppercase; margin-top:2px;">EEG-to-Text Demo</div>
    </div>
    """, unsafe_allow_html=True)

    # Backend status pill
    health = fetch_health()
    if health:
        mock = health.get("using_mock", True)
        c    = "#f39c12" if mock else "#2ecc71"
        tag  = "MOCK" if mock else "REAL"
        st.markdown(f"""
        <div style="background:rgba(26,29,31,0.95); border:1px solid rgba(0,212,232,0.12);
                    border-radius:7px; padding:0.6rem 0.9rem; margin-bottom:0.9rem;">
          <div style="display:flex; align-items:center; gap:7px; margin-bottom:4px;">
            <span class="status-dot" style="background:{c}; box-shadow:0 0 5px {c};"></span>
            <span style="font-family:JetBrains Mono,monospace; font-size:10px;
                         color:{c}; font-weight:600; letter-spacing:0.1em;">BACKEND ACTIVE — {tag}</span>
          </div>
          <div style="font-family:JetBrains Mono,monospace; font-size:8.5px; color:#8fa3a6;">
            {health.get('model_version')} &nbsp;|&nbsp;
            Vocab: {health.get('vocab_size')} &nbsp;|&nbsp;
            CH: {health.get('n_channels')}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(231,76,60,0.07); border:1px solid rgba(231,76,60,0.28);
                    border-radius:7px; padding:0.6rem 0.9rem; margin-bottom:0.9rem;">
          <span style="font-family:JetBrains Mono,monospace; font-size:10px; color:#e74c3c;
                       letter-spacing:0.1em;">BACKEND OFFLINE</span>
        </div>
        """, unsafe_allow_html=True)
        st.code("uvicorn app.backend.main:app\n  --reload --port 8000", language="bash")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Model selector
    st.markdown('<div class="mono-label">Model</div>', unsafe_allow_html=True)
    model_options = {
        "EEGNet  (Laslo — MSE)":              "eegnet",
        "EEG-Conformer  (Magdas — InfoNCE)":  "conformer",
        "Pretrained fine-tuned  (Lupse)":     "pretrained",
    }
    selected_label = st.selectbox(
        "", list(model_options.keys()), label_visibility="collapsed"
    )
    st.session_state["selected_model"] = model_options[selected_label]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Mode buttons
    st.markdown('<div class="mono-label">Input Mode</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Upload EEG File", use_container_width=True):
            st.session_state["mode"]    = "upload"
            st.session_state["result"]  = None
            st.session_state["revealed"] = False
            st.rerun()
    with col_b:
        if st.button("Test Model", use_container_width=True):
            st.session_state["mode"]    = "test"
            st.session_state["result"]  = None
            st.session_state["revealed"] = False
            st.rerun()

    # Highlight active mode
    active_mode = st.session_state["mode"]
    st.markdown(f"""
    <div style="font-family:JetBrains Mono,monospace; font-size:9px; color:#8fa3a6;
                text-align:center; margin-top:4px; letter-spacing:0.08em;">
      Active: <span style="color:#00d4e8; font-weight:600;">
        {"UPLOAD EEG FILE" if active_mode == "upload" else "TEST MODEL"}
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Info block
    st.markdown("""
    <div style="font-family:JetBrains Mono,monospace; font-size:8.5px; color:#8fa3a6; line-height:1.9;">
      Dataset: ZuCo 1.0<br>
      Channels: 105 (500 Hz)<br>
      Vocabulary: 200 words<br>
      Retrieval: cosine similarity<br>
      Window: -200 to +800 ms
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("Reset Session", use_container_width=True):
        for k in ["result", "true_word", "eeg_array", "revealed",
                  "_last_example_id", "eval_result"]:
            st.session_state[k] = None if k != "revealed" else False
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
# Header
st.markdown("""
<div style="margin-bottom:1rem;">
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:3px;">
    <span style="font-family:Inter,sans-serif; font-size:1.6rem; font-weight:700;
                 color:#00d4e8; letter-spacing:-0.02em;">NeuroText</span>
    <span style="font-family:JetBrains Mono,monospace; font-size:10px; letter-spacing:0.1em;
                 color:#8fa3a6; text-transform:uppercase; background:rgba(0,212,232,0.07);
                 border:1px solid rgba(0,212,232,0.18); padding:2px 9px; border-radius:4px;">
      EEG-to-Text v1.0
    </span>
  </div>
  <p style="font-family:Inter,sans-serif; color:#8fa3a6; font-size:13px; margin:0;">
    Decode EEG signals into text — ZuCo 1.0 · 105 channels · 500 Hz · 200-word vocabulary
  </p>
</div>
""", unsafe_allow_html=True)

health = fetch_health()
st.markdown(html_status_bar(health), unsafe_allow_html=True)

# Accent line
st.markdown("""
<div style="height:1px; background:linear-gradient(90deg,transparent,#00d4e8,#d04fff,transparent);
            opacity:0.5; margin-bottom:1.5rem; border-radius:1px;"></div>
""", unsafe_allow_html=True)

model_key = st.session_state["selected_model"]
mode      = st.session_state["mode"]

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — UPLOAD EEG FILE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "upload":
    st.markdown('<div class="mono-label">Upload EEG Epoch (.npy)</div>', unsafe_allow_html=True)
    st.caption(
        "Upload a single-word EEG epoch as a .npy file — shape (105, T), "
        "T in [50, 700] samples, values in microvolts."
    )

    # How-to note for users who have full .mat subject files
    with st.expander("How to extract a single-word epoch from a ZuCo .mat file"):
        st.code(
            "# Extract one epoch per word from a subject file:\n"
            "python mat_to_npy.py data/zuco1/task1-SR/resultsZAB_SR.mat \\\n"
            "    --word film --word story --word war --out npy_output/\n\n"
            "# Then upload the resulting .npy file (e.g. npy_output/film.npy)",
            language="bash",
        )
        st.caption(
            "Full subject .mat files are typically 200MB–1GB and cannot be uploaded directly. "
            "Use mat_to_npy.py to extract individual word epochs first."
        )

    uploaded = st.file_uploader(
        "Drag and drop .npy file here",
        type=["npy"],
        label_visibility="collapsed",
        help="NumPy array of shape (105, T), T in [50, 700] samples.",
    )

    eeg_bytes  = None
    file_label = ""

    if uploaded is not None:
        raw = uploaded.read()
        filename = uploaded.name.lower()

        if filename.endswith(".npy"):
            try:
                arr = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float32)
                st.success(f"Loaded: {uploaded.name} — shape {arr.shape}")
                eeg_bytes  = raw
                file_label = uploaded.name
                st.session_state["eeg_array"] = arr
                st.session_state["true_word"]  = None
                st.session_state["revealed"]   = False
            except Exception as exc:
                st.error(f"Cannot read file: {exc}")

    # EEG preview for .npy
    eeg_arr = st.session_state.get("eeg_array")
    if eeg_arr is not None:
        fig = plot_eeg_signal(eeg_arr)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    ready = eeg_bytes is not None

    if st.button("Run Prediction", disabled=not ready, use_container_width=True, type="primary"):
        with st.spinner("Processing EEG signal..."):
            result = call_predict_upload(eeg_bytes, file_label or "upload.mat", model_key)
        if result is not None:
            st.session_state["result"]   = result
            st.session_state["revealed"] = False
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — TEST MODEL (predefined examples)
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="mono-label" style="margin-bottom:0.5rem;">Test Model on Predefined Examples</div>', unsafe_allow_html=True)

    tab_single, tab_full = st.tabs(["Single Example", "Full Test Suite"])

    # ── Single example ────────────────────────────────────────────────────────
    with tab_single:
        examples = fetch_examples()
        if not examples:
            st.warning("Could not load examples. Check backend connection.")
        else:
            options = {f"#{ex['id']} — {ex['description']}": ex for ex in examples}
            selected_label = st.selectbox("Select example", list(options.keys()), label_visibility="visible")
            selected_ex    = options[selected_label]
            example_id     = selected_ex["id"]

            st.caption(f"Shape: {selected_ex['shape']}  |  File: {selected_ex['filename']}")

            # Load and preview
            try:
                import sys, os
                sys.path.insert(0, os.path.abspath("."))
                from app import config as _cfg
                _npy = _cfg.EXAMPLES_DIR / selected_ex["filename"]
                if _npy.exists():
                    _arr = np.load(str(_npy)).astype(np.float32)
                    fig  = plot_eeg_signal(_arr)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.session_state["eeg_array"] = _arr
            except Exception:
                pass

            if st.session_state["_last_example_id"] != example_id:
                st.session_state["revealed"] = False
                st.session_state["result"]   = None
                st.session_state["_last_example_id"] = example_id

            if st.button("Run Prediction", use_container_width=True, type="primary", key="btn_single"):
                with st.spinner("Running inference..."):
                    result = call_predict_example(example_id, model_key)
                if result is not None:
                    st.session_state["result"]   = result
                    st.session_state["true_word"] = selected_ex["true_word"]
                    st.session_state["revealed"]  = False
                    st.rerun()

    # ── Full test suite ───────────────────────────────────────────────────────
    with tab_full:
        st.markdown("""
        <p style="font-family:Inter,sans-serif; font-size:13px; color:#8fa3a6; margin-bottom:0.75rem;">
          Evaluates the selected model on all predefined examples and reports
          Top-1 accuracy, Top-5 accuracy, cosine similarity, and semantic similarity.
        </p>
        """, unsafe_allow_html=True)

        if st.button("Run Full Evaluation", use_container_width=True, type="primary", key="btn_eval"):
            with st.spinner("Evaluating all examples..."):
                st.session_state["eval_result"] = call_evaluate(model_key)

        ev = st.session_state.get("eval_result")
        if ev:
            top1 = ev.get("top1_accuracy", 0)
            top5 = ev.get("top5_accuracy", 0)
            cos  = ev.get("avg_cosine_similarity", 0)
            sem  = ev.get("avg_semantic_similarity", 0)
            n    = ev.get("n_examples", 0)

            st.markdown(html_eval_kpi(top1, top5, cos, sem, n), unsafe_allow_html=True)

            # Per-example table
            st.markdown('<div class="mono-label" style="margin-top:1rem;">Per-example results</div>', unsafe_allow_html=True)
            for row in ev.get("per_example", []):
                hit1 = row["top1_hit"]
                hit5 = row["top5_hit"]
                tag  = "TOP-1" if hit1 else ("TOP-5" if hit5 else "MISS")
                tag_color = "#2ecc71" if hit1 else ("#f39c12" if hit5 else "#e74c3c")
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:10px; padding:5px 0;
                            border-bottom:1px solid rgba(53,72,74,0.3); font-family:JetBrains Mono,monospace;
                            font-size:11px; color:#8fa3a6;">
                  <span style="color:{tag_color}; font-weight:600; width:44px;">{tag}</span>
                  <span style="color:#dde3e5; width:80px;">#{row['id']}</span>
                  <span style="width:110px;">true: <strong style="color:#dde3e5;">{row['true_word']}</strong></span>
                  <span style="width:110px;">pred: <strong style="color:{'#00d4e8' if hit1 else '#dde3e5'}">{row['pred_top1']}</strong></span>
                  <span>cos={row['cosine_score']:.3f} &nbsp; sem={row['semantic_sim']:.3f} &nbsp; {row['inference_ms']:.0f}ms</span>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS PANEL (shown for both modes after a prediction)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["result"] is not None:
    result    = st.session_state["result"]
    true_word = st.session_state["true_word"]
    top5      = result.get("top_5_words", [])

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="height:1px; background:linear-gradient(90deg,transparent,#00d4e8,#d04fff,transparent);
                opacity:0.45; margin-bottom:1.25rem; border-radius:1px;"></div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: top-5 word cards + metrics ─────────────────────────────────────
    with col_left:
        st.markdown('<div class="mono-label" style="margin-bottom:0.6rem;">Top-5 Candidates</div>', unsafe_allow_html=True)

        try:
            from app import config as _cfg
            pos_map = _cfg.VOCABULARY_POS
        except Exception:
            pos_map = {}

        words_html = ""
        for rank, ws in enumerate(top5, start=1):
            pos = pos_map.get(ws["word"], "")
            words_html += html_word_card(rank, ws["word"], ws["score"], pos)
        st.markdown(words_html, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Inference", f"{result['inference_time_ms']:.1f} ms")
        with mc2:
            st.metric("Model", result.get("model_version", "N/A"))

    # ── Right: reconstructed sequence + EEG + reveal ─────────────────────────
    with col_right:
        st.markdown(html_sentence_card(result.get("reconstructed_sentence", "")), unsafe_allow_html=True)

        eeg_arr = st.session_state.get("eeg_array")
        if eeg_arr is not None:
            st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="mono-label">EEG Signal — Channel Pz</div>', unsafe_allow_html=True)
            fig = plot_eeg_signal(eeg_arr)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Reveal true word (only for examples where we know the ground truth)
        if true_word is not None:
            st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
            top1_w  = top5[0]["word"]
            top5_ws = [ws["word"] for ws in top5]
            if not st.session_state["revealed"]:
                if st.button("Reveal True Word", use_container_width=True):
                    st.session_state["revealed"] = True
                    st.rerun()
            else:
                st.markdown(html_result_card(true_word, top1_w, top5_ws), unsafe_allow_html=True)
