"""
streamlit_app.py - UI demo EEG-to-Text
Lupse Ioan Victor - Sapt. 14
Design: NeuroText dark futuristic theme
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroText | EEG-to-Text",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8000"

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — NeuroText dark theme
# ══════════════════════════════════════════════════════════════════════════════
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
:root {
    --bg:           #101415;
    --surface:      #1d2022;
    --surface-low:  #191c1e;
    --surface-high: #272a2c;
    --primary:      #00f0ff;
    --primary-dim:  #00dbe9;
    --secondary:    #ff32d0;
    --on-surface:   #e0e3e5;
    --on-muted:     #b9cacb;
    --outline:      rgba(0,240,255,0.15);
    --outline-v:    #3b494b;
    --font-title:   'Sora', sans-serif;
    --font-body:    'Inter', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ── Base ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--on-surface) !important;
    font-family: var(--font-body);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(25,28,30,0.95) !important;
    border-right: 1px solid var(--outline) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: var(--on-surface) !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--outline) !important;
    color: var(--primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
    padding: 0.5rem 1.2rem !important;
}
.stButton > button:hover {
    background: rgba(0,240,255,0.08) !important;
    border-color: var(--primary) !important;
    box-shadow: 0 0 15px rgba(0,240,255,0.2) !important;
}
/* Primary button */
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: #002022 !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px rgba(0,240,255,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 35px rgba(0,240,255,0.5) !important;
    transform: scale(1.01);
}

/* ── Progress bars ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary-dim), var(--primary)) !important;
    border-radius: 2px !important;
}
.stProgress > div > div {
    background: var(--surface-high) !important;
    border-radius: 2px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--outline) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { font-family: var(--font-mono) !important; font-size: 10px !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: var(--on-muted) !important; }
[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; color: var(--primary) !important; font-size: 1.8rem !important; }

/* ── Selectbox / Radio ── */
.stSelectbox select, .stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--outline-v) !important;
    color: var(--on-surface) !important;
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
}
.stRadio > label { color: var(--on-muted) !important; font-family: var(--font-mono) !important; font-size: 11px !important; letter-spacing: 0.05em !important; }
.stRadio > div > label { color: var(--on-surface) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,240,255,0.3) !important;
    border-radius: 8px !important;
    background: rgba(0,240,255,0.02) !important;
}

/* ── Divider ── */
hr { border-color: var(--outline-v) !important; opacity: 0.4 !important; }

/* ── Info / success / warning / error boxes ── */
.stAlert { border-radius: 6px !important; border-width: 1px !important; }
.stInfo  { background: rgba(0,240,255,0.05) !important; border-color: rgba(0,240,255,0.25) !important; color: var(--primary) !important; }
.stSuccess { background: rgba(46,204,113,0.08) !important; border-color: rgba(46,204,113,0.3) !important; }
.stWarning { background: rgba(243,156,18,0.08) !important; border-color: rgba(243,156,18,0.3) !important; }
.stError   { background: rgba(231,76,60,0.08) !important; border-color: rgba(231,76,60,0.3) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--primary) !important; }

/* ── Matplotlib figure background ── */
.stImage img, .stPyplotChart { background: transparent !important; }

/* ── Custom classes ── */
.glass {
    background: rgba(29,32,34,0.85);
    backdrop-filter: blur(20px);
    border: 1px solid var(--outline);
    border-radius: 12px;
    padding: 1.5rem;
}
.label-mono {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--on-muted);
}
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--primary);
    animation: neural-pulse 2s infinite ease-in-out;
    box-shadow: 0 0 8px rgba(0,240,255,0.6);
}
@keyframes neural-pulse {
    0%,100% { opacity: 0.7; transform: scale(1); box-shadow: 0 0 8px rgba(0,240,255,0.4); }
    50%      { opacity: 1;   transform: scale(1.3); box-shadow: 0 0 18px rgba(0,240,255,0.8); }
}
.scan-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
    animation: scan 3s linear infinite;
    margin-bottom: 1.5rem;
    border-radius: 1px;
}
@keyframes scan { 0%{opacity:0.2} 50%{opacity:1} 100%{opacity:0.2} }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
for key, val in [("result", None), ("true_word", None),
                  ("eeg_array", None), ("revealed", False),
                  ("_last_example_id", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ══════════════════════════════════════════════════════════════════════════════
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

def call_predict_upload(file_bytes, filename):
    try:
        r = requests.post(
            f"{BACKEND_URL}/predict",
            files={"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")},
            timeout=15,
        )
        if r.status_code == 422:
            st.error(f"Eroare validare: {r.json().get('detail', r.text)}")
            return None
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Nu pot conecta la backend. Porneste uvicorn pe portul 8000.")
        return None
    except Exception as exc:
        st.error(f"Eroare: {exc}")
        return None

def call_evaluate():
    try:
        r = requests.get(f"{BACKEND_URL}/evaluate", timeout=60)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def call_predict_example(example_id):
    try:
        r = requests.post(f"{BACKEND_URL}/predict/example/{example_id}", timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Nu pot conecta la backend.")
        return None
    except Exception as exc:
        st.error(f"Eroare: {exc}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# EEG plot — dark theme NeuroText
# ══════════════════════════════════════════════════════════════════════════════
def plot_eeg_signal(eeg, fs=500):
    pz_idx = min(62, eeg.shape[0] - 1)
    signal = eeg[pz_idx]
    n_t = signal.shape[0]
    t_ms = np.linspace(-200, -200 + n_t / fs * 1000, n_t)

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#101415")
    ax.set_facecolor("#1d2022")

    ax.plot(t_ms, signal, color="#00f0ff", linewidth=0.9, alpha=0.9)
    ax.fill_between(t_ms, signal, 0, alpha=0.06, color="#00f0ff")

    ax.axvline(x=0,   color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7, label="Onset (0ms)")
    ax.axvline(x=300, color="#f39c12", linestyle=":",  linewidth=1.0, alpha=0.8, label="P300 (~300ms)")
    ax.axvline(x=400, color="#ff32d0", linestyle=":",  linewidth=1.0, alpha=0.8, label="N400 (~400ms)")

    ax.fill_between(t_ms, signal, 0, where=(t_ms >= 250) & (t_ms <= 350), alpha=0.12, color="#f39c12")
    ax.fill_between(t_ms, signal, 0, where=(t_ms >= 350) & (t_ms <= 500), alpha=0.10, color="#ff32d0")

    ax.set_xlabel("Timp (ms)", fontsize=9, color="#b9cacb", fontfamily="monospace")
    ax.set_ylabel("Amplitudine (uV)", fontsize=9, color="#b9cacb", fontfamily="monospace")
    ax.set_title("WAVEFORM MONITOR  —  Canal Pz (canal 62)", fontsize=10,
                 color="#00f0ff", fontfamily="monospace", loc="left", pad=10)

    ax.tick_params(colors="#b9cacb", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3b494b")

    ax.legend(fontsize=8, loc="upper right", facecolor="#1d2022",
              edgecolor="#3b494b", labelcolor="#b9cacb", framealpha=0.9)
    ax.grid(True, alpha=0.1, color="#3b494b", linestyle="--")
    ax.set_xlim(t_ms[0], t_ms[-1])

    fig.tight_layout(pad=0.5)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Custom HTML components
# ══════════════════════════════════════════════════════════════════════════════
def html_header():
    return """
    <div style="font-family:'Sora',sans-serif; margin-bottom:1.5rem;">
      <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
        <span style="font-size:2rem; font-weight:800; color:#00f0ff; letter-spacing:-0.02em;">NeuroText</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:10px;
               letter-spacing:0.12em; color:#b9cacb; text-transform:uppercase;
               background:rgba(0,240,255,0.08); border:1px solid rgba(0,240,255,0.2);
               padding:2px 10px; border-radius:4px;">EEG-to-Text v1.0</span>
      </div>
      <p style="font-family:'Inter',sans-serif; color:#b9cacb; font-size:14px; margin:0;">
        Decodifica semnalele EEG in text — ZuCo 1.0 · 105 canale · 500 Hz · Vocabular 200 cuvinte
      </p>
    </div>
    """

def html_status_hud(health):
    if health:
        mock = health.get("using_mock", True)
        mode_color = "#f39c12" if mock else "#2ecc71"
        mode_text  = "MOCK ACTIVE" if mock else "REAL MODEL"
        dot_color  = mode_color
        signal = "Optimal"
        ver = health.get("model_version", "N/A")
        lat = "12ms"
    else:
        mode_color = "#e74c3c"; mode_text = "BACKEND OFFLINE"
        dot_color = "#e74c3c"; signal = "No Signal"; ver = "N/A"; lat = "—"

    return f"""
    <div style="
        background:rgba(29,32,34,0.85); backdrop-filter:blur(20px);
        border:1px solid rgba(0,240,255,0.15); border-radius:10px;
        padding:1rem 1.5rem; margin-bottom:1rem;
        display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">

      <div style="display:flex; gap:2rem; flex-wrap:wrap;">
        <div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:0.12em;
                      text-transform:uppercase; color:#b9cacb; margin-bottom:2px;">Neural Link</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:14px; color:#00dbe9; font-weight:500;">
            Active / 98% Sync</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:0.12em;
                      text-transform:uppercase; color:#b9cacb; margin-bottom:2px;">Latency</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:14px; color:#00dbe9; font-weight:500;">{lat}</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:0.12em;
                      text-transform:uppercase; color:#b9cacb; margin-bottom:2px;">Model</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:14px; color:{mode_color}; font-weight:500;">{ver}</div>
        </div>
      </div>

      <div style="display:flex; align-items:center; gap:10px;">
        <div class="pulse-dot" style="background:{dot_color}; box-shadow:0 0 8px {dot_color};"></div>
        <span style="font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700;
                     letter-spacing:0.1em; text-transform:uppercase; color:{mode_color};">{mode_text}</span>
      </div>
    </div>
    """

def html_word_card(rank, word, score, pos_tag=""):
    pos_colors = {"NOUN":"#e74c3c","VERB":"#3498db","ADJ":"#2ecc71","PROPN":"#9b59b6",
                  "ADV":"#f39c12","NUM":"#1abc9c","ADP":"#95a5a6","PRON":"#e67e22"}
    pos_color = pos_colors.get(pos_tag, "#849495")
    bar_w = int(score * 100)
    is_first = rank == 1
    border = "rgba(0,240,255,0.4)" if is_first else "rgba(59,73,75,0.5)"
    bg = "rgba(0,240,255,0.06)" if is_first else "rgba(29,32,34,0.7)"
    glow = "box-shadow:0 0 15px rgba(0,240,255,0.2);" if is_first else ""
    rank_label = "🥇" if is_first else f"#{rank}"

    return f"""
    <div style="
        background:{bg}; border:1px solid {border}; border-radius:8px;
        padding:0.75rem 1rem; margin-bottom:0.5rem; {glow}
        transition:all 0.2s;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#b9cacb;">{rank_label}</span>
          <span style="font-family:'Sora',sans-serif; font-size:15px; font-weight:700;
                       color:{'#00f0ff' if is_first else '#e0e3e5'};">{word}</span>
          {"<span style='font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:0.1em;color:"+pos_color+";background:rgba(0,0,0,0.3);border:1px solid "+pos_color+"40;padding:1px 6px;border-radius:3px;'>"+pos_tag+"</span>" if pos_tag else ""}
        </div>
        <span style="font-family:'JetBrains Mono',monospace; font-size:12px;
                     color:{'#00f0ff' if is_first else '#b9cacb'}; font-weight:500;">{score:.4f}</span>
      </div>
      <div style="height:3px; background:#1d2022; border-radius:2px; overflow:hidden;">
        <div style="height:100%; width:{bar_w}%;
                    background:{'linear-gradient(90deg,#00dbe9,#00f0ff)' if is_first else 'rgba(0,240,255,0.4)'};
                    border-radius:2px; transition:width 0.5s ease;"></div>
      </div>
    </div>
    """

def html_sentence_card(sentence):
    return f"""
    <div style="
        background:rgba(0,240,255,0.04); border:1px solid rgba(0,240,255,0.2);
        border-radius:10px; padding:1.25rem 1.5rem; margin-top:0.5rem;
        position:relative; overflow:hidden;">
      <div style="position:absolute; top:0; left:0; right:0; height:2px;
                  background:linear-gradient(90deg,transparent,#00f0ff,transparent);
                  opacity:0.6;"></div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:0.12em;
                  text-transform:uppercase; color:#b9cacb; margin-bottom:8px;">
        Propozitie reconstruita
      </div>
      <div style="font-family:'Sora',sans-serif; font-size:16px; font-weight:600;
                  color:#e0e3e5; line-height:1.5;">{sentence}</div>
    </div>
    """

def html_reveal_correct(true_word, top1_word, top5_words):
    correct = top1_word == true_word
    in_top5 = true_word in top5_words
    rank = top5_words.index(true_word) + 1 if in_top5 else None

    if correct:
        bg = "rgba(46,204,113,0.08)"; border = "rgba(46,204,113,0.4)"; icon = "✅"
        msg = f"Top-1 CORECT! Cuvant real: <strong style='color:#2ecc71;'>{true_word}</strong>"
    elif in_top5:
        bg = "rgba(243,156,18,0.08)"; border = "rgba(243,156,18,0.4)"; icon = "⚠️"
        msg = f"Cuvant real: <strong style='color:#f39c12;'>{true_word}</strong> — pe locul #{rank} in top-5. Top-1 prezis: <em>{top1_word}</em>"
    else:
        bg = "rgba(231,76,60,0.08)"; border = "rgba(231,76,60,0.4)"; icon = "❌"
        msg = f"Cuvant real: <strong style='color:#e74c3c;'>{true_word}</strong> — nu e in top-5. Top-1 prezis: <em>{top1_word}</em>"

    return f"""
    <div style="background:{bg}; border:1px solid {border}; border-radius:8px;
                padding:1rem 1.25rem; margin-top:0.75rem;">
      <div style="font-family:'Inter',sans-serif; font-size:14px; color:#e0e3e5;">
        {icon} {msg}
      </div>
    </div>
    """

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Sora',sans-serif; padding:0.5rem 0 1rem;">
      <div style="font-size:1.5rem; font-weight:800; color:#00f0ff; letter-spacing:-0.02em;">NeuroText</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:0.12em;
                  color:#b9cacb; text-transform:uppercase; margin-top:2px;">EEG-to-Text Demo</div>
    </div>
    """, unsafe_allow_html=True)

    health = fetch_health()
    if health:
        mock = health.get("using_mock", True)
        c = "#f39c12" if mock else "#2ecc71"
        tag = "MOCK" if mock else "REAL"
        st.markdown(f"""
        <div style="background:rgba(29,32,34,0.9); border:1px solid rgba(0,240,255,0.15);
                    border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem;">
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <div class="pulse-dot" style="background:{c}; box-shadow:0 0 6px {c};"></div>
            <span style="font-family:'JetBrains Mono',monospace; font-size:10px;
                         color:{c}; font-weight:700; letter-spacing:0.1em;">BACKEND ACTIVE — {tag}</span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:9px; color:#b9cacb;">
            {health.get('model_version')} &nbsp;|&nbsp;
            Vocab: {health.get('vocab_size')} &nbsp;|&nbsp;
            CH: {health.get('n_channels')}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(231,76,60,0.08); border:1px solid rgba(231,76,60,0.3);
                    border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem;">
          <span style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#e74c3c;
                       letter-spacing:0.1em;">⚠ BACKEND OFFLINE</span>
        </div>
        """, unsafe_allow_html=True)
        st.code("python -m uvicorn app.backend.main:app\n  --reload --port 8000", language="bash")

    st.markdown('<hr style="border-color:rgba(59,73,75,0.5);margin:0.5rem 0;">', unsafe_allow_html=True)

    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:#b9cacb;margin-bottom:0.5rem;">Mod Input</p>', unsafe_allow_html=True)
    input_mode = st.radio(
        "",
        ["Upload .npy", "Exemple predefinite"],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color:rgba(59,73,75,0.5);margin:0.5rem 0;">', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:0.1em;
                text-transform:uppercase; color:#b9cacb; margin-bottom:0.5rem;">Neural Stack</div>
    <div style="font-size:12px; color:#e0e3e5; line-height:1.8;">
      📊 ZuCo 1.0 (105 ch, 500 Hz)<br>
      🤖 EEGNet + BERT retrieval<br>
      📝 Vocabular 200 cuvinte<br>
      🔬 Cosine similarity
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(59,73,75,0.5);margin:0.75rem 0;">', unsafe_allow_html=True)

    # ── Sectiunea Evaluare ────────────────────────────────────────────────────
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:#b9cacb;margin-bottom:0.5rem;">Evaluare Model (KPI-uri)</p>', unsafe_allow_html=True)

    if "eval_result" not in st.session_state:
        st.session_state["eval_result"] = None

    if st.button("▶  Ruleaza Evaluare", use_container_width=True):
        with st.spinner("Evaluez toate exemplele..."):
            st.session_state["eval_result"] = call_evaluate()

    ev = st.session_state.get("eval_result")
    if ev:
        top1  = ev.get("top1_accuracy", 0)
        top5  = ev.get("top5_accuracy", 0)
        cos   = ev.get("avg_cosine_similarity", 0)
        sem   = ev.get("avg_semantic_similarity", 0)
        n     = ev.get("n_examples", 0)

        def _kpi_color(v, hi=0.7, lo=0.3):
            if v >= hi: return "#2ecc71"
            if v >= lo: return "#f39c12"
            return "#e74c3c"

        st.markdown(f"""
        <div style="background:rgba(29,32,34,0.9);border:1px solid rgba(0,240,255,0.15);
                    border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#b9cacb;
                      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
            n = {n} exemple
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
            <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.1);
                        border-radius:6px;padding:8px;text-align:center;">
              <div style="font-size:1.2rem;font-weight:700;color:{_kpi_color(top1)};">{top1*100:.0f}%</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#b9cacb;
                          text-transform:uppercase;letter-spacing:0.08em;">Top-1 Acc</div>
            </div>
            <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.1);
                        border-radius:6px;padding:8px;text-align:center;">
              <div style="font-size:1.2rem;font-weight:700;color:{_kpi_color(top5)};">{top5*100:.0f}%</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#b9cacb;
                          text-transform:uppercase;letter-spacing:0.08em;">Top-5 Acc</div>
            </div>
            <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.1);
                        border-radius:6px;padding:8px;text-align:center;">
              <div style="font-size:1.2rem;font-weight:700;color:{_kpi_color(cos)};">{cos:.3f}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#b9cacb;
                          text-transform:uppercase;letter-spacing:0.08em;">Avg Cosine</div>
            </div>
            <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.1);
                        border-radius:6px;padding:8px;text-align:center;">
              <div style="font-size:1.2rem;font-weight:700;color:{_kpi_color(sem)};">{sem:.3f}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#b9cacb;
                          text-transform:uppercase;letter-spacing:0.08em;">Sem Sim</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabel detalii per exemplu
        with st.expander("Detalii per exemplu", expanded=False):
            for row in ev.get("per_example", []):
                icon = "✅" if row["top1_hit"] else ("🟡" if row["top5_hit"] else "❌")
                st.markdown(
                    f"**{icon} #{row['id']}** `{row['true_word']}` → "
                    f"`{row['pred_top1']}` | cos={row['cosine_score']:.3f} "
                    f"| sem={row['semantic_sim']:.3f} | {row['inference_ms']:.0f}ms"
                )

    st.markdown('<hr style="border-color:rgba(59,73,75,0.5);margin:0.75rem 0;">', unsafe_allow_html=True)
    if st.button("↺  Reseteaza sesiunea", use_container_width=True):
        for k in ["result","true_word","eeg_array","revealed","_last_example_id"]:
            st.session_state[k] = None if k != "revealed" else False
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(html_header(), unsafe_allow_html=True)

health = fetch_health()
st.markdown(html_status_hud(health), unsafe_allow_html=True)

st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

eeg_bytes  = None
file_label = ""
example_id = None

# ── Mod 1: Upload ─────────────────────────────────────────────────────────────
if input_mode == "Upload .npy":
    st.markdown('<p class="label-mono" style="margin-bottom:0.5rem;">Input EEG — Upload fisier .npy</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop fisier .npy",
        type=["npy"],
        help="Shape (105, T), T in [50,700] samples, valori in microvolti.",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        raw = uploaded.read()
        try:
            arr = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float32)
            st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                color:#2ecc71;padding:4px 0;">✓ {uploaded.name} — shape {arr.shape}</div>""",
                unsafe_allow_html=True)
            eeg_bytes = raw
            file_label = uploaded.name
            st.session_state["eeg_array"] = arr
            st.session_state["true_word"] = None
            st.session_state["revealed"]  = False
        except Exception as exc:
            st.error(f"Nu pot citi fisierul: {exc}")

# ── Mod 2: Exemple predefinite ────────────────────────────────────────────────
else:
    examples = fetch_examples()
    if not examples:
        st.warning("Nu am putut incarca exemplele. Verifica backend-ul.")
    else:
        st.markdown('<p class="label-mono" style="margin-bottom:0.5rem;">Selecteaza exemplu predefin</p>', unsafe_allow_html=True)
        options = {f"#{ex['id']} — {ex['description']}": ex for ex in examples}
        selected_label = st.selectbox("", list(options.keys()), label_visibility="collapsed")
        selected_ex    = options[selected_label]
        example_id     = selected_ex["id"]

        st.markdown(f"""<div class="label-mono" style="margin:4px 0 12px;">
            Shape: {selected_ex['shape']} &nbsp;|&nbsp; File: {selected_ex['filename']}</div>""",
            unsafe_allow_html=True)

        # Incarca si vizualizeaza EEG
        try:
            import sys, os; sys.path.insert(0, os.path.abspath("."))
            from app import config as _cfg
            _npy = _cfg.EXAMPLES_DIR / selected_ex["filename"]
            if _npy.exists():
                _arr = np.load(str(_npy)).astype(np.float32)
                _fig = plot_eeg_signal(_arr)
                st.pyplot(_fig, use_container_width=True)
                plt.close(_fig)
                st.session_state["eeg_array"] = _arr
        except Exception:
            pass

        if st.session_state["_last_example_id"] != example_id:
            st.session_state["revealed"] = False
            st.session_state["result"]   = None
            st.session_state["_last_example_id"] = example_id

# ── Buton Analizeaza ──────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
ready = (eeg_bytes is not None) or (example_id is not None)

if st.button("⬡  ANALIZEAZA EEG", disabled=not ready, use_container_width=True, type="primary"):
    with st.spinner("Procesare semnal neural..."):
        if eeg_bytes is not None:
            buf = io.BytesIO(); np.save(buf, st.session_state["eeg_array"])
            result = call_predict_upload(buf.getvalue(), file_label or "upload.npy")
        else:
            result = call_predict_example(example_id)

    if result is not None:
        st.session_state["result"]   = result
        st.session_state["revealed"] = False
        if example_id is not None and examples:
            for ex in examples:
                if ex["id"] == example_id:
                    st.session_state["true_word"] = ex["true_word"]
                    break
        st.rerun()

# ── Afisare rezultate ─────────────────────────────────────────────────────────
if st.session_state["result"] is not None:
    result    = st.session_state["result"]
    true_word = st.session_state["true_word"]
    top5      = result.get("top_5_words", [])

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Tenta superioară
    st.markdown("""
    <div style="height:2px; background:linear-gradient(90deg,transparent,#00f0ff,#ff32d0,transparent);
                border-radius:1px; margin-bottom:1.5rem; opacity:0.7;"></div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Coloana stanga: Top-5 + metrici ──
    with col_left:
        st.markdown('<p class="label-mono" style="margin-bottom:0.75rem;">Top-5 cuvinte candidate</p>', unsafe_allow_html=True)

        # Incarca POS tags din config pentru fiecare cuvant
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

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        # Metrici
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Inferenta", f"{result['inference_time_ms']:.1f} ms")
        with mc2:
            st.metric("Model", result.get("model_version", "N/A"))

    # ── Coloana dreapta: propozitie + EEG + reveal ──
    with col_right:
        st.markdown(html_sentence_card(result.get("reconstructed_sentence", "")), unsafe_allow_html=True)
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        # EEG signal (din session_state)
        eeg_arr = st.session_state.get("eeg_array")
        eeg_arr = st.session_state.get("eeg_array")
        if eeg_arr is not None:
            st.markdown('<p class="label-mono" style="margin-bottom:0.5rem;">Semnal EEG (canale 0-4)</p>', unsafe_allow_html=True)
            fig = plot_eeg_signal(eeg_arr)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Reveleaza cuvant real
        if true_word is not None:
            st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
            top1_w  = top5[0]["word"]
            top5_ws = [ws["word"] for ws in top5]
            if not st.session_state["revealed"]:
                if st.button("👁  Reveleaza cuvantul real", use_container_width=True):
                    st.session_state["revealed"] = True
                    st.rerun()
            else:
                st.markdown(html_reveal_correct(true_word, top1_w, top5_ws), unsafe_allow_html=True)
