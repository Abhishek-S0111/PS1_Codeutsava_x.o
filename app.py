"""
Steel Billet Inspection & Traceability System
CodeUtsava X.0 - Team INIT

Main Streamlit application entry point.
"""
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import APP_TITLE, APP_VERSION, APP_TEAM, THEME, is_qc_suspended

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────
# ─── Global CSS (Executive Enterprise Light) ─────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* ── Root variables ────────────────────────────── */
    :root {{
        --bg-primary: {THEME['bg_primary']};
        --bg-secondary: {THEME['bg_secondary']};
        --bg-card: {THEME['bg_card']};
        --bg-elevated: {THEME['bg_elevated']};
        --accent: {THEME['accent_primary']};
        --success: {THEME['accent_success']};
        --danger: {THEME['accent_danger']};
        --warning: {THEME['accent_warning']};
        --info: {THEME['accent_info']};
        --text-primary: {THEME['text_primary']};
        --text-secondary: {THEME['text_secondary']};
        --border: {THEME['border']};
    }}

    /* ── Global resets ─────────────────────────────── */
    .stApp {{
        background-color: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-primary);
    }}

    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid var(--border);
        box-shadow: 1px 0 4px rgba(0, 0, 0, 0.02);
    }}

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {{
        color: var(--text-secondary);
        font-size: 0.88rem;
    }}

    /* ── Cards & Metrics ───────────────────────────── */
    .metric-card {{
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.1rem 1rem;
        text-align: left;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
        position: relative;
        overflow: hidden;
    }}
    .metric-card:hover {{
        border-color: #CBD5E1;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }}
    .metric-value {{
        font-size: 1.9rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }}
    .metric-label {{
        font-size: 0.76rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* ── Status badges ────────────────────────────── */
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
    }}
    .badge-pass {{ background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }}
    .badge-fail {{ background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }}
    .badge-review {{ background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }}
    .badge-rework {{ background: #FFF7ED; color: #9A3412; border: 1px solid #FED7AA; }}
    .badge-unverified {{ background: #F8FAFC; color: #475569; border: 1px solid #E2E8F0; }}
    .badge-processing {{ background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }}

    /* ── Headings ──────────────────────────────────── */
    .page-title {{
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }}
    .page-subtitle {{
        color: var(--text-secondary);
        font-size: 0.92rem;
        margin-bottom: 1.25rem;
    }}

    /* ── Data table styling ────────────────────────── */
    .stDataFrame {{
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }}

    /* ── Clean Industrial Buttons ───────────────────── */
    .stButton > button {{
        background-color: #0F2942;
        color: #FFFFFF;
        border: 1px solid #0F2942;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.45rem 1.1rem;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }}
    .stButton > button:hover {{
        background-color: #1E3A5F;
        border-color: #1E3A5F;
        color: #FFFFFF;
    }}

    /* ── Sidebar Navigation Buttons ─────────────────── */
    section[data-testid="stSidebar"] .stButton > button {{
        background-color: #F8FAFC;
        color: #1E293B;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        font-weight: 500;
        text-align: left;
        margin-bottom: 4px;
        transition: background-color 0.15s, border-color 0.15s;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background-color: #EFF6FF;
        border-color: #BFDBFE;
        color: #1E40AF;
    }}

    /* ── Alert bar ─────────────────────────────────── */
    .alert-bar {{
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: #FFFFFF;
        border: 1px solid var(--border);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }}
    .alert-critical {{ border-left: 4px solid var(--danger); background: #FEF2F2; color: #991B1B; }}
    .alert-warning  {{ border-left: 4px solid var(--warning); background: #FFFBEB; color: #92400E; }}
    .alert-info     {{ border-left: 4px solid var(--info); background: #EFF6FF; color: #1E40AF; }}

    /* ── Hide default Streamlit decorations ─────── */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 0.8rem 0 0.5rem;">
            <div style="display:flex; align-items:center; gap:0.6rem;">
                <div style="font-size:1.8rem; background:#0F2942; border-radius:8px; width:44px; height:44px; display:flex; align-items:center; justify-content:center; color:#FFFFFF;">🏭</div>
                <div>
                    <div style="font-size:1.05rem; font-weight:700; color:#0F172A; line-height:1.2;">
                        STEEL INSPECTOR
                    </div>
                    <div style="font-size:0.75rem; color:{THEME['text_secondary']}; font-weight:500;">
                        Automated QC &amp; Traceability
                    </div>
                </div>
            </div>
            <div style="display:inline-block; margin-top:0.6rem; font-size:0.7rem; font-weight:600; background:#E2E8F0; color:#334155; padding:2px 8px; border-radius:4px;">
                v{APP_VERSION} &bull; {APP_TEAM}
            </div>
        </div>
        <hr style="border-color:{THEME['border']}; margin:0.8rem 0;">
        """, unsafe_allow_html=True)

        if is_qc_suspended():
            st.markdown("""
            <div style="background:#FFFBEB; border:1px solid #FCD34D; border-radius:6px; padding:6px 10px; margin-bottom:0.8rem; font-size:0.75rem; color:#B45309; text-align:center; font-weight:700;">
                🧪 Testing Mode (QC Suspended)
            </div>
            """, unsafe_allow_html=True)

        pages = {
            "🏠 Dashboard": "dashboard",
            "📹 Upload & Process": "upload",
            "🔬 Calibration": "calibration",
            "📊 History & Analysis": "history",
            "🤖 AI Chatbot": "chatbot",
            "⚙️ Settings": "settings",
        }

        if "current_page" not in st.session_state:
            st.session_state.current_page = "dashboard"

        for label, key in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = key

        st.markdown(f"""
        <hr style="border-color:{THEME['border']}; margin:1rem 0 0.6rem;">
        <div style="font-size:0.75rem; color:{THEME['text_muted']}; text-align:center; line-height:1.4;">
            <span style="color:#16A34A; font-weight:600;">● System Ready</span> &bull; 100% Offline<br>
            SQLite WAL &bull; OpenCV &bull; YOLO &bull; EasyOCR
        </div>
        """, unsafe_allow_html=True)


# ─── Page router ──────────────────────────────────────────────────
def main():
    render_sidebar()

    page = st.session_state.get("current_page", "dashboard")

    if page == "dashboard":
        from pages.dashboard import render_dashboard
        render_dashboard()
    elif page == "upload":
        from pages.upload_process import render_upload_process
        render_upload_process()
    elif page == "calibration":
        from pages.calibration import render_calibration
        render_calibration()
    elif page == "history":
        from pages.history_analysis import render_history
        render_history()
    elif page == "chatbot":
        from pages.chatbot_page import render_chatbot
        render_chatbot()
    elif page == "settings":
        from pages.settings import render_settings
        render_settings()
    else:
        from pages.dashboard import render_dashboard
        render_dashboard()


if __name__ == "__main__":
    main()
