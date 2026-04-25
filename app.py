# =========================
# app.py  —  AI Personal Finance Analyzer
# Dark Neon Dashboard  •  Deep Purple / Cyan / Pink
# =========================

import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
matplotlib.use('Agg')

st.set_page_config(
    page_title="FinanceAI — Personal Finance Analyzer",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

from pipeline import load_data, main_pipeline

# ════════════════════════════════════════════════════════════
# GLOBAL CSS  —  single block, no duplicate rules
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── palette ─────────────────────────────────────────────── */
:root {
    --bg-deep:     #0d0d1a;
    --bg-card:     #13132b;
    --bg-card2:    #1a1a35;
    --border:      rgba(100,80,255,0.25);
    --border-glow: rgba(100,80,255,0.6);
    --purple:      #6450ff;
    --cyan:        #00d4ff;
    --pink:        #ff3dac;
    --green:       #00ffb3;
    --yellow:      #ffd166;
    --txt:         #e8e8ff;
    --muted:       #7b7ba8;
    --dim:         #4a4a72;
}

/* ── base ────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--bg-deep) !important;
    font-family: 'Inter', sans-serif;
    color: var(--txt);
}

/* push all content below the fixed top bar — SINGLE rule */
.block-container {
    padding-top: 5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2rem !important;
}

/* ── top bar  ────────────────────────────────────────────── */
header[data-testid="stHeader"] {
    background: rgba(13,13,26,0.97) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid rgba(100,80,255,0.2) !important;
}
[data-testid="stDecoration"] { display: none; }
footer, #MainMenu { visibility: hidden; }

/* ── sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0a0a18 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--txt) !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 13px !important;
    padding: 6px 0 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem !important; }

/* ── metric cards ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 0 20px rgba(100,80,255,0.08) !important;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--purple), var(--cyan));
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--cyan) !important;
}

/* ── tabs ────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 18px !important;
}
button[data-baseweb="tab"]:hover { color: var(--cyan) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
    background: rgba(0,212,255,0.05) !important;
}
[data-testid="stTabPanel"] {
    background: transparent !important;
    padding-top: 1rem !important;
}

/* ── dataframes ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: var(--bg-card2) !important;
    color: var(--cyan) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 700 !important;
}
[data-testid="stDataFrame"] td {
    color: var(--txt) !important;
    font-size: 13px !important;
}

/* ── buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--purple) 0%, #4f35cc 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(100,80,255,0.4) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 30px rgba(100,80,255,0.7) !important;
    transform: translateY(-1px) !important;
}

/* ── file uploader ───────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-glow) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* ── selectbox ───────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 10px !important;
    color: var(--txt) !important;
}

/* ── expander ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── alerts ──────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--txt) !important;
}

/* ── scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--purple); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME
# ════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  "#13132b",
    "axes.facecolor":    "#13132b",
    "axes.edgecolor":    "#2a2a4a",
    "axes.labelcolor":   "#7b7ba8",
    "axes.titlecolor":   "#e8e8ff",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.grid":         True,
    "grid.color":        "#1e1e3a",
    "grid.linewidth":    0.8,
    "xtick.color":       "#7b7ba8",
    "ytick.color":       "#7b7ba8",
    "text.color":        "#e8e8ff",
    "legend.facecolor":  "#1a1a35",
    "legend.edgecolor":  "#2a2a4a",
    "legend.labelcolor": "#e8e8ff",
    "legend.fontsize":   9,
    "figure.dpi":        120,
    "lines.linewidth":   2,
})

NEON = ["#6450ff", "#00d4ff", "#ff3dac", "#00ffb3", "#ffd166", "#ff6b6b", "#c77dff"]

# ════════════════════════════════════════════════════════════
# CACHING
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_cached_pipeline(file_bytes: bytes):
    import io
    df_raw = load_data(io.BytesIO(file_bytes))
    return main_pipeline(df_raw)

# ════════════════════════════════════════════════════════════
# UI HELPERS
# NOTE: Every helper uses st.markdown(..., unsafe_allow_html=True)
#       directly — never passes HTML to another function that
#       then calls st.markdown, because that breaks rendering
#       inside st.columns / st.expander contexts.
# ════════════════════════════════════════════════════════════

def neon_header(title, subtitle="", icon=""):
    sub = f'<p style="color:#7b7ba8;font-size:14px;margin:4px 0 0 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div style="margin-bottom:24px;">'
        f'<h2 style="font-family:Rajdhani,sans-serif;font-size:30px;font-weight:700;'
        f'background:linear-gradient(135deg,#e8e8ff 0%,#00d4ff 60%,#6450ff 100%);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        f'background-clip:text;margin:0;line-height:1.2;">{icon} {title}</h2>'
        f'{sub}</div>',
        unsafe_allow_html=True
    )


def section_label(text):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:22px 0 10px 0;">'
        f'<div style="width:3px;height:18px;background:linear-gradient(180deg,#6450ff,#00d4ff);'
        f'border-radius:2px;flex-shrink:0;"></div>'
        f'<span style="font-size:11px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:#7b7ba8;">{text}</span></div>',
        unsafe_allow_html=True
    )


def dark_card(inner_html, border_color="#6450ff"):
    """Renders a dark neon card. Always call directly — never nest inside another helper."""
    st.markdown(
        f'<div style="background:#13132b;border:1px solid {border_color}40;'
        f'border-radius:14px;padding:18px 22px;'
        f'box-shadow:0 0 24px {border_color}18;margin-bottom:12px;">'
        f'{inner_html}</div>',
        unsafe_allow_html=True
    )


def insight_row(text, color="#6450ff"):
    st.markdown(
        f'<div style="display:flex;align-items:flex-start;gap:12px;background:#13132b;'
        f'border:1px solid {color}30;border-radius:10px;padding:12px 16px;'
        f'margin-bottom:8px;">'
        f'<div style="width:3px;min-height:20px;background:{color};border-radius:2px;'
        f'margin-top:2px;flex-shrink:0;"></div>'
        f'<span style="font-size:14px;color:#c8c8e8;line-height:1.5;">{text}</span></div>',
        unsafe_allow_html=True
    )


def mini_stat_card(label, value, accent="#6450ff", icon=""):
    st.markdown(
        f'<div style="background:#13132b;border:1px solid {accent}35;border-radius:14px;'
        f'padding:18px 20px;position:relative;overflow:hidden;'
        f'box-shadow:0 0 20px {accent}12;margin-bottom:4px;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
        f'background:linear-gradient(90deg,{accent},{accent}00);"></div>'
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.1em;color:#7b7ba8;margin-bottom:8px;">{icon} {label}</div>'
        f'<div style="font-family:Rajdhani,sans-serif;font-size:24px;font-weight:700;'
        f'color:{accent};line-height:1;">{value}</div></div>',
        unsafe_allow_html=True
    )


def progress_bar(pct, color):
    st.markdown(
        f'<div style="background:#0a0a18;border-radius:999px;height:20px;'
        f'width:100%;overflow:hidden;border:1px solid #2a2a4a;">'
        f'<div style="background:linear-gradient(90deg,{color}aa,{color});height:100%;'
        f'width:{min(pct,100):.1f}%;border-radius:999px;box-shadow:0 0 12px {color}88;"></div></div>'
        f'<p style="font-size:12px;color:#7b7ba8;margin:6px 0 0 0;">'
        f'{min(pct,100):.1f}% of recommended budget utilised</p>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="padding:0 8px 20px 8px;">'
        '<p style="font-family:Rajdhani,sans-serif;font-size:26px;font-weight:700;'
        'background:linear-gradient(135deg,#fff,#00d4ff);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;margin:0;line-height:1;">💎 FinanceAI</p>'
        '<p style="font-size:10px;color:#4a4a72;margin:4px 0 0 0;'
        'letter-spacing:0.12em;text-transform:uppercase;">Personal Finance Analyzer</p>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio(
        "NAVIGATE",
        ["🏠  Home", "📊  Overview", "👤  User Analysis", "🤖  Model Comparison"],
        key="nav"
    )
    st.markdown("---")

    if "results" in st.session_state:
        s = st.session_state["results"]["summary"]
        rows = [
            ("Accounts",     str(s["Total Accounts"])),
            ("Transactions", f"{s['Total Transactions']:,}"),
            ("From",         s["Date Range Start"]),
            ("To",           s["Date Range End"]),
        ]
        rows_html = "".join(
            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
            f'<span style="font-size:12px;color:#7b7ba8;">{k}</span>'
            f'<span style="font-size:12px;font-weight:700;color:#00d4ff;">{v}</span></div>'
            for k, v in rows
        )
        st.markdown(
            '<div style="padding:0 4px;">'
            '<p style="font-size:10px;color:#4a4a72;text-transform:uppercase;'
            'letter-spacing:0.1em;margin:0 0 10px 0;font-weight:700;">Dataset</p>'
            + rows_html + '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

    st.markdown(
        '<p style="font-size:10px;color:#4a4a72;text-align:center;margin:0;">'
        'Prophet · LSTM · CatBoost · Streamlit</p>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════
def page_home():
    st.markdown(
        '<div style="text-align:center;padding:32px 0 24px 0;">'
        '<p style="font-size:52px;margin:0 0 10px 0;">💎</p>'
        '<h1 style="font-family:Rajdhani,sans-serif;font-size:40px;font-weight:700;'
        'background:linear-gradient(135deg,#ffffff 0%,#00d4ff 50%,#6450ff 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;margin:0 0 12px 0;line-height:1.1;">'
        'AI Personal Finance Analyzer</h1>'
        '<p style="font-size:15px;color:#7b7ba8;max-width:520px;margin:0 auto;line-height:1.7;">'
        'Upload your bank statement to get instant AI-powered financial insights, '
        'smart forecasts, health scores, and personalised recommendations.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    features = [
        ("#6450ff", "📂", "ML Categorization",  "Auto-labels transactions using CatBoost & Random Forest"),
        ("#00d4ff", "📈", "6-Month Forecast",    "Prophet & LSTM predict future net savings"),
        ("#ff3dac", "🏥", "Health Scoring",      "0–100 financial health score per account"),
        ("#00ffb3", "🚨", "Smart Alerts",        "Real-time overspending alerts with actionable tips"),
    ]
    cols = st.columns(4)
    for col, (color, icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f'<div style="background:#13132b;border:1px solid {color}40;border-radius:16px;'
                f'padding:24px 16px;text-align:center;box-shadow:0 0 24px {color}15;'
                f'height:168px;position:relative;overflow:hidden;">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
                f'background:linear-gradient(90deg,{color},{color}00);"></div>'
                f'<p style="font-size:28px;margin:0 0 10px 0;">{icon}</p>'
                f'<p style="font-weight:700;color:#e8e8ff;font-size:14px;margin:0 0 6px 0;">{title}</p>'
                f'<p style="font-size:12px;color:#7b7ba8;margin:0;line-height:1.5;">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<hr style="border-color:#1e1e3a;margin:24px 0;">', unsafe_allow_html=True)
    section_label("Upload Your Dataset")

    upload_col, info_col = st.columns([3, 2])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Drop your Excel file here (.xlsx)",
            type=["xlsx"],
            help="Needs: Account No, DATE, TRANSACTION DETAILS, WITHDRAWAL AMT, DEPOSIT AMT"
        )
    with info_col:
        # Direct st.markdown — no helper wrapper — guarantees rendering inside columns
        st.markdown(
            '<div style="background:#13132b;border:1px solid #6450ff40;border-radius:14px;'
            'padding:18px 22px;box-shadow:0 0 24px #6450ff18;">'
            '<p style="font-size:11px;color:#7b7ba8;text-transform:uppercase;'
            'letter-spacing:0.1em;margin:0 0 10px 0;font-weight:700;">Required Columns</p>'
            '<p style="font-size:13px;color:#c8c8e8;line-height:2.1;margin:0;">'
            '📋 <span style="color:#00d4ff;font-family:monospace;">Account No</span> — user identifier<br>'
            '📅 <span style="color:#00d4ff;font-family:monospace;">DATE</span> — transaction date<br>'
            '📝 <span style="color:#00d4ff;font-family:monospace;">TRANSACTION DETAILS</span> — description<br>'
            '💸 <span style="color:#00d4ff;font-family:monospace;">WITHDRAWAL AMT</span> — expense<br>'
            '💰 <span style="color:#00d4ff;font-family:monospace;">DEPOSIT AMT</span> — income'
            '</p></div>',
            unsafe_allow_html=True
        )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        st.session_state["file_bytes"] = file_bytes
        st.session_state["file_name"]  = uploaded_file.name

        import io
        preview_df = pd.read_excel(io.BytesIO(file_bytes), nrows=5)
        section_label("Data Preview — First 5 Rows")
        st.dataframe(preview_df, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_col, _ = st.columns([1, 3])
        with run_col:
            run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

        if run_btn:
            if "results" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
                with st.spinner("⏳ Running AI pipeline — Prophet, LSTM, ML Categorization... (~2–5 min first run)"):
                    try:
                        results = run_cached_pipeline(file_bytes)
                        st.session_state["results"]   = results
                        st.session_state["last_file"] = uploaded_file.name
                        st.success("✅ Analysis complete! Navigate using the sidebar.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Pipeline error: {e}")
                        st.exception(e)
            else:
                st.success("✅ Already loaded. Use the sidebar to navigate.")

    elif "results" in st.session_state:
        st.info(f"✅ Loaded: **{st.session_state.get('file_name','')}** — use the sidebar to navigate.")


# ════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════
def page_overview():
    if "results" not in st.session_state:
        st.warning("⚠️ No data loaded. Go to **Home** and upload a file first.")
        return

    results = st.session_state["results"]
    s       = results["summary"]

    neon_header("Overview Dashboard", "High-level snapshot across all accounts", "📊")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: mini_stat_card("Total Accounts",     str(s["Total Accounts"]),            "#6450ff", "👥")
    with k2: mini_stat_card("Transactions",       f"{s['Total Transactions']:,}",       "#00d4ff", "📋")
    with k3: mini_stat_card("Avg Monthly Saving", f"₹{s['Avg Monthly Savings']:,.0f}", "#00ffb3", "💰")
    with k4: mini_stat_card("Period Start",       s["Date Range Start"],               "#ffd166", "📅")
    with k5: mini_stat_card("Period End",         s["Date Range End"],                 "#ff3dac", "📅")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        section_label("Financial Health Score — All Accounts")
        hdf    = results["health"].copy()
        accs   = [str(a) for a in hdf["Account No"].tolist()]
        scores = hdf["Overall Health Score"].tolist()
        bcols  = ["#00ffb3" if s >= 60 else ("#ffd166" if s >= 40 else "#ff3dac") for s in scores]
        fig_h, ax_h = plt.subplots(figsize=(9, 4))
        bars = ax_h.barh(accs, scores, color=bcols, height=0.55, edgecolor="none")
        ax_h.set_xlim(0, 110)
        ax_h.axvline(60, color="#00ffb3", linestyle="--", lw=1, alpha=0.5, label="Good ≥60")
        ax_h.axvline(40, color="#ffd166", linestyle="--", lw=1, alpha=0.5, label="Moderate ≥40")
        for bar, sc in zip(bars, scores):
            ax_h.text(bar.get_width()+1.5, bar.get_y()+bar.get_height()/2,
                      f"{sc:.0f}", va="center", fontsize=10, fontweight="bold", color="#e8e8ff")
        ax_h.set_xlabel("Health Score (0–100)", fontsize=10)
        ax_h.set_title("Financial Health Score per Account")
        ax_h.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

    with right:
        section_label("Transaction Category Split")
        cat = results["category_data"]
        fig_p, ax_p = plt.subplots(figsize=(5, 5))
        _, _, autos = ax_p.pie(
            cat.values, labels=cat.index,
            colors=NEON[:len(cat)], autopct="%1.1f%%", startangle=140,
            wedgeprops={"linewidth": 1.5, "edgecolor": "#0d0d1a"},
            textprops={"fontsize": 9, "color": "#c8c8e8"}
        )
        for at in autos:
            at.set_color("#fff"); at.set_fontweight("bold"); at.set_fontsize(9)
        ax_p.set_title("All Accounts Combined")
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        section_label("Health Scores")
        hs = results["health"].copy()
        hs["Grade"] = hs["Overall Health Score"].apply(
            lambda x: "🟢 Good" if x >= 60 else ("🟡 Moderate" if x >= 40 else "🔴 Poor"))
        st.dataframe(hs, use_container_width=True, hide_index=True)
    with t2:
        section_label("Budget Status")
        st.dataframe(results["budget"][["Account No","Status","Spending Ratio"]],
                     use_container_width=True, hide_index=True)
    with t3:
        section_label("Alert Summary")
        st.dataframe(results["alerts"][["Account No","Alert","Ratio","Top Category"]],
                     use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Insights Summary — All Accounts")
    st.dataframe(results["insights"], use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# PAGE: USER ANALYSIS
# ════════════════════════════════════════════════════════════
def page_user_analysis():
    if "results" not in st.session_state:
        st.warning("⚠️ No data loaded. Go to **Home** and upload a file first.")
        return

    results = st.session_state["results"]
    df      = results["df"]

    neon_header("User Analysis", "Deep-dive into a single account's financial profile", "👤")

    accounts = df["Account No"].unique().tolist()
    sel_acc  = st.selectbox("Select Account", accounts,
                             format_func=lambda x: f"Account  {x}", key="acc_selector")
    if not sel_acc:
        return

    st.markdown("<br>", unsafe_allow_html=True)

    health_row  = results["health"][results["health"]["Account No"] == sel_acc]
    score       = float(health_row["Overall Health Score"].values[0]) if len(health_row) else 0
    s_col       = "#00ffb3" if score >= 60 else ("#ffd166" if score >= 40 else "#ff3dac")
    s_txt       = "Good" if score >= 60 else ("Moderate" if score >= 40 else "Poor")
    alert_d     = results["alert_text"].get(sel_acc, {})
    a_level     = alert_d.get("level", "GOOD")
    a_col       = {"HIGH":"#ff3dac","MEDIUM":"#ffd166","GOOD":"#00ffb3"}.get(a_level, "#6450ff")
    m_user      = results["monthly"][results["monthly"]["Account No"] == sel_acc]
    avg_inc     = m_user["total_credit"].mean()
    avg_exp     = m_user["total_debit"].mean()
    avg_sav     = m_user["monthly_net"].mean()

    st.markdown(
        f'<div style="background:#13132b;border:1px solid #6450ff40;border-radius:18px;'
        f'padding:24px 32px;margin-bottom:24px;box-shadow:0 0 40px #6450ff15;'
        f'position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,#6450ff,#00d4ff,#ff3dac);"></div>'
        f'<div style="display:flex;align-items:center;gap:36px;flex-wrap:wrap;">'

        # Score
        f'<div style="text-align:center;min-width:100px;">'
        f'<p style="font-family:Rajdhani,sans-serif;font-size:52px;font-weight:700;'
        f'color:{s_col};margin:0;line-height:1;">{score:.0f}</p>'
        f'<p style="font-size:11px;color:#7b7ba8;margin:2px 0;">/ 100 HEALTH</p>'
        f'<span style="display:inline-block;background:{s_col}22;border:1px solid {s_col}60;'
        f'border-radius:999px;padding:3px 12px;font-size:11px;font-weight:700;color:{s_col};">'
        f'{s_txt}</span></div>'

        # Divider
        f'<div style="width:1px;height:70px;background:#2a2a4a;"></div>'

        # Stats
        f'<div style="display:flex;gap:32px;flex-wrap:wrap;">'
        f'<div><p style="font-size:10px;color:#7b7ba8;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin:0 0 4px 0;">Avg Income / Mo</p>'
        f'<p style="font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;'
        f'color:#00d4ff;margin:0;">₹{avg_inc:,.0f}</p></div>'
        f'<div><p style="font-size:10px;color:#7b7ba8;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin:0 0 4px 0;">Avg Expense / Mo</p>'
        f'<p style="font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;'
        f'color:#ff3dac;margin:0;">₹{avg_exp:,.0f}</p></div>'
        f'<div><p style="font-size:10px;color:#7b7ba8;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin:0 0 4px 0;">Avg Savings / Mo</p>'
        f'<p style="font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;'
        f'color:#00ffb3;margin:0;">₹{avg_sav:,.0f}</p></div>'
        f'<div><p style="font-size:10px;color:#7b7ba8;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin:0 0 4px 0;">Alert Level</p>'
        f'<p style="font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;'
        f'color:{a_col};margin:0;">{a_level}</p></div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💡 Insights","💰 Budget","🚨 Alerts",
        "🏥 Health Score","📈 Prophet Forecast","🤖 LSTM Forecast"
    ])

    # ── TAB 1 ──────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns([3, 2])
        with c1:
            section_label("Key Financial Insights")
            ins   = results["insights_text"].get(sel_acc, [])
            accs2 = ["#6450ff","#00d4ff","#ff3dac","#00ffb3","#ffd166","#c77dff","#ff6b6b"]
            if not ins:
                st.info("No insights available.")
            else:
                for i, line in enumerate(ins):
                    insight_row(line, accs2[i % len(accs2)])
        with c2:
            section_label("Monthly Summary Table")
            ms = m_user[["ds","total_credit","total_debit","monthly_net","txn_count"]].copy()
            ms = ms.rename(columns={"ds":"Month","total_credit":"Income (₹)",
                                    "total_debit":"Expense (₹)","monthly_net":"Net (₹)","txn_count":"Txns"})
            ms["Month"] = pd.to_datetime(ms["Month"]).dt.strftime("%b %Y")
            for c in ["Income (₹)","Expense (₹)","Net (₹)"]:
                ms[c] = ms[c].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(ms, use_container_width=True, hide_index=True, height=340)

        section_label("Income vs Expense Trend")
        mc  = m_user.sort_values("ds").copy()
        mst = [pd.Timestamp(d).strftime("%b %Y") for d in mc["ds"]]
        xr  = range(len(mst))
        fig_ie, ax_ie = plt.subplots(figsize=(12, 4))
        ax_ie.fill_between(xr, mc["total_credit"], alpha=0.12, color="#00d4ff")
        ax_ie.fill_between(xr, mc["total_debit"],  alpha=0.12, color="#ff3dac")
        ax_ie.plot(xr, mc["total_credit"], color="#00d4ff", lw=2, label="Income",      marker="o", ms=3)
        ax_ie.plot(xr, mc["total_debit"],  color="#ff3dac", lw=2, label="Expense",     marker="o", ms=3)
        ax_ie.plot(xr, mc["monthly_net"],  color="#00ffb3", lw=2, label="Net Savings", linestyle="--")
        ax_ie.axhline(0, color="#4a4a72", lw=0.8)
        ax_ie.set_xticks(list(xr))
        ax_ie.set_xticklabels(mst, rotation=45, ha="right", fontsize=8)
        ax_ie.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"₹{v/1e3:.0f}K" if abs(v) >= 1000 else f"₹{v:.0f}"))
        ax_ie.set_title(f"Income vs Expense — Account {sel_acc}")
        ax_ie.legend()
        plt.tight_layout()
        st.pyplot(fig_ie)
        plt.close(fig_ie)

    # ── TAB 2 ──────────────────────────────────────────────
    with tab2:
        section_label("Budget Recommendation")
        binfo = results["budget_text"].get(sel_acc)
        if binfo is None:
            st.info("Not enough data for budget analysis.")
        else:
            status = binfo["status"]
            sc2 = "#00ffb3" if "🟢" in status else ("#ffd166" if "🟡" in status else "#ff3dac")
            # Direct st.markdown — no wrapper function
            st.markdown(
                f'<div style="background:#13132b;border:1px solid {sc2}40;border-radius:14px;'
                f'padding:18px 22px;box-shadow:0 0 24px {sc2}18;margin-bottom:12px;">'
                f'<p style="font-size:16px;font-weight:700;color:{sc2};margin:0 0 4px 0;">{status}</p>'
                f'<p style="font-size:13px;color:#c8c8e8;margin:0;">{binfo["message"]}</p></div>',
                unsafe_allow_html=True
            )
            b1, b2, b3, b4 = st.columns(4)
            with b1: mini_stat_card("Avg Income (3M)",    f"₹{binfo['avg_income']:,.0f}",         "#00d4ff","💰")
            with b2: mini_stat_card("Avg Expense (3M)",   f"₹{binfo['avg_expense']:,.0f}",        "#ff3dac","💸")
            with b3: mini_stat_card("Recommended Budget", f"₹{binfo['recommended_budget']:,.0f}", "#6450ff","🎯")
            with b4: mini_stat_card("Spending Ratio",     f"{binfo['spending_ratio']:.2f}",        "#ffd166","📊")
            st.markdown("<br>", unsafe_allow_html=True)
            section_label("Budget Utilization")
            ratio = binfo["spending_ratio"]
            bc    = "#00ffb3" if ratio < 0.65 else ("#ffd166" if ratio < 0.9 else "#ff3dac")
            progress_bar(ratio * 100, bc)

    # ── TAB 3 ──────────────────────────────────────────────
    with tab3:
        section_label("Spending Alerts")
        ad = results["alert_text"].get(sel_acc)
        if ad is None:
            st.info("Not enough data for alerts.")
        else:
            alc = {"HIGH":"#ff3dac","MEDIUM":"#ffd166","GOOD":"#00ffb3"}.get(ad["level"],"#6450ff")
            for line in ad["lines"]:
                insight_row(line, alc)
            st.markdown("<br>", unsafe_allow_html=True)
            a1, a2, a3, a4 = st.columns(4)
            with a1: mini_stat_card("Avg Income (3M)",  f"₹{ad['avg_income']:,.0f}",  "#00d4ff","💰")
            with a2: mini_stat_card("Avg Expense (3M)", f"₹{ad['avg_expense']:,.0f}", "#ff3dac","💸")
            with a3: mini_stat_card("Expense Ratio",    f"{ad['ratio']:.2f}",           "#ffd166","📊")
            with a4: mini_stat_card("Top Category",     ad["top_category"].title(),     "#c77dff","📂")

    # ── TAB 4 ──────────────────────────────────────────────
    with tab4:
        section_label("Monthly Health Score — Last 6 Months")
        hm_raw = results["health_results"].get(sel_acc)
        if hm_raw is None:
            st.info("Not enough data (minimum 6 months required).")
        else:
            hm = hm_raw.copy()
            hm["Month"] = pd.to_datetime(hm["Month"]).dt.strftime("%b %Y")
            hm["Grade"] = hm["Health Score"].apply(
                lambda x: "🟢 Good" if x >= 60 else ("🟡 Moderate" if x >= 40 else "🔴 Poor"))
            st.dataframe(hm, use_container_width=True, hide_index=True)

            section_label("Health Score Trend Chart")
            hs2   = hm_raw["Health Score"].values
            hmon  = [pd.Timestamp(d).strftime("%b %Y") for d in hm_raw["Month"]]
            hcols = ["#00ffb3" if sc >= 60 else ("#ffd166" if sc >= 40 else "#ff3dac") for sc in hs2]
            xp    = range(len(hmon))
            fig_hs, ax_hs = plt.subplots(figsize=(10, 4))
            brs = ax_hs.bar(xp, hs2, color=hcols, width=0.5, edgecolor="none")
            for br, sc in zip(brs, hs2):
                ax_hs.text(br.get_x()+br.get_width()/2, br.get_height()+1.5,
                           f"{sc:.0f}", ha="center", fontsize=10, fontweight="bold", color="#e8e8ff")
            ax_hs.set_xticks(list(xp)); ax_hs.set_xticklabels(hmon, fontsize=10)
            ax_hs.set_ylim(0, 112)
            ax_hs.axhline(60, color="#00ffb3", linestyle="--", lw=1, alpha=0.5, label="Good (60)")
            ax_hs.axhline(40, color="#ffd166", linestyle="--", lw=1, alpha=0.5, label="Moderate (40)")
            ax_hs.set_title(f"Monthly Health Score — Account {sel_acc}")
            ax_hs.set_ylabel("Score"); ax_hs.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig_hs); plt.close(fig_hs)

    # ── TAB 5 ──────────────────────────────────────────────
    with tab5:
        section_label("Prophet Forecast — Next 6 Months")
        if sel_acc in results["prophet_figs"]:
            st.pyplot(results["prophet_figs"][sel_acc])
            section_label("Predicted Values")
            pf = results["prophet"].get(sel_acc, pd.DataFrame())
            if not pf.empty:
                pf = pf.copy()
                pf["Month"] = pd.to_datetime(pf["Month"]).dt.strftime("%b %Y")
                st.dataframe(pf, use_container_width=True, hide_index=True)
        else:
            st.info("⚠️ Not enough historical data for Prophet forecast. (Minimum: 12 months)")

    # ── TAB 6 ──────────────────────────────────────────────
    with tab6:
        section_label("LSTM Forecast — Next 6 Months")
        if sel_acc in results["lstm_figs"]:
            st.pyplot(results["lstm_figs"][sel_acc])
            section_label("Predicted Values")
            lf = results["lstm"].get(sel_acc, pd.DataFrame())
            if not lf.empty:
                lf = lf.copy()
                lf["Month"] = pd.to_datetime(lf["Month"]).dt.strftime("%b %Y")
                st.dataframe(lf, use_container_width=True, hide_index=True)
        else:
            st.info("⚠️ Not enough historical data for LSTM forecast. (Minimum: 15 months)")


# ════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ════════════════════════════════════════════════════════════
def page_model_comparison():
    if "results" not in st.session_state:
        st.warning("⚠️ No data loaded. Go to **Home** and upload a file first.")
        return

    results = st.session_state["results"]
    neon_header("Model Comparison", "Benchmarking all ML models used in the pipeline", "🤖")

    section_label("Transaction Categorization — Model Leaderboard")
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<span style="font-size:12px;color:#7b7ba8;">Best model selected: </span>'
        f'<span style="background:#6450ff22;color:#8b7bff;border:1px solid #6450ff50;'
        f'border-radius:6px;padding:2px 10px;font-size:12px;font-weight:700;">'
        f'{results["best_model"]}</span></div>',
        unsafe_allow_html=True
    )
    st.dataframe(results["model_results"], use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Forecasting — Prophet vs LSTM Error Metrics")
    comp = results["comparison"].copy()
    st.dataframe(
        comp.style.map(
            lambda v: "background-color:#00332244;color:#00ffb3;font-weight:700" if v == "LSTM" else (
                      "background-color:#1d4ed822;color:#00d4ff;font-weight:700" if v == "Prophet" else (
                      "background-color:#6450ff22;color:#c77dff;font-weight:700" if v == "Both" else "")),
            subset=["Best Model"]
        ),
        use_container_width=True, hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Winner Summary")
    if "Best Model" in comp.columns:
        counts = comp["Best Model"].value_counts()
        w1, w2, w3 = st.columns(3)
        with w1: mini_stat_card("LSTM Wins",    str(int(counts.get("LSTM",    0))), "#00ffb3","🏆")
        with w2: mini_stat_card("Prophet Wins", str(int(counts.get("Prophet", 0))), "#00d4ff","📈")
        with w3: mini_stat_card("Ties (Both)",  str(int(counts.get("Both",    0))), "#c77dff","🤝")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️  How are models evaluated?"):
        # Direct st.markdown inside expander — no wrapper
        st.markdown(
            '<div style="background:#13132b;border:1px solid #6450ff40;border-radius:14px;'
            'padding:18px 22px;box-shadow:0 0 24px #6450ff18;">'
            '<p style="font-size:13px;color:#c8c8e8;line-height:1.9;margin:0;">'
            '<b style="color:#00d4ff;">MAE</b> — Mean Absolute Error: avg prediction error in ₹<br>'
            '<b style="color:#00d4ff;">RMSE</b> — Root Mean Squared Error: penalises large errors more<br>'
            '<b style="color:#00d4ff;">Error Ratio</b> — MAE ÷ avg absolute value (lower = better)<br><br>'
            '<b style="color:#e8e8ff;">Best model selection:</b><br>'
            '&bull; Error Ratio diff &lt; 0.01 → <span style="color:#c77dff;">Both (Tie)</span><br>'
            '&bull; LSTM Error Ratio lower → <span style="color:#00ffb3;">LSTM wins</span><br>'
            '&bull; Prophet Error Ratio lower → <span style="color:#00d4ff;">Prophet wins</span><br><br>'
            'Both models train on 80% of history, tested on remaining 20%.'
            '</p></div>',
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════
page_map = {
    "🏠  Home":             page_home,
    "📊  Overview":         page_overview,
    "👤  User Analysis":    page_user_analysis,
    "🤖  Model Comparison": page_model_comparison,
}
page_map[page]()