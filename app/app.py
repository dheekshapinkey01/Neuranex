import sys
import html
import json
import hashlib
import hmac
import inspect
import os
import re
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="NeuraNex",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# HTML RENDERING FIX  (this is what was breaking your page)
# ------------------------------------------------------------
# Streamlit's markdown treats (a) any line indented 4+ spaces and
# (b) any blank line inside an HTML block as the start of a CODE BLOCK.
# textwrap.dedent only removes the COMMON indent, so nested tags stayed
# indented and the page printed raw <div> text.
# Fix: strip every line's indentation and drop blank lines.
# ============================================================

_original_markdown = st.markdown


def _flatten_html(body: str) -> str:
    return "\n".join(line.strip() for line in body.splitlines() if line.strip())


def fixed_markdown(body, *args, **kwargs):
    if isinstance(body, str) and kwargs.get("unsafe_allow_html", False):
        body = _flatten_html(body)
    return _original_markdown(body, *args, **kwargs)


st.markdown = fixed_markdown

# ============================================================
# OPTIONAL PROJECT MODULES
# ============================================================

try:
    from resume_parser import extract_resume_text
except Exception:
    extract_resume_text = None

try:
    from text_preprocessing import clean_text
except Exception:
    clean_text = None

try:
    import skill_extractor as _skill_module
    from skill_extractor import extract_skills
except Exception:
    _skill_module = None
    extract_skills = None

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "Dashboard",
    "resume_text": "",
    "cleaned_text": "",
    "resume_name": "",
    "predicted_category": "",
    "extracted_skills": [],
    "analysis_complete": False,
    "selected_job": "Data Scientist",
    "theme": "dark",
    "confidence": None,
    "users": {},
    "is_logged_in": False,
    "username": "",
    "history": [],
    "skill_debug": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# USER ACCOUNTS (saved to disk so they survive page refreshes)
# ============================================================

USERS_FILE = DATA_DIR / "users.json"


def load_users() -> dict:
    try:
        if USERS_FILE.exists():
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print("User file read error:", exc)
    return {}


def save_users(users: dict) -> bool:
    try:
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        print("User file write error:", exc)
        return False


def hash_password(password: str, salt: bytes = None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    _, digest = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(digest, hash_hex)


# ============================================================
# MODELS AND JOB DATA
# ============================================================


@st.cache_resource
def load_models():
    loaded_model = None
    loaded_vectorizer = None
    try:
        model_path = MODELS_DIR / "job_category_model.pkl"
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        if model_path.exists():
            loaded_model = joblib.load(model_path)
        if vectorizer_path.exists():
            loaded_vectorizer = joblib.load(vectorizer_path)
    except Exception as exc:
        print("Model loading error:", exc)
    return loaded_model, loaded_vectorizer


@st.cache_data
def load_jobs():
    path = DATA_DIR / "raw" / "job_descriptions.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            print("Job data loading error:", exc)
    return pd.DataFrame()


model, vectorizer = load_models()
jobs_df = load_jobs()

# ============================================================
# DESIGN SYSTEM
# ============================================================

CURRENT = st.session_state.page
TOP_KEY = {
    "Dashboard": "top_home",
    "About": "top_about",
    "How It Works": "top_how",
    "Features": "top_features",
    "Contact": "top_contact",
}.get(CURRENT, "top_none")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #070d22;
    --border: #1b2850;
    --text: #eef1ff;
    --muted: #9fabcb;
    --cyan: #38bdf8;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #080e26; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: #dce4f8; }
.block-container { max-width: 1600px; padding-top: 4.5rem; padding-bottom: 2rem; }

/* ---------- brand ---------- */
.brand { font-size: 28px; font-weight: 800; letter-spacing: -1px; white-space: nowrap; }
.brand-gradient {
    background: linear-gradient(90deg, #5aa9ff, #9a63ff);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}

/* ---------- top nav buttons look like links ---------- */
[class*="st-key-top_"] button {
    background: transparent !important; border: none !important;
    border-radius: 0 !important; box-shadow: none !important;
    color: #e3e8f8 !important; font-size: 15px; font-weight: 500;
}
[class*="st-key-top_"] button:hover { color: #ffffff !important; }
[class*="st-key-top_"] button { padding: 0.25rem 0.25rem !important; min-width: 0; }
[class*="st-key-top_"] button p,
.st-key-get_started button p,
.st-key-theme_toggle button p { white-space: nowrap !important; overflow: visible !important; text-overflow: clip !important; font-size: 14px; }
.st-key-""" + TOP_KEY + """ button {
    color: #ffffff !important; border-bottom: 2px solid #4f8bff !important;
}
.st-key-theme_toggle button {
    background: #111b42 !important; border: 1px solid var(--border) !important;
    border-radius: 20px !important; color: #fff !important;
}
.st-key-get_started button {
    background: linear-gradient(90deg, #5b5bf5, #8b5cf6) !important;
    border: none !important; color: #fff !important; font-weight: 600;
    border-radius: 10px !important; min-height: 44px;
}

/* ---------- sidebar nav ---------- */
[class*="st-key-side_"] button {
    justify-content: flex-start; background: transparent !important;
    border: none !important; box-shadow: none !important;
    padding: 10px 14px; font-size: 15px; font-weight: 500; border-radius: 10px !important;
}
[class*="st-key-side_"] button:hover { background: #111b42 !important; }
[class*="st-key-side_"] button[kind="primary"],
[class*="st-key-side_"] button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #1b2670, #141d52) !important;
    box-shadow: inset 3px 0 0 #6c63ff !important;
}
.side-quote {
    margin-top: 28px; padding: 18px; border-radius: 12px;
    border: 1px solid var(--border);
    background: linear-gradient(160deg, #0c1538, #2a1d6b);
    font-size: 17px; line-height: 1.5; color: #c9d1ee;
}

/* ---------- hero ---------- */
.hero {
    display: grid; grid-template-columns: 1.05fr 1fr; align-items: center;
    gap: 12px; min-height: 360px; padding: 30px 30px 24px 34px;
    margin: 12px 0 18px 0;
    background:
        radial-gradient(ellipse at 85% 35%, #2c2175 0%, transparent 55%),
        linear-gradient(110deg, #0a1230, #0d1740 55%, #1d1650);
    border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
}
.hero h1 {
    color: #fafbff; font-size: clamp(32px, 3.6vw, 56px); line-height: 1.08;
    letter-spacing: -1.5px; font-weight: 800; margin: 0 0 16px 0;
}
.gradient-text {
    background: linear-gradient(90deg, #7c5cff, #38bdf8);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-lead { font-size: 20px; color: #eef1ff; margin-bottom: 6px; }
.hero-description { font-size: 15px; line-height: 1.6; color: #aab6d3; max-width: 540px; }
.hero-features { display: flex; flex-wrap: wrap; gap: 30px; margin-top: 24px; }
.hero-feature { display: flex; align-items: center; gap: 10px; color: #f0f3ff; font-size: 13px; font-weight: 600; line-height: 1.35; }
.hero-feature-icon { font-size: 26px; color: #8b6cff; }

.hero-visual { position: relative; height: 310px; min-width: 0; display: flex; align-items: center; justify-content: center; }
.orbit { position: absolute; width: 88%; height: 52%; border: 1.5px solid rgba(124,92,255,.6); border-radius: 50%; transform: rotate(-14deg); box-shadow: 0 0 25px #583dff30; }
.orbit-two { position: absolute; width: 74%; height: 66%; border: 1.5px solid rgba(56,189,248,.5); border-radius: 50%; transform: rotate(32deg); }
.resume-illustration {
    position: relative; z-index: 2; width: 185px; height: 245px; padding: 22px 20px;
    border-radius: 16px; transform: rotate(10deg);
    background: linear-gradient(150deg, #9b8cff, #5b54ee 60%, #3b82f6);
    box-shadow: 0 25px 60px rgba(99,102,241,.55);
}
.resume-label { color: #e0dcff; font-size: 16px; font-weight: 700; font-style: italic; margin-bottom: 12px; }
.resume-avatar { width: 48px; height: 48px; border-radius: 10px; background: rgba(255,255,255,.4); margin-bottom: 12px; }
.resume-line { height: 7px; background: rgba(255,255,255,.55); border-radius: 5px; margin: 10px 0; }
.resume-line.short { width: 50%; }
.resume-line.medium { width: 75%; }

.float-card {
    position: absolute; z-index: 3; display: flex; align-items: center; gap: 10px;
    background: rgba(14,22,58,.88); border: 1px solid rgba(124,140,255,.35);
    border-radius: 10px; padding: 10px 14px; color: #f4f6ff;
    font-size: 13px; line-height: 1.3; box-shadow: 0 10px 35px #0005;
}
.fc-icon {
    width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 17px;
    background: linear-gradient(135deg, #6c63ff, #3b82f6);
}
.float-one { left: 2%; top: 14%; }
.float-two { right: 0; top: 16%; }
.float-three { left: 14%; bottom: 20%; }
.float-four { right: 0; bottom: 14%; }

/* ---------- cards ---------- */
.content-card {
    background: linear-gradient(180deg, #0d1633, #0a1229);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 20px; height: 100%; min-width: 0;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, #0d1633, #0a1229);
    border: 1px solid var(--border) !important; border-radius: 14px;
}
.card-heading {
    display: flex; align-items: center; gap: 10px; font-size: 15px;
    font-weight: 600; color: #f7f8ff; margin-bottom: 14px;
}
.card-heading .sub { color: var(--muted); font-weight: 400; }
.card-heading .viewall { margin-left: auto; color: var(--cyan); font-size: 13px; font-weight: 500; }
.muted { color: var(--muted); font-size: 13px; line-height: 1.6; }
.category-name { color: #7c8cff; font-size: 30px; font-weight: 800; line-height: 1.2; margin: 6px 0 8px 0; }
.confidence {
    display: inline-block; color: #9fb0ff; border: 1px solid #3a46a8;
    background: #18215a; border-radius: 20px; padding: 7px 16px; font-size: 13px; margin-top: 10px;
}
.section-title { color: #f4f6ff; font-size: clamp(25px, 3vw, 36px); font-weight: 800; letter-spacing: -1px; margin: 12px 0; }
.section-subtitle { color: var(--muted); font-size: 14px; line-height: 1.7; margin-bottom: 22px; }
.eyebrow { color: #58dcff; font-size: 11px; font-weight: 800; letter-spacing: 2px; }

.skill {
    display: inline-block; padding: 7px 15px; margin: 4px 6px 4px 0;
    background: #17224f; border: 1px solid #223070; color: #e4ecff;
    border-radius: 25px; font-size: 13px;
}
.skill-green { background: #0f2a2c; border-color: #17504a; }
.skill-red { background: #2b1530; border-color: #5a2455; }

/* ---------- score ---------- */
.score-layout { display: flex; align-items: center; gap: 26px; flex-wrap: wrap; }
.score-ring { width: 118px; height: 118px; flex-shrink: 0; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
.score-inner { width: 92px; height: 92px; border-radius: 50%; background: #0b1330; display: flex; align-items: center; justify-content: center; font-size: 26px; font-weight: 800; color: white; }
.legend { display: flex; align-items: center; gap: 10px; margin: 9px 0; color: #cbd4e7; font-size: 14px; min-width: 190px; }
.legend b { margin-left: auto; font-weight: 500; }
.dot { width: 11px; height: 11px; display: inline-block; border-radius: 50%; flex-shrink: 0; }
.dot-green { background: #31dbb0; } .dot-red { background: #ef4444; } .dot-yellow { background: #facc15; }

.bar-row { display: flex; align-items: center; gap: 14px; margin-top: 12px; }
.bar { flex: 1; height: 14px; border-radius: 8px; background: #15204a; overflow: hidden; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #5b5bf5, #38bdf8); }
.bar-label { font-size: 18px; font-weight: 700; white-space: nowrap; }

/* ---------- how it works ---------- */
.workflow { display: grid; grid-template-columns: 1fr 18px 1fr 18px 1fr; align-items: center; gap: 6px; }
.workflow-step { position: relative; text-align: center; background: #0b1535; border: 1px solid var(--border); border-radius: 12px; padding: 18px 6px 12px 6px; min-width: 0; }
.workflow-number { position: absolute; top: -11px; left: 50%; transform: translateX(-50%); width: 23px; height: 23px; border-radius: 50%; color: white; font-weight: 700; font-size: 12px; line-height: 23px; background: #4f46e5; }
.workflow-icon { font-size: 28px; margin-bottom: 6px; }
.workflow-title { font-size: 13px; font-weight: 600; color: white; }
.workflow-subtitle { color: var(--muted); font-size: 11px; line-height: 1.4; margin-top: 4px; }
.workflow-arrow { color: #dce6ff; font-size: 20px; text-align: center; }

/* ---------- upload ---------- */
.upload-title { text-align: center; font-size: 20px; font-weight: 700; margin-top: 4px; }
.upload-icon { text-align: center; font-size: 34px; }
[data-testid="stFileUploader"] section { background: #0a1129; border: 1.5px dashed #4f56e0; border-radius: 12px; }
[data-testid="stFileUploader"] button { background: linear-gradient(90deg, #5b5bf5, #8b5cf6); color: white; border: none; border-radius: 20px; }

/* ---------- misc ---------- */
.footer { text-align: center; color: #7787a3; font-size: 12px; padding: 22px 0; margin-top: 25px; border-top: 1px solid var(--border); }
div.stButton > button { border: 1px solid #2b3a58; border-radius: 9px; background: #14213a; color: #e8eeff; min-height: 38px; font-weight: 600; }
div.stButton > button:hover { color: white; border-color: #8371ff; background: #202d4d; }
.stSelectbox > div > div { background: #121c40; border-color: #293955; }

@media (max-width: 900px) {
    .hero { grid-template-columns: 1fr; padding: 23px; }
    .hero-visual { min-height: 260px; }
}
@media (max-width: 560px) {
    .workflow { grid-template-columns: 1fr; }
    .workflow-arrow { transform: rotate(90deg); }
}
</style>
""", unsafe_allow_html=True)


# ---------- light theme overrides ----------
def _important(css: str) -> str:
    """Force every declaration to win over Streamlit's own theme CSS."""
    css = css.replace(" !important", "")
    return re.sub(r"\s*;", " !important;", css)


if st.session_state.theme == "light":
    _light_css = """
<style>
:root { --bg: #f3f6fd; --border: #d6def0; --text: #14203f; --muted: #5b6a8c; --cyan: #0284c7; }
.stApp { background: #f3f6fd; color: #14203f; }
[data-testid="stHeader"] { background: transparent; }
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stExpander"] summary p { color: #14203f; }
.stApp .muted, .stApp .muted * { color: #5b6a8c; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #d6def0; }
[data-testid="stSidebar"] * { color: #14203f; }
.brand-gradient { background: linear-gradient(90deg, #2563eb, #7c3aed); -webkit-background-clip: text; background-clip: text; color: transparent !important; -webkit-text-fill-color: transparent; }
.gradient-text { background: linear-gradient(90deg, #5b3df5, #0ea5e9); -webkit-background-clip: text; background-clip: text; color: transparent !important; -webkit-text-fill-color: transparent; }

[class*="st-key-top_"] button { color: #26324f !important; }
[class*="st-key-top_"] button:hover { color: #000 !important; }
.st-key-""" + TOP_KEY + """ button { color: #0b1330 !important; border-bottom: 2px solid #4f46e5 !important; }
.st-key-theme_toggle button { background: #ffffff !important; border: 1px solid #d6def0 !important; color: #14203f !important; }
[class*="st-key-side_"] button:hover { background: #eaf0ff !important; }
[class*="st-key-side_"] button[kind="primary"],
[class*="st-key-side_"] button[data-testid="stBaseButton-primary"] { background: linear-gradient(90deg, #dfe6ff, #eef1ff) !important; box-shadow: inset 3px 0 0 #4f46e5 !important; }
.side-quote { background: linear-gradient(160deg, #e8edff, #d9ccff); border-color: #d6def0; color: #2b2f66; }

.hero { background: radial-gradient(ellipse at 85% 35%, #d8ccff 0%, transparent 55%), linear-gradient(110deg, #ffffff, #eaf0ff 60%, #e4dcff); border-color: #d6def0; }
.hero h1 { color: #0b1330; }
.hero-lead { color: #14203f; }
.hero-description { color: #4b5a7d; }
.hero-feature { color: #14203f; }
.float-card { background: rgba(255,255,255,.92); border-color: #c9d4f2; color: #14203f; box-shadow: 0 10px 30px #4f46e522; }
.orbit { border-color: rgba(99,102,241,.55); box-shadow: 0 0 25px #6366f122; }
.orbit-two { border-color: rgba(14,165,233,.5); }

.content-card, div[data-testid="stVerticalBlockBorderWrapper"] { background: #ffffff; border: 1px solid #d6def0 !important; box-shadow: 0 4px 18px rgba(30,50,110,.06); }
.card-heading { color: #0b1330; }
.card-heading .sub { color: #5b6a8c; }
.category-name { color: #4f46e5; }
.confidence { background: #eef0ff; border-color: #b9c0f5; color: #4338ca; }
.section-title { color: #0b1330; }
.section-subtitle { color: #5b6a8c; }
.eyebrow { color: #0284c7; }

.skill { background: #eef2ff; border-color: #cdd6f6; color: #1e2a55; }
.skill-green { background: #e3f8f1; border-color: #a7e3cc; color: #0b5a40; }
.skill-red { background: #fdeaf0; border-color: #f4b6c9; color: #8a1d44; }

.score-inner { background: #ffffff; color: #0b1330; }
.legend { color: #26324f; }
.bar { background: #e3e9fa; }
.bar-label { color: #0b1330; }

.workflow-step { background: #f5f7ff; border-color: #d6def0; }
.workflow-title { color: #0b1330; }
.workflow-subtitle { color: #5b6a8c; }
.workflow-arrow { color: #5b6a8c; }

[data-testid="stFileUploader"] section { background: #ffffff; border-color: #7c83f0; }
[data-testid="stFileUploader"] section * { color: #26324f; }
[data-testid="stFileUploader"] button { color: #ffffff; }
.stSelectbox > div > div { background: #ffffff; border-color: #cdd6f6; color: #14203f; }
.stSelectbox [data-baseweb="select"] * { color: #14203f; }
div.stButton > button { background: #ffffff; border-color: #cdd6f6; color: #14203f; }
div.stButton > button:hover { background: #eef2ff; border-color: #6366f1; color: #0b1330; }
.st-key-get_started button { background: linear-gradient(90deg, #5b5bf5, #8b5cf6) !important; color: #fff !important; border: none !important; }
[data-testid="stTextInput"] input { background: #ffffff; color: #14203f; }
.footer { color: #7a88a8; border-color: #d6def0; }
</style>
"""
    st.markdown(_important(_light_css), unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================


def go(page):
    st.session_state.page = page
    st.rerun()


def skill_chips(skills, variant=""):
    cls = f"skill {variant}".strip()
    return "".join(f'<span class="{cls}">{html.escape(str(s))}</span>' for s in skills)


def norm(skill):
    return " ".join(str(skill).casefold().split()).strip()


def job_columns():
    normalized = {
        str(c).strip().lower().replace(" ", "_"): c for c in jobs_df.columns
    }

    def find(cands):
        for c in cands:
            if c in normalized:
                return normalized[c]
        return None

    return {
        "title": find(["job_title", "title", "job", "role", "category", "job_category", "job_role"]),
        "skills": find(["required_skills", "skills", "job_skills", "skills_required", "required_skills_list"]),
        "desc": find(["job_description", "description", "job_summary", "summary"]),
    }


def job_titles():
    if jobs_df.empty:
        return ["Data Scientist", "ML Engineer", "Data Analyst", "Backend Developer"]
    col = job_columns()["title"]
    if col is None:
        return ["Data Scientist"]
    return list(dict.fromkeys(jobs_df[col].dropna().astype(str).tolist())) or ["Data Scientist"]


def compute_match(job_title):
    out = {"required": [], "matched": [], "partial": [], "missing": [], "score": 0, "desc": ""}
    if jobs_df.empty:
        return out
    cols = job_columns()
    if cols["title"] is None or cols["skills"] is None:
        return out

    rows = jobs_df[jobs_df[cols["title"]].astype(str) == job_title]
    if rows.empty:
        return out
    row = rows.iloc[0]

    raw = row[cols["skills"]]
    if pd.isna(raw):
        required = []
    elif isinstance(raw, (list, tuple, set)):
        required = [str(s).strip() for s in raw if str(s).strip()]
    else:
        required = [
            s.strip().strip("'\"[]")
            for s in str(raw).replace(";", ",").split(",")
            if s.strip().strip("'\"[]")
        ]
    required = list(dict.fromkeys(required))

    resume = [norm(s) for s in st.session_state.extracted_skills or []]
    resume_set = set(resume)

    matched = [s for s in required if norm(s) in resume_set]
    partial = [
        s for s in required
        if s not in matched and any(norm(s) in r or r in norm(s) for r in resume if r)
    ]
    missing = [s for s in required if s not in matched and s not in partial]

    out.update(
        required=required, matched=matched, partial=partial, missing=missing,
        score=round(len(matched) / len(required) * 100) if required else 0,
    )
    if cols["desc"] is not None and pd.notna(row[cols["desc"]]):
        out["desc"] = str(row[cols["desc"]])
    return out


@st.cache_data(show_spinner=False)
def load_skill_dictionary():
    """Find the skill dictionary. Returns (dictionary, where_it_came_from)."""
    # 1) something exposed by your own skill_extractor module
    if _skill_module is not None:
        for name in ("SKILL_DICTIONARY", "skill_dictionary", "SKILLS", "SKILL_LIST"):
            value = getattr(_skill_module, name, None)
            if value:
                return value, f"skill_extractor.{name}"
        for name in ("load_skill_dictionary", "get_skill_dictionary",
                     "load_skills", "load_skill_list", "get_skills"):
            fn = getattr(_skill_module, name, None)
            if callable(fn):
                try:
                    value = fn()
                    if value:
                        return value, f"skill_extractor.{name}()"
                except Exception:
                    pass

    # 2) an explicit skill-dictionary JSON file (not resumes_with_skills.csv etc.)
    for folder in (DATA_DIR, MODELS_DIR):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.json")):
            if "skill" not in path.name.lower():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and value:
                    return value, str(path.relative_to(BASE_DIR))
            except Exception:
                continue

    # 3) build {job title: [required skills]} from job_descriptions.csv.
    #    This is exactly the job_skills dictionary used in your notebooks.
    if not jobs_df.empty:
        cols = job_columns()
        if cols["title"] is not None and cols["skills"] is not None:
            built = {}
            for title, raw in zip(jobs_df[cols["title"]].astype(str), jobs_df[cols["skills"]].fillna("").astype(str)):
                skills = [x.strip().strip("'\"[]") for x in raw.replace(";", ",").split(",")]
                skills = [x for x in dict.fromkeys(skills) if x]
                if skills:
                    built[title] = skills
            if built:
                return built, "job_descriptions.csv (required_skills per job)"
    return [], "none found"


def normalize_resume_text(text: str) -> str:
    """PDFs often contain odd spaces, ligatures and line breaks - tidy them up."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", text)


def tolerant_match(text: str, dictionary: dict) -> list:
    """Backup matcher: works for skills with symbols (c++, node.js) and flexible spacing."""
    low = text.lower()
    found = []
    for skills in dictionary.values():
        for skill in skills:
            sk = str(skill).strip().lower()
            if not sk:
                continue
            body = r"[\s\-_/]*".join(re.escape(part) for part in re.split(r"[\s\-_/]+", sk) if part)
            if re.search(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])", low):
                found.append(str(skill))
    return list(dict.fromkeys(found))


def run_extract_skills(text):
    """Call your extract_skills, supplying the skill dictionary if it asks for one."""
    st.session_state.skill_debug = {
        "source": "n/a", "categories": 0, "total": 0,
        "text_len": len(text), "used_backup": False,
    }
    if extract_skills is None:
        return []

    text = normalize_resume_text(text)
    st.session_state.skill_debug["text_len"] = len(text)

    try:
        params = inspect.signature(extract_skills).parameters.values()
        required = [
            p for p in params
            if p.default is p.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        needs_dictionary = len(required) >= 2
    except (TypeError, ValueError):
        needs_dictionary = False

    if not needs_dictionary:
        return extract_skills(text)

    dictionary, source = load_skill_dictionary()
    if not dictionary:
        raise ValueError(
            "no skill dictionary found. Put a skills file (e.g. data/skills.csv or "
            "data/skills.json) in your data folder."
        )

    # your extract_skills loops over dictionary.values(), so it needs {category: [skills]}
    if isinstance(dictionary, dict):
        dictionary = {
            str(k): (list(v) if isinstance(v, (list, tuple, set)) else [str(v)])
            for k, v in dictionary.items()
        }
    else:
        dictionary = {"skills": [str(x) for x in dictionary]}

    result = extract_skills(text, dictionary)

    used_backup = False
    if not result:
        result = tolerant_match(text, dictionary)
        used_backup = bool(result)

    st.session_state.skill_debug.update(
        source=source,
        categories=len(dictionary),
        total=sum(len(v) for v in dictionary.values()),
        used_backup=used_backup,
    )
    return result


# ============================================================
# OPEN-ENDED SKILL EXTRACTION (reads whatever skills the resume lists)
# ============================================================

BROAD_SKILLS = """
Python|Java|JavaScript|TypeScript|C++|C#|Kotlin|Swift|Dart|PHP|Ruby|Rust|Scala|MATLAB|Perl|Bash|Shell Scripting|PowerShell|SQL|NoSQL|HTML|CSS|HTML5|CSS3|Sass|R Programming
Machine Learning|Deep Learning|Artificial Intelligence|AI|ML|NLP|Natural Language Processing|Computer Vision|Neural Networks|CNN|RNN|LSTM|Transformers|BERT|GPT|LLM|Generative AI|Reinforcement Learning|Data Science|Data Mining|Data Analysis|Data Analytics|Data Visualization|Data Cleaning|Data Preprocessing|Data Engineering|Data Wrangling|Feature Engineering|Model Deployment|MLOps|Time Series|Predictive Modeling|Statistics|Statistical Analysis|Probability|Linear Algebra|A/B Testing|Hypothesis Testing|Regression|Classification|Clustering|Random Forest|XGBoost|LightGBM|Decision Trees|SVM
TensorFlow|PyTorch|Keras|Scikit-learn|sklearn|Pandas|NumPy|SciPy|Matplotlib|Seaborn|Plotly|OpenCV|NLTK|spaCy|Hugging Face|LangChain|Streamlit|Flask|Django|FastAPI|Jupyter|Jupyter Notebook|Google Colab|Anaconda|Beautiful Soup|Selenium|Scrapy
Power BI|Tableau|Excel|Microsoft Excel|Advanced Excel|Google Sheets|Looker|Qlik|Alteryx|SAS|SPSS|Google Analytics|Microsoft Office|MS Office|Word|PowerPoint|Outlook
React|React.js|Angular|Vue.js|Vue|Next.js|Node.js|Express.js|Express|jQuery|Bootstrap|Tailwind CSS|Tailwind|Redux|REST API|REST APIs|RESTful APIs|GraphQL|JSON|XML|AJAX|Web Development|Frontend|Backend|Full Stack|Responsive Design|WordPress|Spring Boot|Spring|Hibernate|.NET|ASP.NET|Laravel
MySQL|PostgreSQL|MongoDB|Oracle|SQLite|SQL Server|MS SQL|Redis|Cassandra|DynamoDB|Firebase|Elasticsearch|Snowflake|BigQuery|Hadoop|Spark|PySpark|Kafka|Airflow|ETL|Data Warehousing|Database Management|Database Security|Backup|Hive|Databricks
AWS|Azure|GCP|Google Cloud|Cloud Computing|Cloud Security|EC2|S3|Lambda|Docker|Kubernetes|Terraform|Ansible|Jenkins|CI/CD|GitHub Actions|GitLab|Linux|Unix|Ubuntu|Windows Server|DevOps|Nginx|Apache|Microservices|Serverless|Networking|TCP/IP|DNS|VPN|Firewall
Git|GitHub|Bitbucket|SVN|Jira|Confluence|Trello|Postman|Swagger|VS Code|Visual Studio|IntelliJ|Eclipse|PyCharm|Figma|Canva|Photoshop|Adobe XD
Android|iOS|Flutter|React Native|Mobile Development|Android Studio|Xcode
Manual Testing|Test Automation|Automation Testing|API Testing|Software Testing|JUnit|TestNG|Pytest|Cypress|Appium|Unit Testing|Regression Testing|Performance Testing|JMeter|Test Cases|Bug Tracking|QA
Cybersecurity|Network Security|Ethical Hacking|Penetration Testing|Cryptography|SIEM|Vulnerability Assessment|Wireshark|Nmap|Burp Suite|Metasploit|OWASP|Information Security|Incident Response|Splunk|Kali Linux
Data Structures|Algorithms|OOP|Object-Oriented Programming|Design Patterns|System Design|Operating Systems|Computer Networks|DBMS|Software Development|SDLC|Agile|Scrum|Kanban|Waterfall|Requirements Analysis|Requirement Gathering|Business Analysis|Documentation|UML|Project Management|Product Management
Communication|Teamwork|Leadership|Problem Solving|Critical Thinking|Time Management|Adaptability|Creativity|Collaboration|Presentation|Public Speaking|Decision Making|Analytical Skills|Research|Technical Writing|Mentoring|Negotiation|Customer Service|Stakeholder Management
""".replace("\n", "|").split("|")
BROAD_SKILLS = [x.strip() for x in dict.fromkeys(BROAD_SKILLS) if x.strip()]

SKILL_HEADINGS = {
    "skills", "technical skills", "key skills", "core skills", "core competencies",
    "skills & tools", "skills and tools", "tools & technologies", "tools and technologies",
    "technologies", "tech stack", "technical proficiency", "technical expertise",
    "programming languages", "tools", "competencies", "areas of expertise", "expertise",
    "soft skills", "software skills", "it skills", "skill set", "skills summary",
    "technical competencies", "professional skills", "other skills", "frameworks",
}
STOP_HEADINGS = {
    "education", "experience", "work experience", "professional experience", "projects",
    "academic projects", "personal projects", "certifications", "certificates", "achievements",
    "summary", "professional summary", "objective", "career objective", "profile", "internships",
    "internship", "awards", "publications", "interests", "hobbies", "references", "declaration",
    "personal details", "personal information", "languages", "extracurricular activities",
    "responsibilities", "employment history", "academic details", "courses", "training",
}
NOISE_ITEMS = {
    "and", "etc", "others", "other", "basic", "advanced", "intermediate", "proficient", "good",
    "knowledge", "familiar", "experience", "skills", "tools", "technologies", "languages",
    "expert", "beginner", "working", "strong", "excellent", "various", "etc.",
}


def _skill_key(skill: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", skill.lower())


def _heading_text(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z&/ ]", " ", line.lower())).strip()


def _squash(text: str) -> str:
    return re.sub(r"[^a-z&/]", "", text.lower())


_SKILL_HEADINGS_SQ = {_squash(h) for h in SKILL_HEADINGS}
_STOP_HEADINGS_SQ = {_squash(h) for h in STOP_HEADINGS}


def _split_items(chunk: str) -> list:
    chunk = re.sub(r"[()\[\]{}]", ",", chunk)
    if ":" in chunk:
        head, tail = chunk.split(":", 1)
        if len(head.split()) <= 4:
            chunk = tail
    parts = re.split(r"[,;|•·●▪■◦‣⁃]|\s{2,}|\s[–—-]\s", chunk)
    items = []
    for part in parts:
        item = part.strip(" .:-•*\t")
        if not (2 <= len(item) <= 35) or not re.search(r"[A-Za-z]", item):
            continue
        if len(item.split()) > 4 or item.lower() in NOISE_ITEMS:
            continue
        items.append(item)
    return items


def parse_skill_sections(raw: str) -> list:
    """Pull items out of the resume's own Skills / Technical Skills sections."""
    lines = [ln.strip() for ln in raw.replace("\r", "\n").split("\n")]
    items, in_section, taken = [], False, 0
    for line in lines:
        if not line:
            continue
        heading = _squash(line)
        if heading in _SKILL_HEADINGS_SQ:
            in_section, taken = True, 0
            continue
        inline = re.match(r"^([A-Za-z &/]{3,40}?)\s*[:\-–]\s*(.+)$", line)
        if inline and _squash(inline.group(1)) in _SKILL_HEADINGS_SQ:
            in_section, taken = True, 0
            items += _split_items(inline.group(2))
            continue
        if in_section:
            if heading in _STOP_HEADINGS_SQ or taken > 40:
                in_section = False
                continue
            items += _split_items(line)
            taken += 1
    return items


def vocab_match(text: str) -> list:
    """Match the broad skill vocabulary anywhere in the resume text."""
    low = text.lower()
    hits = []
    for skill in BROAD_SKILLS:
        if len(skill) <= 2:  # AI / ML: case-sensitive to avoid noise
            m = re.search(r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])", text)
        else:
            body = r"[\s\-_/]*".join(re.escape(x) for x in re.split(r"[\s\-_/]+", skill.lower()) if x)
            m = re.search(r"(?<![a-z0-9])" + body + r"(?![a-z0-9+#])" if skill[-1] not in "+#" else
                          r"(?<![a-z0-9])" + body + r"(?![a-z0-9])", low)
        if m:
            hits.append((m.start(), skill))
    return [s for _, s in sorted(hits)]


def compact_match(text: str) -> list:
    """PDF parsers often insert stray spaces inside words ("Pyth on", "Tensor Flow").
    Compare with ALL whitespace removed so those skills are still found."""
    squashed = re.sub(r"[\s\-_/.]+", "", text.lower())
    hits = []
    for skill in BROAD_SKILLS:
        key = re.sub(r"[\s\-_/.]+", "", skill.lower())
        if len(key) < 5:  # short names (AI, SQL, Git...) are too risky without spaces
            continue
        pos = squashed.find(key)
        if pos >= 0:
            hits.append((pos, skill, key))
    # drop skills that are only part of a longer match (Java inside JavaScript)
    keys = [k for _, _, k in hits]
    return [sk for _, sk, k in sorted(hits) if not any(k != o and k in o for o in keys)]


def smart_extract_skills(raw_text: str) -> list:
    """Skills listed in the resume's skills section + every known skill found in the text."""
    normalized = normalize_resume_text(raw_text)

    base = []
    try:
        base = list(run_extract_skills(raw_text) or [])
    except Exception:
        base = []

    vocab = list(dict.fromkeys(vocab_match(normalized) + compact_match(normalized)))
    section = parse_skill_sections(raw_text)

    canonical = {_skill_key(sk): sk for sk in BROAD_SKILLS + base}
    merged, seen = [], set()

    def add(skill):
        key = _skill_key(skill)
        if not key or key in seen:
            return
        seen.add(key)
        label = canonical.get(key) or (skill.title() if skill.islower() else skill)
        merged.append(label)

    for item in section:      # what the candidate listed under Skills
        add(item)
    low = normalized.lower()
    others = sorted(set(base + vocab), key=lambda x: low.find(x.lower()) if low.find(x.lower()) >= 0 else 10**9)
    for item in others:       # everything else recognised in the resume
        add(item)

    dbg = st.session_state.get("skill_debug") or {}
    dbg.update(from_dictionary=len(base), from_vocabulary=len(vocab), from_section=len(section))
    st.session_state.skill_debug = dbg
    return merged



def analyze_resume(upload):
    if extract_resume_text is None:
        st.error("Resume parser is unavailable. Check your src module.")
        return

    temp_path = None
    try:
        suffix = Path(upload.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(upload.getvalue())
            temp_path = Path(temp.name)

        text = str(extract_resume_text(str(temp_path)) or "")
        if not text.strip():
            st.error("No readable text was extracted from the resume.")
            return

        cleaned = text
        if clean_text:
            try:
                cleaned = str(clean_text(text) or text)
            except Exception:
                cleaned = text

        def _as_list(result):
            if isinstance(result, dict):
                return list(result.keys())
            if isinstance(result, (list, tuple, set)):
                return list(result)
            return [str(result)] if result is not None else []

        skills = []
        # 1) your original code: extract_skills(cleaned)
        #    (your skill_extractor now also wants a skill_dictionary, so pass it)
        if extract_skills is not None:
            try:
                try:
                    result = extract_skills(cleaned)
                except TypeError:
                    dictionary, _src = load_skill_dictionary()
                    if isinstance(dictionary, dict):
                        dictionary = {
                            str(k): (list(v) if isinstance(v, (list, tuple, set)) else [str(v)])
                            for k, v in dictionary.items()
                        }
                    else:
                        dictionary = {"skills": [str(x) for x in dictionary]}
                    result = extract_skills(cleaned, dictionary)
                skills = _as_list(result)
            except Exception as exc:
                st.warning(f"Skill extraction error: {exc}")
        # 2) only if that finds nothing, fall back to the broader matchers
        if not skills:
            try:
                skills = _as_list(smart_extract_skills(text))
            except Exception as exc:
                st.warning(f"Fallback skill extraction error: {exc}")
        if not skills:
            st.warning("No skills were detected in this resume.")

        category, confidence = "Not predicted", None
        if model is not None and vectorizer is not None:
            try:
                matrix = vectorizer.transform([cleaned])
                prediction = model.predict(matrix)
                if len(prediction):
                    category = str(prediction[0])
                if hasattr(model, "predict_proba"):
                    confidence = round(float(max(model.predict_proba(matrix)[0])) * 100)
            except Exception as exc:
                st.warning(f"Prediction error: {exc}")

        st.session_state.resume_text = text
        st.session_state.cleaned_text = cleaned
        st.session_state.resume_name = upload.name
        st.session_state.extracted_skills = skills
        st.session_state.predicted_category = category
        st.session_state.analysis_complete = True
        st.session_state.confidence = confidence
        st.session_state.history.append({
            "time": datetime.now().strftime("%d %b %Y, %H:%M"),
            "file": upload.name,
            "category": category,
            "skills": len(skills),
        })
        st.success("Resume analyzed successfully.")

    except Exception as exc:
        st.error(f"Resume analysis failed: {exc}")
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


SKILL_LOGIC_VERSION = "v5"  # bump this whenever you change skill extraction


def handle_upload(upload):
    if upload is None:
        return
    signature = hashlib.md5(upload.getvalue()).hexdigest() + SKILL_LOGIC_VERSION
    if st.session_state.get("resume_sig") != signature:
        st.session_state.resume_sig = signature
        analyze_resume(upload)


# ============================================================
# TOP NAVIGATION
# ============================================================

brand_col, nav_col, theme_col, action_col = st.columns(
    [1.7, 5.0, 1.15, 1.3], vertical_alignment="center"
)

with brand_col:
    st.markdown(
        '<div class="brand">🧠 Neura<span class="brand-gradient">Nex</span></div>',
        unsafe_allow_html=True,
    )

with nav_col:
    n1, n2, n3, n4, n5 = st.columns(5)
    for col, label, key, target in [
        (n1, "Home", "top_home", "Dashboard"),
        (n2, "About", "top_about", "About"),
        (n3, "How It Works", "top_how", "How It Works"),
        (n4, "Features", "top_features", "Features"),
        (n5, "Contact", "top_contact", "Contact"),
    ]:
        with col:
            if st.button(label, key=key, use_container_width=True):
                go(target)

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


with theme_col:
    st.button(
        "☾ Dark" if st.session_state.theme == "light" else "☼ Light",
        key="theme_toggle",
        help="Switch to dark mode" if st.session_state.theme == "light" else "Switch to light mode",
        on_click=toggle_theme,
        use_container_width=True,
    )

with action_col:
    if st.button("Get Started", key="get_started", use_container_width=True):
        go("Upload Resume")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    side_pages = [
        ("🏠", "Dashboard"),
        ("📄", "Upload Resume"),
        ("◎", "Job Matching"),
        ("♧", "Skill Analysis"),
        ("▤", "Job Categories"),
        ("✧", "Recommendations"),
        ("◷", "History"),
        ("♙", "Profile"),
    ]
    for icon, name in side_pages:
        if st.button(
            f"{icon}   {name}",
            key=f"side_{name}",
            use_container_width=True,
            type="primary" if st.session_state.page == name else "secondary",
        ):
            go(name)

    st.markdown(
        '<div class="side-quote">“A smarter resume today, a brighter career tomorrow.”</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <div>
            <h1>AI-Powered<br>
            <span class="gradient-text">Resume Screening</span><br>
            &amp; Job Matching</h1>
            <div class="hero-lead">Turn your resume into your next career opportunity.</div>
            <div class="hero-description">
                Get AI-driven job recommendations, skill insights and
                personalized career guidance in seconds.
            </div>
            <div class="hero-features">
                <div class="hero-feature"><span class="hero-feature-icon">ϟ</span><span>AI-Powered<br>Analysis</span></div>
                <div class="hero-feature"><span class="hero-feature-icon">◎</span><span>Accurate Job<br>Matching</span></div>
                <div class="hero-feature"><span class="hero-feature-icon">▥</span><span>Personalized<br>Skill Recommendations</span></div>
            </div>
        </div>
        <div class="hero-visual">
            <div class="orbit"></div>
            <div class="orbit-two"></div>
            <div class="resume-illustration">
                <div class="resume-label">RESUME</div>
                <div class="resume-avatar"></div>
                <div class="resume-line"></div>
                <div class="resume-line medium"></div>
                <div class="resume-line"></div>
                <div class="resume-line short"></div>
                <div class="resume-line"></div>
                <div class="resume-line medium"></div>
            </div>
            <div class="float-card float-one"><span class="fc-icon">🧠</span><span>Skills<br>Extraction</span></div>
            <div class="float-card float-two"><span class="fc-icon">▤</span><span>Job Category<br>Prediction</span></div>
            <div class="float-card float-three"><span class="fc-icon">◎</span><span>Match Score</span></div>
            <div class="float-card float-four"><span class="fc-icon">✦</span><span>Career<br>Recommendations</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- upload + how it works ----------
    up_col, how_col = st.columns(2, gap="medium")

    with up_col:
        st.markdown("""
        <div class="upload-icon">☁️</div>
        <div class="upload-title">Upload Your Resume</div>
        <div class="muted" style="text-align:center;margin-bottom:10px;">
            Drag and drop your file here or click to browse<br>Supports PDF and DOCX files
        </div>
        """, unsafe_allow_html=True)
        dash_upload = st.file_uploader(
            "Upload resume", type=["pdf", "docx"],
            key="dash_uploader", label_visibility="collapsed",
        )
        handle_upload(dash_upload)

    with how_col:
        st.markdown("""
        <div class="content-card">
            <div class="card-heading">How It Works?</div>
            <div class="workflow">
                <div class="workflow-step"><span class="workflow-number">1</span>
                    <div class="workflow-icon">📄</div>
                    <div class="workflow-title">Upload Resume</div>
                    <div class="workflow-subtitle">PDF/DOCX</div></div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-step"><span class="workflow-number" style="background:#2563eb">2</span>
                    <div class="workflow-icon">🧠</div>
                    <div class="workflow-title">AI Analysis</div>
                    <div class="workflow-subtitle">Extract skills &amp; predict job category</div></div>
                <div class="workflow-arrow">→</div>
                <div class="workflow-step"><span class="workflow-number" style="background:#14b8a6">3</span>
                    <div class="workflow-icon">🎯</div>
                    <div class="workflow-title">Get Results</div>
                    <div class="workflow-subtitle">Match score, missing skills &amp; recommendations</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ---------- compute results (after any upload so they are fresh) ----------
    titles = job_titles()
    chosen = st.session_state.get("dash_job")
    if chosen not in titles:
        chosen = st.session_state.selected_job if st.session_state.selected_job in titles else titles[0]
    match = compute_match(chosen)

    category = st.session_state.predicted_category or "—"
    skills_top = (st.session_state.extracted_skills or [])[:10]
    conf = st.session_state.confidence

    r1, r2, r3 = st.columns([1, 1.25, 1.1], gap="medium")

    with r1:
        conf_html = f'<span class="confidence">Confidence: {conf}%</span>' if conf is not None else ""
        note = (
            "Based on your skills and experience, your profile best matches this role."
            if st.session_state.analysis_complete
            else "Upload a resume to see your predicted job category."
        )
        st.markdown(f"""
        <div class="content-card">
            <div class="card-heading">🎯 Predicted Job Category</div>
            <div class="category-name">{html.escape(category)}</div>
            <div class="muted">{note}</div>
            {conf_html}
        </div>
        """, unsafe_allow_html=True)

    with r2:
        body = skill_chips(skills_top) if skills_top else '<span class="muted">Upload a resume to display extracted skills.</span>'
        st.markdown(f"""
        <div class="content-card">
            <div class="card-heading">⚙️ Extracted Skills <span class="sub">(Top 10)</span><span class="viewall">View All</span></div>
            {body}
        </div>
        """, unsafe_allow_html=True)

    with r3:
        s = match["score"]
        ring = f"conic-gradient(#a05aff 0%, #4b8eff {s * 0.5}%, #31dbb0 {s}%, #243050 {s}% 100%)"
        st.markdown(f"""
        <div class="content-card">
            <div class="card-heading">📊 Overall Match Score</div>
            <div class="score-layout">
                <div class="score-ring" style="background:{ring}"><div class="score-inner">{s}%</div></div>
                <div>
                    <div class="legend"><span class="dot dot-green"></span>Matching Skills<b>{len(match['matched'])}</b></div>
                    <div class="legend"><span class="dot dot-red"></span>Missing Skills<b>{len(match['missing'])}</b></div>
                    <div class="legend"><span class="dot dot-yellow"></span>Partial Match<b>{len(match['partial'])}</b></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ---------- job matching row ----------
    b1, b2, b3, b4 = st.columns([1.25, 1, 1, 1], gap="medium")

    with b1:
        with st.container(border=True):
            st.markdown(
                '<div class="card-heading">💼 Job Matching</div>'
                '<div class="muted">Select a job to see how well your skills match.</div>',
                unsafe_allow_html=True,
            )
            picked = st.selectbox(
                "Job", titles, index=titles.index(chosen),
                key="dash_job", label_visibility="collapsed",
            )
            st.session_state.selected_job = picked
            st.markdown(
                f'<div class="bar-row"><div class="bar"><div class="bar-fill" style="width:{s}%"></div></div>'
                f'<div class="bar-label">{s}% Match</div></div>',
                unsafe_allow_html=True,
            )

    def empty_hint(items, variant):
        return skill_chips(items, variant) if items else '<span class="muted">Nothing to show yet.</span>'

    with b2:
        st.markdown(
            f'<div class="content-card"><div class="card-heading">✅ Matching Skills</div>'
            f'{empty_hint(match["matched"], "skill-green")}</div>',
            unsafe_allow_html=True,
        )
    with b3:
        st.markdown(
            f'<div class="content-card"><div class="card-heading">❌ Missing Skills</div>'
            f'{empty_hint(match["missing"], "skill-red")}</div>',
            unsafe_allow_html=True,
        )
    with b4:
        st.markdown(
            f'<div class="content-card"><div class="card-heading">💡 Recommended Skills</div>'
            f'{empty_hint(match["missing"][:6], "")}</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# UPLOAD RESUME
# ============================================================

elif st.session_state.page == "Upload Resume":

    st.markdown(
        '<div class="eyebrow">NEURANEX / UPLOAD</div>'
        '<div class="section-title">Upload Resume</div>'
        '<div class="section-subtitle">Upload a PDF or DOCX resume to extract skills and predict your job category.</div>',
        unsafe_allow_html=True,
    )

    page_upload = st.file_uploader(
        "Upload resume", type=["pdf", "docx"],
        key="page_uploader", label_visibility="collapsed",
    )
    handle_upload(page_upload)

    if st.session_state.analysis_complete:
        st.markdown(
            f'<div class="content-card"><div class="card-heading">📄 {html.escape(st.session_state.resume_name)}</div>'
            f'<div class="muted">Predicted category: <b>{html.escape(st.session_state.predicted_category)}</b> · '
            f'{len(st.session_state.extracted_skills)} skills extracted</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("View dashboard", key="upload_to_dash"):
            go("Dashboard")

# ============================================================
# JOB MATCHING
# ============================================================

elif st.session_state.page == "Job Matching":

    st.markdown(
        '<div class="eyebrow">NEURANEX / CAREER MATCHING</div>'
        '<div class="section-title">Job Matching</div>'
        '<div class="section-subtitle">Compare your resume skills with job requirements and identify the skills you need to develop.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.analysis_complete:
        st.info("Upload and analyze your resume before calculating job matches.")
        if st.button("Go to Upload Resume", key="matching_upload_button"):
            go("Upload Resume")
    elif jobs_df.empty:
        st.error("Job data could not be loaded. Check data/raw/job_descriptions.csv.")
    else:
        cols = job_columns()
        if cols["title"] is None or cols["skills"] is None:
            st.error(f"Could not identify job-title / skills columns. Available: {', '.join(map(str, jobs_df.columns))}")
        else:
            titles = job_titles()
            prev = st.session_state.selected_job
            selected = st.selectbox(
                "Select a job role", titles,
                index=titles.index(prev) if prev in titles else 0,
                key="matching_job_selector",
            )
            st.session_state.selected_job = selected
            m = compute_match(selected)
            desc = m["desc"] or "Build your skills for this role by comparing your profile with its requirements."

            st.markdown(
                f'<div class="content-card"><div class="card-heading">💼 Selected Job</div>'
                f'<div class="category-name">{html.escape(selected)}</div>'
                f'<p class="muted">{html.escape(desc)}</p></div>',
                unsafe_allow_html=True,
            )
            st.write("")

            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.markdown(
                    f'<div class="content-card"><div class="card-heading">🎯 Job Match Score</div>'
                    f'<div class="category-name">{m["score"]}%</div>'
                    f'<p class="muted">{len(m["matched"])} of {len(m["required"])} listed skills matched.</p></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                body = skill_chips(m["matched"], "skill-green") or '<span class="muted">No matching skills.</span>'
                st.markdown(f'<div class="content-card"><div class="card-heading">🟢 Matching Skills</div>{body}</div>', unsafe_allow_html=True)
            with c3:
                body = skill_chips(m["missing"], "skill-red") or '<span class="muted">No missing skills.</span>'
                st.markdown(f'<div class="content-card"><div class="card-heading">❌ Missing Skills</div>{body}</div>', unsafe_allow_html=True)

            if not m["required"]:
                st.warning("No required skills were found for this job in your CSV. Check the format of its skills field.")

# ============================================================
# SKILL ANALYSIS
# ============================================================

elif st.session_state.page == "Skill Analysis":

    st.markdown(
        '<div class="eyebrow">NEURANEX / SKILLS</div>'
        '<div class="section-title">Skill Analysis</div>'
        '<div class="section-subtitle">Skills detected in your resume.</div>',
        unsafe_allow_html=True,
    )
    skills = st.session_state.extracted_skills
    if skills:
        st.markdown(
            f'<div class="content-card"><div class="card-heading">⚙️ Extracted Skills <span class="sub">({len(skills)})</span></div>'
            f'{skill_chips(skills)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload a resume to see your extracted skills.")

# ============================================================
# JOB CATEGORIES
# ============================================================

elif st.session_state.page == "Job Categories":

    st.markdown('<div class="section-title">Job Categories</div>', unsafe_allow_html=True)

    if st.session_state.predicted_category:
        conf = st.session_state.confidence
        conf_html = f'<span class="confidence">Confidence: {conf}%</span>' if conf is not None else ""
        st.markdown(
            f'<div class="content-card"><div class="card-heading">🎯 Predicted Category</div>'
            f'<div class="category-name">{html.escape(st.session_state.predicted_category)}</div>{conf_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Analyze a resume to get a category prediction.")

    if not jobs_df.empty and job_columns()["title"] is not None:
        st.write("")
        st.markdown(
            f'<div class="content-card"><div class="card-heading">▤ Available Job Roles</div>{skill_chips(job_titles()[:40])}</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# RECOMMENDATIONS
# ============================================================

elif st.session_state.page == "Recommendations":

    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_complete:
        st.info("Analyze a resume to get skill recommendations.")
    else:
        m = compute_match(st.session_state.selected_job)
        body = skill_chips(m["missing"], "skill-red") or '<span class="muted">No gaps found for this role.</span>'
        st.markdown(
            f'<div class="content-card"><div class="card-heading">💡 Skills to develop for {html.escape(st.session_state.selected_job)}</div>{body}</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# HISTORY
# ============================================================

elif st.session_state.page == "History":

    st.markdown('<div class="section-title">History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.dataframe(
            pd.DataFrame(st.session_state.history[::-1]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No resume has been analyzed in this session.")

# ============================================================
# PROFILE — LOGIN AND REGISTER
# ============================================================

elif st.session_state.page == "Profile":

    st.markdown(
        '<div class="eyebrow">NEURANEX / YOUR ACCOUNT</div>'
        '<div class="section-title">Your Profile</div>'
        '<div class="section-subtitle">Sign in or create an account to access your career workspace.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.is_logged_in:
        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            with st.form("profile_login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                login_submit = st.form_submit_button("Login", use_container_width=True)

            if login_submit:
                key = login_username.strip().lower()
                user = load_users().get(key)
                if user and verify_password(login_password, user["salt"], user["hash"]):
                    st.session_state.is_logged_in = True
                    st.session_state.username = user["display"]
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        with register_tab:
            with st.form("profile_register_form"):
                new_username = st.text_input("Choose username")
                email = st.text_input("Email address")
                new_password = st.text_input("Create password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                register_submit = st.form_submit_button("Create Account", use_container_width=True)

            if register_submit:
                new_username, email = new_username.strip(), email.strip()
                users = load_users()
                if not new_username:
                    st.error("Enter a username.")
                elif "@" not in email or "." not in email.split("@")[-1]:
                    st.error("Enter a valid email address.")
                elif len(new_password) < 8:
                    st.error("Use a password with at least 8 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif new_username.lower() in users:
                    st.error("That username is already registered.")
                else:
                    salt, digest = hash_password(new_password)
                    users[new_username.lower()] = {
                        "display": new_username,
                        "email": email,
                        "salt": salt,
                        "hash": digest,
                    }
                    if save_users(users):
                        st.session_state.is_logged_in = True
                        st.session_state.username = new_username
                        st.rerun()
                    else:
                        st.error("Could not save your account. Check write permission for the data folder.")
    else:
        st.markdown(
            f'<div class="content-card"><div class="eyebrow">SIGNED IN</div>'
            f'<h2>Welcome, {html.escape(st.session_state.username)}</h2>'
            f'<p class="muted">Your career workspace is ready.</p></div>',
            unsafe_allow_html=True,
        )
        st.write("Resume:", st.session_state.resume_name or "Not uploaded")
        st.write("Predicted category:", st.session_state.predicted_category or "Not available")
        if st.button("Logout", use_container_width=True):
            st.session_state.is_logged_in = False
            st.session_state.username = ""
            st.rerun()

# ============================================================
# STATIC PAGES
# ============================================================

elif st.session_state.page == "About":
    st.markdown('<div class="section-title">About NeuraNex</div>', unsafe_allow_html=True)
    st.write("NeuraNex is an AI-powered resume screening and job matching project.")

elif st.session_state.page == "How It Works":
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.write("1. Upload your resume.")
    st.write("2. Extract and preprocess the resume text.")
    st.write("3. Extract skills and predict the job category.")
    st.write("4. Compare the profile against job requirements.")

elif st.session_state.page == "Features":
    st.markdown('<div class="section-title">Features</div>', unsafe_allow_html=True)
    for item in ["Resume parsing", "Skill extraction", "ML-based job-category prediction",
                 "Job dataset exploration", "Career recommendations"]:
        st.write(f"- {item}")

elif st.session_state.page == "Contact":
    st.markdown('<div class="section-title">Contact</div>', unsafe_allow_html=True)
    st.write("NeuraNex — AI Resume Screening & Job Matching.")

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">NeuraNex · AI Resume Screening &amp; Job Matching</div>',
    unsafe_allow_html=True,
)
