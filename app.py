"""
app.py — NLP Autofill App (Improved)
Streamlit UI for the N-gram autocomplete system.
"""

import streamlit as st
from model import build_bigram_model, build_trigram_model, load_corpus, predict_with_prefix

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NLP Autofill",
    layout="centered",
    page_icon="🔍"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header Logo ── */
.nlp-logo {
    font-size: 64px;
    font-weight: 700;
    text-align: center;
    letter-spacing: -2px;
    margin-bottom: 4px;
    line-height: 1;
}
.subtitle {
    text-align: center;
    font-size: 14px;
    color: #9aa0a6;
    margin-bottom: 32px;
    letter-spacing: 0.3px;
}

/* ── Search Box ── */
div[data-baseweb="input"] {
    border-radius: 28px !important;
    border: 1px solid #dfe1e5 !important;
    box-shadow: 0 2px 8px rgba(32,33,36,.18) !important;
    padding: 4px 16px !important;
    transition: box-shadow 0.2s ease;
}
div[data-baseweb="input"]:focus-within {
    box-shadow: 0 4px 16px rgba(32,33,36,.28) !important;
}
div[data-baseweb="input"] > input {
    font-size: 20px !important;
    padding: 10px 4px !important;
}

/* ── Suggestions Dropdown ── */
.suggestion-box {
    background: #fff;
    border: 1px solid #dfe1e5;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(32,33,36,.18);
    overflow: hidden;
    margin-top: 8px;
}
.suggestion-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 11px 20px;
    font-size: 16px;
    color: #202124;
    border-bottom: 1px solid #f1f3f4;
    cursor: pointer;
    transition: background 0.15s;
}
.suggestion-item:last-child { border-bottom: none; }
.suggestion-item:hover { background: #f8f9fa; }
.sug-left { display: flex; align-items: center; gap: 12px; }
.sug-icon { color: #9aa0a6; font-size: 15px; }
.sug-text { font-size: 16px; }
.sug-badge {
    font-size: 11px;
    font-weight: 600;
    background: #e8f0fe;
    color: #1a73e8;
    padding: 2px 8px;
    border-radius: 10px;
    letter-spacing: 0.3px;
}
.no-result {
    text-align: center;
    color: #9aa0a6;
    font-size: 14px;
    padding: 20px;
}
.mode-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 8px;
    letter-spacing: 0.5px;
    margin-left: 6px;
    vertical-align: middle;
}
.bigram-badge  { background: #fce8e6; color: #d93025; }
.trigram-badge { background: #e6f4ea; color: #1e8e3e; }

/* ── Buttons ── */
div.stButton > button {
    display: flex;
    align-items: center;
    width: 100%;
    text-align: left;
    font-size: 16px !important;
    padding: 11px 20px !important;
    border: none !important;
    border-radius: 0 !important;
    background-color: transparent !important;
    color: #202124 !important;
    border-bottom: 1px solid #f1f3f4 !important;
    transition: background 0.15s;
}
div.stButton > button:hover {
    background-color: #f8f9fa !important;
}

/* ── Stats Bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 32px;
    font-size: 13px;
    color: #9aa0a6;
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #f1f3f4;
}
.stats-bar span b { color: #5f6368; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Model Loading  (cached — runs only once)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    tokens = load_corpus()
    bigram  = build_bigram_model(tokens)
    trigram = build_trigram_model(tokens)
    return bigram, trigram, len(tokens)

with st.spinner("⚙️ Loading language model… (first launch only)"):
    bigram_model, trigram_model, vocab_size = load_models()

# ─────────────────────────────────────────
# Session State — keeps input across reruns
# ─────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""

def apply_suggestion(full_text: str):
    """Called when a suggestion button is clicked."""
    st.session_state.query = full_text + " "

# ─────────────────────────────────────────
# Header Logo
# ─────────────────────────────────────────
st.markdown("""
<div class="nlp-logo">
    <span style="color:#4285F4">N</span><span style="color:#EA4335">L</span><span style="color:#FBBC05">P</span>
    <span style="color:#4285F4"> </span>
    <span style="color:#4285F4">S</span><span style="color:#34A853">e</span><span style="color:#EA4335">a</span><span style="color:#FBBC05">r</span><span style="color:#4285F4">c</span><span style="color:#34A853">h</span>
</div>
<p class="subtitle">N-Gram Autocomplete · Powered by NLTK Brown Corpus</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Search Input
# ─────────────────────────────────────────
user_input = st.text_input(
    label="search",
    label_visibility="hidden",
    placeholder="Type a word or phrase…",
    key="query"
)

# ─────────────────────────────────────────
# Suggestions
# ─────────────────────────────────────────
if user_input and user_input.strip():
    words = user_input.strip().split()

    # Detect which model is active for the badge
    using_trigram = (
        len(words) >= 2
        and (words[-2], words[-1]) in trigram_model
    )
    model_label = (
        '<span class="mode-badge trigram-badge">TRIGRAM</span>'
        if using_trigram else
        '<span class="mode-badge bigram-badge">BIGRAM</span>'
    )

    suggestions = predict_with_prefix(bigram_model, trigram_model, user_input, top_n=6)

    if suggestions:
        st.markdown(
            f'<p style="font-size:12px;color:#9aa0a6;margin:12px 0 4px 4px;">'
            f'Suggestions {model_label}</p>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)

        for word, confidence in suggestions:
            # Build the full suggested text
            if user_input.endswith(" "):
                full_text = user_input.strip() + " " + word
            else:
                # Replace the partial last word with the completed word
                parts = user_input.split()
                parts[-1] = word
                full_text = " ".join(parts)

            st.button(
                f"🔍  {full_text}   ({confidence}%)",
                key=f"btn_{word}_{confidence}",
                on_click=apply_suggestion,
                args=(full_text,)
            )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="suggestion-box">'
            '<p class="no-result">🤔 No suggestions found — try a different word</p>'
            '</div>',
            unsafe_allow_html=True
        )

    # ─────────────────────────────────────
    # Confidence Bar Chart
    # ─────────────────────────────────────
    if suggestions:
        with st.expander("📊 Prediction confidence", expanded=False):
            import pandas as pd
            df = pd.DataFrame(suggestions, columns=["Word", "Confidence (%)"])
            df = df.set_index("Word")
            st.bar_chart(df, color="#4285F4")

# ─────────────────────────────────────────
# Footer Stats
# ─────────────────────────────────────────
st.markdown(f"""
<div class="stats-bar">
    <span>📚 Corpus: <b>Brown</b></span>
    <span>🔤 Tokens: <b>{vocab_size:,}</b></span>
    <span>🧠 Model: <b>Bigram + Trigram</b></span>
</div>
""", unsafe_allow_html=True)
