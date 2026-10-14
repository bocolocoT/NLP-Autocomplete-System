import streamlit as st
import re
import nltk
from nltk.corpus import brown
from nltk.util import ngrams
from collections import Counter


# ==========================================
# Module 1: N-Gram (Bigram) Probability Model
# ==========================================
def build_bigram_model(tokens):
    bigrams = list(ngrams(tokens, 2))
    model = {}
    for w1, w2 in bigrams:
        if w1 not in model:
            model[w1] = []
        model[w1].append(w2)
    
    return model

def predict_next_words(model, word, top_n=5):
    word = word.lower().strip()
    if word in model:
        return [w for w, _ in Counter(model[word]).most_common(top_n)]
    return []

# ==========================================
# Module 3: Streamlit User Interface (Google Style)
# ==========================================
st.set_page_config(page_title="Google NLP Autofill", layout="centered", page_icon="🔍")

# --- Custom CSS for Google Style ---
st.markdown("""
    <style>
    /* تصميم شعار جوجل بألوانه */
    .google-logo {
        font-family: 'Arial', sans-serif;
        font-size: 70px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .g-blue { color: #4285F4; }
    .g-red { color: #EA4335; }
    .g-yellow { color: #FBBC05; }
    .g-green { color: #34A853; }

    /* تكبير مربع البحث وتدوير حوافه زي جوجل */
    div[data-baseweb="input"] {
        border-radius: 30px !important;
        border: 1px solid #dfe1e5 !important;
        padding: 5px 15px !important;
        box-shadow: 0 1px 6px rgba(32,33,36,.28) !important;
    }
    div[data-baseweb="input"] > input {
        font-size: 22px !important;
        padding: 10px !important;
    }

    /* تكبير الخطوط في الزراير (الاقتراحات) */
    div.stButton > button {
        width: 100%;
        text-align: left;
        font-size: 20px !important;
        padding: 10px 20px !important;
        border: none !important;
        border-radius: 0px !important;
        background-color: transparent !important;
        color: #1a0dab !important;
        border-bottom: 1px solid #eee !important;
    }
    div.stButton > button:hover {
        background-color: #f8f9fa !important;
        text-decoration: underline !important;
    }
    
    /* إخفاء شريط Streamlit العلوي */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# طباعة الشعار
st.markdown("""
    <div class="google-logo">
        <span class="g-blue">N</span><span class="g-red">L</span><span class="g-yellow">P</span>
        <span class="g-blue"> </span>
        <span class="g-blue">S</span><span class="g-green">e</span><span class="g-red">a</span><span class="g-yellow">r</span><span class="g-blue">c</span><span class="g-green">h</span>
    </div>
    """, unsafe_allow_html=True)

#بيانات علي ما تفرججججججججججججججج
nltk.download('brown')

tokens = [w.lower() for w in brown.words() if w.isalpha()]
# تنظيف وتقطيع البيانات

bigram_model = build_bigram_model(tokens)

# مربع البحث
user_input = st.text_input("", placeholder="Search or type a word (e.g., we, the)...")

# عرض الاقتراحات تحت مربع البحث مباشرة
if user_input:
    last_word = user_input.split()[-1]
    suggestions = predict_next_words(bigram_model, last_word)
    
    if suggestions:
        st.markdown("<br>", unsafe_allow_html=True)
        for sug in suggestions:
            # زرار لكل اقتراح بستايل نتيجة بحث جوجل
            st.button(f"🔍 {user_input} {sug}", key=sug)