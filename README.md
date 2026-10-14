# 🔍 NLP Autofill — Setup & Run Guide

A Google-style autocomplete app built with Python, NLTK, and Streamlit.  
Uses **Bigram + Trigram** language models trained on the Brown corpus.

---

## 📁 Project Files

```
TS/
├── app.py              ← Main app (run this)
├── model.py            ← NLP model logic (bigram + trigram)
├── requirements.txt    ← Python dependencies
├── app (1).py          ← Original version (kept for reference)
└── README.md           ← This file
```

---

## ✅ Requirements

- Python **3.10 or newer**
- pip (comes with Python)

> Check your Python version by running: `python --version`

---

## 🚀 How to Run (Step by Step)

### Step 1 — Open a terminal in the project folder

- On **Windows**: open the `TS` folder → right-click inside it → **"Open in Terminal"**  
  *(or open PowerShell and navigate: `cd "C:\Users\medoo\OneDrive\Desktop\TS"`)*

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `nltk`, and `pandas`.  
You only need to do this **once**.

### Step 3 — Run the app

```bash
streamlit run app.py
```

### Step 4 — Open in browser

After running, you'll see:

```
Local URL: http://localhost:8501
```

Open that link in your browser. The app will load automatically.

> ⚠️ **First launch is slow** — it downloads the Brown corpus (~5 MB) and builds the language model. After that, it's instant every time.

---

## 🧠 How It Works

| Component | What it does |
|-----------|-------------|
| **Brown Corpus** | ~1 million words of English text used to train the model |
| **Bigram Model** | Predicts the next word based on the **last 1 word** |
| **Trigram Model** | Predicts the next word based on the **last 2 words** (smarter) |
| **Prefix Filter** | If you're mid-word, it completes the current word instead |
| **Confidence %** | Shows how likely each suggestion is based on word frequency |

---

## 🖱️ How to Use the App

1. Type any word or phrase into the search box (e.g. `the`, `we are`, `it is`)
2. Suggestions appear automatically below the box
3. Click any suggestion to auto-fill it into the search box
4. Keep typing to refine suggestions
5. Click **"Prediction confidence"** to see a bar chart of suggestion scores

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit: command not found` | Run `pip install streamlit` first |
| App is blank / spinning forever | Wait ~15 seconds on first launch |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| NLTK download error | Make sure you have internet access on first launch |

---

## 📦 Dependencies

```
streamlit   — Web app framework
nltk        — Natural Language Toolkit (corpus + n-grams)
pandas      — Used for the confidence bar chart
```
