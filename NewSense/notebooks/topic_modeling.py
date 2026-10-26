# === notebooks/topic_modeling.py ===
# Re-trained to be compatible with the current environment (sentence-transformers==2.6.1)

# ─── MUST be first – fixes WinError 1114 / c10.dll on Windows ────────────────
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch   # force torch to load cleanly before any other heavy lib
# ─────────────────────────────────────────────────────────────────────────────

import sys
import pandas as pd
from bertopic import BERTopic
import spacy

# ------------------------------------------------------------------
# Resolve paths relative to this script's location so it works
# regardless of the working directory.
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)   # NewSense/

CSV_PATH = os.path.join(
    SCRIPT_DIR,
    "bbc_news_text_complexity_summarization.csv"
)
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "bertopic_model")

print(f"[INFO] Loading CSV from: {CSV_PATH}")
print(f"[INFO] Model will be saved to: {MODEL_SAVE_PATH}")

# Load dataset
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Columns: {df.columns.tolist()}")

# ------------------------------------------------------------------
# SpaCy preprocessing
# ------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.like_num
    ]
    return " ".join(tokens)

print("[INFO] Preprocessing text (this may take a few minutes)...")
df['clean_text'] = df['text'].apply(preprocess)
print(f"[INFO] Preprocessing done. Sample: {df['clean_text'].iloc[0][:120]}")

# ------------------------------------------------------------------
# Train BERTopic
# ------------------------------------------------------------------
print("[INFO] Initialising BERTopic...")
topic_model = BERTopic(language="english", verbose=True)

print("[INFO] Fitting BERTopic model...")
topics, probs = topic_model.fit_transform(df['clean_text'].tolist())

# ------------------------------------------------------------------
# Save model
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
topic_model.save(MODEL_SAVE_PATH)
print(f"\n[SUCCESS] BERTopic model saved to: {MODEL_SAVE_PATH}")

# Print top topics
topic_info = topic_model.get_topic_info()
print("\n[INFO] Top Topics:")
print(topic_info.head(10))
