# === app/utils.py ===

import fitz  # PyMuPDF
from newspaper import Article
import spacy
from transformers import pipeline
import joblib
import string
import numpy as np
from googletrans import Translator

translator = Translator()
from langdetect import detect

def ensure_english(text: str) -> str:
    """
    Detects the language of the input text and translates to English if not already in English.
    """
    try:
        lang = detect(text)
        if lang != 'en':
            translated = translator.translate(text, dest='en')
            return translated.text
        return text
    except Exception as e:
        return f"Language detection/translation error: {e}"


def translate_text(text: str, target_lang: str) -> str:
    """
    Translate English text to the specified language using Google Translate.
    Supported: 'hi' (Hindi), 'te' (Telugu), 'kn' (Kannada)
    """
    if target_lang not in ['hi', 'te', 'kn']:
        return "Unsupported language."

    try:
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as e:
        return f"Translation error: {e}"

# Load SpaCy model for preprocessing once globally
nlp = spacy.load("en_core_web_sm")

# Load summarizer pipeline once globally for efficiency
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load classifier and TF-IDF vectorizer once globally for efficiency
clf = joblib.load("models/classifier.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

def extract_from_url(url: str) -> str:
    """Extract full text from a news article URL using newspaper3k."""
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def extract_from_pdf(file) -> str:
    """Extract text from uploaded PDF file using PyMuPDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess(text: str) -> str:
    """Preprocess text: lemmatization, lowercase, remove stopwords, punctuation, and numbers."""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.like_num and token.text not in string.punctuation
    ]
    return " ".join(tokens)

def predict_category(text: str, return_confidence: bool = False):
    """Predict news category label with optional confidence score."""
    X = tfidf.transform([text])
    pred = clf.predict(X)[0]
    if return_confidence:
        proba = clf.predict_proba(X).max()
        return pred, proba
    return pred
def get_category_description(category):
    descriptions = {
        "business": "Covers financial markets, corporate news, economic policies, and business trends.",
        "entertainment": "Focuses on movies, music, celebrities, arts, and pop culture.",
        "politics": "Reports on government policies, elections, political parties, and leaders.",
        "sport": "Includes news on sports events, teams, players, scores, and tournaments.",
        "tech": "Covers advancements in technology, gadgets, software, and tech companies.",
        "weather": "Reports on climate conditions, forecasts, natural events, and environmental updates."
    }
    return descriptions.get(category.lower(), "No description available for this category.")


def generate_summary(text: str) -> str:
    """
    Generate a summary of the text using BART summarizer.
    If text is too short (less than 50 words), return original text.
    Limit input length for summarizer to 1024 tokens to avoid errors.
    """
    if len(text.split()) < 50:
        return text
    # Summarize only first 1024 tokens to avoid exceeding model limits
    return summarizer(text[:1024])[0]['summary_text']
