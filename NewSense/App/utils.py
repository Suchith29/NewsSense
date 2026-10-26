# === app/utils.py ===

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import string
import fitz  # PyMuPDF
from newspaper import Article, Config
from googletrans import Translator
from langdetect import detect

# Lazy-loaded model cache to conserve memory and optimize Streamlit execution
_model_cache = {}

def get_spacy_nlp():
    """Lazily load and cache the spaCy English pipeline."""
    if "spacy" not in _model_cache:
        import spacy
        _model_cache["spacy"] = spacy.load("en_core_web_sm")
    return _model_cache["spacy"]

def get_summarizer():
    """Lazily load and cache the BART summarization pipeline."""
    if "summarizer" not in _model_cache:
        from transformers import pipeline
        _model_cache["summarizer"] = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # Standard CPU execution
        )
    return _model_cache["summarizer"]

def get_classifier_and_vectorizer():
    """Lazily load and cache the trained classifier and TF-IDF vectorizer with robust path resolution."""
    if "clf" not in _model_cache or "tfidf" not in _model_cache:
        import joblib
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clf_path = os.path.join(base_dir, "models", "classifier.pkl")
        tfidf_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
        
        if not os.path.exists(clf_path) or not os.path.exists(tfidf_path):
            raise FileNotFoundError(f"Model artifacts not found at {clf_path} or {tfidf_path}")
            
        _model_cache["clf"] = joblib.load(clf_path)
        _model_cache["tfidf"] = joblib.load(tfidf_path)
    return _model_cache["clf"], _model_cache["tfidf"]

translator = Translator()

def ensure_english(text: str) -> str:
    """
    Detects the language of the input text and translates to English if not already in English.
    Includes error handling and graceful fallback.
    """
    if not text or not text.strip():
        return text
    try:
        lang = detect(text)
        if lang != 'en':
            translated = translator.translate(text, dest='en')
            return translated.text
        return text
    except Exception as e:
        # Fallback to original text if detection or translation encounters network issues
        return text

def translate_text(text: str, target_lang: str) -> str:
    """
    Translate English text to the specified language using Google Translate with retries.
    Supported: 'hi' (Hindi), 'te' (Telugu), 'kn' (Kannada)
    """
    if not text or not text.strip():
        return ""
    if target_lang not in ['hi', 'te', 'kn']:
        return "Unsupported language."

    for attempt in range(2):
        try:
            result = translator.translate(text, dest=target_lang)
            if result and result.text:
                return result.text
        except Exception:
            if attempt == 1:
                return f"[Translation temporarily unavailable. Showing original]:\n\n{text}"
    return text

def extract_from_url(url: str) -> str:
    """
    Extract full text from a news article URL using newspaper3k with custom desktop browser headers.
    """
    try:
        config = Config()
        config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        config.request_timeout = 10

        article = Article(url, config=config)
        article.download()
        article.parse()
        if not article.text or len(article.text.strip()) == 0:
            return "Unable to extract article text from the provided URL. The page may require JavaScript or authentication."
        return article.text
    except Exception as e:
        return f"Error extracting article from URL: {e}"

def extract_from_pdf(file) -> str:
    """Extract text from uploaded PDF file stream using PyMuPDF."""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"Error reading PDF document: {e}"

def preprocess(text: str) -> str:
    """Preprocess text: lemmatization, lowercase, remove stopwords, punctuation, and numbers."""
    nlp = get_spacy_nlp()
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.like_num and token.text not in string.punctuation
    ]
    return " ".join(tokens)

def predict_category(text: str, return_confidence: bool = False):
    """Predict news category label with optional confidence score."""
    clf, tfidf = get_classifier_and_vectorizer()
    X = tfidf.transform([text])
    pred = clf.predict(X)[0]
    if return_confidence:
        proba = clf.predict_proba(X).max()
        return pred, float(proba)
    return pred

def get_category_description(category: str) -> str:
    """Returns human-readable explanation of predicted news category."""
    descriptions = {
        "business": "Covers financial markets, corporate news, economic policies, and business trends.",
        "entertainment": "Focuses on movies, music, celebrities, arts, and pop culture.",
        "politics": "Reports on government policies, elections, political parties, and leaders.",
        "sport": "Includes news on sports events, teams, players, scores, and tournaments.",
        "tech": "Covers advancements in technology, gadgets, software, and tech companies.",
        "others": "General interest or cross-domain reporting with low category specificity.",
        "weather": "Reports on climate conditions, forecasts, natural events, and environmental updates."
    }
    return descriptions.get(category.lower(), "General news report.")

def generate_summary(text: str) -> str:
    """
    Generate an abstractive summary of the text using BART.
    Fixes previous character-truncation bug (text[:1024]) by properly handling
    token lengths and employing hierarchical chunking for long-form articles.
    """
    words = text.split()
    if len(words) < 45:
        return text

    summarizer = get_summarizer()

    # Case 1: Standard article length (up to ~650 words) fits within 1024 BART tokens
    if len(words) <= 650:
        res = summarizer(
            text,
            max_length=150,
            min_length=40,
            truncation=True,
            do_sample=False
        )
        return res[0]['summary_text']

    # Case 2: Long-form journalism (> 650 words) — Chunked Map-Reduce Summarization
    chunks = []
    chunk_size = 500
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) > 40:
            chunks.append(" ".join(chunk_words))

    partial_summaries = []
    # Process up to first 3 chunks (~1500 words) to avoid excessive CPU latency
    for chunk in chunks[:3]:
        res = summarizer(
            chunk,
            max_length=80,
            min_length=25,
            truncation=True,
            do_sample=False
        )
        partial_summaries.append(res[0]['summary_text'])

    combined_summary = " ".join(partial_summaries)
    
    # If partial summary is still lengthy, do a final synthesis pass
    if len(partial_summaries) > 1 and len(combined_summary.split()) > 100:
        final_res = summarizer(
            combined_summary,
            max_length=130,
            min_length=35,
            truncation=True,
            do_sample=False
        )
        return final_res[0]['summary_text']

    return combined_summary

def extract_keywords(text: str, top_n: int = 20):
    """
    Extract key terms and their frequency directly from the article text.
    Filters out generic reporting words and stopwords using spaCy POS tagging.
    """
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords

    try:
        nltk_stops = set(stopwords.words("english"))
    except Exception:
        nltk_stops = set()

    news_noise = {
        "say", "says", "said", "mr", "mrs", "ms", "also", "would", "could",
        "told", "get", "like", "one", "two", "new", "year", "years", "time",
        "first", "last", "people", "many", "well", "way", "make", "may",
        "take", "come", "go", "see", "know", "back", "much", "day", "days",
        "uk", "eu", "report", "reported", "spokesman", "statement"
    }
    stop_words = nltk_stops.union(news_noise)

    nlp = get_spacy_nlp()
    doc = nlp(text)
    words = []
    for token in doc:
        w = token.lemma_.lower().strip()
        if (
            token.pos_ in ["NOUN", "PROPN", "ADJ"]
            and len(w) > 2
            and w not in stop_words
            and w.isalpha()
        ):
            words.append(w)

    counts = Counter(words)
    return counts.most_common(top_n)
