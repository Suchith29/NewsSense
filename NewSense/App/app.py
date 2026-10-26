import os
# Prevent OpenMP runtime library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # Crucial: Load PyTorch runtime before any other numerical libraries to avoid WinError 1114

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from bertopic import BERTopic
from fpdf import FPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from utils import (
    extract_from_url, extract_from_pdf, preprocess, predict_category, generate_summary,
    get_category_description, translate_text, ensure_english, extract_keywords
)

# Download necessary NLTK data (quiet=True → no-op if already present)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))

# Streamlit page configuration and modern custom styling
st.set_page_config(page_title="NewsSense | AI News Intelligence", page_icon="📰", layout="wide")
st.markdown("""
<style>
body { background-color: #F9FAFB; color: #2E3440; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
h1, h4 { color: #0F52BA; font-weight: 700; letter-spacing: 1px; }
h4 { color: #6B7280; }
[data-testid="stSidebar"] { background-color: #E8F0FE; padding: 20px; border-radius: 12px; }
button, .stButton>button {
    background-color: #FF6F61 !important; border: none !important; color: white !important;
    padding: 8px 18px !important; font-weight: 600; border-radius: 8px !important; transition: all 0.2s ease;
}
button:hover, .stButton>button:hover { background-color: #e85c50 !important; transform: translateY(-1px); }
div[role="main"] > div > div > div:nth-child(4) {
    background-color: #ffffff; padding: 15px 20px; border-radius: 12px;
    box-shadow: 0 4px 10px rgba(15, 82, 186, 0.08);
}
figure {
    border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.06); background: white; padding: 12px;
}
.stDownloadButton > button {
    background-color: #0F52BA !important; color: white !important; border-radius: 8px !important; font-weight: 700;
}
.stDownloadButton > button:hover { background-color: #0b3e85 !important; }
</style>
""", unsafe_allow_html=True)

# App Title & Subtitle
st.markdown("<h1 style='text-align: center;'>📰 NewsSense</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>AI-Powered News Analysis, Summarization & Topic Intelligence</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for multi-source ingestion
with st.sidebar:
    st.header("🔍 Ingestion Source")
    input_type = st.radio("Choose input method:", ["URL", "PDF Upload", "Direct Text Input"])
    
    if input_type == "URL":
        url = st.text_input("🔗 Enter News Article URL")
        if st.button("Analyze Article") and url:
            with st.spinner("Scraping and analyzing URL..."):
                raw_text = extract_from_url(url)
                normalized_text = ensure_english(raw_text)
                st.session_state["text"] = normalized_text

    elif input_type == "PDF Upload":
        uploaded_file = st.file_uploader("📄 Upload PDF Document", type="pdf")
        if uploaded_file and st.button("Analyze PDF"):
            with st.spinner("Extracting text from PDF..."):
                raw_text = extract_from_pdf(uploaded_file)
                normalized_text = ensure_english(raw_text)
                st.session_state["text"] = normalized_text

    elif input_type == "Direct Text Input":
        text_input = st.text_area("✍️ Paste or type news article here", height=200)
        if st.button("Analyze Text") and text_input:
            normalized_text = ensure_english(text_input)
            st.session_state["text"] = normalized_text
            st.info("Article submitted successfully.")

# Retrieve current text from session state
text = st.session_state.get("text", "")

@st.cache_resource
def load_topic_model():
    """Load serialized BERTopic model with absolute path safety."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "bertopic_model")
    return BERTopic.load(model_path)

# Function to compile executive PDF with TrueType Unicode font support
def create_pdf(summary_text: str, category_name: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_name = "Helvetica"
    font_loaded = False

    # 1. Search for DejaVu Sans (universal Unicode support, Indian Rupee ₹, directional quotes)
    try:
        font_path = fm.findfont("DejaVu Sans")
        if font_path and os.path.exists(font_path):
            pdf.add_font("DejaVu", fname=font_path)
            font_name = "DejaVu"
            font_loaded = True
    except Exception:
        font_loaded = False

    # 2. Windows Arial / Segoe UI fallback
    if not font_loaded:
        for win_font in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
            if os.path.exists(win_font):
                try:
                    pdf.add_font("WinUnicode", fname=win_font)
                    font_name = "WinUnicode"
                    font_loaded = True
                    break
                except Exception:
                    pass

    # Header section
    pdf.set_font(font_name, size=16)
    pdf.set_text_color(15, 82, 186)
    pdf.cell(0, 12, "NewsSense Executive Brief", align='C')
    pdf.ln(8)
    
    pdf.set_font(font_name, size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Category: {category_name.title()} | Generated by NewsSense AI", align='C')
    pdf.ln(12)

    # Summary Body
    pdf.set_font(font_name, size=11)
    pdf.set_text_color(30, 30, 30)

    try:
        pdf.multi_cell(0, 7, summary_text)
    except Exception:
        # Emergency ASCII fallback
        safe_text = summary_text.replace("₹", "Rs. ").encode("latin-1", "replace").decode("latin-1")
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, safe_text)

    try:
        return bytes(pdf.output())
    except TypeError:
        return bytes(pdf.output(dest='S'))

# Main Processing Pipeline
if text:
    st.success("✅ Article ingested successfully!")

    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    with st.expander("📄 Ingested Article Preview"):
        st.write(text)
        st.caption(f"📊 Word Count: {len(words)} | 📝 Sentence Count: {len(sentences)}")

    # PERFORMANCE OPTIMIZATION: Only run heavy transformer/model inference when text changes
    if st.session_state.get("analyzed_text") != text:
        with st.spinner("Processing article with NLP pipelines..."):
            topic_model = load_topic_model()
            preprocessed = preprocess(text)

            # 1. Supervised Classification & Confidence
            category, confidence = predict_category(preprocessed, return_confidence=True)
            if confidence < 0.30:
                category = 'others'

            # 2. Abstractive Summarization with BART
            summary = generate_summary(text)

            # 3. Dynamic Semantic Keywords & BERTopic
            article_kw = extract_keywords(text, top_n=20)
            keyword_scores = {w: float(count) for w, count in article_kw}

            topics, _ = topic_model.transform([preprocessed])
            assigned_topic = topics[0]

            # Semantic fallback for outlier cluster (-1)
            if assigned_topic == -1:
                try:
                    sim_topics, _ = topic_model.find_topics(preprocessed, top_n=1)
                    if sim_topics and sim_topics[0] != -1:
                        assigned_topic = sim_topics[0]
                except Exception:
                    pass

            news_noise = {
                "say", "says", "said", "mr", "mrs", "ms", "also", "would", "could",
                "told", "get", "like", "one", "two", "new", "year", "years", "time",
                "first", "last", "people", "many", "well", "way", "make", "may",
                "take", "come", "go", "see", "know", "back", "much", "day", "days",
                "uk", "eu", "report", "reported", "spokesman", "statement"
            }
            all_stops = stop_words.union(news_noise)

            if assigned_topic is not None and assigned_topic != -1:
                raw_topic_words = topic_model.get_topic(assigned_topic) or []
                for w, score in raw_topic_words:
                    w_lower = w.lower()
                    if w_lower.isalpha() and w_lower not in all_stops and len(w_lower) > 2:
                        keyword_scores[w_lower] = keyword_scores.get(w_lower, 0.0) + (score * 5.0)

            explanation = get_category_description(category)

            # Pre-translate summary to regional Indian languages
            translated_hi = translate_text(summary, "hi")
            translated_te = translate_text(summary, "te")
            translated_kn = translate_text(summary, "kn")

            # Store in session state for instant language toggling
            st.session_state.update({
                "analyzed_text": text,
                "summary": summary,
                "translated_hi": translated_hi,
                "translated_te": translated_te,
                "translated_kn": translated_kn,
                "category": category,
                "confidence": confidence,
                "keyword_scores": keyword_scores,
                "category_explanation": explanation,
                "current_lang": "en"
            })

    # Retrieve cached results
    category = st.session_state.get("category", "General")
    confidence = st.session_state.get("confidence", 1.0)
    explanation = st.session_state.get("category_explanation", "")
    summary = st.session_state.get("summary", "")
    keyword_scores = st.session_state.get("keyword_scores", {})

    # Display Category & Summary in two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🏷️ News Category")
        conf_pct = f"({confidence * 100:.1f}% confidence)" if category != "others" else "(Low confidence fallback)"
        st.markdown(f"### **{category.title()}** `{conf_pct}`")
        st.caption(explanation)

    with col2:
        st.subheader("📑 Executive Summary")

        # Instant Language Toggle Buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        with btn_col1:
            if st.button("English 🇬🇧", key="btn_en"):
                st.session_state["current_lang"] = "en"
        with btn_col2:
            if st.button("Hindi 🇮🇳", key="btn_hi"):
                st.session_state["current_lang"] = "hi"
        with btn_col3:
            if st.button("Telugu 🇮🇳", key="btn_te"):
                st.session_state["current_lang"] = "te"
        with btn_col4:
            if st.button("Kannada 🇮🇳", key="btn_kn"):
                st.session_state["current_lang"] = "kn"

        curr_lang = st.session_state.get("current_lang", "en")
        if curr_lang == "hi":
            display_summary = st.session_state.get("translated_hi", summary)
            st.info("Language: **Hindi**")
        elif curr_lang == "te":
            display_summary = st.session_state.get("translated_te", summary)
            st.info("Language: **Telugu**")
        elif curr_lang == "kn":
            display_summary = st.session_state.get("translated_kn", summary)
            st.info("Language: **Kannada**")
        else:
            display_summary = summary
            st.info("Language: **English (Default)**")

        st.markdown(f"> {display_summary}")

        # PDF Download Section
        if curr_lang == "en":
            pdf_data = create_pdf(display_summary, category)
            st.download_button(
                label="📥 Download Executive Brief (PDF)",
                data=bytes(pdf_data),
                file_name=f"NewsSense_{category.lower()}_summary.pdf",
                mime="application/pdf"
            )
        else:
            st.caption("ℹ️ PDF download is formatted for the English executive brief.")

    # Topic Word Cloud & Key Terms Visualization
    if keyword_scores:
        st.markdown("---")
        st.subheader("☁️ Semantic Topic Word Cloud")
        wc = WordCloud(
            width=900, height=320, background_color='white', colormap='Set2'
        ).generate_from_frequencies(keyword_scores)

        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        sorted_terms = sorted(keyword_scores.keys(), key=lambda k: keyword_scores[k], reverse=True)[:15]
        st.subheader("🔑 Top Extracted Key Terms")
        st.markdown(" • ".join([f"`{term}`" for term in sorted_terms]))
    else:
        st.warning("No significant keywords could be extracted from this text.")

else:
    st.info("👈 Use the sidebar to enter a live article URL, upload a PDF, or paste text to begin analysis.")

st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 13px; color: gray;'>NewsSense NLP Intelligence Platform • Built with Streamlit, Hugging Face Transformers & BERTopic</div>", unsafe_allow_html=True)
