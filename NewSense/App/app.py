import streamlit as st
from utils import (
    extract_from_url, extract_from_pdf, preprocess, predict_category, generate_summary,
    get_category_description, translate_text, ensure_english, extract_keywords
)
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from fpdf import FPDF
import time
import os
import matplotlib.font_manager as fm

# Download necessary NLTK data if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Streamlit page configuration and custom styles
st.set_page_config(page_title="NewsSense", page_icon="🧠", layout="wide")
st.markdown("""
<style>
body { background-color: #F9FAFB; color: #2E3440; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
h1, h4 { color: #0F52BA; font-weight: 700; letter-spacing: 1.2px; }
h4 { color: #6B7280; }
[data-testid="stSidebar"] { background-color: #E8F0FE; padding: 20px; border-radius: 12px; }
button, .stButton>button {
    background-color: #FF6F61 !important; border: none !important; color: white !important;
    padding: 8px 18px !important; font-weight: 600; border-radius: 8px !important; transition: background-color 0.3s ease;
}
button:hover, .stButton>button:hover { background-color: #e85c50 !important; }
div[role="main"] > div > div > div:nth-child(4) {
    background-color: #ffffff; padding: 15px 20px; border-radius: 12px;
    box-shadow: 0 4px 10px rgba(15, 82, 186, 0.1);
}
.css-1v0mbdj.e1fqkh3o3 { color: #0F52BA !important; font-weight: 700 !important; }
.css-1q8dd3e.egzxvld1 { background-color: #0F52BA !important; }
figure {
    border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); background: white; padding: 12px;
}
.stDownloadButton > button {
    background-color: #0F52BA !important; color: white !important; border-radius: 8px !important; font-weight: 700;
}
.stDownloadButton > button:hover { background-color: #0b3e85 !important; }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 style='text-align: center;'>📰 NewsSense</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>AI - Powered News Analysis</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for input selection and text extraction
with st.sidebar:
    st.header("🔍 Analyze News")
    input_type = st.radio("Choose input type:", ["URL", "PDF", "Text Input"])
    text = ""

    if input_type == "URL":
        url = st.text_input("🔗 Enter article URL")
        if st.button("Analyze URL") and url:
            with st.spinner("Extracting from URL..."):
                raw_text = extract_from_url(url)
                text = ensure_english(raw_text)
                st.session_state["text"] = text

    elif input_type == "PDF":
        uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")
        if uploaded_file and st.button("Analyze PDF"):
            with st.spinner("Extracting from PDF..."):
                raw_text = extract_from_pdf(uploaded_file)
                text = ensure_english(raw_text)
                st.session_state["text"] = text

    elif input_type == "Text Input":
        text_input = st.text_area("✍️ Paste or type your news article here")
        if st.button("Analyze Text") and text_input:
            st.session_state["text"] = text_input
            st.info("Text submitted successfully.")

# Retrieve extracted/input text from session state
text = st.session_state.get("text", "")


@st.cache_resource
def load_topic_model():
    return BERTopic.load("models/bertopic_model")

# Function to create a PDF summary with Unicode font support
def create_pdf(summary_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_name = "Helvetica"
    font_loaded = False

    # 1. Try finding DejaVu Sans from matplotlib (supports Unicode, symbols, currency like ₹)
    try:
        font_path = fm.findfont("DejaVu Sans")
        if font_path and os.path.exists(font_path):
            pdf.add_font("DejaVu", fname=font_path)
            font_name = "DejaVu"
            font_loaded = True
    except Exception:
        font_loaded = False

    # 2. Try Windows Arial / Segoe UI TTF as fallback
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

    # 3. If no Unicode font could be loaded, replace incompatible characters for core font
    if not font_loaded:
        summary_text = (
            summary_text.replace("₹", "Rs. ")
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", "-")
            .replace("–", "-")
            .encode("latin-1", "replace")
            .decode("latin-1")
        )

    pdf.set_font(font_name, size=14)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "NewsSense Summary", align='C')
    pdf.ln(10)
    pdf.set_font(font_name, size=11)
    pdf.set_text_color(0, 0, 0)

    try:
        pdf.multi_cell(0, 7, summary_text)
    except Exception:
        safe_text = summary_text.replace("₹", "Rs. ").encode("latin-1", "replace").decode("latin-1")
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, safe_text)

    try:
        return bytes(pdf.output())
    except TypeError:
        return bytes(pdf.output(dest='S'))

# Main processing block
if text:
    st.success("✅ Text received successfully!")

    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    with st.expander("📄 View Text"):
        st.write(text)
        st.caption(f"📝 Word Count: {len(words)} | 🧾 Sentence Count: {len(sentences)}")

    st.subheader("🚀 Processing...")
    with st.spinner("Loading model and predicting..."):
        time.sleep(0.5)  # simulate loading

        topic_model = load_topic_model()
        preprocessed = preprocess(text)

        category, confidence = predict_category(preprocessed, True)
        if confidence < 0.3:
            category = 'others'

        summary = generate_summary(text)

        # 1. Extract article-specific key terms directly from the article
        article_kw = extract_keywords(text, top_n=20)
        keyword_scores = {w: float(count) for w, count in article_kw}

        # 2. Predict topic with BERTopic
        topics, _ = topic_model.transform([preprocessed])
        assigned_topic = topics[0]

        # If BERTopic assigned outlier cluster (-1), find closest semantic topic cluster
        if assigned_topic == -1:
            try:
                sim_topics, _ = topic_model.find_topics(preprocessed, top_n=1)
                if sim_topics and sim_topics[0] != -1:
                    assigned_topic = sim_topics[0]
            except Exception:
                pass

        # 3. Retrieve topic keywords from cluster (ignoring generic news noise)
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

        # Translate summary into Hindi, Telugu, Kannada
        translated_hi = translate_text(summary, "hi")
        translated_te = translate_text(summary, "te")
        translated_kn = translate_text(summary, "kn")

        # Store results in session state
        st.session_state.update({
            "summary": summary,
            "translated_hi": translated_hi,
            "translated_te": translated_te,
            "translated_kn": translated_kn,
            "category": category,
            "keyword_scores": keyword_scores,
            "category_explanation": explanation,
            "current_lang": st.session_state.get("current_lang", "en")
        })

    # Layout for category and summary display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Category")
        st.markdown(f"**{category.title()}**")
        st.caption(explanation)

    with col2:
        st.subheader("🧠 Summary")

        # Language toggle buttons
        col_hin, col_tel, col_kan = st.columns(3)
        with col_hin:
            if st.button("Hindi 🇮🇳", key="btn_hi"):
                st.session_state["current_lang"] = "hi" if st.session_state["current_lang"] != "hi" else "en"
        with col_tel:
            if st.button("Telugu 🇮🇳", key="btn_te"):
                st.session_state["current_lang"] = "te" if st.session_state["current_lang"] != "te" else "en"
        with col_kan:
            if st.button("Kannada 🇮🇳", key="btn_kn"):
                st.session_state["current_lang"] = "kn" if st.session_state["current_lang"] != "kn" else "en"

        lang = st.session_state["current_lang"]
        if lang == "hi":
            display_summary = st.session_state.get("translated_hi", summary)
        elif lang == "te":
            display_summary = st.session_state.get("translated_te", summary)
        elif lang == "kn":
            display_summary = st.session_state.get("translated_kn", summary)
        else:
            display_summary = summary

        st.markdown(f"> {display_summary}")

        # PDF download only for English summary
        if lang == "en":
            pdf_data = create_pdf(display_summary)
            st.download_button(
                label="📥 Download Summary as PDF",
                data=bytes(pdf_data),
                file_name="summary.pdf",
                mime="application/pdf"
            )
        else:
            st.info("📄 PDF download is only available for the English summary.")

    # Word cloud and key terms for topics
    keyword_scores = st.session_state.get("keyword_scores", {})
    if keyword_scores:
        st.subheader("🌤️ Topic Word Cloud")
        wc = WordCloud(
            width=800, height=300, background_color='white', colormap='Set2'
        ).generate_from_frequencies(keyword_scores)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        sorted_terms = sorted(keyword_scores.keys(), key=lambda k: keyword_scores[k], reverse=True)[:15]
        st.subheader("🔑 Key Terms")
        st.markdown(", ".join(sorted_terms))
    else:
        st.warning("No keywords could be extracted from this text.")

else:
    st.info("👈 Use the sidebar to provide a URL, upload a PDF, or paste text to begin.")

st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 14px;'>Built with Streamlit</div>", unsafe_allow_html=True)
