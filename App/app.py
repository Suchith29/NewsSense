import streamlit as st
from utils import (
    extract_from_url, extract_from_pdf, preprocess, predict_category, generate_summary,
    get_category_description, translate_text, ensure_english
)
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from fpdf import FPDF
import time

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

# Function to create a PDF summary
def create_pdf(summary_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "NewsSense Summary", ln=True, align='C')
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 10, summary_text)
    return pdf.output(dest='S')

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

        topic_model = BERTopic.load("models/bertopic_model")
        preprocessed = preprocess(text)

        category, confidence = predict_category(preprocessed, True)
        if confidence < 0.3:
            category = 'others'

        summary = generate_summary(text)
        topics, _ = topic_model.transform([preprocessed])
        topic_keywords = topic_model.get_topic(topics[0])
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
            "topic_keywords": topic_keywords,
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
    if topic_keywords:
        filtered_keywords = [
            word for word, _ in topic_keywords if word.lower() not in stop_words and word.isalpha()
        ]
        st.subheader("🌤️ Topic Word Cloud")
        wc = WordCloud(
            width=800, height=300, background_color='white', colormap='Set2'
        ).generate(" ".join(filtered_keywords))

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        st.subheader("🔑 Key Terms")
        st.markdown(", ".join(filtered_keywords))
    else:
        st.warning("No topics could be extracted.")

else:
    st.info("👈 Use the sidebar to provide a URL, upload a PDF, or paste text to begin.")

st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 14px;'>Built with Streamlit</div>", unsafe_allow_html=True)
