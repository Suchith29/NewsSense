# === notebooks/topic_modeling.py ===

import pandas as pd
from bertopic import BERTopic
import spacy
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\marri.s.reddy\Downloads\NewsSense\notebooks\bbc_news_text_complexity_summarization.csv")

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return " ".join(tokens)

# Preprocess text
df['clean_text'] = df['text'].apply(preprocess)

# Initialize BERTopic
topic_model = BERTopic(language="english")

# Fit model
topics, probs = topic_model.fit_transform(df['clean_text'])

# Save model
topic_model.save("models/bertopic_model")

# Optional: Print top topics
topic_info = topic_model.get_topic_info()
print(topic_info.head())
