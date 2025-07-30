# # === notebooks/train_classifier.py ===

# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report
# import spacy

# # Load BBC dataset
# df = pd.read_csv(r"C:\Users\marri.s.reddy\Downloads\NewsSense\notebooks\bbc_news_text_complexity_summarization.csv")





# # Load SpaCy for preprocessing
# nlp = spacy.load("en_core_web_sm")

# def preprocess(text):
#     doc = nlp(text)
#     tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
#     return " ".join(tokens)

# # Preprocess text
# df['clean_text'] = df['text'].apply(preprocess)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['labels'], test_size=0.2, random_state=42)

# # TF-IDF + Logistic Regression pipeline
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=5000)),
#     ('clf', LogisticRegression(max_iter=1000))
# ])

# # Train
# pipeline.fit(X_train, y_train)

# # Evaluate
# y_pred = pipeline.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Save model and vectorizer
# joblib.dump(pipeline.named_steps['clf'], "models/classifier.pkl")
# joblib.dump(pipeline.named_steps['tfidf'], "models/tfidf_vectorizer.pkl")


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import spacy
import re

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv(r"C:\Users\marri.s.reddy\Downloads\NewsSense\notebooks\bbc_news_text_complexity_summarization.csv")

# Improved Preprocessing
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop and not token.ent_type_
    ]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['labels'], test_size=0.2, random_state=42, stratify=df['labels'])

# TF-IDF + Logistic Regression Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        stop_words='english'
    )),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear'
    ))
])

# Hyperparameter tuning
param_grid = {
    'clf__C': [0.1, 1, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluate
y_pred = grid.predict(X_test)
print("Best Params:", grid.best_params_)
print(classification_report(y_test, y_pred, digits=3))

# Save the model and vectorizer
joblib.dump(grid.best_estimator_.named_steps['clf'], "models/classifier.pkl")
joblib.dump(grid.best_estimator_.named_steps['tfidf'], "models/tfidf_vectorizer.pkl")
