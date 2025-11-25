from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.preprocessing import clean_text

def build_pipeline():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, preprocessor=clean_text)),
        ("model", LogisticRegression(max_iter=1000))
    ])
    return pipe
