import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.pipeline import build_pipeline

def train_model():
    df = pd.read_csv("data/fake_or_real_news.csv")
    X = df["text"]
    y = df["label"].map({"REAL":1,"FAKE":0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Évaluation
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred)*100)
    print(classification_report(y_test, y_pred))

    # Sauvegarde
    joblib.dump(pipeline, "models/fake_news_model.pkl")
    print("✅ Model saved in models/fake_news_model.pkl")

if __name__=="__main__":
    train_model()
