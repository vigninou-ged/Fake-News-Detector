import joblib

def predict(text):
    pipeline = joblib.load("models/fake_news_model.pkl")
    prediction = pipeline.predict([text])[0]
    return "REAL" if prediction==1 else "FAKE"
