from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Fake News Detector API")
model = joblib.load("models/fake_news_model.pkl")

class News(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API Fake News Detector running"}

@app.post("/predict")
def predict_news(news: News):
    prediction = model.predict([news.text])[0]
    result = "REAL" if prediction==1 else "FAKE"
    return {"prediction": result}
