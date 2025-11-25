import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector")

model = joblib.load("models/fake_news_model.pkl")
user_input = st.text_area("Entrez votre texte de news:")

if st.button("Prédire"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte valide.")
    else:
        result = "✅ REAL" if model.predict([user_input])[0]==1 else "❌ FAKE"
        st.success(f"Résultat: {result}")
