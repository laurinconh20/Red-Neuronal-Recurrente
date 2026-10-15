"""Interfaz web para probar el clasificador de sentimientos."""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import load_model, predict_text  # noqa: E402

st.set_page_config(page_title="Sentimientos LSTM", page_icon="💬")
st.title("Clasificador de sentimientos")
st.write("Escribe una reseña y el modelo LSTM estimará su sentimiento.")

@st.cache_resource
def get_model():
    return load_model()

text = st.text_area("Reseña", placeholder="This movie was fantastic...")
if st.button("Analizar", type="primary"):
    if not text.strip():
        st.warning("Escribe una reseña antes de analizarla.")
    else:
        with st.spinner("Analizando..."):
            result = predict_text(text, get_model())
        st.subheader(result["sentiment"])
        st.metric("Confianza", f"{result['confidence']:.1%}")
        st.progress(result["probability_positive"], text="Probabilidad de sentimiento positivo")
