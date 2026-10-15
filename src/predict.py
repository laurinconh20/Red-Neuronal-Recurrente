"""Predicción del modelo LSTM de sentimientos en español."""
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 80
BASE = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = BASE / "modelo_sentimientos_lstm_espanol.keras"
TOKENIZER_PATH = BASE / "tokenizer_espanol.pkl"

def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)

def text_to_sequence(text: str):
    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)
    return tokenizer.texts_to_sequences([text])[0]

def predict_text(text: str, model=None) -> dict:
    if not text or not text.strip():
        raise ValueError("El texto no puede estar vacío")
    model = model if model is not None else load_model()
    encoded = pad_sequences([text_to_sequence(text)], maxlen=MAX_LEN, padding="post", truncating="post")
    probability = float(np.asarray(model.predict(encoded, verbose=0)).ravel()[0])
    return {"sentiment": "Positivo" if probability >= 0.5 else "Negativo", "probability_positive": probability, "confidence": max(probability, 1 - probability)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+")
    result = predict_text(" ".join(parser.parse_args().text))
    print(f"Sentimiento: {result['sentiment']}\nConfianza: {result['confidence']:.2%}")
