# Clasificación de sentimientos con LSTM

El modelo se entrenó con IMDB usando un vocabulario de 10.000 palabras y secuencias de 250 tokens.

## Instalación

```bash
pip install -r requirements.txt
```

## Aplicación web

```bash
streamlit run app/streamlit_app.py
```

## Predicción desde consola

```bash
python src/predict.py "This movie was excellent"
```
