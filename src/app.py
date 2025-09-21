from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# BASE_DIR = root project (folder paling atas)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model, vectorizer, dan encoder
model = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_logreg_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))

# Inisialisasi FastAPI
app = FastAPI(title="Thesis Topic Classifier API")

# Schema input
class TextIn(BaseModel):
    text: str

# Endpoint prediksi
@app.post("/predict")
def predict(data: TextIn):
    # Transform input ke TF-IDF
    X = tfidf.transform([data.text])
    # Prediksi label numerik
    pred_num = model.predict(X)[0]
    # Konversi ke label asli
    pred_label = le.inverse_transform([pred_num])[0]

    return {
        "input_text": data.text,
        "prediction": pred_label
    }
