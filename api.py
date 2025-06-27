from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import numpy as np

app = FastAPI(title="IMDB Review Sentiment Classifier API")

# Load model and vectorizer at startup
try:
    clf = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
except Exception as e:
    clf = None
    vectorizer = None
    print(f"Error loading model/vectorizer: {e}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

class ReviewRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    if clf is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model or vectorizer not loaded.")
    review = request.text
    if not review or not review.strip():
        raise HTTPException(status_code=400, detail="Text field is empty.")
    review_clean = clean_text(review)
    X = vectorizer.transform([review_clean])
    proba = clf.predict_proba(X)[0]
    label_idx = int(np.argmax(proba))
    label = "positive" if label_idx == 1 else "negative"
    confidence = float(proba[label_idx])
    return {"label": label, "confidence": round(confidence, 2)} 