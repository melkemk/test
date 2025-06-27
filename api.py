from flask import Flask, request, jsonify
import joblib
import re
import numpy as np

api = Flask(__name__)

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

@api.route('/predict', methods=['POST'])
def predict_sentiment():
    if clf is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded."}), 500
    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({"error": "Text field is empty."}), 400
    review = data['text']
    review_clean = clean_text(review)
    X = vectorizer.transform([review_clean])
    proba = clf.predict_proba(X)[0]
    label_idx = int(np.argmax(proba))
    label = "positive" if label_idx == 1 else "negative"
    confidence = float(proba[label_idx])
    return jsonify({"label": label, "confidence": round(confidence, 2)})

if __name__ == "__main__":
    api.run(debug=True)