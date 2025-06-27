import sys
import re
import joblib
import numpy as np

# Basic text cleaning function (should match train.py)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<review text>\"")
        sys.exit(1)
    review = sys.argv[1]
    if not review.strip():
        print("Error: Review text is empty.")
        sys.exit(1)

    # Load model and vectorizer
    try:
        clf = joblib.load("model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
    except Exception as e:
        print(f"Error loading model/vectorizer: {e}")
        sys.exit(1)

    # Preprocess and vectorize
    review_clean = clean_text(review)
    X = vectorizer.transform([review_clean])

    # Predict
    proba = clf.predict_proba(X)[0]
    label_idx = np.argmax(proba)
    label = "positive" if label_idx == 1 else "negative"
    confidence = proba[label_idx]

    print(f"Label: {label}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main() 