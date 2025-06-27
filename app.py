from flask import Flask, request, render_template_string
import joblib
import re
import numpy as np

app = Flask(__name__)

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

# Load model and vectorizer once at startup
try:
    clf, vectorizer = joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
except Exception as e:
    clf = vectorizer = None
    print(f"Error loading model/vectorizer: {e}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

HTML = '''
<!doctype html>
<html>
<head>
    <title>IMDB Movie Review Sentiment Classifier</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
<div class="container">
<h1>IMDB Movie Review Sentiment Classifier</h1>
<p>Enter a movie review below to predict if it's positive or negative.</p>
<form method=post>
  <textarea name=review rows=8 cols=60>{{ review }}</textarea><br>
  <input type=submit value=Predict>
</form>
{% if label %}
  <h2 class="success">Label: {{ label }}</h2>
  <p class="info">Confidence: {{ confidence }}</p>
{% endif %}
{% if warning %}
  <p class="warning">{{ warning }}</p>
{% endif %}
{% if error %}
  <p class="error">{{ error }}</p>
{% endif %}
</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    label = confidence = warning = error = None
    review = ""
    if request.method == 'POST':
        review = request.form.get('review', '')
        if not review.strip():
            warning = "Please enter a review."
        else:
            try:
                if clf is None or vectorizer is None:
                    raise Exception("Model or vectorizer not loaded.")
                review_clean = clean_text(review)
                X = vectorizer.transform([review_clean])
                proba = clf.predict_proba(X)[0]
                label_idx = int(np.argmax(proba))
                label = "positive" if label_idx == 1 else "negative"
                confidence = f"{proba[label_idx]:.2f}"
            except Exception as e:
                error = f"Error: {e}"
    return render_template_string(HTML, label=label, confidence=confidence, warning=warning, error=error, review=review)

if __name__ == "__main__":
    app.run(debug=True)
