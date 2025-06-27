import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Try to import the IMDB dataset from the datasets library
try:
    from datasets import load_dataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("Warning: 'datasets' library not found. Please install it or use another dataset.")

def clean_text(text):
    """Basic text cleaning: lowercase, remove special characters."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def load_imdb_subset(n_samples=5000):
    """Load a subset of the IMDB dataset using the datasets library."""
    if not DATASET_AVAILABLE:
        raise ImportError("datasets library is not installed.")
    dataset = load_dataset('imdb')
    # Combine train and test, shuffle, and sample
    df = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.head(n_samples)
    return df['text'].tolist(), df['label'].tolist()

def fallback_dataset():
    """Fallback: create a tiny dataset if IMDB is not available."""
    texts = [
        "I loved this movie, it was fantastic!",
        "Absolutely terrible. Waste of time.",
        "Best film I've seen this year.",
        "Awful acting and bad plot.",
        "A masterpiece of cinema.",
        "Not good. Would not recommend.",
        "Wonderful performances and great story.",
        "Boring and predictable.",
        "An enjoyable experience.",
        "Horrible. I walked out halfway."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    return texts, labels

def main():
    print("Loading dataset...")
    try:
        texts, labels = load_imdb_subset(5000)
        print("Loaded IMDB dataset subset.")
    except Exception as e:
        print(f"Could not load IMDB dataset: {e}")
        print("Using fallback tiny dataset.")
        texts, labels = fallback_dataset()

    # Preprocess text
    print("Preprocessing text...")
    texts_clean = [clean_text(t) for t in texts]

    # Vectorize text
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(texts_clean)
    y = np.array(labels)

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    print("Training Logistic Regression model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Classification report on holdout set:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    # Save model and vectorizer
    print("Saving model and vectorizer...")
    joblib.dump(clf, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("Done. Model and vectorizer saved as 'model.joblib' and 'vectorizer.joblib'.")

if __name__ == "__main__":
    main()