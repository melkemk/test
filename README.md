# IMDB Movie Review Sentiment Classifier

This project is a Python-based machine learning pipeline for classifying IMDB movie reviews as positive or negative using scikit-learn. It demonstrates a complete workflow from data loading and preprocessing to model training, prediction, and optional API deployment.

## Features
- Loads a subset of the IMDB reviews dataset (5,000 samples)
- Preprocesses and vectorizes text using TF-IDF
- Trains a Logistic Regression classifier
- Command-line prediction script
- (Optional) FastAPI endpoint for predictions

## Setup

1. **Clone the repository and navigate to the project directory.**
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

Run the following command to train the model and save the artifacts:
```bash
python train.py
```

## Making Predictions from the Command Line

Use the `predict.py` script to classify a review:
```bash
python predict.py "I loved this movie!"
```
Output:
```
Label: positive
Confidence: 0.92
```

## Running the API (Optional Bonus)

Start the FastAPI server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Send a POST request to `/predict`:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "I loved this movie!"}'
```
Response:
```
{"label": "positive", "confidence": 0.92}
```

## Project Structure
- `train.py`: Training script
- `predict.py`: Command-line prediction script
- `api.py`: FastAPI app (optional)
- `requirements.txt`: Dependencies
- `README.md`: Project documentation

## Notes
- Requires Python 3.8+
- Handles errors gracefully (e.g., missing arguments, invalid input)
- All code follows PEP 8 style guidelines # test
