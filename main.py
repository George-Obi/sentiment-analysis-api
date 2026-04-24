from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np

app = FastAPI()

class PredictionInput(BaseModel):
    text: str

# Train simple model (runs once)
texts = [
    "I love this", "Great product", "Feels good", "Awesome",
    "I hate it", "Bad experience", "Terrible", "I don't like it", "Not good"
]
labels = [1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=positive, 0=negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

@app.get("/")
def home():
    return {"message": "GEORGE AI Backend API Live!"}

@app.post("/predict")
def predict_sentiment(input: PredictionInput):
    X_test = vectorizer.transform([input.text])
    sentiment = model.predict(X_test)[0]
    confidence = model.predict_proba(X_test)[0].max()
    return {
        "input": input.text, 
        "sentiment": "positive" if sentiment == 1 else "negative",
        "confidence": float(confidence)
    }