# app/routes/sentiment.py

from fastapi import APIRouter, Body
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# ✅ THIS IS REQUIRED
router = APIRouter()

# ✅ Load your saved tokenizer and model (make sure this path is valid)
tokenizer = DistilBertTokenizerFast.from_pretrained("app/models/sentiment_model")
model = DistilBertForSequenceClassification.from_pretrained("app/models/sentiment_model")

@router.post("/")
def analyze_sentiment(text: str = Body(..., embed=True)):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = round(probs[0][pred].item(), 4)

    label = "Positive" if pred == 1 else "Negative"
    return {"label": label, "confidence": confidence}
