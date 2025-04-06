from fastapi import APIRouter, Body
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from pathlib import Path
import torch

from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from fastapi import APIRouter

router = APIRouter()

# Use pathlib to resolve absolute path to model dir
MODEL_DIR = "app/models/sentiment_model"

from pathlib import Path

MODEL_DIR = Path("app/models/sentiment_model").resolve().as_posix()

# ✅ Load from local path with local_files_only flag
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

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
