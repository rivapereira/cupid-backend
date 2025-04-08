# --- backend: app/routes/predict.py ---

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import pickle

router = APIRouter()

MODEL_PATH = os.path.join("app", "models", "cupid_match_model_best.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

class Profile(BaseModel):
    age: int
    gender: str
    orientation: str
    essay: str
    traits: list[str]

@router.post("/")
def predict_match(profile: Profile):
    try:
        # Basic placeholder feature engineering
        features = [
            profile.age,
            len(profile.essay),
            len(profile.traits)
        ]

        features += [0] * (6 - len(features)) if len(features) < 6 else []
        features = features[:6]  # trim if over

        match_score = model.predict_proba([features])[0][1] * 100

        sentiment = (
            "positive" if "love" in profile.essay.lower()
            else "neutral"
        )

        return {
            "match_score": round(float(match_score), 2),
            "sentiment": str(sentiment)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()  # 👈 This prints error details to your terminal
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

