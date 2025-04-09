from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel


router = APIRouter()

# Load model
MODEL_PATH = os.path.join("app", "models", "cupid_match_model_best.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Store last input for SHAP
last_input_df = None

# Input schema
class Profile(BaseModel):
    age: int
    gender: str
    orientation: str
    essay: str
    traits: list[str]

@router.post("/")
def predict_match(profile: Profile):
    global last_input_df
    try:
        # Feature engineering with meaningful flags
        features = {
            "age": profile.age,
            "essay_length": len(profile.essay),
            "num_traits": len(profile.traits),
            "has_love_word": int("love" in profile.essay.lower()),
            "has_fun_word": int("fun" in profile.essay.lower()),
            "has_chill_word": int("chill" in profile.essay.lower())
        }

        X = pd.DataFrame([features])
        last_input_df = X

        # Predict match score
        match_score = model.predict_proba(X)[0][1] * 100

        # Basic sentiment rule
        sentiment = "positive" if "love" in profile.essay.lower() else "neutral"

        return {
            "match_score": round(float(match_score), 2),
            "sentiment": sentiment
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

from fastapi.responses import FileResponse
import os
import shap
import matplotlib.pyplot as plt

# Assuming you have already loaded the model, and 'last_input_df' contains the necessary data.

# Ensure the static folder exists
static_path = os.path.join("app", "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)

import logging

logging.basicConfig(level=logging.DEBUG)

@router.get("/predict/explanation/")
async def get_shap_explanation():
    global last_input_df
    if last_input_df is None:
        raise HTTPException(status_code=400, detail="No input data for SHAP explanation.")
    
    try:
        explainer = shap.Explainer(model, last_input_df)
        shap_values = explainer(last_input_df)

        # Generate waterfall plot
        plt.clf()
        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()

        # Save SHAP explanation image to static folder
        image_path = os.path.join(static_path, "shap_explanation.png")
        plt.savefig(image_path, bbox_inches="tight", dpi=200)

        return FileResponse(image_path, media_type="image/png")

    except Exception as e:
        print(f"Error generating SHAP explanation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate SHAP explanation.")
