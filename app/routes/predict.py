from fastapi import APIRouter, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from typing import List
import pickle  # For loading the .pkl model
import logging

import pickle
import logging


router = APIRouter()

# Correct path to the model
model_path = "app/models/cupid_match_model_best.pkl"

# Load the model
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None


# Input schema
class Profile(BaseModel):
    age: int
    gender: str
    orientation: str
    essay: str
    traits: list[str]  # User selected traits

mockProfiles = [
    {
        "name": "Alex",
        "image": "https://randomuser.me/api/portraits/men/32.jpg",
        "match": "92%",
        "sentiment": "😊",
        "reason": "You both love nature walks and deep chats.",
        "traits": ["romantic", "adventurous"]
    },
    {
        "name": "Sam",
        "image": "https://randomuser.me/api/portraits/women/45.jpg",
        "match": "87%",
        "sentiment": "😐",
        "reason": "You have similar humor and music taste.",
        "traits": ["chaotic", "creative"]
    },
    {
        "name": "Jules",
        "image": "https://randomuser.me/api/portraits/men/76.jpg",
        "match": "90%",
        "sentiment": "😊",
        "reason": "You both enjoy lowkey weekends and comfort food.",
        "traits": ["introvert", "dreamboat"]
    },
    {
        "name": "Lana",
        "image": "https://randomuser.me/api/portraits/women/23.jpg",
        "match": "85%",
        "sentiment": "😊",
        "reason": "You share a love for music and spontaneous road trips.",
        "traits": ["musical", "adventurous"]
    },
    {
        "name": "Ryan",
        "image": "https://randomuser.me/api/portraits/men/56.jpg",
        "match": "75%",
        "sentiment": "😐",
        "reason": "You both enjoy working out and exploring new tech.",
        "traits": ["sporty", "tech-savvy"]
    },
    {
        "name": "Sophia",
        "image": "https://randomuser.me/api/portraits/women/74.jpg",
        "match": "88%",
        "sentiment": "😊",
        "reason": "You're both optimists and enjoy outdoor activities.",
        "traits": ["optimistic", "outdoorsy"]
    },
    {
        "name": "Olivia",
        "image": "https://randomuser.me/api/portraits/women/15.jpg",
        "match": "92%",
        "sentiment": "😊",
        "reason": "You're both empathetic and passionate about helping others.",
        "traits": ["empathetic", "creative"]
    },
    {
        "name": "Ethan",
        "image": "https://randomuser.me/api/portraits/men/63.jpg",
        "match": "78%",
        "sentiment": "😐",
        "reason": "You both love to travel and enjoy trying new foods.",
        "traits": ["adventurous", "musical"]

    },
  {
    "name": "Jester",
    "image": "https://randomuser.me/api/portraits/men/32.jpg",
    "match": "92%",
    "sentiment": "🎶",
    "reason": "You both share a love for music, anime, and creative expression.",
    "traits": ["Creative", "Adventurous", "Anime Enthusiast"]
  },
  {
    "name": "Abyaz",
    "image": "https://randomuser.me/api/portraits/men/33.jpg",
    "match": "87%",
    "sentiment": "🤓",
    "reason": "You both are driven, intellectual, and value practical knowledge.",
    "traits": ["Intelligent", "Logical", "Cool-headed"]
  },
  {
    "name": "Jade",
    "image": "https://randomuser.me/api/portraits/women/34.jpg",
    "match": "90%",
    "sentiment": "🍝",
    "reason": "You both are into tech, problem-solving, and good food.",
    "traits": ["Analytical", "Tech-savvy", "Foodie"]
  },
  {
    "name": "BB",
    "image": "https://randomuser.me/api/portraits/women/35.jpg",
    "match": "88%",
    "sentiment": "🎨",
    "reason": "You both share a creative passion and understand the value of visual aesthetics.",
    "traits": ["Creative", "Design-focused", "Independent"]
  },
  {
    "name": "Qasim",
    "image": "https://randomuser.me/api/portraits/men/36.jpg",
    "match": "85%",
    "sentiment": "🎮",
    "reason": "You both enjoy Pokemon, anime, and the escapism that comes with them.",
    "traits": ["Geeky", "Anime lover", "Gaming enthusiast"]
  },
  {
    "name": "Nao",
    "image": "https://randomuser.me/api/portraits/men/37.jpg",
    "match": "91%",
    "sentiment": "🕹️",
    "reason": "You both have a shared passion for gaming, especially Pokemon and rhythm games.",
    "traits": ["Energetic", "Multicultural", "Music lover"]
  },
  {
    "name": "Hiba",
    "image": "https://randomuser.me/api/portraits/women/38.jpg",
    "match": "93%",
    "sentiment": "💻",
    "reason": "You both value independence, creativity, and are driven by your passions.",
    "traits": ["Strong-willed", "Tech-savvy", "Creative"]
  }
]

    


@router.get("/mock-profiles")
async def get_mock_profiles():
    return mockProfiles

def calculate_cosine_similarity(user_traits: list[str], profile_traits: list[str]) -> float:
    # Convert traits to binary vectors based on presence or absence of traits
    all_traits = list(set(user_traits + profile_traits))  # All unique traits
    user_vector = [1 if trait in user_traits else 0 for trait in all_traits]
    profile_vector = [1 if trait in profile_traits else 0 for trait in all_traits]

    # Log the vectors to check their values
    logging.debug(f"User Vector: {user_vector}")
    logging.debug(f"Profile Vector: {profile_vector}")

    # Compute the cosine similarity
    similarity = cosine_similarity([user_vector], [profile_vector])[0][0]

    # Log the similarity score
    logging.debug(f"Cosine Similarity: {similarity}")
    return similarity

# Function to handle prediction
@router.post("/")
def predict_match(profile: Profile):
    try:
        updatedProfiles = []

        for p in mockProfiles:
            # Calculate cosine similarity between user traits and profile traits
            similarity_score = calculate_cosine_similarity(profile.traits, p['traits'])
            
            # Calculate match percentage
            match_percentage = round(similarity_score * 100, 2)
            
            # Determine sentiment based on match score
            sentiment = "😊" if match_percentage > 70 else "😐" if match_percentage > 40 else "😞"
            
            # Update profile with match score and sentiment
            updatedProfile = {
                **p,
                "match": f"{match_percentage}%",
                "sentiment": sentiment,
                "reason": f"You both share similar traits: {', '.join(profile.traits)}."
            }

            updatedProfiles.append(updatedProfile)

        return {"profiles": updatedProfiles}  # Return the updated profiles with match data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
