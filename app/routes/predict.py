from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle  # For loading the .pkl model
import logging
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

router = APIRouter()

# Model paths
model_path = "app/models/cupid_match_model_best.pkl"
sentiment_model_path = "app/models/sentiment_model/"  

# Load the Cupid Match Model
try:
    with open(model_path, "rb") as model_file:
        match_model = pickle.load(model_file)
    logging.info("Cupid Match Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading Cupid Match Model: {e}")
    match_model = None

# Load the Sentiment Model
try:
    sentiment_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_path)
    sentiment_tokenizer = DistilBertTokenizerFast.from_pretrained(sentiment_model_path)
    logging.info("Sentiment Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading Sentiment Model: {e}")
    sentiment_model = None
    
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

# Function to calculate cosine similarity between two arrays (user traits vs profile traits)
def calculate_cosine_similarity(user_traits: list[str], profile_traits: list[str]) -> float:
    all_traits = list(set(user_traits + profile_traits))
    user_vector = [1 if trait in user_traits else 0 for trait in all_traits]
    profile_vector = [1 if trait in profile_traits else 0 for trait in all_traits]
    similarity = cosine_similarity([user_vector], [profile_vector])[0][0]
    return similarity

# Function to predict sentiment of an essay using the sentiment model
def predict_sentiment(essay: str) -> str:
    # Tokenize the essay
    inputs = sentiment_tokenizer(essay, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Make prediction
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
    
    # Get the sentiment label (0 = negative, 1 = positive)
    sentiment = torch.argmax(logits, dim=1).item()
    return "😊" if sentiment == 1 else "😞"  # Emoji for positive/negative sentiment

# Endpoint to get mock profiles
@router.get("/mock-profiles")
async def get_mock_profiles():
    return mockProfiles

# Function to handle prediction
@router.post("/")
def predict_match(profile: Profile):
    try:
        updatedProfiles = []
        
        # Get the sentiment of the user's essay
        user_sentiment = predict_sentiment(profile.essay)
        
        for p in mockProfiles:
            # Calculate cosine similarity between user traits and profile traits
            similarity_score = calculate_cosine_similarity(profile.traits, p['traits'])
            
            # Calculate match percentage
            match_percentage = round(similarity_score * 100, 2)
            
            # Update profile with match score, sentiment, and reason
            updatedProfile = {
                **p,
                "match": f"{match_percentage}%",
                "sentiment": user_sentiment,  # Using sentiment from user's essay
                "reason": f"You both share similar traits: {', '.join(profile.traits)}."
            }

            updatedProfiles.append(updatedProfile)

        return {"profiles": updatedProfiles}  # Return the updated profiles with match data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")