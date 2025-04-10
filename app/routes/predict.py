from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import logging
import joblib
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

router = APIRouter()

# Model paths
model_path = "app/models/cupid_match_model_best.pkl"
sentiment_model_path = "app/models/sentiment_model/"

# Example mock profiles
mockProfiles = [
  {
    "name": "Alex",
    "image": "https://randomuser.me/api/portraits/men/32.jpg",
    "match": "92%",
    "sentiment": "😊",
    "reason": "You both love nature walks and deep chats.",
    "traits": ["romantic", "adventurous"],
    "bio": "I enjoy long walks in nature...",
    "hobbies": ["Hiking", "Photography", "Stargazing", "Traveling"],
    "ideal_date": "A surprise weekend getaway...",
    "favorite_music": "Indie Folk, Acoustic, Classical",
    "personality_traits": ["Romantic", "Adventurous", "Deep thinker"],
    "age": 30,
    "gender": "male",
    "orientation": "single",
    "essay": "I love hiking and photography, capturing the beauty of nature in my free time."
  },
  {
    "name": "Ryan",
    "image": "https://randomuser.me/api/portraits/men/56.jpg",
    "match": "75%",
    "sentiment": "😐",
    "reason": "You both enjoy working out and exploring new tech.",
    "traits": ["sporty", "tech-savvy"],
    "bio": "I'm a fitness enthusiast who loves hitting the gym and exploring the latest tech gadgets. Whether it's a new smartwatch or the latest gaming console, I’m always into the newest innovations. On weekends, I usually work on my fitness goals or try out the latest tech products.",
    "hobbies": ["Working out", "Tech reviews", "Cycling", "Video gaming"],
    "ideal_date": "An afternoon cycling trip followed by testing out new gadgets at a tech store and grabbing dinner at a new, trendy restaurant.",
    "favorite_music": "Hip Hop, EDM, Alternative",
    "personality_traits": ["Energetic", "Curious", "Focused"],
    "age": 28,
    "gender": "male",
    "orientation": "single",
    "essay": "I love fitness challenges and trying out the latest gadgets."
  },
  {
    "name": "Sophia",
    "image": "https://randomuser.me/api/portraits/women/74.jpg",
    "match": "88%",
    "sentiment": "😊",
    "reason": "You're both optimists and enjoy outdoor activities.",
    "traits": ["optimistic", "outdoorsy"],
    "bio": "I'm a free spirit who loves being outside and soaking in the beauty of nature. Whether it's a hike in the mountains or a walk in the park, I find peace in the outdoors. I also have a positive outlook on life and enjoy surrounding myself with people who share that energy.",
    "hobbies": ["Hiking", "Camping", "Photography", "Gardening"],
    "ideal_date": "A long hike in the mountains, followed by a picnic by a serene lake, enjoying the view and talking about our dreams and ambitions.",
    "favorite_music": "Indie, Folk, Nature sounds",
    "personality_traits": ["Optimistic", "Adventurous", "Empathetic"],
    "age": 27,
    "gender": "female",
    "orientation": "single",
    "essay": "Being outdoors brings me peace, and I love capturing nature through my camera lens."
  },
  {
    "name": "Olivia",
    "image": "https://randomuser.me/api/portraits/women/15.jpg",
    "match": "92%",
    "sentiment": "😊",
    "reason": "You're both empathetic and passionate about helping others.",
    "traits": ["empathetic", "creative"],
    "bio": "I’m a people person who enjoys helping others and making a difference in their lives. Creativity fuels my soul, whether it’s through art or writing. My friends know me for my deep empathy and understanding, always ready to lend a listening ear or offer advice.",
    "hobbies": ["Writing", "Volunteer work", "Painting", "Community organizing"],
    "ideal_date": "A visit to a local art gallery, followed by a volunteer session at a community center, and finishing the evening with a cozy dinner in a quiet café.",
    "favorite_music": "Jazz, Classical, Pop",
    "personality_traits": ["Empathetic", "Creative", "Supportive"],
    "age": 29,
    "gender": "female",
    "orientation": "single",
    "essay": "I enjoy helping others and using creativity to make a positive impact in the community."
  },
  {
    "name": "Ethan",
    "image": "https://randomuser.me/api/portraits/men/63.jpg",
    "match": "78%",
    "sentiment": "😐",
    "reason": "You both love to travel and enjoy trying new foods.",
    "traits": ["adventurous", "musical"],
    "bio": "I'm always looking for my next travel destination and love exploring new cultures through food. Whether it's sampling street food in Bangkok or dining at a Michelin-starred restaurant in Paris, I’m all in for trying something new. I’m also passionate about music and often find myself exploring different genres to match my moods.",
    "hobbies": ["Traveling", "Food tasting", "Photography", "Music exploration"],
    "ideal_date": "A trip to a bustling city where we can sample street food, explore local sights, and end the day with live music at a cool venue.",
    "favorite_music": "Indie Pop, Folk, Reggae",
    "personality_traits": ["Adventurous", "Curious", "Easygoing"],
    "age": 31,
    "gender": "male",
    "orientation": "single",
    "essay": "Traveling, exploring new food, and capturing moments are my passions."
  },
  {
    "name": "Jester",
    "image": "https://randomuser.me/api/portraits/men/32.jpg",
    "match": "92%",
    "sentiment": "🎶",
    "reason": "You both share a love for music, anime, and creative expression.",
    "traits": ["Creative", "Adventurous", "Anime Enthusiast"],
    "bio": "I live for creativity and self-expression. Whether it’s playing guitar, watching anime, or creating digital art, I’m always exploring new ways to express myself. My weekends often involve jamming with friends, drawing characters, and catching up on the latest anime releases.",
    "hobbies": ["Music", "Anime", "Art", "Gaming"],
    "ideal_date": "A day spent at an anime convention, followed by a cozy evening watching our favorite series and creating some art together.",
    "favorite_music": "Rock, J-Pop, Lo-fi Beats",
    "personality_traits": ["Creative", "Adventurous", "Playful"],
    "age": 26,
    "gender": "male",
    "orientation": "single",
    "essay": "I love expressing myself through music and art, with anime as my biggest inspiration."
  }
]


# Load the Cupid Match Model
try:
    with open(model_path, "rb") as model_file:
        match_model = joblib.load(model_file)  # Use joblib to load the XGBoost model
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

# Endpoint to get mock profiles
@router.get("/mock-profiles")
async def get_mock_profiles():
    return mockProfiles  # Ensure mockProfiles are available for prediction

# Function to prepare the input for the model
def prepare_input_for_model(user_traits, additional_features):
    # Convert gender to a numerical value (0 for male, 1 for female)
    gender = additional_features[1]
    gender = 0 if gender.lower() == "male" else 1  # Example mapping

    # Define a list of possible traits (you can expand this as needed)
    all_possible_traits = ["romantic", "adventurous", "funny", "outgoing", "creative", "intellectual"]
    
    # One-hot encode user traits
    user_traits_vector = [1 if trait in user_traits else 0 for trait in all_possible_traits]

    # Combine traits with additional features (like age, gender)
    all_features = user_traits_vector + [additional_features[0], gender]  # Add age and gender as features
    
    # Ensure the vector has exactly 6 features (pad with zeros or truncate if necessary)
    if len(all_features) < 6:
        all_features.extend([0] * (6 - len(all_features)))  # Pad with zeros if fewer than 6 features
    elif len(all_features) > 6:
        all_features = all_features[:6]  # Truncate to 6 features if there are more

    return all_features

# Function to calculate cosine similarity between two arrays
def calculate_cosine_similarity(user_input, profile_input):
    # Cosine similarity calculation
    similarity = cosine_similarity([user_input], [profile_input])[0][0]
    return similarity

# Function to predict sentiment of an essay using the sentiment model
def predict_sentiment(essay: str):
    inputs = sentiment_tokenizer(essay, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
    sentiment = torch.argmax(logits, dim=1).item()

    # Return sentiment and confidence score
    confidence = torch.softmax(logits, dim=1).max().item()
    sentiment_label = "😊" if sentiment == 1 else "😞"
    
    return sentiment_label, confidence

# Function to handle prediction
@router.post("/predict/")
def predict_match(profile: Profile):
    try:
        updatedProfiles = []

        # Log the incoming profile for debugging
        logging.debug(f"Received profile: {profile.dict()}")

        # Get the sentiment of the user's essay
        user_sentiment, confidence = predict_sentiment(profile.essay)
        logging.debug(f"User sentiment: {user_sentiment} with confidence: {confidence}")

        # Prepare the feature vector for the user's traits
        user_input = prepare_input_for_model(profile.traits, [profile.age, profile.gender])

        for p in mockProfiles:
            # Log each mock profile for debugging
            logging.debug(f"Processing profile: {p['name']} with traits: {p['traits']}")

            # Prepare input features for the user and profile
            profile_input = prepare_input_for_model(p['traits'], [p['age'], p['gender']])

            # Calculate cosine similarity between user traits and profile traits
            similarity_score = calculate_cosine_similarity(user_input, profile_input)
            logging.debug(f"Cosine similarity for {profile.traits} and {p['traits']}: {similarity_score}")

            # Adjust match percentage based on similarity score
            match_percentage = round(similarity_score * 100, 2)
            logging.debug(f"Prediction match percentage: {match_percentage}%")

            # Update profile with match score, sentiment, and reason
            updatedProfile = {
                **p,
                "match": f"{match_percentage}%",
                "sentiment": user_sentiment,
                "confidence": confidence,
                "reason": f"You both share similar traits: {', '.join(profile.traits)}."
            }

            updatedProfiles.append(updatedProfile)

        return {"profiles": updatedProfiles}  # Return the updated profiles with match data

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
