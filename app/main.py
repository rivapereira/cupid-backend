from fastapi import FastAPI
from app.routes import predict, sentiment  # Import your routes
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# Ensure the static folder exists
static_path = os.path.join("app", "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)  # Create the folder if it doesn't exist

# Path to save the SHAP explanation image
image_path = os.path.join(static_path, "shap_explanation.png")


app = FastAPI(
    title="Cupid AI Backend",
    description="Compatibility & Sentiment Predictor API",
    version="1.0.0"
)

# Mount static directory to serve files like images
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/predict")  # Ensure the prefix is correct
app.include_router(sentiment.router, prefix="/sentiment")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello from Cupid API!"}
