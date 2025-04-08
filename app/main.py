from fastapi import FastAPI
from app.routes import predict
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict  # this assumes app/routes/predict.py exists and has a router
from app.routes import sentiment
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware



router = APIRouter()

app = FastAPI(
    title="Cupid AI Backend",
    description="Compatibility & Sentiment Predictor API",
    version="1.0.0"
)

# Allow local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/predict")  # this line is crashing currently
app.include_router(sentiment.router, prefix="/sentiment")

from app.routes import sentiment
print("Sentiment module:", dir(sentiment))  # <-- Add this


@app.get("/")
def read_root():
    return {"message": "Hello from Cupid API!"}


