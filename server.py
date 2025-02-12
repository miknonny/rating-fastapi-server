from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import asyncio

# Initialize the FastAPI app
app = FastAPI(title="Rating Prediction API")

# Load the nlptown sentiment analysis model as a text classification pipeline
# This model returns ratings like "1 star", "2 stars", ... "5 stars"
rating_model = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Define the request body schema using Pydantic
class ReviewRequest(BaseModel):
    review: str

# Create a POST endpoint to get the rating prediction
@app.post("/review", summary="Predict the rating from a review")
async def predict_rating(request: ReviewRequest):
    # Offload the blocking inference to a thread so as not to block the event loop
    result = await asyncio.to_thread(rating_model, request.review)
    return result

# A simple health check endpoint
@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "ok"}

# For local testing, run the server with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8002, workers=4)