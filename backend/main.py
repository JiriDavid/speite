"""Entry point for the FastAPI application."""

from fastapi import FastAPI

from backend.api.routes import router as api_router


app = FastAPI(
    title="Offline Speech-to-Text API",
    description="CPU-only, offline Whisper-based speech-to-text with keyword detection.",
    version="0.1.0",
)

app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health() -> dict:
    """Lightweight health check to verify the API is running."""
    return {"status": "ok"}
