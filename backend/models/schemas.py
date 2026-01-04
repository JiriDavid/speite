"""API schemas used by the FastAPI layer."""

from typing import Dict, List

from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """Response model for transcription requests."""

    text: str
    timestamps: List[Dict[str, float]]
    keywords: List[str]
