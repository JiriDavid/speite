"""API schemas used by the FastAPI layer."""

from typing import List

from pydantic import BaseModel


class TimestampSegment(BaseModel):
    """Segment-level timestamp produced by the transcription model."""

    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    """Response model for transcription requests."""

    text: str
    timestamps: List[TimestampSegment]
    keywords: List[str]
