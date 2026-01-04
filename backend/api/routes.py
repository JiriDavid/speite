"""API routes for the offline speech-to-text service."""

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.models.schemas import TranscriptionResponse
from backend.services.keyword_detection import detect_keywords
from backend.services.speech_to_text import transcribe_audio
from backend.utils.audio_preprocessing import preprocess_audio

router = APIRouter(tags=["speech-to-text"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(..., description="WAV audio file"),
    keywords: List[str] | None = None,
) -> TranscriptionResponse:
    """
    Accept a WAV file, preprocess audio, run offline speech-to-text, and detect keywords.
    """
    if file.content_type not in {"audio/wav", "audio/x-wav"}:
        raise HTTPException(status_code=400, detail="Only WAV audio is supported.")

    audio_bytes = await file.read()
    waveform, sample_rate = preprocess_audio(audio_bytes)
    text, timestamps = transcribe_audio(waveform, sample_rate)
    detected = detect_keywords(text, keywords or [])

    return TranscriptionResponse(text=text, timestamps=timestamps, keywords=detected)
