"""
FastAPI backend for offline speech-to-text system

This module provides REST API endpoints for speech-to-text transcription.
All processing is done locally with no external API calls.
"""

import logging
from typing import Optional, Dict, Any
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from speite.config import settings
from speite.core import SpeechToTextService
from speite.utils import load_audio_from_bytes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Speite - Offline Speech-to-Text API",
    description="Fully offline speech-to-text system using open-source Whisper",
    version="0.1.0",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize speech-to-text service
stt_service = SpeechToTextService()


class TranscriptionResponse(BaseModel):
    """Response model for transcription requests"""
    text: str
    language: str
    segments: Optional[list] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """
    Initialize the service on startup.
    Loads the Whisper model into memory.
    """
    logger.info("Starting Speite API server...")
    try:
        stt_service.load_model()
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "name": "Speite - Offline Speech-to-Text API",
        "version": "0.1.0",
        "status": "online",
        "description": "Fully offline speech-to-text using open-source Whisper"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status and model information
    """
    model_info = stt_service.get_model_info()
    
    return HealthResponse(
        status="healthy" if model_info["loaded"] else "not_ready",
        model_loaded=model_info["loaded"],
        model_info=model_info
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    include_timestamps: bool = Form(False, description="Include segment timestamps in response")
):
    """
    Transcribe audio file to text.
    
    Args:
        file: Audio file (WAV, MP3, FLAC, OGG, etc.)
        include_timestamps: Whether to include detailed timestamps for each segment
        
    Returns:
        Transcription result with text and optional timestamps
        
    Raises:
        HTTPException: If transcription fails
    """
    logger.info(f"Received transcription request for file: {file.filename}")
    
    # Validate file size
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size / 1024 / 1024:.2f} MB"
        )
    
    try:
        # Load and preprocess audio from bytes
        audio_data = load_audio_from_bytes(content, file.filename)
        
        # Transcribe
        if include_timestamps:
            result = stt_service.transcribe_with_timestamps(audio_data)
        else:
            result = stt_service.transcribe(audio_data)
            # Remove segments if timestamps not requested
            if not include_timestamps and "segments" in result:
                result["segments"] = None
        
        logger.info(f"Transcription completed for {file.filename}")
        
        return TranscriptionResponse(
            text=result["text"],
            language=result["language"],
            segments=result.get("segments")
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/models")
async def get_model_info():
    """
    Get information about the current model configuration.
    
    Returns:
        Model configuration details
    """
    return stt_service.get_model_info()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )
