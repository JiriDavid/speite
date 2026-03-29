"""
FastAPI backend for offline speech-to-text system

This module provides REST API endpoints for speech-to-text transcription.
All processing is done locally with no external API calls.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from speite.config import settings
from speite.core import DEFAULT_KEYWORDS, SpeechToTextService
from speite.utils import AudioPreprocessor, load_audio_from_bytes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Static assets live in the project root /static directory
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

# Initialize speech-to-text service
stt_service = SpeechToTextService()
audio_preprocessor = AudioPreprocessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Speite API server...")
    try:
        stt_service.load_model()
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        raise
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down Speite API server...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Speite - Offline Speech-to-Text API",
    description="Fully offline speech-to-text system using open-source Whisper",
    version="0.1.0",
    lifespan=lifespan
)

# Mount static assets for the browser UI when present
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionResponse(BaseModel):
    """Response model for transcription requests"""
    text: str
    language: str
    segments: Optional[list] = None
    keyword_hits: Optional[list] = None


def parse_keywords(raw_keywords: Optional[str], fallback: Optional[List[str]] = None) -> List[str]:
    """Parse comma/newline separated keywords or phrases with optional fallback."""
    if not raw_keywords:
        return list(fallback or [])

    keywords: List[str] = []
    for item in raw_keywords.replace("\n", ",").split(","):
        normalized = " ".join(item.strip().split())
        if normalized:
            keywords.append(normalized)
    return keywords


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]


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


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """
    Serve the built-in web UI for interactive transcription.
    """
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(index.read_text(encoding="utf-8"))


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
    include_timestamps: bool = Form(False, description="Include segment timestamps in response"),
    keywords: Optional[str] = Form(
        None,
        description="Optional comma/newline separated keywords or phrases for spotting",
    ),
    prompt: Optional[str] = Form(
        None,
        description="Optional initial prompt text to bias decoding",
    ),
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
        audio_data = load_audio_from_bytes(content, file.filename, profile="offline")
        
        parsed_keywords = parse_keywords(keywords, fallback=DEFAULT_KEYWORDS)

        decode_kwargs = {"initial_prompt": prompt} if prompt else {}

        # Transcribe
        if include_timestamps:
            result = stt_service.transcribe_with_timestamps(audio_data, **decode_kwargs)
        else:
            result = stt_service.transcribe(audio_data, **decode_kwargs)

        keyword_hits = stt_service.detect_keywords(result, parsed_keywords)

        # Remove segments if timestamps are not requested
        if not include_timestamps and "segments" in result:
            result["segments"] = None
        
        logger.info(f"Transcription completed for {file.filename}")
        
        return TranscriptionResponse(
            text=result["text"],
            language=result["language"],
            segments=result.get("segments"),
            keyword_hits=keyword_hits,
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


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming transcription.
    
    Expects audio chunks as binary data, responds with transcription updates.
    """
    await websocket.accept()
    logger.info("WebSocket connection established for streaming")
    
    accumulated_text = ""
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            logger.info(f"Received audio chunk: {len(data)} bytes")
            
            if not data:
                continue
            
            # Load audio from bytes
            try:
                # Data is raw 16-bit PCM at 16kHz mono
                pcm_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                logger.info(f"Loaded PCM audio chunk: {len(pcm_data)} samples")

                pcm_data = audio_preprocessor.preprocess_array(pcm_data, profile="live")
                
                # Transcribe the chunk
                result = await asyncio.to_thread(
                    stt_service.streaming_transcribe,
                    pcm_data,
                    accumulated_text,
                )
                logger.info(f"Transcription result: '{result['text']}'")
                
                if result["text"]:
                    # Update accumulated text
                    accumulated_text += " " + result["text"]
                    accumulated_text = accumulated_text.strip()
                    
                    # Send transcription update
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result["text"],
                        "accumulated_text": accumulated_text,
                        "segments": result["segments"],
                        "language": result["language"]
                    })
                    logger.info(f"Sent transcription update: '{accumulated_text}'")
                    
            except Exception as e:
                logger.error(f"Error processing audio chunk: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")


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
