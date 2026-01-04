"""
Configuration management for Speite

This module handles configuration for the offline speech-to-text system.
All settings can be overridden via environment variables.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """
    Application settings with sensible defaults for offline operation.
    
    All settings can be overridden via environment variables with SPEITE_ prefix.
    """
    
    model_config = ConfigDict(
        env_prefix="SPEITE_",
        env_file=".env",
        case_sensitive=False
    )
    
    # Model configuration
    whisper_model_name: str = Field(
        default="base",
        description="Whisper model size: tiny, base, small, medium, large"
    )
    
    device: str = Field(
        default="cpu",
        description="Device for inference: cpu only (no GPU support in offline mode)"
    )
    
    language: str = Field(
        default="en",
        description="Language for transcription (English only for MVP)"
    )
    
    # Audio preprocessing
    sample_rate: int = Field(
        default=16000,
        description="Target sample rate for audio preprocessing"
    )
    
    max_audio_duration: int = Field(
        default=300,
        description="Maximum audio duration in seconds (5 minutes default)"
    )
    
    # API configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI host binding"
    )
    
    api_port: int = Field(
        default=8000,
        description="FastAPI port"
    )
    
    max_upload_size: int = Field(
        default=50 * 1024 * 1024,  # 50 MB
        description="Maximum upload file size in bytes"
    )
    
    # Model cache directory
    model_cache_dir: Optional[str] = Field(
        default=None,
        description="Directory to cache Whisper models (default: ~/.cache/whisper)"
    )


# Global settings instance
settings = Settings()
