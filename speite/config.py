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

    audio_trim_silence: bool = Field(
        default=True,
        description="Trim leading and trailing silence before transcription"
    )

    audio_trim_top_db: int = Field(
        default=24,
        description="Silence trimming aggressiveness in decibels"
    )

    audio_preemphasis: float = Field(
        default=0.97,
        description="Pre-emphasis coefficient to sharpen speech consonants"
    )

    audio_noise_floor_percentile: float = Field(
        default=15.0,
        description="Percentile used to estimate background noise floor"
    )

    audio_noise_gate_multiplier: float = Field(
        default=1.4,
        description="How strongly to gate low-energy background noise"
    )

    audio_noise_gate_attenuation: float = Field(
        default=0.25,
        description="Residual gain applied to samples below the offline noise gate"
    )

    audio_normalization_peak: float = Field(
        default=0.92,
        description="Peak level target for normalized audio"
    )

    live_audio_trim_silence: bool = Field(
        default=False,
        description="Trim leading and trailing silence during live chunk preprocessing"
    )

    live_audio_trim_top_db: int = Field(
        default=18,
        description="Silence trimming aggressiveness for live chunks"
    )

    live_audio_preemphasis: float = Field(
        default=0.985,
        description="Pre-emphasis coefficient for live speech enhancement"
    )

    live_audio_noise_floor_percentile: float = Field(
        default=35.0,
        description="Percentile used to estimate live background noise floor"
    )

    live_audio_noise_gate_multiplier: float = Field(
        default=2.8,
        description="Noise gate strength for live transcription"
    )

    live_audio_noise_gate_attenuation: float = Field(
        default=0.05,
        description="Residual gain applied below the live noise gate"
    )

    live_audio_normalization_peak: float = Field(
        default=0.82,
        description="Peak level target for live transcription chunks"
    )
    
    max_audio_duration: int = Field(
        default=900,
        description="Maximum audio duration in seconds (5 minutes default)"
    )

    streaming_min_audio_seconds: float = Field(
        default=0.75,
        description="Minimum buffered audio length before live transcription runs"
    )

    streaming_silence_threshold: float = Field(
        default=0.01,
        description="RMS threshold below which live audio chunks are treated as silence"
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
