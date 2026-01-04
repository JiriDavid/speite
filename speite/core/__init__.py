"""
Speech-to-text service using open-source Whisper model

This module provides the core speech-to-text functionality using
the locally-run open-source Whisper model (no cloud APIs).
"""

import logging
from typing import Dict, Optional, Any
import numpy as np
import torch
import whisper

from speite.config import settings

logger = logging.getLogger(__name__)


class SpeechToTextService:
    """
    Offline speech-to-text service using open-source Whisper.
    
    This service runs entirely offline with CPU-only inference,
    suitable for low-connectivity environments.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Initialize the speech-to-text service.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device for inference (cpu only)
            language: Target language (English only for MVP)
        """
        self.model_name = model_name or settings.whisper_model_name
        self.device = device or settings.device
        self.language = language or settings.language
        
        # Enforce CPU-only inference for offline compatibility
        if self.device != "cpu":
            logger.warning(f"Device {self.device} requested, but forcing CPU for offline mode")
            self.device = "cpu"
        
        # Enforce English language for MVP
        if self.language != "en":
            logger.warning(f"Language {self.language} requested, but only English supported in MVP")
            self.language = "en"
        
        self.model = None
        logger.info(
            f"SpeechToTextService initialized: model={self.model_name}, "
            f"device={self.device}, language={self.language}"
        )
    
    def load_model(self) -> None:
        """
        Load the Whisper model.
        
        This downloads the model on first use and caches it locally.
        Subsequent calls will use the cached model.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Download and cache the model
            # Models are cached in ~/.cache/whisper by default
            download_root = settings.model_cache_dir
            
            self.model = whisper.load_model(
                self.model_name,
                device=self.device,
                download_root=download_root
            )
            
            logger.info(f"Model {self.model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_data: Preprocessed audio as numpy array (16kHz mono)
            task: Task type - "transcribe" or "translate" (default: transcribe)
            **kwargs: Additional arguments to pass to Whisper
            
        Returns:
            Dictionary containing:
                - text: Transcribed text
                - segments: List of segments with timestamps
                - language: Detected language
                
        Raises:
            RuntimeError: If transcription fails
            ValueError: If model is not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            logger.info(f"Starting transcription (task={task})")
            
            # Prepare transcription options
            options = {
                "language": self.language,
                "task": task,
                "fp16": False,  # Disable FP16 for CPU compatibility
            }
            
            # Merge with any additional kwargs
            options.update(kwargs)
            
            # Perform transcription
            result = self.model.transcribe(
                audio_data,
                **options
            )
            
            # Extract relevant information
            transcription = {
                "text": result["text"].strip(),
                "segments": result.get("segments", []),
                "language": result.get("language", self.language),
            }
            
            logger.info(f"Transcription completed: {len(transcription['text'])} characters")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription error: {str(e)}")
    
    def transcribe_with_timestamps(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio with detailed timestamp information.
        
        Args:
            audio_data: Preprocessed audio as numpy array
            
        Returns:
            Dictionary with text and detailed segment timestamps
        """
        result = self.transcribe(audio_data)
        
        # Format segments with timestamps
        formatted_segments = []
        for segment in result.get("segments", []):
            formatted_segments.append({
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", "").strip(),
            })
        
        return {
            "text": result["text"],
            "segments": formatted_segments,
            "language": result["language"],
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "language": self.language,
            "loaded": self.model is not None,
        }
