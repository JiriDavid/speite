"""
Audio preprocessing utilities

This module provides functions for loading, validating, and preprocessing
audio files for speech-to-text transcription.
"""

import os
import logging
from typing import Tuple, Optional
import numpy as np
import librosa
import soundfile as sf

from speite.config import settings

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Audio preprocessing class for speech-to-text pipeline.
    
    Handles loading various audio formats, resampling to target sample rate,
    and validation of audio files.
    """
    
    def __init__(self, target_sample_rate: int = None):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sample_rate: Target sample rate for audio (default from settings)
        """
        self.target_sample_rate = target_sample_rate or settings.sample_rate
        logger.info(f"AudioPreprocessor initialized with sample rate: {self.target_sample_rate} Hz")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data with sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file cannot be loaded
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio using librosa (supports various formats)
            audio_data, sample_rate = librosa.load(
                audio_path,
                sr=None,  # Keep original sample rate initially
                mono=True  # Convert to mono
            )
            
            logger.info(f"Loaded audio from {audio_path}: {len(audio_data)} samples at {sample_rate} Hz")
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {str(e)}")
            raise ValueError(f"Cannot load audio file: {str(e)}")
    
    def resample_audio(self, audio_data: np.ndarray, original_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio_data: Audio samples as numpy array
            original_sr: Original sample rate
            
        Returns:
            Resampled audio data
        """
        if original_sr == self.target_sample_rate:
            logger.debug("Audio already at target sample rate, skipping resampling")
            return audio_data
        
        logger.info(f"Resampling audio from {original_sr} Hz to {self.target_sample_rate} Hz")
        resampled_audio = librosa.resample(
            audio_data,
            orig_sr=original_sr,
            target_sr=self.target_sample_rate
        )
        
        return resampled_audio
    
    def validate_audio(self, audio_data: np.ndarray) -> bool:
        """
        Validate audio data for processing.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            True if audio is valid
            
        Raises:
            ValueError: If audio validation fails
        """
        # Check if audio is empty
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")
        
        # Check duration
        duration = len(audio_data) / self.target_sample_rate
        if duration > settings.max_audio_duration:
            raise ValueError(
                f"Audio duration ({duration:.2f}s) exceeds maximum allowed "
                f"({settings.max_audio_duration}s)"
            )
        
        # Check for NaN or Inf values
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            raise ValueError("Audio contains invalid values (NaN or Inf)")
        
        logger.info(f"Audio validation passed: duration={duration:.2f}s")
        return True
    
    def preprocess(self, audio_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline: load, resample, and validate audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio data as numpy array
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio processing or validation fails
        """
        # Load audio
        audio_data, original_sr = self.load_audio(audio_path)
        
        # Resample if needed
        audio_data = self.resample_audio(audio_data, original_sr)
        
        # Validate
        self.validate_audio(audio_data)
        
        logger.info(f"Audio preprocessing completed for {audio_path}")
        return audio_data
    
    def save_audio(self, audio_data: np.ndarray, output_path: str) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio samples as numpy array
            output_path: Path to save audio file
        """
        sf.write(output_path, audio_data, self.target_sample_rate)
        logger.info(f"Audio saved to {output_path}")


def load_audio_from_bytes(audio_bytes: bytes, filename: str = "temp.wav") -> np.ndarray:
    """
    Load audio from bytes (useful for API uploads).
    
    Args:
        audio_bytes: Audio file content as bytes
        filename: Optional filename for format detection
        
    Returns:
        Preprocessed audio data as numpy array
        
    Raises:
        ValueError: If audio cannot be loaded from bytes
    """
    import tempfile
    
    preprocessor = AudioPreprocessor()
    
    # Create a secure temporary file with proper permissions
    # Use mkstemp for secure temp file creation
    file_extension = os.path.splitext(filename)[1]
    fd, temp_path = tempfile.mkstemp(suffix=file_extension)
    
    try:
        # Write bytes to temporary file using the file descriptor
        os.write(fd, audio_bytes)
        os.close(fd)  # Close the file descriptor
        
        # Process the temporary file
        audio_data = preprocessor.preprocess(temp_path)
        return audio_data
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
