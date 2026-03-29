"""
Audio preprocessing utilities

This module provides functions for loading, validating, and preprocessing
audio files for speech-to-text transcription.
"""

import os
import logging
import subprocess
import tempfile
from typing import Tuple
import numpy as np
import librosa
import soundfile as sf

from speite.config import settings

logger = logging.getLogger(__name__)


def _convert_audio_to_wav_with_ffmpeg(audio_path: str) -> str:
    """
    Convert unsupported audio formats to mono 16k WAV using bundled ffmpeg.

    Returns:
        Path to a temporary WAV file.
    """
    try:
        import imageio_ffmpeg
    except Exception as exc:
        raise ValueError(
            "Audio format is not supported by current backend. "
            "Install imageio-ffmpeg to enable fallback decoding."
        ) from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        temp_wav,
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        stderr_tail = completed.stderr[-1000:] if completed.stderr else ""
        raise ValueError(
            "ffmpeg fallback conversion failed while decoding audio. "
            f"Details: {stderr_tail}"
        )

    return temp_wav


def get_preprocessing_profile(profile: str = "offline") -> dict:
    """
    Return the configured preprocessing profile.
    """
    if profile == "live":
        return {
            "trim_silence": settings.live_audio_trim_silence,
            "trim_top_db": settings.live_audio_trim_top_db,
            "noise_floor_percentile": settings.live_audio_noise_floor_percentile,
            "noise_gate_multiplier": settings.live_audio_noise_gate_multiplier,
            "noise_gate_attenuation": settings.live_audio_noise_gate_attenuation,
            "normalization_peak": settings.live_audio_normalization_peak,
            "preemphasis": settings.live_audio_preemphasis,
        }

    return {
        "trim_silence": settings.audio_trim_silence,
        "trim_top_db": settings.audio_trim_top_db,
        "noise_floor_percentile": settings.audio_noise_floor_percentile,
        "noise_gate_multiplier": settings.audio_noise_gate_multiplier,
        "noise_gate_attenuation": settings.audio_noise_gate_attenuation,
        "normalization_peak": settings.audio_normalization_peak,
        "preemphasis": settings.audio_preemphasis,
    }


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
            # Load audio using librosa (supports many formats with available backends)
            audio_data, sample_rate = librosa.load(
                audio_path,
                sr=None,  # Keep original sample rate initially
                mono=True,  # Convert to mono
            )

            logger.info(f"Loaded audio from {audio_path}: {len(audio_data)} samples at {sample_rate} Hz")
            return audio_data, sample_rate

        except Exception as first_error:
            logger.warning(
                "Primary audio load failed for %s. Trying ffmpeg fallback decode. Error: %s",
                audio_path,
                str(first_error),
            )

            temp_wav_path = None
            try:
                temp_wav_path = _convert_audio_to_wav_with_ffmpeg(audio_path)
                audio_data, sample_rate = librosa.load(
                    temp_wav_path,
                    sr=None,
                    mono=True,
                )
                logger.info(
                    "Loaded audio via ffmpeg fallback from %s: %s samples at %s Hz",
                    audio_path,
                    len(audio_data),
                    sample_rate,
                )
                return audio_data, sample_rate
            except Exception as second_error:
                logger.error(f"Error loading audio file {audio_path}: {str(second_error)}")
                raise ValueError(f"Cannot load audio file: {str(second_error)}")
            finally:
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
    
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

    def trim_silence(self, audio_data: np.ndarray, enabled: bool, top_db: int) -> np.ndarray:
        """
        Trim leading and trailing silence to focus inference on speech.
        """
        if not enabled or len(audio_data) == 0:
            return audio_data

        trimmed_audio, _ = librosa.effects.trim(
            audio_data,
            top_db=top_db,
        )

        if len(trimmed_audio) == 0:
            logger.warning("Silence trimming removed all audio; keeping original signal")
            return audio_data

        return trimmed_audio

    def reduce_noise(
        self,
        audio_data: np.ndarray,
        noise_floor_percentile: float,
        noise_gate_multiplier: float,
        noise_gate_attenuation: float,
    ) -> np.ndarray:
        """
        Apply a lightweight noise gate based on the estimated background floor.
        """
        if len(audio_data) == 0:
            return audio_data

        abs_audio = np.abs(audio_data)
        noise_floor = np.percentile(abs_audio, noise_floor_percentile)
        gate_threshold = noise_floor * noise_gate_multiplier

        if gate_threshold <= 0:
            return audio_data

        gated_audio = audio_data.copy()
        below_gate = abs_audio < gate_threshold
        gated_audio[below_gate] *= noise_gate_attenuation

        return np.asarray(gated_audio, dtype=np.float32)

    def normalize_audio(self, audio_data: np.ndarray, normalization_peak: float) -> np.ndarray:
        """
        Remove DC offset and normalize peak amplitude.
        """
        if len(audio_data) == 0:
            return audio_data

        centered_audio = audio_data - np.mean(audio_data)
        peak = np.max(np.abs(centered_audio))
        if peak <= 0:
            return centered_audio

        scale = normalization_peak / peak
        return centered_audio * scale

    def enhance_speech(self, audio_data: np.ndarray, profile: str = "offline") -> np.ndarray:
        """
        Run the speech enhancement pipeline used before Whisper inference.
        """
        config = get_preprocessing_profile(profile)
        processed_audio = np.asarray(audio_data, dtype=np.float32)
        processed_audio = self.trim_silence(
            processed_audio,
            enabled=config["trim_silence"],
            top_db=config["trim_top_db"],
        )
        processed_audio = self.reduce_noise(
            processed_audio,
            noise_floor_percentile=config["noise_floor_percentile"],
            noise_gate_multiplier=config["noise_gate_multiplier"],
            noise_gate_attenuation=config["noise_gate_attenuation"],
        )
        processed_audio = self.normalize_audio(
            processed_audio,
            normalization_peak=config["normalization_peak"],
        )

        if 0.0 < config["preemphasis"] < 1.0:
            processed_audio = librosa.effects.preemphasis(
                processed_audio,
                coef=config["preemphasis"],
            )

        return np.clip(processed_audio, -1.0, 1.0).astype(np.float32)

    def preprocess_array(self, audio_data: np.ndarray, profile: str = "offline") -> np.ndarray:
        """
        Preprocess an already-loaded waveform.
        """
        processed_audio = self.enhance_speech(audio_data, profile=profile)
        self.validate_audio(processed_audio)
        return processed_audio
    
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
    
    def preprocess(self, audio_path: str, profile: str = "offline") -> np.ndarray:
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

        # Speech-focused cleanup
        audio_data = self.enhance_speech(audio_data, profile=profile)
        
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


def load_audio_from_bytes(
    audio_bytes: bytes,
    filename: str = "temp.wav",
    profile: str = "offline",
) -> np.ndarray:
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
        audio_data = preprocessor.preprocess(temp_path, profile=profile)
        return audio_data
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
