"""
Speech-to-text service using open-source Whisper model

This module provides the core speech-to-text functionality using
the locally-run open-source Whisper model (no cloud APIs).
"""

import logging
import re
from typing import Dict, Optional, Any, List
import numpy as np
import whisper

from speite.config import settings

# Default keywords for emergency and medical scenarios
DEFAULT_KEYWORDS: List[str] = [
    "emergency",
    "fire",
    "smoke",
    "explosion",
    "hazard",
    "danger",
    "alarm",
    "evacuate",
    "evacuation",
    "mayday",
    "help",
    "medical emergency",
    "injury",
    "bleeding",
    "unconscious",
    "CPR",
    "defibrillator",
    "AED",
    "cardiac arrest",
    "breathing difficulty",
    "shortness of breath",
]


def get_default_keywords() -> List[str]:
    """Return a copy of the built-in keyword list."""
    return list(DEFAULT_KEYWORDS)

logger = logging.getLogger(__name__)


def _normalize_keyword(keyword: str) -> str:
    """Normalize user-supplied keyword input for matching."""
    return " ".join(keyword.strip().split())


def _build_keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Compile a case-insensitive boundary-aware regex for a keyword or phrase."""
    normalized = _normalize_keyword(keyword)
    escaped = re.escape(normalized)
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


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
        self.streaming_min_samples = int(
            settings.sample_rate * settings.streaming_min_audio_seconds
        )
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
    
    def transcribe_with_timestamps(self, audio_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio with detailed timestamp information.
        
        Args:
            audio_data: Preprocessed audio as numpy array
            **kwargs: Additional arguments to pass through to Whisper
            
        Returns:
            Dictionary with text and detailed segment timestamps
        """
        result = self.transcribe(audio_data, **kwargs)
        
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

    @staticmethod
    def detect_keywords(result: Dict[str, Any], keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Detect keyword and keyphrase hits in transcript segments.

        Args:
            result: Transcription dictionary containing a segments list.
            keywords: List of keywords or multi-word phrases.

        Returns:
            List of keyword hit dictionaries with timestamps.
        """
        if not result or not result.get("segments"):
            return []

        prepared_keywords = []
        for item in keywords:
            normalized = _normalize_keyword(item)
            if not normalized:
                continue
            prepared_keywords.append((normalized, _build_keyword_pattern(normalized)))

        if not prepared_keywords:
            return []

        hits: List[Dict[str, Any]] = []
        for segment_index, segment in enumerate(result.get("segments", [])):
            segment_text = (segment.get("text") or "").strip()
            if not segment_text:
                continue

            segment_start = float(segment.get("start", 0.0) or 0.0)
            segment_end = float(segment.get("end", segment_start) or segment_start)
            segment_duration = max(0.0, segment_end - segment_start)
            char_count = len(segment_text)

            for keyword, pattern in prepared_keywords:
                for match in pattern.finditer(segment_text):
                    if char_count > 0 and segment_duration > 0:
                        hit_start = segment_start + (match.start() / char_count) * segment_duration
                        hit_end = segment_start + (match.end() / char_count) * segment_duration
                    else:
                        hit_start = segment_start
                        hit_end = segment_end

                    hits.append(
                        {
                            "keyword": keyword,
                            "match": match.group(0),
                            "start": round(hit_start, 3),
                            "end": round(max(hit_start, hit_end), 3),
                            "segment_index": segment_index,
                            "segment_start": round(segment_start, 3),
                            "segment_end": round(segment_end, 3),
                            "segment_text": segment_text,
                        }
                    )

        hits.sort(key=lambda item: (item["start"], item["keyword"]))
        return hits
    
    def streaming_transcribe(self, audio_chunk: np.ndarray, previous_text: str = "") -> Dict[str, Any]:
        """
        Transcribe a chunk of audio in streaming mode.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            previous_text: Previous transcribed text for context
            
        Returns:
            Dictionary with new text and segments
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if len(audio_chunk) < self.streaming_min_samples:
            return {
                "text": "",
                "segments": [],
                "language": self.language,
            }

        rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
        if rms < settings.streaming_silence_threshold:
            logger.debug("Skipping streaming chunk below silence threshold")
            return {
                "text": "",
                "segments": [],
                "language": self.language,
            }
        
        try:
            logger.debug(f"Streaming transcription of chunk: {len(audio_chunk)} samples")
            
            prompt_words = previous_text.split()
            prompt = " ".join(prompt_words[-16:]) if prompt_words else None

            options = {
                "language": self.language,
                "task": "transcribe",
                "fp16": False,
                "initial_prompt": prompt,
                "beam_size": 1,
                "best_of": 1,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "without_timestamps": True,
                "verbose": False,
            }
            
            # Perform transcription
            result = self.model.transcribe(
                audio_chunk,
                **options
            )
            
            # Extract relevant information
            transcription = {
                "text": result["text"].strip(),
                "segments": result.get("segments", []),
                "language": result.get("language", self.language),
            }
            
            return transcription
            
        except Exception as e:
            logger.error(f"Streaming transcription failed: {str(e)}")
            raise RuntimeError(f"Streaming transcription error: {str(e)}")
    
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
