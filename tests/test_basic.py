"""
Tests for Speite offline speech-to-text system

This module contains basic tests for the audio preprocessing
and configuration components.
"""

import pytest
import numpy as np
from speite.config import settings
from speite.utils import AudioPreprocessor, get_preprocessing_profile
from speite.core import SpeechToTextService


class TestConfiguration:
    """Test configuration management"""
    
    def test_default_settings(self):
        """Test that default settings are correct"""
        assert settings.device == "cpu"
        assert settings.language == "en"
        assert settings.sample_rate == 16000
        assert settings.whisper_model_name in ["tiny", "base", "small", "medium", "large"]
    
    def test_api_settings(self):
        """Test API configuration"""
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.max_upload_size > 0

    def test_live_profile_is_more_aggressive(self):
        """Test live preprocessing profile uses stricter denoise settings"""
        offline = get_preprocessing_profile("offline")
        live = get_preprocessing_profile("live")

        assert live["noise_gate_multiplier"] > offline["noise_gate_multiplier"]
        assert live["noise_gate_attenuation"] < offline["noise_gate_attenuation"]


class TestAudioPreprocessor:
    """Test audio preprocessing utilities"""
    
    def test_preprocessor_initialization(self):
        """Test AudioPreprocessor initialization"""
        preprocessor = AudioPreprocessor()
        assert preprocessor.target_sample_rate == settings.sample_rate
    
    def test_preprocessor_custom_sample_rate(self):
        """Test AudioPreprocessor with custom sample rate"""
        custom_sr = 8000
        preprocessor = AudioPreprocessor(target_sample_rate=custom_sr)
        assert preprocessor.target_sample_rate == custom_sr
    
    def test_validate_audio_empty(self):
        """Test validation fails for empty audio"""
        preprocessor = AudioPreprocessor()
        empty_audio = np.array([])
        
        with pytest.raises(ValueError, match="empty"):
            preprocessor.validate_audio(empty_audio)
    
    def test_validate_audio_with_nan(self):
        """Test validation fails for audio with NaN values"""
        preprocessor = AudioPreprocessor()
        audio_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
        
        with pytest.raises(ValueError, match="invalid values"):
            preprocessor.validate_audio(audio_with_nan)
    
    def test_validate_audio_with_inf(self):
        """Test validation fails for audio with Inf values"""
        preprocessor = AudioPreprocessor()
        audio_with_inf = np.array([1.0, 2.0, np.inf, 4.0])
        
        with pytest.raises(ValueError, match="invalid values"):
            preprocessor.validate_audio(audio_with_inf)
    
    def test_validate_audio_duration_exceeds_max(self):
        """Test validation fails for audio exceeding maximum duration"""
        preprocessor = AudioPreprocessor()
        # Create audio longer than max duration
        long_audio = np.zeros(settings.sample_rate * (settings.max_audio_duration + 10))
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            preprocessor.validate_audio(long_audio)
    
    def test_validate_audio_success(self):
        """Test validation passes for valid audio"""
        preprocessor = AudioPreprocessor()
        # Create valid audio (1 second)
        valid_audio = np.random.randn(settings.sample_rate)
        
        assert preprocessor.validate_audio(valid_audio) is True

    def test_normalize_audio_removes_dc_offset(self):
        """Test normalization centers the waveform and constrains peak level"""
        preprocessor = AudioPreprocessor()
        audio = np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32)

        normalized = preprocessor.normalize_audio(
            audio,
            normalization_peak=settings.audio_normalization_peak,
        )

        assert abs(float(np.mean(normalized))) < 1e-6
        assert np.max(np.abs(normalized)) <= settings.audio_normalization_peak + 1e-6

    def test_reduce_noise_attenuates_background_floor(self):
        """Test lightweight denoising attenuates very quiet samples"""
        preprocessor = AudioPreprocessor()
        audio = np.array([0.001, 0.002, 0.25, 0.4], dtype=np.float32)

        reduced = preprocessor.reduce_noise(
            audio,
            noise_floor_percentile=settings.audio_noise_floor_percentile,
            noise_gate_multiplier=settings.audio_noise_gate_multiplier,
            noise_gate_attenuation=settings.audio_noise_gate_attenuation,
        )

        assert abs(reduced[0]) < abs(audio[0])
        assert abs(reduced[1]) < abs(audio[1])
        assert reduced.dtype == np.float32

    def test_trim_silence_reduces_leading_padding(self):
        """Test silence trimming removes quiet leading and trailing regions"""
        preprocessor = AudioPreprocessor()
        padded = np.concatenate(
            [
                np.zeros(settings.sample_rate // 4, dtype=np.float32),
                np.ones(settings.sample_rate // 2, dtype=np.float32) * 0.2,
                np.zeros(settings.sample_rate // 4, dtype=np.float32),
            ]
        )

        trimmed = preprocessor.trim_silence(
            padded,
            enabled=True,
            top_db=settings.audio_trim_top_db,
        )

        assert len(trimmed) < len(padded)

    def test_enhance_speech_returns_float32_audio(self):
        """Test speech enhancement preserves a valid float32 waveform"""
        preprocessor = AudioPreprocessor()
        audio = np.sin(
            np.linspace(0, 4 * np.pi, settings.sample_rate, dtype=np.float32)
        ) * 0.2

        enhanced = preprocessor.enhance_speech(audio)

        assert enhanced.dtype == np.float32
        assert np.max(np.abs(enhanced)) <= 1.0

    def test_live_enhance_speech_is_supported(self):
        """Test live enhancement profile returns a valid waveform"""
        preprocessor = AudioPreprocessor()
        audio = np.random.randn(settings.sample_rate).astype(np.float32) * 0.01

        enhanced = preprocessor.enhance_speech(audio, profile="live")

        assert enhanced.dtype == np.float32
        assert np.max(np.abs(enhanced)) <= 1.0
    
    def test_resample_audio_same_rate(self):
        """Test resampling when rates are the same"""
        preprocessor = AudioPreprocessor(target_sample_rate=16000)
        audio = np.random.randn(16000)
        
        resampled = preprocessor.resample_audio(audio, 16000)
        np.testing.assert_array_equal(audio, resampled)
    
    def test_load_audio_nonexistent_file(self):
        """Test loading non-existent audio file"""
        preprocessor = AudioPreprocessor()
        
        with pytest.raises(FileNotFoundError):
            preprocessor.load_audio("nonexistent_file.wav")


class TestKeywordSpotting:
    """Test keyword and phrase detection over transcript segments"""

    def test_detect_keywords_returns_expected_hits(self):
        result = {
            "text": "Fire near the emergency exit. Mark hazard zone.",
            "segments": [
                {"start": 0.0, "end": 4.0, "text": "Fire near the emergency exit."},
                {"start": 4.0, "end": 7.0, "text": "Mark hazard zone."},
            ],
            "language": "en",
        }

        hits = SpeechToTextService.detect_keywords(
            result,
            ["fire", "emergency exit", "hazard"],
        )

        assert len(hits) == 3
        assert {hit["keyword"] for hit in hits} == {"fire", "emergency exit", "hazard"}
        assert all(hit["start"] <= hit["end"] for hit in hits)
        assert all(hit["segment_index"] in (0, 1) for hit in hits)

    def test_detect_keywords_is_case_insensitive_and_boundary_aware(self):
        result = {
            "text": "The FIRE drill was confirmed. Firewall stayed unchanged.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "The FIRE drill was confirmed. Firewall stayed unchanged.",
                }
            ],
            "language": "en",
        }

        hits = SpeechToTextService.detect_keywords(result, ["fire"])

        assert len(hits) == 1
        assert hits[0]["match"].lower() == "fire"

    def test_detect_keywords_handles_empty_segments(self):
        result = {"text": "", "segments": [], "language": "en"}

        hits = SpeechToTextService.detect_keywords(result, ["safety"])

        assert hits == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
