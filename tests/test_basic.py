"""
Tests for Speite offline speech-to-text system

This module contains basic tests for the audio preprocessing
and configuration components.
"""

import pytest
import numpy as np
from speite.config import settings
from speite.utils import AudioPreprocessor


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
