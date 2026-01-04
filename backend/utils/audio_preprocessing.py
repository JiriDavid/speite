"""Audio preprocessing utilities."""

import io
from typing import Tuple

import librosa
import numpy as np

EPSILON = 1e-9


def preprocess_audio(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Convert raw audio bytes to a normalized mono waveform at the target sample rate.

    The function keeps processing CPU-only and offline by relying on librosa.
    """
    with io.BytesIO(audio_bytes) as buffer:
        waveform, _ = librosa.load(buffer, sr=target_sr, mono=True)

    if waveform.size == 0:
        raise ValueError(
            "Audio file is empty or could not be loaded. Ensure the upload is a valid WAV file."
        )

    peak = np.max(np.abs(waveform))
    if peak < EPSILON:
        return waveform, target_sr

    waveform = waveform / peak
    return waveform, target_sr
