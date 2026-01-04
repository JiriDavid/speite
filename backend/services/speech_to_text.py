"""Offline speech-to-text utilities built around open-source Whisper."""

from typing import List, Tuple


def load_model(model_name: str = "base"):
    """
    Lazily load a Whisper model for CPU-only inference.

    This keeps initialization separate so that the API can decide when
    to load the model (e.g., on startup) and ensures the code path stays
    offline by using the local open-source Whisper package.
    """
    import whisper

    return whisper.load_model(model_name, device="cpu")


def transcribe_audio(
    waveform, sample_rate: int, model_name: str = "base"
) -> Tuple[str, List[dict]]:
    """
    Run offline transcription and return text plus rough timestamps.

    The default implementation uses Whisper in CPU mode. This placeholder
    keeps the signature and return format stable so it can be expanded
    later with full decoding and timestamp support.
    """
    _ = sample_rate  # placeholder until full implementation
    _ = model_name

    # NOTE: Real implementation will load a Whisper model and call model.transcribe.
    text = "placeholder transcription"
    timestamps = [{"start": 0.0, "end": 0.0, "text": text}]
    return text, timestamps
