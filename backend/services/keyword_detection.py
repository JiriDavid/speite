"""Simple keyword detection on transcribed text."""

from typing import Iterable, List


def detect_keywords(transcript: str, keywords: Iterable[str]) -> List[str]:
    """
    Perform lightweight keyword matching on the transcript.

    The logic is intentionally simple (case-insensitive substring checks)
    to keep the CPU-only constraint and allow later upgrades to more
    advanced keyword spotting models if needed.
    """
    transcript_lower = transcript.lower()
    return [kw for kw in keywords if kw.lower() in transcript_lower]
