"""Placeholder utilities for evaluating transcription quality."""

from typing import Dict


def evaluate_transcription(reference: str, hypothesis: str) -> Dict[str, float | None]:
    """
    Compare reference and hypothesis transcripts.

    This stub leaves room for WER/CER computations using open-source metrics
    while keeping the current implementation simple for the project scaffold.
    """
    _ = (reference, hypothesis)
    return {"wer": None, "cer": None}
