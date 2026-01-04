#!/usr/bin/env python3
"""
Example: Simple Speech-to-Text Transcription

This example demonstrates how to use Speite for basic audio transcription.
"""

import sys
from speite.core import SpeechToTextService
from speite.utils import AudioPreprocessor


def main():
    """
    Example transcription workflow.
    """
    # Check if audio file provided
    if len(sys.argv) < 2:
        print("Usage: python example_simple.py <audio_file>")
        print("\nExample:")
        print("  python example_simple.py sample.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print("=" * 60)
    print("Speite - Simple Transcription Example")
    print("=" * 60)
    
    # Step 1: Initialize the speech-to-text service
    print("\n[1/4] Initializing speech-to-text service...")
    stt_service = SpeechToTextService(model_name="base")
    
    # Step 2: Load the Whisper model
    print("[2/4] Loading Whisper model (this may take a moment)...")
    stt_service.load_model()
    print("      Model loaded successfully!")
    
    # Step 3: Preprocess the audio file
    print(f"[3/4] Processing audio file: {audio_file}")
    preprocessor = AudioPreprocessor()
    
    try:
        audio_data = preprocessor.preprocess(audio_file)
        print(f"      Audio loaded: {len(audio_data)} samples")
    except FileNotFoundError:
        print(f"      ERROR: Audio file not found: {audio_file}")
        sys.exit(1)
    except ValueError as e:
        print(f"      ERROR: {str(e)}")
        sys.exit(1)
    
    # Step 4: Transcribe the audio
    print("[4/4] Transcribing audio...")
    result = stt_service.transcribe(audio_data)
    
    # Display results
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)
    print(f"\nText: {result['text']}\n")
    print(f"Language: {result['language']}")
    print(f"Number of segments: {len(result.get('segments', []))}")
    print("=" * 60)
    
    print("\n✓ Transcription completed successfully!")


if __name__ == "__main__":
    main()
