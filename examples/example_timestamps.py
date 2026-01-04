#!/usr/bin/env python3
"""
Example: Transcription with Timestamps

This example demonstrates how to get detailed timestamps for each segment
of the transcription.
"""

import sys
from speite.core import SpeechToTextService
from speite.utils import AudioPreprocessor


def main():
    """
    Example transcription with timestamps.
    """
    # Check if audio file provided
    if len(sys.argv) < 2:
        print("Usage: python example_timestamps.py <audio_file>")
        print("\nExample:")
        print("  python example_timestamps.py sample.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print("=" * 60)
    print("Speite - Transcription with Timestamps Example")
    print("=" * 60)
    
    # Initialize and load model
    print("\nInitializing and loading model...")
    stt_service = SpeechToTextService(model_name="base")
    stt_service.load_model()
    
    # Preprocess audio
    print(f"Processing audio file: {audio_file}")
    preprocessor = AudioPreprocessor()
    
    try:
        audio_data = preprocessor.preprocess(audio_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)
    
    # Transcribe with timestamps
    print("Transcribing with timestamps...")
    result = stt_service.transcribe_with_timestamps(audio_data)
    
    # Display results
    print("\n" + "=" * 60)
    print("TRANSCRIPTION WITH TIMESTAMPS")
    print("=" * 60)
    
    if result.get("segments"):
        for i, segment in enumerate(result["segments"], 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            print(f"\n[Segment {i}]")
            print(f"  Time: {start:.2f}s - {end:.2f}s")
            print(f"  Text: {text}")
    else:
        print(f"\n{result['text']}")
    
    print("\n" + "=" * 60)
    print(f"Full text: {result['text']}")
    print(f"Language: {result['language']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
