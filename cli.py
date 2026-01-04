#!/usr/bin/env python3
"""
Command-line interface for offline speech-to-text

This script provides a CLI tool for transcribing audio files
without running the API server.
"""

import sys
import argparse
import logging
from pathlib import Path

from speite.core import SpeechToTextService
from speite.utils import AudioPreprocessor
from speite.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """
    CLI entry point for speech-to-text transcription.
    """
    parser = argparse.ArgumentParser(
        description="Offline Speech-to-Text using open-source Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a single audio file
  python cli.py input.wav
  
  # Transcribe with timestamps
  python cli.py input.wav --timestamps
  
  # Use a different model size
  python cli.py input.wav --model small
  
  # Save output to file
  python cli.py input.wav --output transcription.txt
        """
    )
    
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file to transcribe"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=settings.whisper_model_name,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: %(default)s)"
    )
    
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in output"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: print to stdout)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    try:
        # Initialize components
        logger.info("Initializing speech-to-text service...")
        stt_service = SpeechToTextService(model_name=args.model)
        stt_service.load_model()
        
        preprocessor = AudioPreprocessor()
        
        # Process audio
        logger.info(f"Processing audio file: {args.audio_file}")
        audio_data = preprocessor.preprocess(str(audio_path))
        
        # Transcribe
        logger.info("Transcribing audio...")
        if args.timestamps:
            result = stt_service.transcribe_with_timestamps(audio_data)
        else:
            result = stt_service.transcribe(audio_data)
        
        # Format output
        output_text = format_output(result, args.timestamps)
        
        # Write output
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(output_text)
            logger.info(f"Transcription saved to: {args.output}")
        else:
            print("\n" + "=" * 60)
            print("TRANSCRIPTION")
            print("=" * 60)
            print(output_text)
            print("=" * 60)
        
        logger.info("Transcription completed successfully")
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=args.verbose)
        sys.exit(1)


def format_output(result, include_timestamps):
    """
    Format transcription result for output.
    
    Args:
        result: Transcription result dictionary
        include_timestamps: Whether to include timestamps
        
    Returns:
        Formatted output string
    """
    output_lines = []
    
    if include_timestamps and result.get("segments"):
        # Format with timestamps
        for segment in result["segments"]:
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            text = segment.get("text", "").strip()
            output_lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")
    else:
        # Simple text output
        output_lines.append(result["text"])
    
    return "\n".join(output_lines)


if __name__ == "__main__":
    main()
