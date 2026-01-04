# Speite Examples

This directory contains example scripts demonstrating how to use Speite for speech-to-text transcription.

## Available Examples

### 1. Simple Transcription (`example_simple.py`)

Basic transcription example showing the complete workflow.

```bash
python examples/example_simple.py audio_file.wav
```

### 2. Transcription with Timestamps (`example_timestamps.py`)

Get detailed timestamps for each segment of the transcription.

```bash
python examples/example_timestamps.py audio_file.wav
```

## Creating Sample Audio

If you don't have an audio file to test with, you can:

1. Record a short audio clip using your system's recorder
2. Download a sample from the internet
3. Use text-to-speech to generate audio:

```python
# Example using pyttsx3 (install with: pip install pyttsx3)
import pyttsx3

engine = pyttsx3.init()
engine.save_to_file("Hello, this is a test of the speech recognition system.", "sample.wav")
engine.runAndWait()
```

## Running the Examples

1. Make sure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

2. Run any example with an audio file:
```bash
python examples/example_simple.py your_audio.wav
```

Note: The first run will download the Whisper model (base model is ~140MB).
