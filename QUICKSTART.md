# Speite Quick Start Guide

Get up and running with Speite in under 5 minutes!

## Step 1: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/JiriDavid/speite.git
cd speite

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The first time you run transcription, Whisper will download the model (~140MB for base model).

## Step 2: Choose Your Interface

### Option A: Command-Line Interface (Quickest)

Transcribe an audio file directly:

```bash
python cli.py your_audio.wav
```

With timestamps:

```bash
python cli.py your_audio.wav --timestamps
```

Save to file:

```bash
python cli.py your_audio.wav --output transcript.txt
```

### Option B: REST API Server

1. Start the server:

```bash
python main.py
```

2. In another terminal, send a request:

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@your_audio.wav"
```

3. Check the health:

```bash
curl http://localhost:8000/health
```

## Step 3: Test with Sample Audio

If you don't have an audio file, create a simple test:

### Using Python (with pyttsx3):

```python
# Install: pip install pyttsx3
import pyttsx3

engine = pyttsx3.init()
text = "Hello, this is a test of the Speite speech recognition system."
engine.save_to_file(text, "test_audio.wav")
engine.runAndWait()
```

Then transcribe:

```bash
python cli.py test_audio.wav
```

### Using Online Sample:

Download a sample from [Common Voice](https://commonvoice.mozilla.org/) or record your own using:
- Windows: Voice Recorder
- Mac: QuickTime Player
- Linux: Audacity or `arecord`

## Step 4: Explore Examples

Check out the example scripts:

```bash
# Simple transcription
python examples/example_simple.py your_audio.wav

# With detailed timestamps
python examples/example_timestamps.py your_audio.wav
```

## Configuration

### Basic Configuration

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` to customize:

```env
# Use a smaller/faster model
SPEITE_WHISPER_MODEL_NAME=tiny

# Change API port
SPEITE_API_PORT=9000
```

### Model Sizes

Choose based on your needs:

| Model | Speed | Accuracy | Recommended For |
|-------|-------|----------|-----------------|
| tiny  | Fast  | Good     | Testing, quick transcriptions |
| base  | Medium| Better   | **Default - balanced** |
| small | Slow  | Very Good| High accuracy needs |

## Common Use Cases

### Use Case 1: Batch Transcription

```bash
for file in *.wav; do
    python cli.py "$file" --output "${file%.wav}.txt"
done
```

### Use Case 2: API Integration

```python
import requests

files = {"file": open("audio.wav", "rb")}
response = requests.post("http://localhost:8000/transcribe", files=files)
print(response.json()["text"])
```

### Use Case 3: Python Script

```python
from speite.core import SpeechToTextService
from speite.utils import AudioPreprocessor

# Initialize
stt = SpeechToTextService(model_name="base")
stt.load_model()

# Process
preprocessor = AudioPreprocessor()
audio_data = preprocessor.preprocess("audio.wav")

# Transcribe
result = stt.transcribe(audio_data)
print(result["text"])
```

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution**: Install PyTorch:
```bash
pip install torch torchaudio
```

### Issue: Model download fails

**Solution**: 
1. Check internet connection (needed for first download)
2. Or manually download model and set cache directory:
```bash
export SPEITE_MODEL_CACHE_DIR=/path/to/models
```

### Issue: Transcription is slow

**Solution**: Use a smaller model:
```bash
export SPEITE_WHISPER_MODEL_NAME=tiny
```

### Issue: Audio file not supported

**Solution**: Convert to WAV using ffmpeg:
```bash
ffmpeg -i input.mp3 output.wav
```

## Next Steps

1. **Read the full documentation**: See [README.md](README.md)
2. **Understand the architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)
4. **Explore API docs**: Visit http://localhost:8000/docs when server is running

## Getting Help

- **Issues**: Open an issue on GitHub
- **Questions**: Check existing documentation
- **Examples**: Look in the `examples/` directory

## Quick Reference

### CLI Commands

```bash
# Basic transcription
python cli.py audio.wav

# With options
python cli.py audio.wav --model small --timestamps --output out.txt
```

### API Endpoints

```bash
# Health check
GET http://localhost:8000/health

# Transcribe
POST http://localhost:8000/transcribe
  - file: audio file
  - include_timestamps: true/false

# Model info
GET http://localhost:8000/models
```

### Python API

```python
from speite.core import SpeechToTextService
from speite.utils import AudioPreprocessor

# Load model
stt = SpeechToTextService()
stt.load_model()

# Transcribe
audio = AudioPreprocessor().preprocess("file.wav")
result = stt.transcribe(audio)
```

---

**You're ready to go!** Start transcribing audio with Speite 🎉
