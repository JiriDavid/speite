# Speite - Offline Speech-to-Text System

An offline speech-to-text system designed for low-connectivity environments. The system runs entirely offline using open-source tools and is suitable for academic review.

## Features

- **Fully Offline**: No cloud APIs or internet connection required after initial setup
- **Open Source**: Uses open-source Whisper model (no OpenAI API)
- **CPU-Only Inference**: Works on systems without GPU
- **English Language Support**: MVP focuses on English transcription
- **FastAPI Backend**: REST API for easy integration
- **Clean & Well-Commented Code**: Designed for academic review and extension

## Technology Stack

- **Python 3.10+**
- **OpenAI Whisper** (open-source, local inference)
- **PyTorch** (deep learning framework)
- **librosa** (audio preprocessing)
- **ONNX Runtime** (optional optimization)
- **FastAPI** (backend API)

## Requirements

- Python 3.10 or higher
- 4GB+ RAM (depending on model size)
- Storage for model files (~100MB - 3GB depending on model)

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/JiriDavid/speite.git
cd speite
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download Whisper model** (happens automatically on first use):
The model will be downloaded and cached locally when you first run the system.

## Usage

### Option 1: REST API Server

Start the FastAPI server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

**API Endpoints:**

- `GET /` - API information
- `GET /health` - Health check and model status
- `POST /transcribe` - Transcribe audio file
  - Upload audio file using multipart/form-data
  - Optional parameter: `include_timestamps` (boolean)
- `GET /models` - Get model configuration

**Example using curl:**

```bash
# Transcribe an audio file
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "include_timestamps=false"

# With timestamps
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "include_timestamps=true"
```

**Example using Python:**

```python
import requests

url = "http://localhost:8000/transcribe"
files = {"file": open("audio.wav", "rb")}
data = {"include_timestamps": False}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### Option 2: Command-Line Interface

Use the CLI tool for direct transcription:

```bash
# Basic transcription
python cli.py audio.wav

# With timestamps
python cli.py audio.wav --timestamps

# Use different model size
python cli.py audio.wav --model small

# Save to file
python cli.py audio.wav --output transcription.txt

# Verbose output
python cli.py audio.wav --verbose
```

## Configuration

The system can be configured using environment variables with the `SPEITE_` prefix:

```bash
# Model configuration
export SPEITE_WHISPER_MODEL_NAME=base  # tiny, base, small, medium, large
export SPEITE_DEVICE=cpu               # cpu only (enforced)
export SPEITE_LANGUAGE=en              # English only (MVP)

# API configuration
export SPEITE_API_HOST=0.0.0.0
export SPEITE_API_PORT=8000
export SPEITE_MAX_UPLOAD_SIZE=52428800  # 50 MB in bytes

# Audio settings
export SPEITE_SAMPLE_RATE=16000
export SPEITE_MAX_AUDIO_DURATION=300    # 5 minutes

# Model cache directory (optional)
export SPEITE_MODEL_CACHE_DIR=/path/to/models
```

Or create a `.env` file in the project root:

```env
SPEITE_WHISPER_MODEL_NAME=base
SPEITE_API_PORT=8000
```

## Whisper Model Sizes

Choose a model based on your requirements and available resources:

| Model  | Parameters | VRAM | Relative Speed | English WER |
|--------|------------|------|----------------|-------------|
| tiny   | 39 M       | ~1 GB | ~32x          | ~10%        |
| base   | 74 M       | ~1 GB | ~16x          | ~7%         |
| small  | 244 M      | ~2 GB | ~6x           | ~5%         |
| medium | 769 M      | ~5 GB | ~2x           | ~4%         |
| large  | 1550 M     | ~10 GB| 1x            | ~3%         |

**Recommendation**: Use `base` for a good balance of speed and accuracy.

## Project Structure

```
speite/
├── speite/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration management
│   ├── api/
│   │   └── __init__.py       # FastAPI endpoints
│   ├── core/
│   │   └── __init__.py       # Speech-to-text service
│   └── utils/
│       └── __init__.py       # Audio preprocessing utilities
├── tests/                    # Test files
├── sample_audio/             # Sample audio files for testing
├── main.py                   # API server entry point
├── cli.py                    # Command-line interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Supported Audio Formats

The system supports various audio formats through librosa:
- WAV
- MP3
- FLAC
- OGG
- M4A
- And more...

Audio is automatically:
- Converted to mono
- Resampled to 16kHz
- Validated for duration and quality

## Core Constraints (Compliance)

✅ **No cloud APIs**: All processing is done locally  
✅ **No OpenAI API**: Uses open-source Whisper with local inference  
✅ **CPU-only inference**: Optimized for CPU, no GPU required  
✅ **English language only**: MVP focuses on English (configurable)  
✅ **Clean, well-commented code**: Comprehensive documentation throughout  

## Examples

### Example 1: Transcribe an Audio File

```python
from speite.core import SpeechToTextService
from speite.utils import AudioPreprocessor

# Initialize
stt_service = SpeechToTextService(model_name="base")
stt_service.load_model()

preprocessor = AudioPreprocessor()

# Process audio
audio_data = preprocessor.preprocess("audio.wav")

# Transcribe
result = stt_service.transcribe(audio_data)
print(result["text"])
```

### Example 2: Transcribe with Timestamps

```python
from speite.core import SpeechToTextService
from speite.utils import AudioPreprocessor

stt_service = SpeechToTextService()
stt_service.load_model()

preprocessor = AudioPreprocessor()
audio_data = preprocessor.preprocess("audio.wav")

# Get detailed timestamps
result = stt_service.transcribe_with_timestamps(audio_data)

for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
```

## Development

### Running Tests

Tests can be added to the `tests/` directory following standard pytest conventions.

### Code Style

The codebase follows Python best practices:
- PEP 8 style guidelines
- Comprehensive docstrings
- Type hints where applicable
- Detailed comments for complex logic

## Troubleshooting

**Issue**: Model download fails  
**Solution**: Ensure you have a stable internet connection for the initial model download. Models are cached locally after first download.

**Issue**: Out of memory error  
**Solution**: Try a smaller model size (e.g., `tiny` or `base`) or reduce `max_audio_duration`.

**Issue**: Transcription is slow  
**Solution**: This is expected with CPU inference. Consider using a smaller model for faster processing.

**Issue**: Audio file not supported  
**Solution**: Convert your audio to a supported format (WAV, MP3, FLAC) using ffmpeg or similar tools.

## License

[Specify your license here]

## Academic Use

This system is designed for academic review and research. The codebase is clean, well-commented, and follows best practices for:
- Code organization
- Error handling
- Documentation
- Modularity and extensibility

## Contributing

Contributions are welcome! Please ensure code follows the existing style and includes appropriate documentation.

## Acknowledgments

- OpenAI Whisper team for the open-source model
- FastAPI team for the excellent web framework
- librosa developers for audio processing capabilities
