# Speite System Architecture

## Overview

Speite is an offline speech-to-text system designed for low-connectivity environments. It uses the open-source Whisper model for local inference with CPU-only processing.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Client Applications                   │
│  (CLI, Web Browser, Mobile App, Custom Integration)     │
└───────────────┬─────────────────────────────────────────┘
                │
                │ HTTP/REST
                │
┌───────────────▼─────────────────────────────────────────┐
│                  FastAPI Server                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  API Endpoints (/transcribe, /health, /models)    │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────┬─────────────────────────────────────────┘
                │
                │
┌───────────────▼─────────────────────────────────────────┐
│              Application Layer                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │        Speech-to-Text Service                      │ │
│  │  - Model loading and caching                       │ │
│  │  - Transcription orchestration                     │ │
│  │  - CPU-only inference                              │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────┬─────────────────────────────────────────┘
                │
                │
┌───────────────▼─────────────────────────────────────────┐
│            Processing Layer                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │       Audio Preprocessor                           │ │
│  │  - Format conversion                               │ │
│  │  - Resampling to 16kHz                            │ │
│  │  - Mono conversion                                │ │
│  │  - Validation                                      │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────┬─────────────────────────────────────────┘
                │
                │
┌───────────────▼─────────────────────────────────────────┐
│              ML Model Layer                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │       Open-Source Whisper Model                    │ │
│  │  - Local inference (no cloud)                      │ │
│  │  - CPU-only processing                             │ │
│  │  - English language support                        │ │
│  │  - Multiple model sizes available                  │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Module Structure

### 1. Configuration Module (`speite/config.py`)

**Purpose**: Centralized configuration management

**Key Features**:
- Environment variable support (SPEITE_* prefix)
- .env file loading
- Sensible defaults for offline operation
- Type validation using Pydantic

**Configuration Options**:
- Model selection (tiny, base, small, medium, large)
- Device selection (CPU enforced)
- Language setting (English for MVP)
- API settings (host, port, upload limits)
- Audio processing parameters

### 2. Audio Processing Module (`speite/utils/`)

**Purpose**: Audio preprocessing and validation

**Key Components**:
- `AudioPreprocessor`: Main preprocessing class
  - Audio loading (supports WAV, MP3, FLAC, OGG, etc.)
  - Format conversion and resampling
  - Validation (duration, quality checks)
  - Mono conversion

**Processing Pipeline**:
1. Load audio file (various formats supported via librosa)
2. Convert to mono channel
3. Resample to 16kHz (Whisper requirement)
4. Validate audio quality and duration
5. Return numpy array ready for inference

### 3. Core STT Service (`speite/core/`)

**Purpose**: Speech-to-text processing using Whisper

**Key Components**:
- `SpeechToTextService`: Main service class
  - Model loading and caching
  - Transcription execution
  - Timestamp generation
  - CPU-only enforcement

**Features**:
- Lazy model loading (on first use)
- Model caching (local storage)
- Support for different model sizes
- Detailed segment information
- Configurable transcription options

### 4. API Layer (`speite/api/`)

**Purpose**: REST API interface using FastAPI

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and model status |
| `/transcribe` | POST | Transcribe audio file |
| `/models` | GET | Current model configuration |

**API Features**:
- CORS support for cross-origin requests
- File upload handling (multipart/form-data)
- Request validation
- Error handling and logging
- Response streaming support

### 5. CLI Tool (`cli.py`)

**Purpose**: Command-line interface for direct transcription

**Features**:
- Single-file transcription
- Multiple model size support
- Timestamp option
- Output to file or stdout
- Verbose logging

### 6. Server Entry Point (`main.py`)

**Purpose**: Start the FastAPI server

**Features**:
- Configuration display
- Server initialization
- Logging setup
- Graceful startup/shutdown

## Data Flow

### Transcription Request Flow

```
1. Client uploads audio file → API endpoint
                ↓
2. File validation (size, format)
                ↓
3. Audio preprocessing (load, resample, validate)
                ↓
4. Audio data → SpeechToTextService
                ↓
5. Whisper model inference (CPU-only)
                ↓
6. Transcription result (text + segments)
                ↓
7. JSON response → Client
```

### Model Loading Flow

```
1. First transcription request
                ↓
2. Check if model loaded
                ↓
3. If not loaded:
   - Download model (if not cached)
   - Load into memory
   - Cache for subsequent requests
                ↓
4. Model ready for inference
```

## Design Decisions

### 1. CPU-Only Inference

**Rationale**: 
- Ensures compatibility with all systems
- No GPU drivers required
- Suitable for low-resource environments
- Predictable performance

**Trade-off**: 
- Slower inference compared to GPU
- Mitigated by smaller model options

### 2. English-Only MVP

**Rationale**:
- Simplified initial implementation
- Reduced complexity
- Focused testing
- Easier academic review

**Future Extension**:
- Multi-language support can be added by removing language constraint
- Whisper supports 99+ languages natively

### 3. Open-Source Whisper

**Rationale**:
- No API keys or cloud dependencies
- Completely offline operation
- Reproducible results
- Academic transparency
- No usage costs

**Benefits**:
- Privacy-preserving
- Works in air-gapped environments
- No rate limits
- Customizable

### 4. FastAPI Framework

**Rationale**:
- Modern, fast, and easy to use
- Automatic API documentation (OpenAPI/Swagger)
- Type validation with Pydantic
- Async support for future scaling
- Excellent developer experience

### 5. Modular Architecture

**Rationale**:
- Separation of concerns
- Easy to test individual components
- Extensible for new features
- Clear code organization
- Facilitates academic review

## Performance Considerations

### Model Size vs. Performance

| Model  | Size  | Speed (CPU) | Accuracy | Use Case |
|--------|-------|-------------|----------|----------|
| tiny   | 39M   | Fast        | Good     | Quick transcription |
| base   | 74M   | Moderate    | Better   | **Recommended default** |
| small  | 244M  | Slow        | Very Good| High accuracy needs |
| medium | 769M  | Very Slow   | Excellent| Production quality |
| large  | 1550M | Slowest     | Best     | Maximum accuracy |

### Optimization Strategies

1. **Model Selection**: Choose smallest model that meets accuracy needs
2. **Batch Processing**: Process multiple files sequentially (CLI mode)
3. **Audio Duration**: Limit input length to reduce processing time
4. **Caching**: Model loaded once and reused for all requests

## Security Considerations

### Input Validation

- File size limits (default: 50MB)
- Audio duration limits (default: 5 minutes)
- Format validation
- Content type verification

### Error Handling

- Graceful error messages (no stack traces to clients)
- Logging for debugging
- Input sanitization
- Resource cleanup

### Privacy

- No data sent to external services
- No telemetry or tracking
- Local processing only
- No persistent storage of audio files

## Testing Strategy

### Unit Tests

- Configuration validation
- Audio preprocessing functions
- Validation logic
- Utility functions

### Integration Tests

- API endpoint testing
- End-to-end transcription flow
- Error handling scenarios

### Manual Testing

- CLI tool functionality
- API server operation
- Different audio formats
- Various model sizes

## Deployment Options

### 1. Standalone Server

```bash
python main.py
```

### 2. Production Server (with Gunicorn)

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker speite.api:app
```

### 3. Docker Container

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### 4. Systemd Service

```ini
[Unit]
Description=Speite Speech-to-Text Service
After=network.target

[Service]
Type=simple
User=speite
WorkingDirectory=/opt/speite
ExecStart=/opt/speite/venv/bin/python main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Future Enhancements

### Potential Improvements

1. **Multi-language Support**: Remove English-only constraint
2. **GPU Acceleration**: Optional CUDA support for faster inference
3. **Streaming Audio**: Real-time transcription support
4. **ONNX Optimization**: Convert model to ONNX for better performance
5. **Speaker Diarization**: Identify different speakers
6. **Punctuation Enhancement**: Improve punctuation in output
7. **Custom Vocabulary**: Fine-tune for specific domains
8. **WebSocket Support**: Real-time streaming transcription
9. **Audio Quality Enhancement**: Pre-processing for noisy audio
10. **Batch Processing**: Process multiple files in one request

## Compliance with Requirements

### Core Constraints

✅ **No cloud APIs**: All processing is local  
✅ **No OpenAI API**: Uses open-source Whisper locally  
✅ **CPU-only inference**: Device forced to "cpu"  
✅ **English language only**: Language forced to "en"  
✅ **Clean, well-commented code**: Comprehensive documentation

### Technology Stack

✅ **Python 3.10+**: Tested with 3.10-3.12  
✅ **Open-source Whisper**: openai-whisper package  
✅ **PyTorch**: Required by Whisper  
✅ **librosa**: Audio preprocessing  
✅ **ONNX Runtime**: Optional (included)  
✅ **FastAPI**: Backend framework  

## Conclusion

Speite provides a complete, offline speech-to-text solution suitable for academic review and low-connectivity environments. The modular architecture allows for easy extension while maintaining simplicity and clarity in the codebase.
