# Offline Speech-to-Text (CPU-only, offline Whisper)

An offline, CPU-only speech-to-text scaffold for low-connectivity environments. The project uses open-source Whisper locally (no cloud APIs) and is structured to cleanly separate preprocessing, inference, keyword detection, backend API, and evaluation utilities.

## Offline and CPU Constraints
- 100% offline: no cloud calls, no OpenAI API usage.
- CPU-only inference to fit constrained environments.
- English-only MVP; extensible to more languages later.

## Tech Stack
- Python 3.10+
- FastAPI for the backend API
- Open-source Whisper + PyTorch for transcription
- librosa for audio preprocessing
- ONNX Runtime (optional) for CPU optimization

## Repository Structure
```
backend/
├── main.py              # FastAPI app entrypoint
├── api/                 # API routes
├── services/            # STT + keyword detection logic
├── models/              # Pydantic schemas
└── utils/               # Audio preprocessing helpers
evaluation/              # Evaluation utilities (WER/CER stubs)
requirements.txt         # Project dependencies
```

## Getting Started
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API locally (CPU-only):
   ```bash
   uvicorn backend.main:app --reload
   ```

## Usage
Send a WAV file to the transcription endpoint:
```bash
curl -X POST "http://localhost:8000/api/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.wav" \
  -F "keywords=hello" \
  -F "keywords=world"
```

Response shape:
```json
{
  "text": "placeholder transcription",
  "timestamps": [{"start": 0.0, "end": 0.0, "text": "placeholder transcription"}],
  "keywords": ["hello"]
}
```

## Next Steps
- Load Whisper models at startup and stream decoded timestamps.
- Add proper WER/CER evaluation utilities in `evaluation/`.
- Expand keyword detection to model-based keyword spotting.
- Add tests once implementation solidifies.
