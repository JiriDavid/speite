#!/usr/bin/env python3
"""
Test script for WebSocket streaming transcription
"""

import asyncio
import websockets
import json
import numpy as np
import librosa

async def test_websocket():
    uri = "ws://localhost:8000/ws/stream"

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")

        # Load a test audio file
        audio_data, sr = librosa.load("sermon.ogg", sr=16000, mono=True)
        print(f"Audio loaded: {len(audio_data)} samples, {sr}Hz")

        # Convert to 16-bit PCM
        pcm_data = (audio_data * 32767).astype(np.int16)

        # Send in chunks
        chunk_size = 16000 * 5  # 5 seconds at 16kHz
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i+chunk_size]
            if len(chunk) == 0:
                break

            # Send as bytes
            await websocket.send(chunk.tobytes())
            print(f"Sent chunk {i//chunk_size + 1}: {len(chunk)} samples")

            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                print(f"Received: {data}")
            except asyncio.TimeoutError:
                print("Timeout waiting for response")

            # Wait a bit between chunks
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_websocket())