"""
RealtimeTTS Production Server

A production-ready FastAPI server for text-to-speech synthesis
with support for multiple streaming protocols.

Features:
- REST API for synchronous generation
- Server-Sent Events (SSE) for streaming
- WebSocket for bidirectional streaming
- Multi-engine support with hot-swapping
- Voice management (CRUD operations)
- Health monitoring

Usage:
    python -m tts_server.app --engine chatterbox --port 8000

    # Or with uvicorn
    uvicorn tts_server.app:app --host 0.0.0.0 --port 8000
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import struct
import time
import wave
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .model_manager import ModelManager, get_model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Request/Response Models

class TTSRequest(BaseModel):
    """Request model for TTS generation."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="Voice ID to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature")
    exaggeration: Optional[float] = Field(None, ge=0.0, le=1.0, description="Exaggeration factor (Chatterbox)")
    cfg_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="CFG weight (Chatterbox)")
    language: Optional[str] = Field(None, description="Language code for multilingual")


class TTSStreamRequest(BaseModel):
    """Request model for streaming TTS."""
    text: str = Field(..., min_length=1, max_length=10000)
    voice_id: Optional[str] = None
    chunk_size: int = Field(4096, ge=1024, le=32768)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)


class VoiceCreateRequest(BaseModel):
    """Request model for voice creation."""
    name: str = Field(..., min_length=1, max_length=100)
    language: str = Field("en", min_length=2, max_length=10)


class EngineConfig(BaseModel):
    """Configuration for engine switching."""
    engine_type: str = Field(..., description="Engine type to use")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    engine: Optional[str]
    device: str
    voices_count: int


class VoiceResponse(BaseModel):
    """Voice information response."""
    id: str
    name: str
    engine: str
    language: str
    is_default: bool


# Audio utilities

def audio_to_wav_bytes(
    audio_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 4,  # float32 = 4 bytes
) -> bytes:
    """Convert raw audio bytes to WAV format."""
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # Convert to 16-bit for WAV
        wav_file.setframerate(sample_rate)

        # Convert float32 to int16 if necessary
        if sample_width == 4:
            import numpy as np
            float_data = np.frombuffer(audio_data, dtype=np.float32)
            int_data = (float_data * 32767).astype(np.int16)
            wav_file.writeframes(int_data.tobytes())
        else:
            wav_file.writeframes(audio_data)

    return wav_buffer.getvalue()


def create_wav_header(
    sample_rate: int = 24000,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Create a WAV header for streaming."""
    # Use a large placeholder for data size (will be actual size for complete files)
    data_size = 0x7FFFFFFF
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_size + 36,
        b"WAVE",
        b"fmt ",
        16,  # Subchunk1Size
        1,   # AudioFormat (PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


# App lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    manager = get_model_manager()

    engine_type = os.environ.get("TTS_ENGINE", "system")
    voices_path = os.environ.get("TTS_VOICES_PATH", "./voices")

    # Parse engine config from environment
    engine_config = {}
    if engine_type == "chatterbox":
        model_type = os.environ.get("CHATTERBOX_MODEL_TYPE", "standard")
        engine_config["model_type"] = model_type

    try:
        await manager.initialize(
            engine_type=engine_type,
            engine_config=engine_config,
            voices_path=voices_path,
        )
        logger.info(f"Server started with engine: {engine_type}")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        # Continue anyway - can be initialized later via API

    yield

    # Shutdown
    manager.shutdown()
    logger.info("Server shutdown complete")


# Create FastAPI app

app = FastAPI(
    title="RealtimeTTS API",
    description="Production-ready text-to-speech API with streaming support",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health & Status Endpoints

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check server health and status."""
    manager = get_model_manager()
    return HealthResponse(
        status="healthy" if manager.current_engine else "no_engine",
        engine=manager.current_engine_name,
        device=manager.device,
        voices_count=len(manager.get_voices()),
    )


@app.get("/v1/engines", tags=["Engines"])
async def list_engines():
    """List available and loaded engines."""
    manager = get_model_manager()
    return {
        "loaded": manager.available_engines,
        "current": manager.current_engine_name,
        "supported": [
            "system", "azure", "openai", "elevenlabs", "coqui",
            "kokoro", "edge", "gtts", "chatterbox"
        ],
    }


@app.post("/v1/engines/switch", tags=["Engines"])
async def switch_engine(config: EngineConfig):
    """Switch to a different TTS engine."""
    manager = get_model_manager()
    try:
        await manager.switch_engine(config.engine_type, config.config)
        return {
            "status": "success",
            "engine": config.engine_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# TTS Generation Endpoints

@app.post("/v1/tts/generate", tags=["TTS"])
async def generate_audio(request: TTSRequest):
    """
    Generate audio from text (synchronous, complete file).

    Returns a complete WAV file.
    """
    manager = get_model_manager()

    if not manager.current_engine:
        raise HTTPException(status_code=503, detail="No engine initialized")

    try:
        # Build kwargs from request
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.exaggeration is not None:
            kwargs["exaggeration"] = request.exaggeration
        if request.cfg_weight is not None:
            kwargs["cfg_weight"] = request.cfg_weight
        if request.language is not None:
            kwargs["language"] = request.language

        start_time = time.time()

        audio_data = await manager.generate(
            text=request.text,
            voice_id=request.voice_id,
            **kwargs,
        )

        generation_time = time.time() - start_time

        # Get stream info for proper conversion
        format_type, channels, sample_rate = manager.get_stream_info()

        # Determine sample width from format
        import pyaudio
        if format_type == pyaudio.paFloat32:
            sample_width = 4
        elif format_type == pyaudio.paInt16:
            sample_width = 2
        else:
            sample_width = 2

        wav_data = audio_to_wav_bytes(
            audio_data,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )

        return Response(
            content=wav_data,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": str(generation_time),
                "X-Audio-Duration": str(len(audio_data) / (sample_rate * sample_width)),
            },
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/tts/stream-sse", tags=["TTS"])
async def stream_sse(
    text: str = Query(..., min_length=1, max_length=10000),
    voice_id: Optional[str] = Query(None),
    chunk_size: int = Query(4096, ge=1024, le=32768),
    temperature: float = Query(0.7, ge=0.0, le=1.0),
):
    """
    Stream audio generation via Server-Sent Events.

    Events:
    - audio: Base64-encoded audio chunk
    - metrics: Generation metrics
    - done: Generation complete
    - error: Error occurred
    """
    manager = get_model_manager()

    if not manager.current_engine:
        raise HTTPException(status_code=503, detail="No engine initialized")

    async def event_generator():
        start_time = time.time()
        total_bytes = 0
        chunk_count = 0

        try:
            # Acquire lock for generation
            async with manager.request_lock:
                loop = asyncio.get_event_loop()

                # Run streaming in thread pool
                def generate():
                    return list(manager.generate_stream(
                        text=text,
                        voice_id=voice_id,
                        chunk_size=chunk_size,
                        temperature=temperature,
                    ))

                chunks = await loop.run_in_executor(None, generate)

                for chunk in chunks:
                    chunk_count += 1
                    total_bytes += len(chunk)

                    # Convert chunk to base64
                    audio_b64 = base64.b64encode(chunk).decode("utf-8")

                    yield f"event: audio\ndata: {json.dumps({'chunk': audio_b64, 'index': chunk_count})}\n\n"

            # Send metrics
            generation_time = time.time() - start_time
            metrics = {
                "generation_time": generation_time,
                "total_bytes": total_bytes,
                "chunk_count": chunk_count,
            }
            yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"

            # Done event
            yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"SSE streaming error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.websocket("/v1/tts/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for bidirectional TTS streaming.

    Client messages:
    - {"action": "generate", "text": "...", "voice_id": "...", "chunk_size": 4096}
    - {"action": "stop"}
    - {"action": "ping"}

    Server messages:
    - Binary audio chunks
    - {"type": "metrics", "data": {...}}
    - {"type": "done"}
    - {"type": "error", "message": "..."}
    - {"type": "pong"}
    """
    await websocket.accept()
    manager = get_model_manager()

    if not manager.current_engine:
        await websocket.send_json({"type": "error", "message": "No engine initialized"})
        await websocket.close()
        return

    stop_flag = asyncio.Event()

    try:
        while True:
            # Receive message
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                break

            action = data.get("action")

            if action == "ping":
                await websocket.send_json({"type": "pong"})

            elif action == "stop":
                stop_flag.set()
                await websocket.send_json({"type": "stopped"})

            elif action == "generate":
                stop_flag.clear()
                text = data.get("text", "")
                voice_id = data.get("voice_id")
                chunk_size = data.get("chunk_size", 4096)

                if not text:
                    await websocket.send_json({"type": "error", "message": "No text provided"})
                    continue

                start_time = time.time()
                total_bytes = 0
                chunk_count = 0

                try:
                    async with manager.request_lock:
                        loop = asyncio.get_event_loop()

                        def generate():
                            return manager.generate_stream(
                                text=text,
                                voice_id=voice_id,
                                chunk_size=chunk_size,
                            )

                        # Run generator in thread
                        generator = await loop.run_in_executor(None, generate)

                        for chunk in generator:
                            if stop_flag.is_set():
                                break

                            chunk_count += 1
                            total_bytes += len(chunk)

                            # Send binary chunk
                            await websocket.send_bytes(chunk)

                    # Send metrics
                    generation_time = time.time() - start_time
                    await websocket.send_json({
                        "type": "metrics",
                        "data": {
                            "generation_time": generation_time,
                            "total_bytes": total_bytes,
                            "chunk_count": chunk_count,
                        },
                    })

                    # Done
                    await websocket.send_json({"type": "done"})

                except Exception as e:
                    logger.error(f"WebSocket generation error: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        stop_flag.set()


# Voice Management Endpoints

@app.get("/v1/voices", response_model=list[VoiceResponse], tags=["Voices"])
async def list_voices():
    """List all available voices."""
    manager = get_model_manager()
    voices = manager.get_voices()

    return [
        VoiceResponse(
            id=v.id,
            name=v.name,
            engine=v.engine,
            language=v.language,
            is_default=v.is_default,
        )
        for v in voices
    ]


@app.post("/v1/voices/create", response_model=VoiceResponse, tags=["Voices"])
async def create_voice(
    name: str = Form(...),
    language: str = Form("en"),
    audio_file: UploadFile = File(...),
):
    """
    Create a new voice from an audio file.

    Audio should be 6-15 seconds for best results.
    """
    manager = get_model_manager()

    # Validate file type
    if not audio_file.filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: wav, mp3, flac, ogg",
        )

    # Read file
    audio_data = await audio_file.read()

    if len(audio_data) < 1000:
        raise HTTPException(status_code=400, detail="Audio file too small")

    if len(audio_data) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="Audio file too large (max 50MB)")

    try:
        voice_info = await manager.create_voice(
            name=name,
            audio_data=audio_data,
            language=language,
        )

        return VoiceResponse(
            id=voice_info.id,
            name=voice_info.name,
            engine=voice_info.engine,
            language=voice_info.language,
            is_default=voice_info.is_default,
        )

    except Exception as e:
        logger.error(f"Voice creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/voices/{voice_id}", tags=["Voices"])
async def delete_voice(voice_id: str):
    """Delete a voice."""
    manager = get_model_manager()

    if manager.delete_voice(voice_id):
        return {"status": "deleted", "voice_id": voice_id}
    else:
        raise HTTPException(status_code=404, detail="Voice not found or cannot be deleted")


@app.post("/v1/voices/{voice_id}/set", tags=["Voices"])
async def set_current_voice(voice_id: str):
    """Set the current voice for synthesis."""
    manager = get_model_manager()

    if manager.set_voice(voice_id):
        return {"status": "success", "voice_id": voice_id}
    else:
        raise HTTPException(status_code=404, detail="Voice not found")


# Main entry point

def main():
    """Run the server from command line."""
    parser = argparse.ArgumentParser(description="RealtimeTTS Production Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--engine", default="system", help="TTS engine to use")
    parser.add_argument("--model-type", default="standard", help="Model type for Chatterbox")
    parser.add_argument("--voices-path", default="./voices", help="Path to voices directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set environment variables for lifespan
    os.environ["TTS_ENGINE"] = args.engine
    os.environ["TTS_VOICES_PATH"] = args.voices_path
    if args.engine == "chatterbox":
        os.environ["CHATTERBOX_MODEL_TYPE"] = args.model_type

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    import uvicorn
    uvicorn.run(
        "tts_server.app:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
    )


if __name__ == "__main__":
    main()
