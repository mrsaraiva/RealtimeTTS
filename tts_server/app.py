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
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


# μ-law and A-law encoding tables and functions (replaces deprecated audioop)
# These are ITU-T G.711 standard implementations

# μ-law encoding parameters
MULAW_BIAS = 0x84
MULAW_CLIP = 32635
MULAW_ENCODE_TABLE = np.array([
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
], dtype=np.uint8)


def _lin2mulaw(samples: np.ndarray) -> np.ndarray:
    """Convert 16-bit linear PCM samples to μ-law (ITU-T G.711)."""
    # Get sign bit
    sign = (samples >> 8) & 0x80
    # Get magnitude (absolute value)
    samples = np.where(samples < 0, -samples, samples)
    # Clip to max value
    samples = np.clip(samples, 0, MULAW_CLIP)
    # Add bias
    samples = samples + MULAW_BIAS
    # Get exponent from lookup table
    exponent = MULAW_ENCODE_TABLE[(samples >> 7) & 0xFF]
    # Get mantissa
    mantissa = (samples >> (exponent + 3)) & 0x0F
    # Combine and complement
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _lin2alaw(samples: np.ndarray) -> np.ndarray:
    """Convert 16-bit linear PCM samples to A-law (ITU-T G.711)."""
    # Get sign
    sign = ((~samples) >> 8) & 0x80
    # Get magnitude
    samples = np.where(samples < 0, -samples, samples)

    result = np.zeros_like(samples, dtype=np.uint8)

    # For samples >= 256
    mask = samples >= 256
    if np.any(mask):
        s = samples[mask]
        # Find exponent (position of highest bit)
        exponent = np.zeros_like(s)
        temp = s.copy()
        for i in range(7, 0, -1):
            bit_mask = temp >= (1 << (i + 8))
            exponent = np.where(bit_mask & (exponent == 0), i, exponent)

        mantissa = (s >> (exponent + 3)) & 0x0F
        result[mask] = ((exponent << 4) | mantissa).astype(np.uint8)

    # For samples < 256
    mask = samples < 256
    if np.any(mask):
        result[mask] = (samples[mask] >> 4).astype(np.uint8)

    return (sign | result) ^ 0x55

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
    engine: Optional[str] = Field(None, description="Engine to use (overrides default)")
    output_format: Optional[str] = Field(None, description="Output audio format (wav, pcm_24000_16, mulaw_8000, etc.)")
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
    loaded_engines: list[str]
    device: str
    voices_count: int


class VoiceResponse(BaseModel):
    """Voice information response."""
    id: str
    name: str
    engine: str
    language: str
    is_default: bool


# Audio format definitions

class AudioFormat(str, Enum):
    """Supported audio output formats."""
    WAV = "wav"                    # WAV file (for /generate endpoint)
    PCM_24000_16 = "pcm_24000_16"  # Raw PCM, 24kHz, 16-bit signed
    PCM_24000_F32 = "pcm_24000_f32"  # Raw PCM, 24kHz, float32
    PCM_16000_16 = "pcm_16000_16"  # Raw PCM, 16kHz, 16-bit signed
    PCM_8000_16 = "pcm_8000_16"    # Raw PCM, 8kHz, 16-bit signed
    MULAW_8000 = "mulaw_8000"      # μ-law, 8kHz (telephony)
    ALAW_8000 = "alaw_8000"        # A-law, 8kHz (telephony)


# Audio format metadata
AUDIO_FORMAT_INFO = {
    AudioFormat.WAV: {"sample_rate": 24000, "bits": 16, "encoding": "pcm", "channels": 1},
    AudioFormat.PCM_24000_16: {"sample_rate": 24000, "bits": 16, "encoding": "pcm", "channels": 1},
    AudioFormat.PCM_24000_F32: {"sample_rate": 24000, "bits": 32, "encoding": "float", "channels": 1},
    AudioFormat.PCM_16000_16: {"sample_rate": 16000, "bits": 16, "encoding": "pcm", "channels": 1},
    AudioFormat.PCM_8000_16: {"sample_rate": 8000, "bits": 16, "encoding": "pcm", "channels": 1},
    AudioFormat.MULAW_8000: {"sample_rate": 8000, "bits": 8, "encoding": "mulaw", "channels": 1},
    AudioFormat.ALAW_8000: {"sample_rate": 8000, "bits": 8, "encoding": "alaw", "channels": 1},
}


# Audio conversion utilities

def resample_audio(
    audio_data: bytes,
    src_rate: int,
    dst_rate: int,
    sample_width: int = 2,
) -> bytes:
    """Resample audio to a different sample rate using linear interpolation."""
    if src_rate == dst_rate:
        return audio_data

    # Convert to numpy for resampling
    if sample_width == 2:
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    elif sample_width == 4:
        samples = np.frombuffer(audio_data, dtype=np.float32)
    else:
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

    # Calculate new length
    new_length = int(len(samples) * dst_rate / src_rate)

    # Resample using linear interpolation
    x_old = np.linspace(0, 1, len(samples))
    x_new = np.linspace(0, 1, new_length)
    resampled = np.interp(x_new, x_old, samples)

    # Convert back to int16
    return resampled.astype(np.int16).tobytes()


def _safe_float32_to_int16(float_data: np.ndarray) -> np.ndarray:
    """
    Safely convert float32 audio data to int16, handling NaN/Inf values.

    Args:
        float_data: Float32 audio samples (expected range [-1.0, 1.0])

    Returns:
        Int16 audio samples
    """
    # Replace NaN with 0
    float_data = np.nan_to_num(float_data, nan=0.0, posinf=1.0, neginf=-1.0)
    # Clip to valid range [-1.0, 1.0]
    float_data = np.clip(float_data, -1.0, 1.0)
    # Convert to int16
    return (float_data * 32767).astype(np.int16)


def convert_to_mulaw(audio_data: bytes, sample_width: int = 2) -> bytes:
    """Convert PCM audio to μ-law encoding (ITU-T G.711)."""
    if sample_width == 4:
        # Convert float32 to int16 first (safely)
        float_data = np.frombuffer(audio_data, dtype=np.float32)
        samples = _safe_float32_to_int16(float_data)
    else:
        samples = np.frombuffer(audio_data, dtype=np.int16)

    # Convert to μ-law using numpy implementation
    mulaw_samples = _lin2mulaw(samples.astype(np.int32))
    return mulaw_samples.astype(np.uint8).tobytes()


def convert_to_alaw(audio_data: bytes, sample_width: int = 2) -> bytes:
    """Convert PCM audio to A-law encoding (ITU-T G.711)."""
    if sample_width == 4:
        # Convert float32 to int16 first (safely)
        float_data = np.frombuffer(audio_data, dtype=np.float32)
        samples = _safe_float32_to_int16(float_data)
    else:
        samples = np.frombuffer(audio_data, dtype=np.int16)

    # Convert to A-law using numpy implementation
    alaw_samples = _lin2alaw(samples.astype(np.int32))
    return alaw_samples.astype(np.uint8).tobytes()


def convert_audio_format(
    audio_data: bytes,
    src_rate: int,
    src_width: int,
    output_format: AudioFormat,
) -> bytes:
    """
    Convert audio data to the specified output format.

    Args:
        audio_data: Raw audio bytes
        src_rate: Source sample rate (e.g., 24000)
        src_width: Source sample width in bytes (2 for int16, 4 for float32)
        output_format: Target audio format

    Returns:
        Converted audio bytes
    """
    format_info = AUDIO_FORMAT_INFO[output_format]
    dst_rate = format_info["sample_rate"]
    encoding = format_info["encoding"]

    # First, convert float32 to int16 if needed
    if src_width == 4:
        float_data = np.frombuffer(audio_data, dtype=np.float32)
        audio_data = _safe_float32_to_int16(float_data).tobytes()
        src_width = 2

    # Resample if needed
    if src_rate != dst_rate:
        audio_data = resample_audio(audio_data, src_rate, dst_rate, src_width)

    # Apply encoding
    if encoding == "mulaw":
        audio_data = convert_to_mulaw(audio_data, sample_width=2)
    elif encoding == "alaw":
        audio_data = convert_to_alaw(audio_data, sample_width=2)
    elif encoding == "float" and format_info["bits"] == 32:
        # Convert int16 back to float32
        int_data = np.frombuffer(audio_data, dtype=np.int16)
        audio_data = (int_data / 32767.0).astype(np.float32).tobytes()

    return audio_data


def get_audio_format_headers(output_format: AudioFormat) -> dict:
    """Get HTTP headers describing the audio format."""
    info = AUDIO_FORMAT_INFO[output_format]
    return {
        "X-Audio-Format": output_format.value,
        "X-Audio-Sample-Rate": str(info["sample_rate"]),
        "X-Audio-Bits": str(info["bits"]),
        "X-Audio-Encoding": info["encoding"],
        "X-Audio-Channels": str(info["channels"]),
    }


# WAV utilities

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

def _parse_engine_config(engine_type: str) -> dict:
    """Parse engine-specific configuration from environment variables."""
    config = {}

    if engine_type == "chatterbox":
        model_type = os.environ.get("CHATTERBOX_MODEL_TYPE", "standard")
        config["model_type"] = model_type

    # Add more engine-specific configs here as needed

    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    manager = get_model_manager()

    voices_path = os.environ.get("TTS_VOICES_PATH", "./voices")

    # Check for multi-engine configuration
    engines_str = os.environ.get("TTS_ENGINES", "")
    default_engine = os.environ.get("TTS_DEFAULT_ENGINE", "")

    if engines_str:
        # Multi-engine mode: pre-load multiple engines
        engine_types = [e.strip() for e in engines_str.split(",") if e.strip()]

        # Build config for each engine
        engine_configs = {}
        for engine_type in engine_types:
            engine_configs[engine_type] = _parse_engine_config(engine_type)

        # Use TTS_DEFAULT_ENGINE or fall back to first engine
        if not default_engine:
            default_engine = engine_types[0] if engine_types else "system"

        try:
            await manager.initialize_engines(
                engine_types=engine_types,
                default_engine=default_engine,
                engine_configs=engine_configs,
                voices_path=voices_path,
            )
            logger.info(f"Server started with engines: {engine_types}, default: {default_engine}")
        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}")
    else:
        # Single engine mode (backwards compatible)
        engine_type = os.environ.get("TTS_ENGINE", "system")
        engine_config = _parse_engine_config(engine_type)

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
        loaded_engines=manager.available_engines,
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


@app.get("/v1/formats", tags=["Audio"])
async def list_audio_formats():
    """
    List supported audio output formats.

    Returns format details including sample rate, bit depth, and encoding.
    """
    return {
        "formats": {
            fmt.value: info
            for fmt, info in AUDIO_FORMAT_INFO.items()
        },
        "streaming_formats": [
            f.value for f in AudioFormat if f != AudioFormat.WAV
        ],
        "notes": {
            "wav": "Complete WAV file, only for /v1/tts/generate",
            "mulaw_8000": "μ-law 8kHz, standard for telephony (Twilio, etc.)",
            "alaw_8000": "A-law 8kHz, European telephony standard",
            "pcm_*": "Raw PCM, specify sample rate and bit depth",
        },
    }


# TTS Generation Endpoints

@app.post("/v1/tts/generate", tags=["TTS"])
async def generate_audio(request: TTSRequest):
    """
    Generate audio from text (synchronous, complete file).

    Returns audio in the requested format (default: WAV).

    Supported formats:
    - wav: WAV file (default)
    - pcm_24000_16: Raw PCM, 24kHz, 16-bit
    - pcm_16000_16: Raw PCM, 16kHz, 16-bit
    - pcm_8000_16: Raw PCM, 8kHz, 16-bit
    - mulaw_8000: μ-law, 8kHz (telephony)
    - alaw_8000: A-law, 8kHz (telephony)
    """
    manager = get_model_manager()

    if not manager.current_engine:
        raise HTTPException(status_code=503, detail="No engine initialized")

    # Parse output format
    output_format = AudioFormat.WAV
    if request.output_format:
        try:
            output_format = AudioFormat(request.output_format)
        except ValueError:
            valid_formats = [f.value for f in AudioFormat]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid output_format. Valid options: {valid_formats}",
            )

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
            engine=request.engine,
            **kwargs,
        )

        generation_time = time.time() - start_time

        # Get stream info for proper conversion (from the target engine)
        format_type, channels, sample_rate = manager.get_stream_info(request.engine)

        # Determine sample width from format
        import pyaudio
        if format_type == pyaudio.paFloat32:
            sample_width = 4
        elif format_type == pyaudio.paInt16:
            sample_width = 2
        else:
            sample_width = 2

        # Calculate duration before conversion
        duration = len(audio_data) / (sample_rate * sample_width)

        # Convert to requested format
        if output_format == AudioFormat.WAV:
            output_data = audio_to_wav_bytes(
                audio_data,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
            )
            media_type = "audio/wav"
        else:
            output_data = convert_audio_format(
                audio_data,
                src_rate=sample_rate,
                src_width=sample_width,
                output_format=output_format,
            )
            # Raw audio formats
            if "mulaw" in output_format.value:
                media_type = "audio/basic"  # Standard for μ-law
            elif "alaw" in output_format.value:
                media_type = "audio/basic"
            else:
                media_type = "audio/pcm"

        # Build response headers
        headers = {
            "X-Generation-Time": str(generation_time),
            "X-Audio-Duration": str(duration),
            **get_audio_format_headers(output_format),
        }

        return Response(
            content=output_data,
            media_type=media_type,
            headers=headers,
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/tts/stream-sse", tags=["TTS"])
async def stream_sse(
    text: str = Query(..., min_length=1, max_length=10000),
    voice_id: Optional[str] = Query(None),
    engine: Optional[str] = Query(None, description="Engine to use (overrides default)"),
    output_format: Optional[str] = Query(None, description="Output format (pcm_24000_16, mulaw_8000, etc.)"),
    chunk_size: int = Query(4096, ge=1024, le=32768),
    temperature: float = Query(0.7, ge=0.0, le=1.0),
    language: Optional[str] = Query(None, description="Language code for multilingual"),
):
    """
    Stream audio generation via Server-Sent Events.

    Events:
    - format: Audio format info (sent first)
    - audio: Base64-encoded audio chunk
    - metrics: Generation metrics
    - done: Generation complete
    - error: Error occurred

    Supported output_format values:
    - pcm_24000_16: Raw PCM, 24kHz, 16-bit (default)
    - pcm_16000_16: Raw PCM, 16kHz, 16-bit
    - pcm_8000_16: Raw PCM, 8kHz, 16-bit
    - mulaw_8000: μ-law, 8kHz (telephony)
    - alaw_8000: A-law, 8kHz (telephony)
    """
    manager = get_model_manager()

    if not manager.current_engine and not engine:
        raise HTTPException(status_code=503, detail="No engine initialized")

    # Parse output format (default to native PCM for streaming)
    fmt = AudioFormat.PCM_24000_16
    if output_format:
        try:
            fmt = AudioFormat(output_format)
            if fmt == AudioFormat.WAV:
                fmt = AudioFormat.PCM_24000_16  # WAV not suitable for streaming
        except ValueError:
            valid_formats = [f.value for f in AudioFormat if f != AudioFormat.WAV]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid output_format. Valid options for streaming: {valid_formats}",
            )

    async def event_generator():
        start_time = time.time()
        total_bytes = 0
        chunk_count = 0

        # Get source format info from the target engine
        format_type, channels, sample_rate = manager.get_stream_info(engine)
        import pyaudio
        if format_type == pyaudio.paFloat32:
            src_width = 4
        else:
            src_width = 2

        # Send format info first
        format_info = AUDIO_FORMAT_INFO[fmt]
        yield f"event: format\ndata: {json.dumps(format_info)}\n\n"

        try:
            # Acquire lock for generation
            async with manager.request_lock:
                loop = asyncio.get_event_loop()

                # Run streaming in thread pool
                def generate():
                    kwargs = {"temperature": temperature}
                    if language:
                        kwargs["language"] = language
                    return list(manager.generate_stream(
                        text=text,
                        voice_id=voice_id,
                        chunk_size=chunk_size,
                        engine=engine,
                        **kwargs,
                    ))

                chunks = await loop.run_in_executor(None, generate)

                for chunk in chunks:
                    chunk_count += 1

                    # Convert chunk to requested format
                    if fmt != AudioFormat.PCM_24000_F32 and fmt != AudioFormat.PCM_24000_16:
                        chunk = convert_audio_format(
                            chunk,
                            src_rate=sample_rate,
                            src_width=src_width,
                            output_format=fmt,
                        )

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

    # Include format info in headers too
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        **get_audio_format_headers(fmt),
    }

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.websocket("/v1/tts/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for bidirectional TTS streaming.

    Client messages:
    - {"action": "generate", "text": "...", "voice_id": "...", "engine": "...", "language": "...", "output_format": "mulaw_8000", "chunk_size": 4096}
    - {"action": "stop"}
    - {"action": "ping"}

    Server messages:
    - {"type": "format", "data": {...}}  (sent first, contains audio format info)
    - Binary audio chunks
    - {"type": "metrics", "data": {...}}
    - {"type": "done"}
    - {"type": "error", "message": "..."}
    - {"type": "pong"}

    Supported output_format values:
    - pcm_24000_16: Raw PCM, 24kHz, 16-bit (default)
    - pcm_16000_16: Raw PCM, 16kHz, 16-bit
    - pcm_8000_16: Raw PCM, 8kHz, 16-bit
    - mulaw_8000: μ-law, 8kHz (telephony)
    - alaw_8000: A-law, 8kHz (telephony)
    """
    await websocket.accept()
    manager = get_model_manager()

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
                engine = data.get("engine")
                language = data.get("language")
                output_format_str = data.get("output_format")
                chunk_size = data.get("chunk_size", 4096)

                if not text:
                    await websocket.send_json({"type": "error", "message": "No text provided"})
                    continue

                # Check engine availability
                if not manager.current_engine and not engine:
                    await websocket.send_json({"type": "error", "message": "No engine initialized"})
                    continue

                # Parse output format
                fmt = AudioFormat.PCM_24000_16
                if output_format_str:
                    try:
                        fmt = AudioFormat(output_format_str)
                        if fmt == AudioFormat.WAV:
                            fmt = AudioFormat.PCM_24000_16
                    except ValueError:
                        valid_formats = [f.value for f in AudioFormat if f != AudioFormat.WAV]
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Invalid output_format. Valid options: {valid_formats}",
                        })
                        continue

                # Get source format info from the target engine
                format_type, channels, sample_rate = manager.get_stream_info(engine)
                import pyaudio
                if format_type == pyaudio.paFloat32:
                    src_width = 4
                else:
                    src_width = 2

                # Send format info first
                format_info = AUDIO_FORMAT_INFO[fmt]
                await websocket.send_json({"type": "format", "data": format_info})

                start_time = time.time()
                total_bytes = 0
                chunk_count = 0

                try:
                    async with manager.request_lock:
                        loop = asyncio.get_event_loop()

                        def generate():
                            kwargs = {}
                            if language:
                                kwargs["language"] = language
                            return manager.generate_stream(
                                text=text,
                                voice_id=voice_id,
                                chunk_size=chunk_size,
                                engine=engine,
                                **kwargs,
                            )

                        # Run generator in thread
                        generator = await loop.run_in_executor(None, generate)

                        for chunk in generator:
                            if stop_flag.is_set():
                                break

                            # Convert chunk to requested format
                            if fmt != AudioFormat.PCM_24000_F32 and fmt != AudioFormat.PCM_24000_16:
                                chunk = convert_audio_format(
                                    chunk,
                                    src_rate=sample_rate,
                                    src_width=src_width,
                                    output_format=fmt,
                                )

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
                            "output_format": fmt.value,
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


@app.get("/v1/engines/{engine_name}/voices", response_model=list[VoiceResponse], tags=["Voices"])
async def list_engine_voices(engine_name: str):
    """
    List all voices available for a specific engine.

    This endpoint directly queries the engine for its built-in voices.

    **Kokoro voices** (54 built-in):
    - American English: af_heart, af_alloy, af_bella, am_adam, am_echo, etc.
    - British English: bf_alice, bf_emma, bm_daniel, bm_george, etc.
    - Japanese: jf_alpha, jf_gongitsune, jm_kumo, etc.
    - Chinese: zf_xiaobei, zm_yunjian, etc.
    - Spanish, French, Hindi, Italian, Portuguese

    **Chatterbox voices**:
    - Uses voice cloning from audio files (no built-in named voices)
    - Returns custom voices from voices directory + "default"
    """
    manager = get_model_manager()
    voices = manager.get_engine_voices(engine_name)

    if not voices:
        # Check if engine exists
        available = manager.available_engines
        if engine_name not in available:
            raise HTTPException(
                status_code=404,
                detail=f"Engine '{engine_name}' not found. Available: {available}",
            )

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
    parser.add_argument("--engine", default=None, help="Single TTS engine to use (use --engines for multiple)")
    parser.add_argument("--engines", default=None, help="Comma-separated list of engines to pre-load (e.g., 'chatterbox,kokoro')")
    parser.add_argument("--default-engine", default=None, help="Default engine when using --engines")
    parser.add_argument("--model-type", default="standard", help="Model type for Chatterbox")
    parser.add_argument("--voices-path", default="./voices", help="Path to voices directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set environment variables for lifespan
    os.environ["TTS_VOICES_PATH"] = args.voices_path

    if args.engines:
        # Multi-engine mode
        os.environ["TTS_ENGINES"] = args.engines
        if args.default_engine:
            os.environ["TTS_DEFAULT_ENGINE"] = args.default_engine
        # Check if chatterbox is in the list
        if "chatterbox" in args.engines:
            os.environ["CHATTERBOX_MODEL_TYPE"] = args.model_type
    elif args.engine:
        # Single engine mode
        os.environ["TTS_ENGINE"] = args.engine
        if args.engine == "chatterbox":
            os.environ["CHATTERBOX_MODEL_TYPE"] = args.model_type
    else:
        # Default to system engine
        os.environ["TTS_ENGINE"] = "system"

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
