# RealtimeTTS

*Easy to use, low-latency text-to-speech library for realtime applications*

> Fork of [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) with additional engines and a production-ready streaming server.

## What's New in This Fork

### ChatterboxEngine

A unified TTS engine supporting three Chatterbox model variants with voice cloning:

| Model | Size | Languages | Features |
|-------|------|-----------|----------|
| `standard` | 500M | English | Creative controls (CFG, exaggeration) |
| `turbo` | 350M | English | Fastest inference, paralinguistic tags (`[laugh]`, `[cough]`) |
| `multilingual` | 500M | 23+ | Zero-shot voice cloning across languages |

```python
from RealtimeTTS import TextToAudioStream, ChatterboxEngine

# Standard model with voice cloning
engine = ChatterboxEngine(model_type="standard", voice="path/to/reference.wav")

# Turbo for low-latency applications
engine = ChatterboxEngine(model_type="turbo")

# Multilingual with Spanish
engine = ChatterboxEngine(model_type="multilingual", language="es")

stream = TextToAudioStream(engine)
stream.feed("Hello world!")
stream.play()
```

### Production TTS Server

A FastAPI server for text-to-speech with multiple streaming protocols:

- **REST API**: Synchronous audio generation
- **SSE Streaming**: Server-Sent Events for real-time audio
- **WebSocket**: Bidirectional streaming with stop/ping support
- **Multi-Engine**: Hot-swap between engines via API
- **Voice Management**: Create, list, delete custom voices
- **Telephony Formats**: μ-law and A-law for Twilio/Vonage integration

```bash
# Run server with Chatterbox
python -m tts_server.app --engine chatterbox --port 8000

# Multi-engine mode (pre-load multiple engines)
python -m tts_server.app --engines chatterbox,kokoro --default-engine chatterbox

# Docker
docker-compose up -d
```

See [tts_server/README.md](tts_server/README.md) for full API documentation.

---

## About the Project

RealtimeTTS is a state-of-the-art text-to-speech (TTS) library designed for real-time applications. It converts text streams into high-quality auditory output with minimal latency, making it ideal for LLM integrations and interactive voice applications.

## Key Features

- **Low Latency**: Almost instantaneous text-to-speech conversion, compatible with LLM outputs
- **High-Quality Audio**: Clear and natural-sounding speech
- **Multiple TTS Engine Support**: 13+ engines including cloud and local options
- **Voice Cloning**: Clone voices from short audio samples (Chatterbox, Coqui)
- **Multilingual**: Support for 23+ languages
- **Production Ready**: Streaming server with REST, SSE, and WebSocket APIs
- **Fallback Mechanism**: Switch to alternative engines in case of disruptions

## Supported Engines

| Engine | Type | Features |
|--------|------|----------|
| **ChatterboxEngine** | 🏠 Local | Voice cloning, 3 model variants, 23+ languages |
| **KokoroEngine** | 🏠 Local | Fast, 54 voices, 9 languages |
| **CoquiEngine** | 🏠 Local | Voice cloning, high quality |
| **OpenAIEngine** | 🌐 Cloud | 6 premium voices |
| **AzureEngine** | 🌐 Cloud | 500k free chars/month |
| **ElevenlabsEngine** | 🌐 Cloud | Premium voice quality |
| **EdgeEngine** | 🌐 Cloud | Free Microsoft TTS |
| **GTTSEngine** | 🌐 Cloud | Free Google TTS |
| **OrpheusEngine** | 🏠 Local | Llama-powered, emotion tags |
| **ZipVoiceEngine** | 🏠 Local | 123M zero-shot model |
| **StyleTTS2Engine** | 🏠 Local | Expressive, natural speech |
| **ParlerEngine** | 🏠 Local | Neural TTS for high-end GPUs |
| **PiperEngine** | 🏠 Local | Fast, runs on Raspberry Pi |
| **SystemEngine** | 🏠 Local | Built-in system TTS |

🏠 Local processing (no internet required) | 🌐 Requires internet connection

## Installation

```bash
# Full installation with all engines
pip install realtimetts[all]

# Specific engines
pip install realtimetts[chatterbox]
pip install realtimetts[chatterbox,kokoro]
pip install realtimetts[azure,openai]

# Minimal (for custom engine development)
pip install realtimetts[minimal]
```

**System dependencies:**
- Linux: `apt-get install -y portaudio19-dev`
- macOS: `brew install portaudio`

## Quick Start

### Basic Usage

```python
from RealtimeTTS import TextToAudioStream, SystemEngine

engine = SystemEngine()  # or ChatterboxEngine(), KokoroEngine(), etc.
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

### Streaming from LLM

```python
from RealtimeTTS import TextToAudioStream, ChatterboxEngine

def llm_generator():
    for chunk in openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True
    ):
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

engine = ChatterboxEngine(model_type="turbo")
stream = TextToAudioStream(engine)
stream.feed(llm_generator())
stream.play_async()
```

### Voice Cloning

```python
from RealtimeTTS import TextToAudioStream, ChatterboxEngine

# Clone a voice from a reference audio file (6-15 seconds recommended)
engine = ChatterboxEngine(
    model_type="standard",
    voice="path/to/voice_sample.wav"
)
stream = TextToAudioStream(engine)
stream.feed("This will sound like the reference voice!")
stream.play()
```

### Multilingual Synthesis

```python
from RealtimeTTS import TextToAudioStream, ChatterboxEngine

engine = ChatterboxEngine(
    model_type="multilingual",
    language="fr",
    voice="path/to/french_speaker.wav"
)
stream = TextToAudioStream(engine)
stream.feed("Bonjour, comment allez-vous?")
stream.play()
```

## Production Server

### Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn python-multipart
pip install realtimetts[chatterbox]

# Run server
python -m tts_server.app --engine chatterbox --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/tts/generate` | POST | Generate complete WAV |
| `/v1/tts/stream-sse` | GET | SSE streaming |
| `/v1/tts/stream` | WS | WebSocket streaming |
| `/v1/voices` | GET | List voices |
| `/v1/voices/create` | POST | Create voice from audio |
| `/v1/engines` | GET | List engines |
| `/v1/engines/switch` | POST | Switch TTS engine |

### Example: Generate Audio

```bash
curl -X POST http://localhost:8000/v1/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "engine": "chatterbox"}' \
  --output hello.wav
```

### Example: SSE Streaming

```python
import requests
import base64

response = requests.get(
    "http://localhost:8000/v1/tts/stream-sse",
    params={"text": "Hello world!", "engine": "chatterbox"},
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b"data:"):
        data = json.loads(line[5:])
        if "chunk" in data:
            audio = base64.b64decode(data["chunk"])
            # Play or process audio chunk
```

### Docker

```bash
# With GPU
docker-compose up -d

# Without GPU
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

See [tts_server/README.md](tts_server/README.md) for complete documentation.

## ChatterboxEngine Reference

### Model Types

| Type | Parameters | Languages | Best For |
|------|------------|-----------|----------|
| `standard` | 500M | English | High quality, creative controls |
| `turbo` | 350M | English | Low latency, paralinguistic tags |
| `multilingual` | 500M | 23+ | Multi-language, zero-shot cloning |

### Supported Languages (Multilingual)

```
en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, ko, hu, hi, vi, sv, da, fi, no, el
```

### Parameters

```python
engine = ChatterboxEngine(
    model_type="standard",      # "standard", "turbo", or "multilingual"
    voice="path/to/ref.wav",    # Voice reference for cloning
    language="en",              # Language code (multilingual only)
    temperature=0.7,            # Sampling temperature (0.0-1.0)
    exaggeration=0.5,           # Expression factor (standard only)
    cfg_weight=0.5,             # CFG weight (standard only)
    device="cuda",              # "cuda", "cpu", or "mps"
)
```

### Paralinguistic Tags (Turbo)

The turbo model supports expressive tags:

```python
engine = ChatterboxEngine(model_type="turbo")
stream.feed("That's hilarious! [laugh] I can't believe it!")
stream.feed("[sigh] What a long day...")
stream.feed("[cough] Excuse me.")
```

## Configuration

### TextToAudioStream Parameters

```python
stream = TextToAudioStream(
    engine=engine,                    # TTS engine or list of engines (fallback)
    on_text_stream_start=callback,    # Called when text streaming starts
    on_text_stream_stop=callback,     # Called when text streaming ends
    on_audio_stream_start=callback,   # Called when audio playback starts
    on_audio_stream_stop=callback,    # Called when audio playback ends
    on_character=callback,            # Called for each character processed
    on_word=callback,                 # Word timing (Azure/Kokoro only)
    output_device_index=None,         # Audio output device index
    tokenizer="nltk",                 # "nltk" or "stanza"
    language="en",                    # Language for sentence splitting
    muted=False,                      # Disable audio playback
)
```

### Play Parameters

```python
stream.play(
    fast_sentence_fragment=True,      # Prioritize speed
    buffer_threshold_seconds=0.0,     # Audio buffering threshold
    minimum_sentence_length=10,       # Min chars for sentence
    log_synthesized_text=False,       # Log synthesized text
    output_wavfile="output.wav",      # Save to file
    on_sentence_synthesized=callback, # Called after each sentence
    on_audio_chunk=callback,          # Called for each audio chunk
)
```

## Requirements

- **Python**: >= 3.9, < 3.13
- **GPU**: CUDA 11.8+ recommended for local neural engines
- **Audio**: PyAudio for playback

## CUDA Installation

For GPU acceleration with local engines:

1. Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install PyTorch with CUDA:
   ```bash
   pip install torch==2.5.1+cu121 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```

## License

This library is open-source. Note that individual TTS engines may have their own licensing requirements:

- **CoquiEngine**: Open-source for noncommercial use
- **ElevenlabsEngine**: Requires paid plan for commercial use
- **AzureEngine**: Free tier available, paid for commercial
- **OpenAIEngine**: See [OpenAI Terms](https://openai.com/policies/terms-of-use)
- **ChatterboxEngine**: See [Chatterbox License](https://github.com/nari-labs/chatterbox)

## Acknowledgements

- [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) - Original project
- [Coqui AI](https://coqui.ai/) - Local neural TTS
- [Nari Labs](https://github.com/nari-labs/chatterbox) - Chatterbox TTS

## Author

Marcos Saraiva

Based on the original work by Kolja Beigel.
