# RealtimeTTS Production Server

A production-ready FastAPI server for text-to-speech synthesis with support for multiple streaming protocols.

## Features

- **Multiple Streaming Protocols**: REST, Server-Sent Events (SSE), and WebSocket
- **Multi-Engine Support**: Switch between TTS engines on-the-fly
- **Voice Management**: Create, list, and delete custom voices
- **Async Architecture**: Non-blocking operations with proper concurrency control
- **Health Monitoring**: Built-in health check endpoint

## Quick Start

### Installation

```bash
# Install server dependencies
pip install fastapi uvicorn python-multipart

# Install RealtimeTTS with your preferred engine
pip install realtimetts[chatterbox]  # For Chatterbox
pip install realtimetts[all]          # For all engines
```

### Running the Server

```bash
# Single engine mode
python -m tts_server.app --engine chatterbox --port 8000

# Multi-engine mode: pre-load multiple engines at startup
python -m tts_server.app --engines chatterbox,kokoro --default-engine chatterbox

# Or with uvicorn directly
TTS_ENGINE=chatterbox uvicorn tts_server.app:app --host 0.0.0.0 --port 8000

# Multi-engine with environment variables
TTS_ENGINES=chatterbox,kokoro TTS_DEFAULT_ENGINE=chatterbox uvicorn tts_server.app:app

# With Chatterbox Turbo model
TTS_ENGINE=chatterbox CHATTERBOX_MODEL_TYPE=turbo uvicorn tts_server.app:app
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENGINE` | `system` | Single TTS engine to use |
| `TTS_ENGINES` | - | Comma-separated list of engines to pre-load (e.g., `chatterbox,kokoro`) |
| `TTS_DEFAULT_ENGINE` | - | Default active engine when using `TTS_ENGINES` |
| `TTS_VOICES_PATH` | `./voices` | Directory for voice files |
| `CHATTERBOX_MODEL_TYPE` | `standard` | Chatterbox model variant (`standard`, `turbo`, `multilingual`) |
| `AZURE_SPEECH_KEY` | - | Azure Speech API key |
| `AZURE_SPEECH_REGION` | - | Azure region |
| `ELEVENLABS_API_KEY` | - | ElevenLabs API key |
| `OPENAI_API_KEY` | - | OpenAI API key |

**Multi-Engine Mode:** When `TTS_ENGINES` is set, all specified engines are pre-loaded at startup. This eliminates cold-start latency when switching between engines via the API. Use `TTS_DEFAULT_ENGINE` to specify which engine is active by default.

## API Reference

### Health Check

```bash
GET /health
```

Returns server status, current engine, and voice count.

### Text-to-Speech

#### Generate (Synchronous)

```bash
POST /v1/tts/generate
Content-Type: application/json

{
  "text": "Hello, world!",
  "voice_id": "default",
  "temperature": 0.7
}
```

**Request Parameters:**
- `text` (required): Text to synthesize
- `voice_id` (optional): Voice ID to use
- `engine` (optional): Engine to use for this request (e.g., "chatterbox", "kokoro")
- `temperature` (optional): Sampling temperature (0.0-1.0)
- `exaggeration` (optional): Exaggeration factor for Chatterbox (0.0-1.0)
- `cfg_weight` (optional): CFG weight for Chatterbox (0.0-1.0)
- `language` (optional): Language code for multilingual engines (e.g., "fr", "es")

**Direct engine selection** (no switch needed):
```bash
POST /v1/tts/generate
Content-Type: application/json

{
  "text": "Bonjour le monde!",
  "engine": "chatterbox",
  "language": "fr"
}
```

Returns a complete WAV file.

#### Stream via SSE

```bash
GET /v1/tts/stream-sse?text=Hello&voice_id=default&chunk_size=4096
```

**Query Parameters:**
- `text` (required): Text to synthesize
- `voice_id` (optional): Voice ID to use
- `engine` (optional): Engine to use for this request
- `chunk_size` (optional): Size of audio chunks (default: 4096)
- `temperature` (optional): Sampling temperature (default: 0.7)
- `language` (optional): Language code for multilingual engines

**Direct engine selection:**
```bash
GET /v1/tts/stream-sse?text=Hello&engine=kokoro
```

Returns Server-Sent Events with base64-encoded audio chunks.

Events:
- `audio`: `{"chunk": "<base64>", "index": 1}`
- `metrics`: `{"generation_time": 1.2, "total_bytes": 48000}`
- `done`: `{"status": "complete"}`

#### Stream via WebSocket

```bash
WS /v1/tts/stream
```

Client messages:
```json
{"action": "generate", "text": "Hello", "voice_id": "default"}
{"action": "generate", "text": "Hola", "engine": "chatterbox", "language": "es"}
{"action": "stop"}
{"action": "ping"}
```

**Generate action parameters:**
- `text` (required): Text to synthesize
- `voice_id` (optional): Voice ID to use
- `engine` (optional): Engine to use for this request
- `language` (optional): Language code for multilingual engines
- `chunk_size` (optional): Size of audio chunks (default: 4096)

Server messages:
- Binary audio chunks
- `{"type": "metrics", "data": {...}}`
- `{"type": "done"}`
- `{"type": "pong"}`

### Engines

#### List Engines

```bash
GET /v1/engines
```

Returns list of loaded engines, current default engine, and supported engines.

#### Switch Default Engine

```bash
POST /v1/engines/switch
Content-Type: application/json

{
  "engine_type": "chatterbox",
  "config": {"model_type": "turbo"}
}
```

Changes the default engine used when no `engine` parameter is specified.

**Note:** When using multi-engine mode (pre-loading multiple engines at startup), you can switch between loaded engines instantly by specifying the `engine` parameter in your requests. The `/v1/engines/switch` endpoint is useful for:
- Changing the default engine for requests that don't specify one
- Loading a new engine that wasn't pre-loaded at startup

### Voice Management

#### List Voices

```bash
GET /v1/voices
```

#### Create Voice

```bash
POST /v1/voices/create
Content-Type: multipart/form-data

name=my_voice
language=en
audio_file=@reference.wav
```

#### Delete Voice

```bash
DELETE /v1/voices/{voice_id}
```

#### Set Current Voice

```bash
POST /v1/voices/{voice_id}/set
```

## Client Examples

### Python (requests)

```python
import requests

# Generate audio with default engine
response = requests.post(
    "http://localhost:8000/v1/tts/generate",
    json={"text": "Hello, world!"}
)
with open("output.wav", "wb") as f:
    f.write(response.content)

# Direct engine selection (no switch needed)
response = requests.post(
    "http://localhost:8000/v1/tts/generate",
    json={
        "text": "Bonjour le monde!",
        "engine": "chatterbox",
        "language": "fr"
    }
)
with open("output_fr.wav", "wb") as f:
    f.write(response.content)
```

### Python (SSE streaming)

```python
import requests
import base64
import json

# Stream with default engine
response = requests.get(
    "http://localhost:8000/v1/tts/stream-sse",
    params={"text": "Hello, world!"},
    stream=True
)

audio_chunks = []
for line in response.iter_lines():
    if line.startswith(b"data:"):
        data = json.loads(line[5:])
        if "chunk" in data:
            audio_chunks.append(base64.b64decode(data["chunk"]))

# Stream with specific engine
response = requests.get(
    "http://localhost:8000/v1/tts/stream-sse",
    params={"text": "Hola mundo!", "engine": "chatterbox", "language": "es"},
    stream=True
)
```

### JavaScript (WebSocket)

```javascript
const ws = new WebSocket("ws://localhost:8000/v1/tts/stream");

ws.onmessage = (event) => {
    if (event.data instanceof Blob) {
        // Audio chunk - play it
        const audioContext = new AudioContext();
        event.data.arrayBuffer().then(buffer => {
            audioContext.decodeAudioData(buffer, decoded => {
                const source = audioContext.createBufferSource();
                source.buffer = decoded;
                source.connect(audioContext.destination);
                source.start();
            });
        });
    } else {
        // JSON message
        const msg = JSON.parse(event.data);
        console.log(msg);
    }
};

ws.onopen = () => {
    // Generate with default engine
    ws.send(JSON.stringify({
        action: "generate",
        text: "Hello, world!"
    }));

    // Or generate with specific engine
    ws.send(JSON.stringify({
        action: "generate",
        text: "Bonjour le monde!",
        engine: "chatterbox",
        language: "fr"
    }));
};
```

## Docker

### Quick Start with Docker Compose

```bash
# With GPU (default)
docker-compose up -d

# Without GPU (CPU only)
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Configuration

Create a `.env` file to customize:

```bash
# Single engine mode
TTS_ENGINE=chatterbox

# OR Multi-engine mode (pre-load multiple engines)
TTS_ENGINES=chatterbox,kokoro
TTS_DEFAULT_ENGINE=chatterbox

# Chatterbox model type: standard, turbo, multilingual
CHATTERBOX_MODEL_TYPE=turbo

# API Keys for cloud engines (optional)
OPENAI_API_KEY=sk-...
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=eastus
ELEVENLABS_API_KEY=...

# Hugging Face token for gated models (optional)
HF_TOKEN=hf_...
```

### Build Manually

```bash
# Build with default engines (chatterbox + kokoro)
docker build -t realtimetts-server -f tts_server/Dockerfile .

# Build with specific engines
docker build --build-arg TTS_EXTRAS="chatterbox,kokoro,coqui" \
  -t realtimetts-server -f tts_server/Dockerfile .

# Build with all engines
docker build --build-arg TTS_EXTRAS="all" \
  -t realtimetts-server -f tts_server/Dockerfile .

# Run with GPU
docker run -p 8000:8000 --gpus all \
  -v ./voices:/app/voices \
  -e TTS_ENGINES=chatterbox,kokoro \
  -e TTS_DEFAULT_ENGINE=chatterbox \
  realtimetts-server

# Run without GPU
docker run -p 8000:8000 \
  -v ./voices:/app/voices \
  -e TTS_ENGINE=system \
  realtimetts-server
```

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `TTS_EXTRAS` | `chatterbox,kokoro` | Comma-separated list of engines to install. Options: `chatterbox`, `kokoro`, `coqui`, `azure`, `openai`, `elevenlabs`, `edge`, `gtts`, `all` |

## Production Deployment

For production deployments, consider:

1. **Use gunicorn with uvicorn workers**:
   ```bash
   gunicorn tts_server.app:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Enable GPU if available** - The server auto-detects CUDA/MPS

3. **Set up a reverse proxy** (nginx/traefik) for TLS termination

4. **Monitor health endpoint** at `/health`

5. **Configure rate limiting** at the proxy level
