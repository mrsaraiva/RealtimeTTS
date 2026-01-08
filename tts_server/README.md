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
# Using the CLI
python -m tts_server.app --engine chatterbox --port 8000

# Or with uvicorn directly
TTS_ENGINE=chatterbox uvicorn tts_server.app:app --host 0.0.0.0 --port 8000

# With Chatterbox Turbo model
TTS_ENGINE=chatterbox CHATTERBOX_MODEL_TYPE=turbo uvicorn tts_server.app:app
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENGINE` | `system` | TTS engine to use |
| `TTS_VOICES_PATH` | `./voices` | Directory for voice files |
| `CHATTERBOX_MODEL_TYPE` | `standard` | Chatterbox model variant |
| `AZURE_SPEECH_KEY` | - | Azure Speech API key |
| `AZURE_SPEECH_REGION` | - | Azure region |
| `ELEVENLABS_API_KEY` | - | ElevenLabs API key |
| `OPENAI_API_KEY` | - | OpenAI API key |

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

Returns a complete WAV file.

#### Stream via SSE

```bash
GET /v1/tts/stream-sse?text=Hello&voice_id=default&chunk_size=4096
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
{"action": "stop"}
{"action": "ping"}
```

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

#### Switch Engine

```bash
POST /v1/engines/switch
Content-Type: application/json

{
  "engine_type": "chatterbox",
  "config": {"model_type": "turbo"}
}
```

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

# Generate audio
response = requests.post(
    "http://localhost:8000/v1/tts/generate",
    json={"text": "Hello, world!"}
)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Python (SSE streaming)

```python
import requests
import base64

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
    ws.send(JSON.stringify({
        action: "generate",
        text: "Hello, world!"
    }));
};
```

## Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r tts_server/requirements.txt
RUN pip install realtimetts[chatterbox]

EXPOSE 8000
CMD ["python", "-m", "tts_server.app", "--engine", "chatterbox"]
```

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
