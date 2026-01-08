# RealtimeTTS Production Server Dockerfile
#
# Build with specific engines:
#   docker build --build-arg TTS_EXTRAS="chatterbox,kokoro" -t realtimetts-server .
#
# Build with all engines:
#   docker build --build-arg TTS_EXTRAS="all" -t realtimetts-server .
#
# Run:
#   docker run -p 8000:8000 --gpus all realtimetts-server

FROM python:3.11-slim

# Build argument for which engine extras to install
# Options: chatterbox, kokoro, coqui, azure, openai, elevenlabs, edge, gtts, all
ARG TTS_EXTRAS="chatterbox,kokoro"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    git \
    espeak-ng \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first for better caching
COPY tts_server/requirements.txt ./tts_server/
COPY requirements.txt ./
COPY setup.py ./
COPY pyproject.toml ./
COPY README.md ./

# Copy the main package
COPY RealtimeTTS/ ./RealtimeTTS/

# Copy the server
COPY tts_server/ ./tts_server/

# Install dependencies with uv (much faster than pip)
RUN uv pip install -r tts_server/requirements.txt && \
    uv pip install -e ".[$TTS_EXTRAS]"

# Create voices directory
RUN mkdir -p /app/voices

# Default environment variables
ENV TTS_ENGINE=chatterbox \
    TTS_VOICES_PATH=/app/voices \
    CHATTERBOX_MODEL_TYPE=standard

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["python", "-m", "tts_server.app", "--host", "0.0.0.0", "--port", "8000"]
