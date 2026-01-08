"""
ModelManager - Singleton pattern for TTS engine management.

Handles:
- Engine initialization and lifecycle
- Voice caching and management
- Concurrency control for GPU operations
- Multi-engine support
"""

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Type, Union

import torch

from RealtimeTTS import (
    BaseEngine,
    TextToAudioStream,
)

logger = logging.getLogger(__name__)


@dataclass
class VoiceInfo:
    """Metadata for a cached voice."""
    id: str
    name: str
    engine: str
    created_at: datetime = field(default_factory=datetime.now)
    audio_path: Optional[str] = None
    language: str = "en"
    is_default: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "engine": self.engine,
            "created_at": self.created_at.isoformat(),
            "audio_path": self.audio_path,
            "language": self.language,
            "is_default": self.is_default,
        }


class ModelManager:
    """
    Singleton manager for TTS engines and voice caching.

    Features:
    - Thread-safe singleton pattern
    - Async-compatible with request locking
    - Multi-engine support with hot-swapping
    - Voice caching with persistence
    - GPU memory management
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._engines: Dict[str, BaseEngine] = {}
        self._current_engine_name: Optional[str] = None
        self._voices: Dict[str, VoiceInfo] = {}
        self._voices_path: Path = Path("./voices")
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Async lock for GPU operations
        self._request_lock: Optional[asyncio.Lock] = None

        # Device detection
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        logger.info(f"ModelManager initialized with device: {self._device}")

    @property
    def request_lock(self) -> asyncio.Lock:
        """Get or create the async request lock."""
        if self._request_lock is None:
            self._request_lock = asyncio.Lock()
        return self._request_lock

    @property
    def device(self) -> str:
        """Current compute device."""
        return self._device

    @property
    def current_engine(self) -> Optional[BaseEngine]:
        """Currently active engine."""
        if self._current_engine_name:
            return self._engines.get(self._current_engine_name)
        return None

    @property
    def current_engine_name(self) -> Optional[str]:
        """Name of currently active engine."""
        return self._current_engine_name

    @property
    def available_engines(self) -> List[str]:
        """List of registered engine names."""
        return list(self._engines.keys())

    async def initialize(
        self,
        engine_type: str = "system",
        engine_config: Optional[Dict[str, Any]] = None,
        voices_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the model manager with a default engine.

        Args:
            engine_type: Type of engine to initialize.
            engine_config: Configuration dict for the engine.
            voices_path: Path to store/load voice files.
        """
        if voices_path:
            self._voices_path = Path(voices_path)

        self._voices_path.mkdir(parents=True, exist_ok=True)

        # Load engine in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._load_engine_sync,
            engine_type,
            engine_config or {},
        )

        # Load cached voices
        self._load_voices_from_disk()

        logger.info(f"ModelManager initialized with engine: {engine_type}")

    def _load_engine_sync(
        self,
        engine_type: str,
        config: Dict[str, Any],
    ) -> None:
        """Synchronously load an engine (runs in thread pool)."""
        engine_type = engine_type.lower()
        engine: BaseEngine

        try:
            if engine_type == "system":
                from RealtimeTTS import SystemEngine
                engine = SystemEngine(**config)

            elif engine_type == "azure":
                from RealtimeTTS import AzureEngine
                speech_key = config.get("speech_key") or os.environ.get("AZURE_SPEECH_KEY")
                region = config.get("region") or os.environ.get("AZURE_SPEECH_REGION")
                if not speech_key or not region:
                    raise ValueError("Azure requires speech_key and region")
                engine = AzureEngine(speech_key, region, **{
                    k: v for k, v in config.items()
                    if k not in ("speech_key", "region")
                })

            elif engine_type == "openai":
                from RealtimeTTS import OpenAIEngine
                engine = OpenAIEngine(**config)

            elif engine_type == "elevenlabs":
                from RealtimeTTS import ElevenlabsEngine
                api_key = config.get("api_key") or os.environ.get("ELEVENLABS_API_KEY")
                engine = ElevenlabsEngine(api_key, **{
                    k: v for k, v in config.items() if k != "api_key"
                })

            elif engine_type == "coqui":
                from RealtimeTTS import CoquiEngine
                engine = CoquiEngine(**config)

            elif engine_type == "kokoro":
                from RealtimeTTS import KokoroEngine
                engine = KokoroEngine(**config)

            elif engine_type == "edge":
                from RealtimeTTS import EdgeEngine
                engine = EdgeEngine(**config)

            elif engine_type == "gtts":
                from RealtimeTTS import GTTSEngine
                engine = GTTSEngine(**config)

            elif engine_type == "chatterbox":
                from RealtimeTTS import ChatterboxEngine
                config.setdefault("device", self._device)
                engine = ChatterboxEngine(**config)

            else:
                raise ValueError(f"Unknown engine type: {engine_type}")

            self._engines[engine_type] = engine
            self._current_engine_name = engine_type

            logger.info(f"Loaded engine: {engine_type}")

        except Exception as e:
            logger.error(f"Failed to load engine {engine_type}: {e}")
            raise

    async def switch_engine(
        self,
        engine_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Switch to a different engine.

        Args:
            engine_type: Type of engine to switch to.
            config: Optional configuration for the engine.
        """
        async with self.request_lock:
            # Check if engine already loaded
            if engine_type in self._engines:
                self._current_engine_name = engine_type
                logger.info(f"Switched to existing engine: {engine_type}")
                return

            # Load new engine
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self._load_engine_sync,
                engine_type,
                config or {},
            )

    async def generate(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """
        Generate audio for text (non-streaming).

        Args:
            text: Text to synthesize.
            voice_id: Optional voice to use.
            **kwargs: Additional parameters passed to engine.

        Returns:
            Complete audio data as bytes.
        """
        async with self.request_lock:
            if not self.current_engine:
                raise RuntimeError("No engine initialized")

            # Set voice if specified
            if voice_id and voice_id in self._voices:
                voice_info = self._voices[voice_id]
                if voice_info.audio_path:
                    self.current_engine.set_voice(voice_info.audio_path)

            # Set additional parameters
            if kwargs:
                try:
                    self.current_engine.set_voice_parameters(**kwargs)
                except NotImplementedError:
                    pass

            # Generate in thread pool
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                self._executor,
                self._generate_sync,
                text,
            )

            return audio_data

    def _generate_sync(self, text: str) -> bytes:
        """Synchronously generate audio (runs in thread pool)."""
        engine = self.current_engine
        if not engine:
            raise RuntimeError("No engine initialized")

        # Clear the queue
        while not engine.queue.empty():
            engine.queue.get()

        # Synthesize
        engine.synthesize(text)

        # Collect all chunks
        chunks = []
        while not engine.queue.empty():
            chunks.append(engine.queue.get())

        return b"".join(chunks)

    def generate_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        chunk_size: int = 4096,
        **kwargs,
    ) -> Generator[bytes, None, None]:
        """
        Generate audio stream (synchronous generator).

        Args:
            text: Text to synthesize.
            voice_id: Optional voice to use.
            chunk_size: Size of chunks to yield.
            **kwargs: Additional parameters.

        Yields:
            Audio data chunks.
        """
        engine = self.current_engine
        if not engine:
            raise RuntimeError("No engine initialized")

        # Set voice if specified
        if voice_id and voice_id in self._voices:
            voice_info = self._voices[voice_id]
            if voice_info.audio_path:
                engine.set_voice(voice_info.audio_path)

        # Set additional parameters
        if kwargs:
            try:
                engine.set_voice_parameters(**kwargs)
            except NotImplementedError:
                pass

        # Check if engine has generate_stream method (like ChatterboxEngine)
        if hasattr(engine, "generate_stream"):
            yield from engine.generate_stream(text, chunk_size=chunk_size)
        else:
            # Fallback: use queue-based synthesis
            # Clear the queue
            while not engine.queue.empty():
                engine.queue.get()

            # Start synthesis in a separate thread
            import threading
            synthesis_done = threading.Event()

            def synthesize_thread():
                try:
                    engine.synthesize(text)
                finally:
                    synthesis_done.set()

            thread = threading.Thread(target=synthesize_thread)
            thread.start()

            # Yield chunks as they become available
            while not synthesis_done.is_set() or not engine.queue.empty():
                try:
                    chunk = engine.queue.get(timeout=0.1)
                    yield chunk
                except Exception:
                    if synthesis_done.is_set():
                        break

            thread.join()

    def get_stream_info(self) -> tuple:
        """Get audio stream configuration from current engine."""
        if not self.current_engine:
            raise RuntimeError("No engine initialized")
        return self.current_engine.get_stream_info()

    # Voice Management

    def _load_voices_from_disk(self) -> None:
        """Load voice metadata from the voices directory."""
        if not self._voices_path.exists():
            return

        # Add default voice
        self._voices["default"] = VoiceInfo(
            id="default",
            name="Default",
            engine="all",
            is_default=True,
        )

        # Scan for voice files
        for ext in ["*.wav", "*.mp3", "*.flac"]:
            for audio_file in self._voices_path.glob(ext):
                voice_id = audio_file.stem
                if voice_id not in self._voices:
                    self._voices[voice_id] = VoiceInfo(
                        id=voice_id,
                        name=voice_id.replace("_", " ").title(),
                        engine=self._current_engine_name or "unknown",
                        audio_path=str(audio_file),
                        created_at=datetime.fromtimestamp(audio_file.stat().st_mtime),
                    )

        logger.info(f"Loaded {len(self._voices)} voices from disk")

    def get_voices(self) -> List[VoiceInfo]:
        """Get all available voices."""
        # Also get voices from current engine
        voices = list(self._voices.values())

        if self.current_engine:
            try:
                engine_voices = self.current_engine.get_voices()
                for ev in engine_voices:
                    voice_id = getattr(ev, "name", str(ev))
                    if voice_id not in self._voices:
                        voices.append(VoiceInfo(
                            id=voice_id,
                            name=voice_id,
                            engine=self._current_engine_name or "unknown",
                        ))
            except Exception as e:
                logger.warning(f"Could not get engine voices: {e}")

        return voices

    async def create_voice(
        self,
        name: str,
        audio_data: bytes,
        language: str = "en",
    ) -> VoiceInfo:
        """
        Create a new voice from audio data.

        Args:
            name: Name for the voice.
            audio_data: Audio file data.
            language: Language code.

        Returns:
            Created VoiceInfo.
        """
        async with self.request_lock:
            # Generate voice ID
            voice_id = name.lower().replace(" ", "_")

            # Save audio file
            audio_path = self._voices_path / f"{voice_id}.wav"
            audio_path.write_bytes(audio_data)

            # Create voice info
            voice_info = VoiceInfo(
                id=voice_id,
                name=name,
                engine=self._current_engine_name or "unknown",
                audio_path=str(audio_path),
                language=language,
            )

            self._voices[voice_id] = voice_info
            logger.info(f"Created voice: {voice_id}")

            return voice_info

    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a voice.

        Args:
            voice_id: ID of voice to delete.

        Returns:
            True if deleted, False if not found or is default.
        """
        if voice_id not in self._voices:
            return False

        voice_info = self._voices[voice_id]
        if voice_info.is_default:
            logger.warning("Cannot delete default voice")
            return False

        # Delete audio file
        if voice_info.audio_path:
            audio_path = Path(voice_info.audio_path)
            if audio_path.exists():
                audio_path.unlink()

        del self._voices[voice_id]
        logger.info(f"Deleted voice: {voice_id}")
        return True

    def set_voice(self, voice_id: str) -> bool:
        """
        Set the current voice on the engine.

        Args:
            voice_id: ID of voice to set.

        Returns:
            True if set successfully.
        """
        if not self.current_engine:
            return False

        if voice_id == "default":
            return True

        if voice_id in self._voices:
            voice_info = self._voices[voice_id]
            if voice_info.audio_path:
                self.current_engine.set_voice(voice_info.audio_path)
                return True

        # Try setting directly on engine
        try:
            self.current_engine.set_voice(voice_id)
            return True
        except Exception as e:
            logger.warning(f"Could not set voice {voice_id}: {e}")
            return False

    # Cleanup

    def shutdown(self) -> None:
        """Shutdown all engines and cleanup resources."""
        for name, engine in self._engines.items():
            try:
                engine.shutdown()
                logger.info(f"Shutdown engine: {name}")
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")

        self._engines.clear()
        self._current_engine_name = None
        self._executor.shutdown(wait=False)

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("ModelManager shutdown complete")


# Global instance accessor
def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    return ModelManager()
