"""
ChatterboxEngine - A unified TTS engine supporting all Chatterbox model variants.

Supports three model types:
- standard: ChatterboxTTS (500M params, English, creative controls)
- turbo: ChatterboxTurboTTS (350M params, English, fastest, paralinguistic tags)
- multilingual: ChatterboxMultilingualTTS (500M params, 23+ languages)

Requires: pip install chatterbox-tts
"""

from .base_engine import BaseEngine
from typing import Union, Optional, List, Generator
from pathlib import Path
from queue import Queue
import numpy as np
import traceback
import logging
import pyaudio
import torch
import os


class ChatterboxVoice:
    """Represents a voice configuration for the Chatterbox engine."""

    def __init__(
        self,
        name: str,
        audio_prompt_path: Optional[str] = None,
        language: str = "en",
    ):
        """
        Initialize a ChatterboxVoice.

        Args:
            name: Display name for the voice.
            audio_prompt_path: Path to audio file for voice cloning (6-15 seconds recommended).
            language: Language code (only used for multilingual model).
        """
        self.name = name
        self.audio_prompt_path = audio_prompt_path
        self.language = language

    def __repr__(self):
        return f"ChatterboxVoice(name={self.name}, language={self.language})"


class ChatterboxEngine(BaseEngine):
    """
    A unified TTS engine for all Chatterbox model variants.

    Supports:
    - standard: High-quality synthesis with creative controls (CFG, exaggeration)
    - turbo: Fastest inference with paralinguistic tags ([laugh], [cough], etc.)
    - multilingual: 23+ language support with zero-shot voice cloning

    Example usage:
        # Standard model
        engine = ChatterboxEngine(model_type="standard")

        # Turbo model for low-latency
        engine = ChatterboxEngine(model_type="turbo")

        # Multilingual model
        engine = ChatterboxEngine(model_type="multilingual", language="es")

        # With voice cloning
        engine = ChatterboxEngine(
            model_type="standard",
            voice="path/to/reference.wav"
        )
    """

    SUPPORTED_MODEL_TYPES = ["standard", "turbo", "multilingual"]

    # Sample rate for all Chatterbox models
    SAMPLE_RATE = 24000

    # Supported languages for multilingual model
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
        "cs", "ar", "zh", "ja", "ko", "hu", "hi", "vi", "sv", "da",
        "fi", "no", "el"
    ]

    def __init__(
        self,
        model_type: str = "standard",
        voice: Optional[Union[str, ChatterboxVoice]] = None,
        language: str = "en",
        device: Optional[str] = None,
        temperature: float = 0.7,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        voices_path: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the ChatterboxEngine.

        Args:
            model_type: Model variant to use ("standard", "turbo", or "multilingual").
            voice: Voice to use - either a path to audio file or ChatterboxVoice instance.
            language: Language code for multilingual model (default: "en").
            device: Device to run inference on ("cuda", "cpu", or None for auto-detect).
            temperature: Sampling temperature for generation (0.0-1.0).
            exaggeration: Exaggeration factor for expressive synthesis (standard model only).
            cfg_weight: Classifier-free guidance weight (standard model only).
            voices_path: Directory to store/load voice files.
            debug: Enable debug logging.
        """
        self.model_type = model_type.lower()
        if self.model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Supported types: {self.SUPPORTED_MODEL_TYPES}"
            )

        self.language = language
        self.temperature = temperature
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.voices_path = voices_path or os.path.join(os.getcwd(), "voices")
        self.debug = debug

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._current_voice: Optional[ChatterboxVoice] = None
        self._voice_cache = {}  # Cache for computed voice conditionals

        # Set initial voice
        if voice:
            self.set_voice(voice)

        # Create voices directory if it doesn't exist
        os.makedirs(self.voices_path, exist_ok=True)

        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
            logging.debug(
                f"[ChatterboxEngine] Initialized with model_type={self.model_type}, "
                f"device={self.device}, language={self.language}"
            )

    def post_init(self):
        """Called after BaseEngine.__init__ to set engine-specific properties."""
        self.engine_name = "chatterbox"
        self._load_model()

    def _load_model(self):
        """Load the appropriate Chatterbox model based on model_type."""
        try:
            if self.model_type == "turbo":
                from chatterbox.tts import ChatterboxTurboTTS
                self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
                if self.debug:
                    logging.debug("[ChatterboxEngine] Loaded ChatterboxTurboTTS model")

            elif self.model_type == "multilingual":
                from chatterbox.tts import ChatterboxMultilingualTTS
                self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                if self.debug:
                    logging.debug("[ChatterboxEngine] Loaded ChatterboxMultilingualTTS model")

            else:  # standard
                from chatterbox.tts import ChatterboxTTS
                self._model = ChatterboxTTS.from_pretrained(device=self.device)
                if self.debug:
                    logging.debug("[ChatterboxEngine] Loaded ChatterboxTTS model")

        except ImportError as e:
            raise ImportError(
                "Failed to import chatterbox. Please install it with:\n"
                "pip install chatterbox-tts"
            ) from e
        except Exception as e:
            logging.error(f"[ChatterboxEngine] Error loading model: {e}")
            traceback.print_exc()
            raise

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration for Chatterbox audio output.

        Returns:
            tuple: (format, channels, sample_rate)
        """
        return (pyaudio.paFloat32, 1, self.SAMPLE_RATE)

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio and puts chunks into the queue.

        Args:
            text: The text to synthesize.

        Returns:
            bool: True if synthesis was successful, False otherwise.
        """
        super().synthesize(text)

        if not text or not text.strip():
            return False

        try:
            if self._model is None:
                self._load_model()

            # Prepare generation kwargs
            gen_kwargs = {
                "text": text,
                "temperature": self.temperature,
            }

            # Add voice reference if available
            if self._current_voice and self._current_voice.audio_prompt_path:
                gen_kwargs["audio_prompt_path"] = self._current_voice.audio_prompt_path

            # Model-specific parameters
            if self.model_type == "standard":
                gen_kwargs["exaggeration"] = self.exaggeration
                gen_kwargs["cfg_weight"] = self.cfg_weight

            elif self.model_type == "multilingual":
                gen_kwargs["language"] = self.language

            if self.debug:
                logging.debug(f"[ChatterboxEngine] Generating audio for: {text[:50]}...")

            # Generate audio
            wav = self._model.generate(**gen_kwargs)

            # Convert to numpy array if it's a tensor
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            # Ensure correct shape (1D array)
            if wav.ndim > 1:
                wav = wav.squeeze()

            # Normalize to float32 range [-1, 1]
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)

            # Check for stop event
            if self.stop_synthesis_event.is_set():
                return False

            # Put audio data into queue as bytes
            audio_bytes = wav.tobytes()
            self.queue.put(audio_bytes)

            # Update audio duration for timing info
            self.audio_duration += len(wav) / self.SAMPLE_RATE

            if self.debug:
                logging.debug(
                    f"[ChatterboxEngine] Generated {len(wav) / self.SAMPLE_RATE:.2f}s of audio"
                )

            return True

        except Exception as e:
            logging.error(f"[ChatterboxEngine] Synthesis error: {e}")
            traceback.print_exc()
            return False

    def generate_stream(
        self,
        text: str,
        chunk_size: int = 4096,
    ) -> Generator[bytes, None, None]:
        """
        Generator that yields audio chunks for streaming.

        This is useful for the production server's streaming endpoints.

        Args:
            text: The text to synthesize.
            chunk_size: Size of audio chunks to yield (in samples).

        Yields:
            bytes: Audio data chunks.
        """
        if not text or not text.strip():
            return

        try:
            if self._model is None:
                self._load_model()

            # Prepare generation kwargs
            gen_kwargs = {
                "text": text,
                "temperature": self.temperature,
            }

            if self._current_voice and self._current_voice.audio_prompt_path:
                gen_kwargs["audio_prompt_path"] = self._current_voice.audio_prompt_path

            if self.model_type == "standard":
                gen_kwargs["exaggeration"] = self.exaggeration
                gen_kwargs["cfg_weight"] = self.cfg_weight
            elif self.model_type == "multilingual":
                gen_kwargs["language"] = self.language

            # Generate audio
            wav = self._model.generate(**gen_kwargs)

            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            if wav.ndim > 1:
                wav = wav.squeeze()

            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)

            # Yield chunks
            for i in range(0, len(wav), chunk_size):
                if self.stop_synthesis_event.is_set():
                    break
                chunk = wav[i:i + chunk_size]
                yield chunk.tobytes()

        except Exception as e:
            logging.error(f"[ChatterboxEngine] Stream generation error: {e}")
            traceback.print_exc()

    def set_voice(self, voice: Union[str, ChatterboxVoice]):
        """
        Sets the voice to use for synthesis.

        Args:
            voice: Either a path to an audio file, a voice name, or a ChatterboxVoice instance.
        """
        if isinstance(voice, ChatterboxVoice):
            self._current_voice = voice
        elif isinstance(voice, str):
            # Check if it's a path to an audio file
            if os.path.isfile(voice):
                self._current_voice = ChatterboxVoice(
                    name=os.path.basename(voice),
                    audio_prompt_path=voice,
                    language=self.language,
                )
            else:
                # Check in voices directory
                voice_path = os.path.join(self.voices_path, voice)
                if os.path.isfile(voice_path):
                    self._current_voice = ChatterboxVoice(
                        name=voice,
                        audio_prompt_path=voice_path,
                        language=self.language,
                    )
                elif os.path.isfile(voice_path + ".wav"):
                    self._current_voice = ChatterboxVoice(
                        name=voice,
                        audio_prompt_path=voice_path + ".wav",
                        language=self.language,
                    )
                else:
                    # Create a voice without audio prompt (will use model default)
                    self._current_voice = ChatterboxVoice(
                        name=voice,
                        audio_prompt_path=None,
                        language=self.language,
                    )
                    if self.debug:
                        logging.debug(
                            f"[ChatterboxEngine] Voice '{voice}' has no audio prompt, "
                            "using model default"
                        )

        if self.debug:
            logging.debug(f"[ChatterboxEngine] Voice set to: {self._current_voice}")

    def get_voices(self) -> List[ChatterboxVoice]:
        """
        Retrieves available voices from the voices directory.

        Returns:
            List of ChatterboxVoice objects.
        """
        voices = []
        seen_names = set()

        if os.path.isdir(self.voices_path):
            for filename in os.listdir(self.voices_path):
                if filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
                    name = os.path.splitext(filename)[0]
                    if name not in seen_names:
                        seen_names.add(name)
                        voices.append(ChatterboxVoice(
                            name=name,
                            audio_prompt_path=os.path.join(self.voices_path, filename),
                            language=self.language,
                        ))

        # Add a default voice entry
        if "default" not in seen_names:
            voices.insert(0, ChatterboxVoice(
                name="default",
                audio_prompt_path=None,
                language=self.language,
            ))

        return voices

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets voice/synthesis parameters.

        Supported parameters:
            - temperature: float (0.0-1.0)
            - exaggeration: float (standard model only)
            - cfg_weight: float (standard model only)
            - language: str (multilingual model only)
        """
        if "temperature" in voice_parameters:
            self.temperature = float(voice_parameters["temperature"])

        if "exaggeration" in voice_parameters and self.model_type == "standard":
            self.exaggeration = float(voice_parameters["exaggeration"])

        if "cfg_weight" in voice_parameters and self.model_type == "standard":
            self.cfg_weight = float(voice_parameters["cfg_weight"])

        if "language" in voice_parameters and self.model_type == "multilingual":
            lang = voice_parameters["language"]
            if lang in self.SUPPORTED_LANGUAGES:
                self.language = lang
            else:
                logging.warning(
                    f"[ChatterboxEngine] Unsupported language: {lang}. "
                    f"Supported: {self.SUPPORTED_LANGUAGES}"
                )

        if self.debug:
            logging.debug(f"[ChatterboxEngine] Voice parameters updated")

    def set_model_type(self, model_type: str):
        """
        Switch to a different model type.

        Note: This will reload the model, which may take some time.

        Args:
            model_type: One of "standard", "turbo", or "multilingual".
        """
        model_type = model_type.lower()
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Supported types: {self.SUPPORTED_MODEL_TYPES}"
            )

        if model_type != self.model_type:
            self.model_type = model_type
            self._model = None  # Clear current model
            self._load_model()  # Load new model

            if self.debug:
                logging.debug(f"[ChatterboxEngine] Switched to model type: {model_type}")

    def create_voice(
        self,
        name: str,
        audio_path: str,
        language: str = "en",
    ) -> ChatterboxVoice:
        """
        Create a new voice from an audio file.

        The audio file will be copied to the voices directory.

        Args:
            name: Name for the new voice.
            audio_path: Path to the source audio file (6-15 seconds recommended).
            language: Language code for the voice.

        Returns:
            The created ChatterboxVoice instance.
        """
        import shutil

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Determine destination path
        ext = os.path.splitext(audio_path)[1]
        dest_path = os.path.join(self.voices_path, f"{name}{ext}")

        # Copy the file
        shutil.copy2(audio_path, dest_path)

        voice = ChatterboxVoice(
            name=name,
            audio_prompt_path=dest_path,
            language=language,
        )

        if self.debug:
            logging.debug(f"[ChatterboxEngine] Created voice: {voice}")

        return voice

    def delete_voice(self, name: str) -> bool:
        """
        Delete a voice from the voices directory.

        Args:
            name: Name of the voice to delete.

        Returns:
            True if the voice was deleted, False if not found.
        """
        if name == "default":
            logging.warning("[ChatterboxEngine] Cannot delete default voice")
            return False

        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            voice_path = os.path.join(self.voices_path, f"{name}{ext}")
            if os.path.isfile(voice_path):
                os.remove(voice_path)
                if self.debug:
                    logging.debug(f"[ChatterboxEngine] Deleted voice: {name}")
                return True

        return False

    def shutdown(self):
        """Cleanup resources."""
        if self._model is not None:
            del self._model
            self._model = None

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.debug:
            logging.debug("[ChatterboxEngine] Shutdown complete")
