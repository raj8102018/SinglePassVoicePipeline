# tts_generator.py

import os
import asyncio
# from dotenv import load_dotenv
import wave
import numpy as np
import sounddevice as sd
from typing import AsyncIterator
from dotenv import load_dotenv
load_dotenv()

# Use the AsyncElevenLabs client for asynchronous operations
from elevenlabs.client import AsyncElevenLabs
from agents.voice.model import TTSModel, TTSModelSettings

# --- Configuration ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_TTS_API_KEY")
DEFAULT_VOICE_ID = "y1adqrqs4jNaANXsIZnD"  # A default voice for testing

# 1. Initialize the ASYNCHRONOUS ElevenLabs client
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in environment variables. Please set it in your .env file.")

elevenlabs_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)


class ElevenLabsTTSModel(TTSModel):
    """A custom, asynchronous TTSModel implementation for ElevenLabs,
    compatible with the OpenAI Agent SDK's VoicePipeline."""

    def __init__(self, model_id: str = "eleven_turbo_v2"):
        """Initializes the ElevenLabs TTS model.
        
        Args:
            model_id: The ElevenLabs model to use for synthesis.
        """
        self._model_id = model_id

    @property
    def model_name(self) -> str:
        """Returns the model identifier for this TTS instance."""
        return self._model_id

    # 2. The 'run' method must be an 'async def' and correctly handle the async generator pattern.
    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        """
        Runs the ElevenLabs TTS model and streams the audio output asynchronously.
        
        Args:
            text: The text to convert to speech.
            settings: The settings to use for the TTS model, including the voice.
        
        Returns:
            An async iterator of audio data chunks in bytes.
        """
        # 3. Correctly handle the case of empty text for an async generator.
        # If the text is empty, we simply do nothing, and the generator will
        # finish without yielding any items.
        if text and text.strip():
            # Use the voice from settings, or fall back to a default.
            voice_id = settings.voice if settings and settings.voice else DEFAULT_VOICE_ID

            # 4. 'await' the asynchronous stream method
            audio_stream = elevenlabs_client.text_to_speech.stream(
                text=text,
                voice_id=voice_id,
                model_id=self.model_name,
                # For telephony with Twilio, you'll eventually need a format like 'pcm_8000'.
                # For local playback, 'pcm_24000' is a good choice.
                output_format="pcm_24000"
            )

            # 5. Use 'async for' to iterate over the asynchronous stream
            async for chunk in audio_stream:
                yield chunk
