"""This module contains the functionality that converts uploaded or streamed .wav/.mp3 files
    to text using the Whisper ASR model or an external STT API"""

import os
import io
import wave
import asyncio
import numpy as np
import sounddevice as sd
from typing import AsyncIterator
from dotenv import load_dotenv
load_dotenv()

from elevenlabs.client import AsyncElevenLabs
from agents.voice.model import STTModel, STTModelSettings, StreamedTranscriptionSession
from agents.voice.pipeline import AudioInput, StreamedAudioInput

# --- Configuration ---
ELEVENLABS_STT_API_KEY = os.environ.get("ELEVENLABS_STT_API_KEY")

if not ELEVENLABS_STT_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found.")

# Initialize the asynchronous client for ElevenLabs
elevenlabs_client = AsyncElevenLabs(api_key=ELEVENLABS_STT_API_KEY)


class ElevenLabsSTTModel(STTModel):
    """A custom STTModel implementation for ElevenLabs, focusing on single API calls."""

    def __init__(self, model_id: str = "scribe_v1"):
        self._model_id = model_id

    @property
    def model_name(self) -> str:
        """Returns the model identifier for this STT instance."""
        return self._model_id

    # This method uses a single API call for transcription.
    async def transcribe(
        self,
        input: AudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool = False,
        trace_include_sensitive_audio_data: bool = False,
    ) -> str:
        """Transcribes a complete audio buffer using the ElevenLabs STT API."""
        try:
            # The ElevenLabs STT API's convert endpoint expects a file format (like WAV),
            # not raw PCM data. We create a WAV file in memory.
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(44100)
                wf.writeframes(input.buffer.tobytes())
            
            # Reset the in-memory file's pointer to the beginning
            wav_io.seek(0)
            
            # Transcribe the audio using ElevenLabs
            response = await elevenlabs_client.speech_to_text.convert(
                file=wav_io, model_id=self.model_name
            )
            transcribed_text = response.text
            print(f"Transcribed (one-shot): '{transcribed_text}'")
            return transcribed_text
        except Exception as e:
            print(f"Error during ElevenLabs STT transcription: {e}")
            return ""

    # SIMPLIFIED: This method is required by the STTModel protocol,
    # but we raise an error because we are deferring the WebSocket implementation.
    async def create_session(
        self,
        input: StreamedAudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool = False,
        trace_include_sensitive_audio_data: bool = False,
    ) -> StreamedTranscriptionSession:
        """Creates a new real-time transcription session."""
        raise NotImplementedError(
            "Real-time streaming transcription is not implemented in this simplified version. "
            "A WebSocket-based implementation is needed for the VoicePipeline's streaming mode."
        )
