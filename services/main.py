"""This module contains the voicepipeline flow"""
import asyncio
import io
import os
import wave
import numpy as np
import sounddevice as sd
from typing import AsyncIterator
from dotenv import load_dotenv

from agents import Agent
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
    VoicePipelineConfig,
    TTSModelSettings,
    STTModel,
    STTModelSettings,
    StreamedAudioInput,
    StreamedTranscriptionSession,
    TTSModel
)
from elevenlabs.client import AsyncElevenLabs
from stt_module import ElevenLabsSTTModel
from tts_module import ElevenLabsTTSModel

# --- 1. Configuration ---
# Load API keys from .env file
load_dotenv()
ELEVENLABS_STT_API_KEY = os.getenv("ELEVENLABS_STT_API_KEY")
ELEVENLABS_TTS_API_KEY = os.getenv("ELEVENLABS_TTS_API_KEY")

DEFAULT_VOICE_ID = "y1adqrqs4jNaANXsIZnD" # A default voice for testing
SAMPLERATE = 24000 # Sample rate for both recording and playback
DURATION = 5 # Recording duration in seconds

# --- 2. Initialize ElevenLabs Clients ---
if not ELEVENLABS_STT_API_KEY:
    raise ValueError("ELEVENLABS_STT_API_KEY not found in .env file.")
stt_client = AsyncElevenLabs(api_key=ELEVENLABS_STT_API_KEY)

if not ELEVENLABS_TTS_API_KEY:
    raise ValueError("ELEVENLABS_TTS_API_KEY not found in .env file.")
tts_client = AsyncElevenLabs(api_key=ELEVENLABS_TTS_API_KEY)


# --- 3. Define the Agent ---
voice_agent = Agent(
    name="voice_agent",
    instructions="You are a voice agent that can answer questions and help with tasks. Your name in ELVIS and you only answer in english",
    model="gpt-4o-mini"
)
# --- 4.instantiate the speech to text and text to speech models ---
speech_to_text = ElevenLabsSTTModel()
text_to_speech = ElevenLabsTTSModel()

# --- 6. Main Execution ---
async def main():
    """Records audio, processes it through the voice pipeline, and plays the response."""
    
    # a. Define the voice pipeline configuration
    pipeline_config = VoicePipelineConfig(
        tts_settings=TTSModelSettings(
            # The agent's instructions are the primary driver for language.
            # This setting can be used for voice selection if needed.
            voice=DEFAULT_VOICE_ID 
        )
    )

    # b. Create the voice pipeline instance
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(voice_agent),
        config=pipeline_config
    )
    
    # c. Set our custom STT and TTS models
    pipeline.stt_model = speech_to_text
    pipeline.tts_model = text_to_speech

    # d. Record audio from the microphone
    print(f"Recording for {DURATION} seconds...")
    audio_data = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete! Processing...")
    
    audio_input = AudioInput(buffer=audio_data.flatten())

    # e. Run the pipeline and get the result
    result = await pipeline.run(audio_input)
    
    # f. Stream the audio response and play it
    response_audio = []
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            response_audio.append(event.data)
            
    if response_audio:
        print("Playing response...")
        full_response = np.concatenate(response_audio, axis=0)
        sd.play(full_response, SAMPLERATE)
        sd.wait()
        print("Playback complete.")
    else:
        print("No audio response was generated.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting program.")
