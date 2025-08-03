# SinglePassVoicePipeline Implementation

This project demonstrates an implementation of a voice pipeline using OpenAI's Agents SDK and the ElevenLabs API for Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities. The pipeline captures user audio, transcribes it to text, processes the text with an AI agent, and responds using synthesized speech.

## Project Structure

- `main.py`: The main entry point for the voice pipeline. It configures and runs the voice pipeline, handling both Speech-to-Text (STT) and Text-to-Speech (TTS) functionality.
- `stt_module.py`: Contains the implementation for the custom STT model using ElevenLabs' STT API.
- `tts_generator.py`: Contains the implementation for the custom TTS model using ElevenLabs' TTS API.
- `.gitignore`: A standard `.gitignore` file to exclude sensitive and unnecessary files.
- `requirements.txt`: The list of dependencies required to run the project.

## Requirements

The project requires the following dependencies:

- `openai-agents[voice]`
- `elevenlabs`
- `sounddevice`
- `numpy`
- `python-dotenv`

To install these dependencies, run:

pip install -r requirements.txt

## Set up Environment Variables
The project requires two API keys from ElevenLabs to interact with the STT and TTS APIs. Ensure you have the following environment variables in your .env file:

ELEVENLABS_STT_API_KEY=your_stt_api_key
ELEVENLABS_TTS_API_KEY=your_tts_api_key

## Running the Pipeline
To run the pipeline and test the voice interaction:

python main.py

This script will:

Record audio for a specified duration.

Use ElevenLabs' Speech-to-Text (STT) API to transcribe the speech.

Use OpenAI's GPT-4o-mini model (via the voice_agent) to process the transcription.

Use ElevenLabs' Text-to-Speech (TTS) API to synthesize the response.

Play the response audio back to the user.

## Customizing the Agent

In the main.py file, you can customize the voice agent settings:

voice_agent = Agent(
    name="voice_agent",
    instructions="You are a voice agent that can answer questions and help with tasks. Your name in ELVIS",
    model="gpt-4o-mini"
)

You can adjust the agent's model, name, and instructions based on your needs.

## File Details
stt_module.py
This module contains the custom implementation of a Speech-to-Text model that interfaces with the ElevenLabs STT API. The model handles audio input, converts it into a WAV format, and sends it to ElevenLabs for transcription.

tts_generator.py
This module contains the custom implementation of a Text-to-Speech model that interfaces with the ElevenLabs TTS API. It takes text as input and streams the synthesized audio to be played back to the user.

.gitignore
The .gitignore file excludes sensitive and unnecessary files from version control, including the .env file and Python's __pycache__.

requirements.txt
This file lists the required Python packages for the project. To set up the environment, use the command pip install -r requirements.txt.

## Notes
Ensure you have valid API keys from ElevenLabs for both STT and TTS.

This implementation records audio, processes it through the pipeline, and plays back the response in real-time.

Modify the DURATION variable in main.py to adjust the length of the recording.

The pipeline uses a default voice (y1adqrqs4jNaANXsIZnD) for TTS. You can change the voice ID in the TTSModelSettings if desired.

## License
This project is licensed under the MIT License - see the LICENSE file for details.