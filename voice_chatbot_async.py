import asyncio
import io
import wave
import sounddevice as sd
import numpy as np
import soundfile as sf
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import platform
import subprocess
import sys

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GENAI_API_KEY not found in environment. Please set it in your .env file.")
    sys.exit(1)

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5
MODEL = "gemini-2.5-flash-preview-native-audio-dialog"

config = {
    "response_modalities": ["AUDIO"],
    "system_instruction": (
        "You are a friendly and patient language learning assistant. "
        "Your job is to help the user improve their language skills by listening carefully to their speech, "
        "providing gentle corrections for pronunciation, grammar, vocabulary, and sentence structure. "
        "When the user speaks, respond with clear explanations and examples, "
        "correct mistakes thoughtfully, and encourage them to practice more. "
        "Be supportive, educational, and conversational."
    ),
}


# Initialize GenAI client with API key from env
client = genai.Client(api_key=API_KEY)

def record_audio_pcm16():
    print(f"Recording {RECORD_SECONDS} seconds of audio... Speak now")
    recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    print("Recording stopped.")
    audio_bytes = recording.tobytes()
    return audio_bytes

def play_audio(filename="response.wav"):
    if platform.system() == "Windows":
        subprocess.call(["powershell", "-c", "Start-Process", filename])
    elif platform.system() == "Darwin":
        subprocess.call(["afplay", filename])
    else:
        subprocess.call(["aplay", filename])

async def main():
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        audio_bytes = record_audio_pcm16()

        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )

        wf = wave.open("response.wav", "wb")
        wf.setnchannels(1)  # mono output
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(24000)  # 24kHz output per model docs

        print("Waiting for response audio...")

        async for response in session.receive():
            if response.data is not None:
                wf.writeframes(response.data)

        wf.close()
        print("Response saved as response.wav")
        play_audio("response.wav")

if __name__ == "__main__":
    asyncio.run(main())
