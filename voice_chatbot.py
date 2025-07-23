import os
from dotenv import load_dotenv
import requests
import sounddevice as sd
import numpy as np
import wave
import io
import base64
import platform
import subprocess
import sys

# Load .env file
load_dotenv()

# Load API key from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set in .env file or system env.")
    sys.exit(1)

MODEL_NAME = "models/gemini-2.5-flash-preview-native-audio-dialog"
API_URL = f"https://api.google.com/v1/{MODEL_NAME}:generate"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

def record_audio():
    print("Recording... Speak now")
    try:
        recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
    except Exception as e:
        print(f"Audio recording error: {e}")
        return None
    print("Recording stopped.")
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(recording.tobytes())
        return wav_buffer.getvalue()

def play_audio(base64_audio):
    audio_bytes = base64.b64decode(base64_audio)
    temp_filename = "response.wav"
    try:
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
        if platform.system() == "Windows":
            subprocess.call(["powershell", "-c", "Start-Process", temp_filename])
        elif platform.system() == "Darwin":
            subprocess.call(["afplay", temp_filename])
        else:
            subprocess.call(["aplay", temp_filename])
    except Exception as e:
        print(f"Error playing audio: {e}")

def chat_with_gemini(audio_bytes):
    if audio_bytes is None:
        return None, None
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    payload = {
        "input": {
            "audio": base64_audio
        },
        "parameters": {
            "enableThinking": True,
            "stylePrompt": "friendly and helpful"
        }
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        text_response = data.get("output", {}).get("text", "")
        audio_response = data.get("output", {}).get("audio", None)
        return text_response, audio_response
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return None, None

def main():
    print("Voice Chatbot (say 'exit' to quit)")
    while True:
        audio_bytes = record_audio()
        if audio_bytes is None:
            continue
        text_resp, audio_resp = chat_with_gemini(audio_bytes)
        if text_resp:
            print("Bot:", text_resp)
            if "exit" in text_resp.lower():
                break
        if audio_resp:
            play_audio(audio_resp)

if __name__ == "__main__":
    main()
