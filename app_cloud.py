import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import tempfile
import wave
import asyncio
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL = "gemini-2.5-flash-preview-native-audio-dialog"

# Gemini Config
config = {
    "response_modalities": ["AUDIO"],
    "system_instruction": (
        "You are a friendly and patient language learning assistant. "
        "Your job is to help the user improve their language skills by listening carefully to their speech, "
        "providing gentle corrections for pronunciation, grammar, vocabulary, and sentence structure. "
        "Respond with clear explanations and encouragement."
    ),
}

# Initialize Gemini client
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ Please set your GEMINI_API_KEY in your environment or .env file.")
    st.stop()
client = genai.Client(api_key=API_KEY)

# App title
st.set_page_config(page_title="🎤 Language Learning Assistant")
st.title("🎤 Voice Language Learning Assistant")
st.markdown("Practice speaking and get feedback from Gemini AI!")

# Audio Processor to record from browser mic
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten()
        self.frames.append(audio)
        return frame

# Start the webrtc mic stream
recorder = AudioProcessor()

ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_processor_factory=lambda: recorder,
)

# Only proceed when stream is active and user clicks button
if ctx.audio_receiver and st.button("✅ Process Recording"):

    if not recorder.frames:
        st.warning("No audio frames recorded yet.")
        st.stop()

    # Combine all audio frames
    audio_np = np.concatenate(recorder.frames).astype(np.int16)
    audio_bytes = audio_np.tobytes()

    # Save to WAV for playback and download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        with wave.open(f.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        wav_data = open(f.name, "rb").read()

    st.success("✅ Audio recorded!")
    st.audio(wav_data, format="audio/wav")

    # Gemini audio processing
    async def process_audio_with_gemini():
        try:
            async with client.aio.live.connect(model=MODEL, config=config) as session:
                await session.send_realtime_input(
                    audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                )
                response_audio = b""
                async for response in session.receive():
                    if response.data:
                        response_audio += response.data
                return response_audio
        except Exception as e:
            st.error(f"❌ Error during Gemini processing: {e}")
            return None

    st.info("⏳ Sending to Gemini...")
    response_audio = asyncio.run(process_audio_with_gemini())

    if response_audio:
        st.success("✅ AI response received!")
        st.audio(response_audio, format="audio/wav")
        st.download_button(
            "💾 Download AI Response",
            data=response_audio,
            file_name="ai_response.wav",
            mime="audio/wav"
        )
    else:
        st.error("❌ Failed to get AI response.")
