import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
import tempfile
import wave
import asyncio
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
with st.sidebar:
    st.header("⚙️ AI & API Settings")

    # Gemini API Key
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get("GEMINI_API_KEY", ""),
        help="Enter your Gemini (Google AI) API key"
    )

    # Save Button
    if st.button("💾 Save Settings"):
        updated = False

        if gemini_api_key:
            st.session_state["GEMINI_API_KEY"] = gemini_api_key
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            updated = True

        if updated:
            st.success("✅ Settings saved successfully!")
        else:
            st.error("❌ Please enter at least one API key or model selection.")

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL = "gemini-2.5-flash-preview-native-audio-dialog"

config = {
    "response_modalities": ["AUDIO"],
    "system_instruction": (
        "You are a friendly and patient language learning assistant. "
        "Your job is to help the user improve their language skills by listening carefully to their speech, "
        "providing gentle corrections for pronunciation, grammar, vocabulary, and sentence structure."
    ),
}

# Init Gemini client
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

# UI
st.title("🎤 Voice Language Learning Assistant (WebRTC)")
st.markdown("Record your voice in-browser and get feedback!")

# WebRTC audio processor
class AudioRecorder:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        audio_data = frame.to_ndarray().flatten()
        self.frames.append(audio_data)
        return frame

recorder = AudioRecorder()

ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    client_settings=ClientSettings(
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_processor_factory=lambda: recorder,
)

# When recording ends
if ctx.audio_receiver and st.button("Process Recording"):
    # Combine audio frames
    audio_np = np.concatenate(recorder.frames).astype(np.int16)

    # Save to temp WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        with wave.open(f, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_np.tobytes())
        wav_bytes = f.read()

    st.success("✅ Audio captured successfully!")
    st.audio(wav_bytes, format="audio/wav")

    # Process with Gemini
    async def process_audio_with_gemini():
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            await session.send_realtime_input(
                audio=types.Blob(data=audio_np.tobytes(), mime_type="audio/pcm;rate=16000")
            )
            response_audio = b""
            async for response in session.receive():
                if response.data is not None:
                    response_audio += response.data
            return response_audio

    st.info("🎧 Sending to Gemini...")
    response_audio = asyncio.run(process_audio_with_gemini())

    if response_audio:
        st.success("✅ AI response received!")
        st.audio(response_audio, format="audio/wav")
        st.download_button(
            "💾 Download Response",
            data=response_audio,
            file_name="gemini_response.wav",
            mime="audio/wav"
        )
    else:
        st.error("❌ Failed to process audio.")

