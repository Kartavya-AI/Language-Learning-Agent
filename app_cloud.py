# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, WebRtcMode
# import av
# import numpy as np
# import tempfile
# import wave
# import asyncio
# from dotenv import load_dotenv
# import os
# from google import genai
# from google.genai import types

# # ------------------------ #
# # Sidebar: API Key Settings
# # ------------------------ #
# with st.sidebar:
#     st.header("⚙️ AI & API Settings")

#     gemini_api_key = st.text_input(
#         "Gemini API Key",
#         type="password",
#         value=st.session_state.get("GEMINI_API_KEY", ""),
#         help="Enter your Gemini (Google AI) API key"
#     )

#     if st.button("💾 Save Settings"):
#         if gemini_api_key:
#             st.session_state["GEMINI_API_KEY"] = gemini_api_key
#             os.environ["GEMINI_API_KEY"] = gemini_api_key
#             st.success("✅ Settings saved successfully!")
#         else:
#             st.error("❌ Please enter a valid Gemini API key.")

# # Load .env variables
# load_dotenv()

# # ------------------------ #
# # Setup Gemini
# # ------------------------ #
# API_KEY = st.session_state.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
# if not API_KEY:
#     st.error("❌ Gemini API key missing! Please set it in the sidebar.")
#     st.stop()

# client = genai.Client(api_key=API_KEY)

# SAMPLE_RATE = 16000
# CHANNELS = 1
# MODEL = "gemini-2.5-flash-preview-native-audio-dialog"

# config = {
#     "response_modalities": ["AUDIO"],
#     "system_instruction": (
#         "You are a friendly and patient language learning assistant. "
#         "Your job is to help the user improve their language skills by listening carefully to their speech, "
#         "providing gentle corrections for pronunciation, grammar, vocabulary, and sentence structure. "
#         "Respond with clear explanations and encouragement."
#     ),
# }

# # ------------------------ #
# # App UI
# # ------------------------ #
# st.set_page_config(page_title="🎤 Language Learning Assistant")
# st.title("🎤 Voice Language Learning Assistant")
# st.markdown("Practice speaking and get feedback from Gemini AI!")

# # ------------------------ #
# # Recorder class (persistent via session_state)
# # ------------------------ #
# class AudioProcessor:
#     def __init__(self) -> None:
#         self.frames = []

#     def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
#         audio = frame.to_ndarray().flatten()
#         self.frames.append(audio)
#         return frame



# # Ensure recorder is initialized before WebRTC uses it
# if "recorder" not in st.session_state:
#     st.session_state["recorder"] = AudioProcessor()

# recorder_instance = st.session_state["recorder"]

# ctx = webrtc_streamer(
#     key="mic",
#     mode=WebRtcMode.SENDONLY,
#     audio_receiver_size=256,
#     media_stream_constraints={"video": False, "audio": True},
#     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     audio_processor_factory=lambda: recorder_instance,
# )

# # ------------------------ #
# # Process Button
# # ------------------------ #
# if ctx.audio_receiver and st.button("✅ Process Recording"):
#     if not st.session_state.recorder.frames:
#         st.warning("⚠️ No audio recorded. Click START on the microphone component above, speak, then click STOP.")
#         st.stop()

#     # Get the recorded frames and immediately clear the buffer to avoid accumulation
#     recorded_frames = st.session_state.recorder.frames
#     st.session_state.recorder.frames = []

#     # Combine audio
#     audio_np = np.concatenate(recorded_frames).astype(np.int16)
#     audio_bytes = audio_np.tobytes()

#     # Save to WAV
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         with wave.open(f.name, "wb") as wf:
#             wf.setnchannels(CHANNELS)
#             wf.setsampwidth(2)
#             wf.setframerate(SAMPLE_RATE)
#             wf.writeframes(audio_bytes)
#         wav_data = open(f.name, "rb").read()

#     st.success("✅ Audio recorded!")
#     st.audio(wav_data, format="audio/wav")

#     # ------------------------ #
#     # Gemini Audio Processing
#     # ------------------------ #
#     async def process_audio_with_gemini():
#         try:
#             async with client.aio.live.connect(model=MODEL, config=config) as session:
#                 await session.send_realtime_input(
#                     audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
#                 )
#                 response_audio = b""
#                 async for response in session.receive():
#                     if response.data:
#                         response_audio += response.data
#                 return response_audio
#         except Exception as e:
#             st.error(f"❌ Gemini processing error: {e}")
#             return None

#     st.info("⏳ Sending to Gemini...")
#     response_audio = asyncio.run(process_audio_with_gemini())

#     if response_audio:
#         st.success("✅ AI response received!")
#         st.audio(response_audio, format="audio/wav")
#         st.download_button("💾 Download AI Response", data=response_audio, file_name="ai_response.wav", mime="audio/wav")
#     else:
#         st.error("❌ Failed to get AI response.")

# # ------------------------ #
# # Optional: Upload Fallback
# # ------------------------ #
# st.markdown("---")
# with st.expander("📤 Or upload a recording instead"):
#     uploaded = st.file_uploader("Upload a .wav file", type=["wav"])
#     if uploaded:
#         st.audio(uploaded, format="audio/wav")
#         bytes_data = uploaded.read()

#         async def process_uploaded_audio():
#             try:
#                 async with client.aio.live.connect(model=MODEL, config=config) as session:
#                     await session.send_realtime_input(
#                         audio=types.Blob(data=bytes_data, mime_type="audio/pcm;rate=16000")
#                     )
#                     response_audio = b""
#                     async for response in session.receive():
#                         if response.data:
#                             response_audio += response.data
#                     return response_audio
#             except Exception as e:
#                 st.error(f"❌ Error: {e}")
#                 return None

#         if st.button("🔄 Process Uploaded File"):
#             st.info("⏳ Sending to Gemini...")
#             result = asyncio.run(process_uploaded_audio())
#             if result:
#                 st.success("✅ AI response received!")
#                 st.audio(result, format="audio/wav")
#                 st.download_button("💾 Download AI Response", data=result, file_name="ai_response.wav", mime="audio/wav")
#             else:
#                 st.error("❌ Failed to process audio.")

# # ------------------------ #
# # Footer
# # ------------------------ #
# st.markdown("---")
# st.markdown("Built with ❤️ using Streamlit & Gemini AI")

# import streamlit as st
# from streamlit_mic_recorder import mic_recorder
# import io

# # Set page config
# st.set_page_config(
#     page_title="Voice Recorder",
#     page_icon="🎤",
#     layout="centered"
# )

# st.title("🎤 Simple Voice Recorder")
# st.markdown("Record your voice and play it back!")

# # Initialize session state
# if 'audio_data' not in st.session_state:
#     st.session_state.audio_data = None

# # Simple audio recorder using streamlit-mic-recorder
# st.markdown("### 🎙️ Click the microphone to record")

# # Record audio
# audio = mic_recorder(
#     start_prompt="🎤 Start Recording",
#     stop_prompt="⏹️ Stop Recording", 
#     just_once=False,
#     use_container_width=True,
#     callback=None,
#     args=(),
#     kwargs={},
#     key='recorder'
# )

# # Process recorded audio
# if audio:
#     st.session_state.audio_data = audio
    
#     st.markdown("### 🔊 Playback")
#     st.success("✅ Recording completed!")
    
#     # Display audio info
#     st.info(f"Audio format: {audio['format']}")
#     st.info(f"Sample rate: {audio['sample_rate']} Hz")
    
#     # Play audio
#     st.audio(audio['bytes'], format=audio['format'])
    
#     # Download button
#     st.download_button(
#         label="💾 Download Recording",
#         data=audio['bytes'],
#         file_name=f"recording.{audio['format']}",
#         mime=f"audio/{audio['format']}",
#         use_container_width=True
#     )
    
#     # Clear button
#     if st.button("🗑️ Clear Recording", use_container_width=True):
#         st.session_state.audio_data = None
#         st.rerun()

# # Instructions
# with st.expander("📝 How to use"):
#     st.markdown("""
#     1. **Click the microphone button** to start recording
#     2. **Speak into your microphone**
#     3. **Click stop** when finished
#     4. **Listen to your recording** using the audio player
#     5. **Download** if needed, or **Clear** to record again
    
#     **Note**: 
#     - Your browser will ask for microphone permission
#     - This works on Streamlit Cloud and local development
#     - Audio is processed entirely in the browser
#     """)

# # Alternative simple recorder if streamlit-mic-recorder doesn't work
# st.markdown("---")
# st.markdown("### 🎵 Alternative: File Upload")
# st.markdown("If the microphone doesn't work, you can upload an audio file:")

# uploaded_file = st.file_uploader(
#     "Choose an audio file", 
#     type=['wav', 'mp3', 'ogg', 'm4a'],
#     key='audio_upload'
# )

# if uploaded_file is not None:
#     st.audio(uploaded_file, format='audio/wav')
    
#     st.download_button(
#         label="💾 Download Uploaded File",
#         data=uploaded_file.getvalue(),
#         file_name=uploaded_file.name,
#         mime="audio/wav"
#     )

# st.markdown("---")
# st.markdown("🎵 Built with Streamlit - Cloud Compatible")

import streamlit as st
from streamlit_mic_recorder import mic_recorder
import os
import tempfile
import wave
from dotenv import load_dotenv
from google.generativeai.types import Blob
import asyncio
from google import genai

# Load .env if available
load_dotenv()

# Constants
MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
SAMPLE_RATE = 16000

# --- Page Config ---
st.set_page_config(page_title="🎤 Voice Feedback with Gemini", layout="centered")
st.title("🎤 Voice Recorder + Gemini Feedback")
st.markdown("Record your voice, and get feedback from Google Gemini!")

# --- Sidebar for API key ---
with st.sidebar:
    st.header("⚙️ AI & API Settings")

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get("GEMINI_API_KEY", ""),
        help="Enter your Gemini (Google AI) API key"
    )

    if st.button("💾 Save Settings"):
        if gemini_api_key:
            st.session_state["GEMINI_API_KEY"] = gemini_api_key
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            st.success("✅ Settings saved!")
        else:
            st.error("❌ Please enter a valid Gemini API key.")

# --- Initialize Gemini client ---
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
else:
    st.warning("⚠️ Please enter your Gemini API key in the sidebar to enable AI feedback.")

# --- Recorder UI ---
st.markdown("### 🎙️ Record your voice")
audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=False,
    use_container_width=True,
    key='recorder'
)

# --- Upload Option ---
st.markdown("### 🎵 Or Upload an Audio File")
uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg", "m4a"])

# --- Handle Audio Input ---
audio_bytes = None
audio_format = "wav"

if audio:
    audio_bytes = audio["bytes"]
    audio_format = audio["format"]
    st.audio(audio_bytes, format=f'audio/{audio_format}')
elif uploaded_file:
    audio_bytes = uploaded_file.read()
    audio_format = uploaded_file.name.split('.')[-1]
    st.audio(audio_bytes, format=f'audio/{audio_format}')

# --- Feedback Button ---
if audio_bytes and os.getenv("GEMINI_API_KEY"):
    st.markdown("### 🤖 Get AI Voice Feedback")
    if st.button("🧠 Send to Gemini AI"):
        with st.spinner("🎧 Gemini is analyzing your voice..."):

            # Save to temp WAV file in PCM16 for Gemini
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                with wave.open(tmp.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_bytes)
                tmp_path = tmp.name

            # Load saved audio and convert to raw PCM16
            with open(tmp_path, 'rb') as f:
                raw_audio = f.read()

            async def get_gemini_response():
                client = genai.AsyncClient()
                config = {
                    "response_modalities": ["AUDIO"],
                    "system_instruction": (
                        "You are a friendly and patient language learning assistant. "
                        "Provide pronunciation feedback, grammar corrections, and encouragement. "
                        "Be kind, helpful, and engaging in your tone."
                    ),
                }

                async with client.aio.live.connect(model=MODEL, config=config) as session:
                    await session.send_realtime_input(
                        audio=Blob(data=raw_audio, mime_type="audio/pcm;rate=16000")
                    )

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as resp_wav:
                        wf = wave.open(resp_wav.name, 'wb')
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)

                        async for response in session.receive():
                            if response.data:
                                wf.writeframes(response.data)

                        wf.close()
                        return resp_wav.name

            response_audio_path = asyncio.run(get_gemini_response())

            # Playback Gemini response
            st.success("✅ Feedback received!")
            with open(response_audio_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")

# --- Instructions ---
with st.expander("📝 How to use"):
    st.markdown("""
    1. 🎤 **Record your voice** or upload a recording
    2. 📩 **Click "Send to Gemini AI"**
    3. 🤖 **Listen to the feedback** generated by Gemini
    4. 🧠 Practice and improve!

    **Note:**
    - Gemini responds in audio (like a language tutor).
    - This app processes raw audio in-browser and via Gemini’s native audio model.
    """)
