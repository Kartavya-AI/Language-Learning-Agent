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
from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.generativeai.types import Blob
import soundfile as sf
import tempfile
import io
import platform
import subprocess
# ------------------------ #
# Sidebar: API Key Settings
# ------------------------ #
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
            st.success("✅ Settings saved successfully!")
        else:
            st.error("❌ Please enter a valid Gemini API key.")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()

# Configure Gemini
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
client = genai.Client(api_key=API_KEY)

st.set_page_config(page_title="Language Learning Assistant 🎧", layout="centered")
st.title("🗣️ Language Learning Assistant with Gemini AI")
st.markdown("Record your voice and get spoken feedback from an AI language tutor.")

# --- Section 1: Microphone Recorder ---
audio_data = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop",
    just_once=False,
    use_container_width=True,
    key='recorder'
)

# OR Section 2: Upload File
uploaded_file = st.file_uploader("Or upload an audio file (wav, mp3, m4a, ogg):", type=["wav", "mp3", "m4a", "ogg"])

# Handle the input
if audio_data or uploaded_file:
    # Determine source
    audio_bytes = None
    sample_rate = 16000  # default fallback

    if audio_data:
        st.success("✅ Recording complete!")
        st.audio(audio_data['bytes'], format=audio_data['format'])
        audio_bytes = audio_data['bytes']
        sample_rate = audio_data['sample_rate']

    elif uploaded_file:
        st.success("✅ File uploaded!")
        st.audio(uploaded_file, format="audio/wav")
        audio_bytes = uploaded_file.read()
        sample_rate = 16000  # may vary

    # Gemini expects PCM format
    st.markdown("🎧 Processing your audio and sending to Gemini...")

    try:
        with client.aio.live.connect(model=MODEL, config=config) as session:
            # Convert to required PCM format
            pcm_buffer = io.BytesIO()
            data, samplerate = sf.read(io.BytesIO(audio_bytes))
            sf.write(pcm_buffer, data, 16000, format='RAW', subtype='PCM_16')
            pcm_bytes = pcm_buffer.getvalue()

            # Send to Gemini
            session.send_realtime_input(
                audio=Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000")
            )

            # Collect audio response
            ai_audio = io.BytesIO()
            async for response in session.receive():
                if response.data:
                    ai_audio.write(response.data)
            ai_audio.seek(0)

        st.success("✅ AI Response received!")
        st.markdown("### 🔁 Gemini's Spoken Feedback")
        st.audio(ai_audio.read(), format="audio/wav")

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")

# Footer Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    - Use the microphone to record your voice or upload an audio file.
    - Your voice is analyzed by Gemini AI for pronunciation and language feedback.
    - You will receive an AI-generated spoken response to help improve your language.
    - Best used with clear, slow speech for optimal feedback.
    """)

