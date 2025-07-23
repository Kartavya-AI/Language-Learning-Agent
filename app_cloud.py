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

import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import io
import time

# Configuration
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = np.int16

# Initialize session state
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = []

def record_audio():
    """Record audio continuously until stopped"""
    st.session_state.audio_data = []
    
    def audio_callback(indata, frames, time, status):
        if status:
            st.warning(f"Audio input status: {status}")
        st.session_state.audio_data.append(indata.copy())
    
    # Start recording
    with sd.InputStream(
        callback=audio_callback,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype=DTYPE
    ):
        while st.session_state.is_recording:
            time.sleep(0.1)

def save_audio_to_bytes():
    """Convert recorded audio to bytes for playback"""
    if not st.session_state.audio_data:
        return None
    
    # Concatenate all audio chunks
    audio_array = np.concatenate(st.session_state.audio_data, axis=0)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_array.tobytes())
    
    buffer.seek(0)
    return buffer.getvalue()

# Streamlit UI
st.set_page_config(
    page_title="Voice Recorder",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Simple Voice Recorder")
st.markdown("Record your voice and play it back!")

# Recording controls
col1, col2 = st.columns(2)

with col1:
    if not st.session_state.is_recording:
        if st.button("🎤 Start Recording", use_container_width=True, type="primary"):
            st.session_state.is_recording = True
            st.session_state.recorded_audio = None
            st.rerun()

with col2:
    if st.session_state.is_recording:
        if st.button("⏹️ Stop Recording", use_container_width=True, type="secondary"):
            st.session_state.is_recording = False
            # Process the recorded audio
            st.session_state.recorded_audio = save_audio_to_bytes()
            st.rerun()

# Recording status
if st.session_state.is_recording:
    st.markdown("### 🔴 Recording... Click 'Stop Recording' when done")
    
    # Show a live indicator
    placeholder = st.empty()
    import threading
    
    def update_recording_status():
        while st.session_state.is_recording:
            for i in range(4):
                if not st.session_state.is_recording:
                    break
                placeholder.info(f"Recording{'.' * (i + 1)}")
                time.sleep(0.5)
    
    # Start recording in a separate thread
    if 'recording_thread' not in st.session_state or not st.session_state.recording_thread.is_alive():
        st.session_state.recording_thread = threading.Thread(target=record_audio)
        st.session_state.recording_thread.daemon = True
        st.session_state.recording_thread.start()
    
    # Start status update thread
    status_thread = threading.Thread(target=update_recording_status)
    status_thread.daemon = True
    status_thread.start()

# Playback section
if st.session_state.recorded_audio and not st.session_state.is_recording:
    st.markdown("### 🔊 Playback")
    
    st.success("✅ Recording completed!")
    
    # Audio player
    st.audio(st.session_state.recorded_audio, format='audio/wav')
    
    # Download button
    st.download_button(
        label="💾 Download Recording",
        data=st.session_state.recorded_audio,
        file_name="recorded_audio.wav",
        mime="audio/wav",
        use_container_width=True
    )
    
    # Clear recording button
    if st.button("🗑️ Clear Recording", use_container_width=True):
        st.session_state.recorded_audio = None
        st.session_state.audio_data = []
        st.rerun()

# Instructions
with st.expander("📝 How to use"):
    st.markdown("""
    1. **Click 'Start Recording'** to begin
    2. **Speak into your microphone**
    3. **Click 'Stop Recording'** when finished
    4. **Use the audio player** to listen to your recording
    5. **Download** your recording if needed
    6. **Clear** to record again
    
    **Note**: Make sure to allow microphone access when prompted by your browser.
    """)

# Footer
st.markdown("---")
st.markdown("🎵 Built with Streamlit")