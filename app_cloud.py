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
import io
import asyncio
import wave
import numpy as np
from google import genai
from google.genai import types
import os
import tempfile
import traceback
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Voice Recorder with Gemini AI",
    page_icon="🎤",
    layout="centered"
)
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

# Load .env variables
load_dotenv()
# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize session state
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'gemini_response' not in st.session_state:
    st.session_state.gemini_response = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

st.title("🎤 Voice Recorder with Gemini AI")
st.markdown("Record your voice, get AI feedback, and play back responses!")

# ------------------------ #
# Sidebar: API Key Settings
# ------------------------ #
with st.sidebar:
    st.header("⚙️ AI & API Settings")
    st.info(f"🔑 Current API Key: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-4:] if GEMINI_API_KEY else 'Not Set'}")
    
    # Test API key
    if st.button("🧪 Test API Key"):
        if GEMINI_API_KEY:
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                st.success("✅ API Key is valid!")
            except Exception as e:
                st.error(f"❌ API Key test failed: {str(e)}")
        else:
            st.error("❌ No API key found")

# Gemini configuration
MODEL = "gemini-2.0-flash-exp"  # Using a more stable model
config = {
    "response_modalities": ["AUDIO"],
    "system_instruction": (
        "You are a friendly and patient language learning assistant. "
        "Your job is to help the user improve their language skills by listening carefully to their speech, "
        "providing gentle corrections for pronunciation, grammar, vocabulary, and sentence structure. "
        "When the user speaks, respond with clear explanations and examples, "
        "correct mistakes thoughtfully, and encourage them to practice more. "
        "Be supportive, educational, and conversational. Keep your responses concise but helpful."
    ),
}

def convert_audio_to_pcm16(audio_bytes, sample_rate=16000):
    """Convert audio bytes to PCM16 format for Gemini"""
    try:
        # Create a temporary file to work with
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        # Try to load the audio using librosa (most robust)
        try:
            import librosa
            # Load audio with librosa (handles most formats including webm)
            data, original_sr = librosa.load(temp_file_path, sr=None, mono=True)
            
            # Resample to 16kHz if needed
            if original_sr != sample_rate:
                data = librosa.resample(data, orig_sr=original_sr, target_sr=sample_rate)
            
            # Convert to int16 (PCM16)
            # librosa loads as float32 in range [-1, 1]
            data = np.clip(data, -1.0, 1.0)  # Ensure values are in valid range
            data = (data * 32767).astype(np.int16)
            
        except Exception as librosa_error:
            st.warning(f"Librosa failed: {librosa_error}, trying alternative method...")
            
            # Fallback: try with soundfile
            try:
                import soundfile as sf
                data, original_sr = sf.read(temp_file_path)
                
                # Resample if needed
                if original_sr != sample_rate:
                    from scipy import signal
                    num_samples = int(len(data) * sample_rate / original_sr)
                    data = signal.resample(data, num_samples)
                
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                # Convert to int16
                if data.dtype != np.int16:
                    if data.dtype == np.float32 or data.dtype == np.float64:
                        data = np.clip(data, -1.0, 1.0)
                        data = (data * 32767).astype(np.int16)
                    else:
                        data = data.astype(np.int16)
                        
            except Exception as sf_error:
                st.error(f"All audio loading methods failed. Librosa: {librosa_error}, Soundfile: {sf_error}")
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                return None
        
        # Clean up temp file
        try:
            os.unlink(temp_file_path)
        except:
            pass
        
        st.success(f"✅ Audio converted successfully: {len(data)} samples at {sample_rate}Hz")
        return data.tobytes()
    
    except Exception as e:
        st.error(f"❌ Error converting audio: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

async def process_audio_with_gemini(client, audio_bytes):
    """Process audio with Gemini and return response audio"""
    try:
        st.info("🔄 Converting audio format...")
        
        # Convert audio to PCM16 format
        pcm_audio = convert_audio_to_pcm16(audio_bytes)
        if pcm_audio is None:
            return None
        
        st.info("🔄 Connecting to Gemini...")
        
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            st.info("🔄 Sending audio to Gemini...")
            
            # Send audio to Gemini
            await session.send_realtime_input(
                audio=types.Blob(data=pcm_audio, mime_type="audio/pcm;rate=16000")
            )
            
            st.info("🔄 Waiting for Gemini response...")
            
            # Collect response audio
            response_audio = io.BytesIO()
            
            # Create WAV file structure
            wf = wave.open(response_audio, "wb")
            wf.setnchannels(1)  # mono output
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(24000)  # 24kHz output per model docs
            
            response_count = 0
            async for response in session.receive():
                if response.data is not None:
                    wf.writeframes(response.data)
                    response_count += 1
            
            wf.close()
            response_audio.seek(0)
            
            if response_count == 0:
                st.warning("⚠️ No audio response received from Gemini")
                return None
            
            st.info(f"✅ Received {response_count} audio chunks from Gemini")
            return response_audio.getvalue()
            
    except Exception as e:
        st.error(f"❌ Error processing with Gemini: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def run_gemini_processing_sync(audio_bytes):
    """Synchronous wrapper for Gemini processing"""
    try:
        if not GEMINI_API_KEY:
            st.error("❌ No Gemini API key found in environment")
            return None
        
        # Create client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(process_audio_with_gemini(client, audio_bytes))
        finally:
            loop.close()
        
        return result
        
    except Exception as e:
        st.error(f"❌ Error in processing: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Main recording interface
st.markdown("### 🎙️ Click the microphone to record")

# Record audio
audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording", 
    just_once=False,
    use_container_width=True,
    callback=None,
    args=(),
    kwargs={},
    key='recorder'
)

# Process recorded audio
if audio and not st.session_state.processing:
    st.session_state.audio_data = audio
    
    st.markdown("### 🔊 Your Recording")
    st.success("✅ Recording completed!")
    
    # Display audio info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🎵 Format: {audio['format']}")
    with col2:
        st.info(f"📊 Sample rate: {audio['sample_rate']} Hz")
    
    # Play original audio
    st.audio(audio['bytes'], format=audio['format'])
    
    # Download button for original
    st.download_button(
        label="💾 Download Original Recording",
        data=audio['bytes'],
        file_name=f"recording.{audio['format']}",
        mime=f"audio/{audio['format']}",
        use_container_width=True
    )

# Gemini AI Processing Section
if st.session_state.audio_data and not st.session_state.processing:
    st.markdown("### 🤖 AI Analysis with Gemini")
    
    if st.button("🧠 Analyze with Gemini AI", use_container_width=True, type="primary"):
        if not GEMINI_API_KEY:
            st.error("❌ No Gemini API key found. Please set GEMINI_API_KEY in your .env file")
        else:
            st.session_state.processing = True
            
            # Create a placeholder for status updates
            status_placeholder = st.empty()
            
            with status_placeholder.container():
                st.info("🚀 Starting Gemini processing...")
                
                # Process the audio
                result = run_gemini_processing_sync(st.session_state.audio_data['bytes'])
                
                if result and len(result) > 0:
                    st.session_state.gemini_response = result
                    st.session_state.processing = False
                    status_placeholder.success("✅ AI analysis complete!")
                    st.rerun()
                else:
                    st.session_state.processing = False
                    status_placeholder.error("❌ Failed to get response from Gemini AI")

# Display Gemini response
if st.session_state.gemini_response:
    st.markdown("### 🎯 AI Feedback & Response")
    st.success("🤖 Gemini has analyzed your speech!")
    
    # Play Gemini's audio response
    st.audio(st.session_state.gemini_response, format="audio/wav")
    
    # Download Gemini response
    st.download_button(
        label="💾 Download AI Response",
        data=st.session_state.gemini_response,
        file_name="gemini_response.wav",
        mime="audio/wav",
        use_container_width=True
    )

# Control buttons
if st.session_state.audio_data or st.session_state.gemini_response:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Recording", use_container_width=True):
            st.session_state.audio_data = None
            st.session_state.gemini_response = None
            st.session_state.processing = False
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.audio_data = None
            st.session_state.gemini_response = None
            st.session_state.processing = False
            st.rerun()

# Alternative file upload
st.markdown("---")
st.markdown("### 🎵 Alternative: File Upload")
st.markdown("If the microphone doesn't work, you can upload an audio file:")

uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=['wav', 'mp3', 'ogg', 'm4a'],
    key='audio_upload'
)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="💾 Download Uploaded File",
            data=uploaded_file.getvalue(),
            file_name=uploaded_file.name,
            mime="audio/wav"
        )
    
    with col2:
        if st.button("🧠 Analyze Uploaded File", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("❌ No Gemini API key found")
            else:
                with st.spinner("Processing uploaded file..."):
                    result = run_gemini_processing_sync(uploaded_file.getvalue())
                    if result:
                        st.session_state.gemini_response = result
                        st.success("✅ Uploaded file analyzed!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process uploaded file")

# Instructions
with st.expander("📝 How to use"):
    st.markdown("""
    ## Setup
    1. Create a `.env` file in your project directory
    2. Add your API key: `GEMINI_API_KEY=your_api_key_here`
    3. Install dependencies: `pip install streamlit-mic-recorder google-genai soundfile librosa numpy python-dotenv`
    
    ## Recording & AI Analysis
    1. **Click the microphone button** to start recording
    2. **Speak clearly** into your microphone
    3. **Click stop** when finished
    4. **Click "Analyze with Gemini AI"** to get AI feedback
    5. **Listen to both** your original recording and AI response
    
    ## Troubleshooting
    - Check the sidebar for API key status
    - Use "Test API Key" button to verify connection
    - Make sure you have a stable internet connection
    - Try uploading a file if microphone doesn't work
    """)

# Debug information
with st.expander("🔧 Debug Information"):
    st.write("**Environment:**")
    st.write(f"- API Key Set: {'✅ Yes' if GEMINI_API_KEY else '❌ No'}")
    st.write(f"- API Key Preview: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-4:] if GEMINI_API_KEY else 'None'}")
    
    st.write("**Session State:**")
    st.write(f"- Has Audio Data: {'✅ Yes' if st.session_state.audio_data else '❌ No'}")
    st.write(f"- Has Gemini Response: {'✅ Yes' if st.session_state.gemini_response else '❌ No'}")
    st.write(f"- Processing: {'🔄 Yes' if st.session_state.processing else '✅ No'}")
    
    if st.session_state.audio_data:
        st.write(f"- Audio Format: {st.session_state.audio_data.get('format', 'Unknown')}")
        st.write(f"- Audio Size: {len(st.session_state.audio_data.get('bytes', []))} bytes")

# Footer
st.markdown("---")
st.markdown("🎵 Enhanced Voice Recorder with Gemini AI - Built with Streamlit")

# Display current status
if st.session_state.processing:
    st.info("🔄 Processing with AI...")
elif st.session_state.gemini_response:
    st.success("✅ AI analysis ready!")
elif st.session_state.audio_data:
    st.info("🎤 Recording ready for analysis")
else:
    st.info("🎙️ Ready to record")