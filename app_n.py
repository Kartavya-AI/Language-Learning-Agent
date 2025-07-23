import streamlit as st
import asyncio
import io
import wave
import sounddevice as sd

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
import tempfile

# Load environment variables
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

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
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

# Initialize session state
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'client' not in st.session_state:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if API_KEY:
        st.session_state.client = genai.Client(api_key=API_KEY)
    else:
        st.session_state.client = None

def record_audio_chunk(duration=1):
    """Record a chunk of audio"""
    recording = sd.rec(
        int(duration * SAMPLE_RATE), 
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        dtype='int16'
    )
    sd.wait()
    return recording.tobytes()

def play_audio_from_bytes(audio_bytes, sample_rate=24000):
    """Play audio from bytes"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            # Write wave file
            with wave.open(tmp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            
            # Play the file
            if platform.system() == "Windows":
                subprocess.Popen([
                    "powershell", "-c", 
                    f"(New-Object Media.SoundPlayer '{tmp_file.name}').PlaySync()"
                ])
            elif platform.system() == "Darwin":
                subprocess.Popen(["afplay", tmp_file.name])
            else:
                subprocess.Popen(["aplay", tmp_file.name])
                
        return tmp_file.name
    except Exception as e:
        st.error(f"Error playing audio: {e}")
        return None

async def process_audio_with_gemini(client, audio_bytes):
    """Process audio with Gemini model"""
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            # Send audio input
            await session.send_realtime_input(
                audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
            )
            
            # Collect response audio
            response_audio = b""
            async for response in session.receive():
                if response.data is not None:
                    response_audio += response.data
            
            return response_audio
    except Exception as e:
        st.error(f"Error processing with Gemini: {e}")
        return None

def run_async_in_thread(coro):
    """Run async function in a thread"""
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return run()

# Streamlit UI
st.set_page_config(
    page_title="Voice Language Assistant",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Voice Language Learning Assistant")
st.markdown("Practice your language skills with AI-powered voice feedback!")

# Check API key
if not st.session_state.client:
    st.error("❌ Gemini API key not found! Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# Create columns for layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 🎙️ Voice Interaction")
    
    # Recording controls
    if not st.session_state.is_recording:
        if st.button("🎤 Start Recording", use_container_width=True, type="primary"):
            st.session_state.is_recording = True
            st.rerun()
    else:
        if st.button("⏹️ Stop Recording", use_container_width=True, type="secondary"):
            st.session_state.is_recording = False
            st.rerun()

# Real-time recording section
if st.session_state.is_recording:
    st.markdown("### 🔴 Recording in progress...")
    
    # Create placeholders for real-time updates
    status_placeholder = st.empty()
    audio_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    # Recording loop
    recorded_chunks = []
    recording_duration = 0
    max_duration = 10  # Maximum recording duration in seconds
    
    while st.session_state.is_recording and recording_duration < max_duration:
        # Record 1-second chunk
        chunk = record_audio_chunk(1)
        recorded_chunks.append(chunk)
        recording_duration += 1
        
        # Update UI
        status_placeholder.info(f"Recording... {recording_duration}s / {max_duration}s")
        progress_placeholder.progress(recording_duration / max_duration)
        
        # Check if user stopped recording
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    # Stop recording
    st.session_state.is_recording = False
    
    if recorded_chunks:
        # Combine all chunks
        combined_audio = b"".join(recorded_chunks)
        st.session_state.audio_data = combined_audio
        
        status_placeholder.success("✅ Recording completed!")
        progress_placeholder.empty()
        
        # Process with Gemini
        st.markdown("### 🤖 Processing with AI...")
        processing_placeholder = st.empty()
        processing_placeholder.info("Sending audio to Gemini...")
        
        # Run async processing
        try:
            response_audio = run_async_in_thread(
                process_audio_with_gemini(st.session_state.client, combined_audio)
            )
            
            if response_audio:
                processing_placeholder.success("✅ Response received!")
                
                # Play response
                st.markdown("### 🔊 AI Response")
                st.info("Playing AI response...")
                
                # Save and play audio
                temp_file = play_audio_from_bytes(response_audio)
                if temp_file:
                    # Also provide download option
                    st.download_button(
                        label="💾 Download Response",
                        data=response_audio,
                        file_name="ai_response.wav",
                        mime="audio/wav"
                    )
                
                # Clean up temp file after a delay
                if temp_file and os.path.exists(temp_file):
                    threading.Timer(10.0, lambda: os.unlink(temp_file) if os.path.exists(temp_file) else None).start()
                    
            else:
                processing_placeholder.error("❌ Failed to process audio")
                
        except Exception as e:
            processing_placeholder.error(f"❌ Error: {e}")

# Instructions and tips
with st.expander("📝 How to use"):
    st.markdown("""
    1. **Click 'Start Recording'** to begin voice input
    2. **Speak clearly** - the AI will listen and analyze your speech
    3. **Click 'Stop Recording'** when you're done (or wait for auto-stop at 10s)
    4. **Wait for processing** - your audio will be sent to Gemini AI
    5. **Listen to feedback** - the AI will respond with corrections and suggestions
    
    **Tips for better results:**
    - Speak in a quiet environment
    - Use clear pronunciation
    - Keep recordings under 10 seconds for best performance
    - Practice regularly for improvement
    """)

with st.expander("⚙️ Settings"):
    st.markdown(f"""
    **Current Configuration:**
    - Sample Rate: {SAMPLE_RATE} Hz
    - Channels: {CHANNELS} (Mono)
    - Model: {MODEL}
    - Max Recording: 10 seconds
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Google Gemini AI")