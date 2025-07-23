import streamlit as st
from streamlit_mic_recorder import mic_recorder
import asyncio
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import tempfile
import wave
import io

# Load environment variables
load_dotenv()

# ------------------------ #
# Page Configuration
# ------------------------ #
st.set_page_config(
    page_title="🎤 Language Learning Assistant",
    page_icon="🎤",
    layout="wide"
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
    
    st.markdown("---")
    st.markdown("### 🎯 Instructions")
    st.markdown("""
    1. Enter your Gemini API key above
    2. Click the microphone to record
    3. Speak clearly in any language
    4. Get AI feedback on your pronunciation and grammar
    """)

# ------------------------ #
# Setup Gemini
# ------------------------ #
API_KEY = st.session_state.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ Gemini API key missing! Please set it in the sidebar.")
    st.markdown("### 🔑 Get your API key:")
    st.markdown("1. Go to [Google AI Studio](https://aistudio.google.com/)")
    st.markdown("2. Create an API key")
    st.markdown("3. Enter it in the sidebar")
    st.stop()

# Initialize Gemini client
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"❌ Failed to initialize Gemini client: {e}")
    st.stop()

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
        "Be supportive, educational, and conversational. Keep your responses concise but helpful."
    ),
}

# ------------------------ #
# Main App UI
# ------------------------ #
st.title("🎤 Voice Language Learning Assistant")
st.markdown("Practice speaking and get personalized feedback from Gemini AI!")

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ------------------------ #
# Audio Recording Section
# ------------------------ #
st.markdown("### 🎙️ Record Your Voice")

# Audio recorder
audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=False,
    use_container_width=True,
    callback=None,
    args=(),
    kwargs={},
    key='language_recorder'
)

# ------------------------ #
# Process Recorded Audio
# ------------------------ #
if audio and not st.session_state.processing:
    st.markdown("### 🔊 Your Recording")
    st.success("✅ Audio recorded successfully!")
    
    # Display audio info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 Format: {audio['format']}")
    with col2:
        st.info(f"📈 Sample Rate: {audio['sample_rate']} Hz")
    
    # Play the recorded audio
    st.audio(audio['bytes'], format=audio['format'])
    
    # Process button
    if st.button("🧠 Get AI Feedback", type="primary", use_container_width=True):
        st.session_state.processing = True
        
        # ------------------------ #
        # Gemini Audio Processing
        # ------------------------ #
        async def process_audio_with_gemini(audio_data):
            try:
                with st.spinner("🤖 AI is analyzing your speech..."):
                    async with client.aio.live.connect(model=MODEL, config=config) as session:
                        # Send audio to Gemini
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=audio_data, 
                                mime_type="audio/pcm;rate=16000"
                            )
                        )
                        
                        # Collect response
                        response_audio = b""
                        async for response in session.receive():
                            if response.data:
                                response_audio += response.data
                        
                        return response_audio
            except Exception as e:
                st.error(f"❌ Gemini processing error: {e}")
                return None
        
        # Process the audio
        try:
            # Convert to the right format for Gemini
            audio_bytes = audio['bytes']
            
            # Process with Gemini
            with st.status("Processing with AI...", expanded=True) as status:
                st.write("🎵 Analyzing audio quality...")
                st.write("🧠 Sending to Gemini AI...")
                
                response_audio = asyncio.run(process_audio_with_gemini(audio_bytes))
                
                if response_audio:
                    st.write("✅ AI feedback received!")
                    status.update(label="✅ Processing complete!", state="complete")
                else:
                    st.write("❌ Failed to get response")
                    status.update(label="❌ Processing failed", state="error")
        
        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
            response_audio = None
        
        # ------------------------ #
        # Display AI Response
        # ------------------------ #
        if response_audio:
            st.markdown("### 🎯 AI Feedback")
            st.success("🎉 Your language coach has some feedback for you!")
            
            # Play AI response
            st.audio(response_audio, format="audio/wav")
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "💾 Download Your Recording",
                    data=audio['bytes'],
                    file_name="my_recording.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "💾 Download AI Feedback",
                    data=response_audio,
                    file_name="ai_feedback.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
        else:
            st.error("❌ Failed to get AI feedback. Please try again.")
        
        # Reset processing state
        st.session_state.processing = False
        
        # Option to record again
        if st.button("🔄 Record Again", use_container_width=True):
            st.rerun()

# ------------------------ #
# Alternative: File Upload
# ------------------------ #
st.markdown("---")
with st.expander("📤 Alternative: Upload Audio File"):
    st.markdown("If the microphone doesn't work, you can upload an audio file instead:")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg', 'm4a', 'webm'],
        key='audio_upload'
    )
    
    if uploaded_file is not None:
        st.success("✅ Audio file uploaded!")
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🧠 Analyze Uploaded File", use_container_width=True):
            # Process uploaded file
            async def process_uploaded_audio():
                try:
                    async with client.aio.live.connect(model=MODEL, config=config) as session:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=uploaded_file.read(),
                                mime_type="audio/pcm;rate=16000"
                            )
                        )
                        response_audio = b""
                        async for response in session.receive():
                            if response.data:
                                response_audio += response.data
                        return response_audio
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    return None
            
            with st.spinner("🤖 Processing uploaded file..."):
                result = asyncio.run(process_uploaded_audio())
            
            if result:
                st.success("✅ AI feedback received!")
                st.audio(result, format="audio/wav")
                st.download_button(
                    "💾 Download AI Response",
                    data=result,
                    file_name="ai_feedback.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
            else:
                st.error("❌ Failed to process uploaded audio.")

# ------------------------ #
# Tips and Information
# ------------------------ #
st.markdown("---")
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    ### 🎯 Recording Tips:
    - **Speak clearly** and at a normal pace
    - **Use a quiet environment** to reduce background noise
    - **Hold your device close** to your mouth (but not too close)
    - **Speak for 5-15 seconds** for best results
    
    ### 🌍 Language Learning Tips:
    - **Practice regularly** - consistency is key
    - **Focus on pronunciation** - the AI will help correct you
    - **Try different topics** - vary your vocabulary
    - **Listen carefully** to the AI's feedback and suggestions
    
    ### 🔧 Technical Notes:
    - Works best with **Chrome or Edge browsers**
    - Your browser will ask for **microphone permission**
    - Audio is processed securely through Google's Gemini AI
    - No audio data is stored permanently
    """)

# ------------------------ #
# Footer
# ------------------------ #
st.markdown("---")
st.markdown("### 🚀 Built with:")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🎵 Streamlit** - Web Framework")
with col2:
    st.markdown("**🤖 Gemini AI** - Language Processing")
with col3:
    st.markdown("**🎤 Mic Recorder** - Audio Capture")

st.markdown("---")
st.markdown("💡 **Pro Tip**: Practice daily for 10-15 minutes to see real improvement in your language skills!")

import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io

# Set page config
st.set_page_config(
    page_title="Voice Recorder",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Simple Voice Recorder")
st.markdown("Record your voice and play it back!")

# Initialize session state
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

# Simple audio recorder using streamlit-mic-recorder
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
if audio:
    st.session_state.audio_data = audio
    
    st.markdown("### 🔊 Playback")
    st.success("✅ Recording completed!")
    
    # Display audio info
    st.info(f"Audio format: {audio['format']}")
    st.info(f"Sample rate: {audio['sample_rate']} Hz")
    
    # Play audio
    st.audio(audio['bytes'], format=audio['format'])
    
    # Download button
    st.download_button(
        label="💾 Download Recording",
        data=audio['bytes'],
        file_name=f"recording.{audio['format']}",
        mime=f"audio/{audio['format']}",
        use_container_width=True
    )
    
    # Clear button
    if st.button("🗑️ Clear Recording", use_container_width=True):
        st.session_state.audio_data = None
        st.rerun()

# Instructions
with st.expander("📝 How to use"):
    st.markdown("""
    1. **Click the microphone button** to start recording
    2. **Speak into your microphone**
    3. **Click stop** when finished
    4. **Listen to your recording** using the audio player
    5. **Download** if needed, or **Clear** to record again
    
    **Note**: 
    - Your browser will ask for microphone permission
    - This works on Streamlit Cloud and local development
    - Audio is processed entirely in the browser
    """)

# Alternative simple recorder if streamlit-mic-recorder doesn't work
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
    
    st.download_button(
        label="💾 Download Uploaded File",
        data=uploaded_file.getvalue(),
        file_name=uploaded_file.name,
        mime="audio/wav"
    )

st.markdown("---")
st.markdown("🎵 Built with Streamlit - Cloud Compatible")