import streamlit as st
from dotenv import load_dotenv
from config.settings import get_api_key, MODEL, GEMINI_CONFIG
from ui.sidebar import render_sidebar
from ui.recorder import handle_recording_ui
from services.gemini import run_gemini_processing_sync

load_dotenv()
st.set_page_config(page_title="Voice Recorder with Gemini AI", page_icon="🎤", layout="centered")
st.title("🎤 Voice Recorder with Gemini AI")

# Render Sidebar & get API key
api_key = render_sidebar()
if not api_key:
    st.stop()

# Handle recording UI and return audio
audio = handle_recording_ui()

# Process audio
if audio and st.button("🧠 Analyze with Gemini AI", use_container_width=True):
    with st.spinner("Processing with Gemini..."):
        result = run_gemini_processing_sync(audio["bytes"], api_key, MODEL, GEMINI_CONFIG)
        if result:
            st.audio(result, format="audio/wav")
            st.download_button("💾 Download AI Response", data=result, file_name="gemini_response.wav", mime="audio/wav")
